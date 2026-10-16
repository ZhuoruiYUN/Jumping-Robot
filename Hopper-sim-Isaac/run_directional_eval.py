"""Focused nominal/robust tests for an all-direction two-circle policy."""
from __future__ import annotations

import argparse
import datetime
import json
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    # Uniform signed [-180, +180] turns without special point masses.
    'uniform_turn180': dict(max_turn=180.0, zero_probability=0.0, reverse_probability=0.0),
    # Every queued target is exactly the current landing point.
    'zero_only': dict(max_turn=180.0, zero_probability=1.0, reverse_probability=0.0),
    # Every nonzero target reverses heading in either 150--180 degree band.
    'reverse_only': dict(max_turn=180.0, zero_probability=0.0, reverse_probability=1.0),
}


def stage_results(root: Path, conditions: tuple[str, ...]) -> dict[str, object]:
    return {
        name: {
            key: value
            for key, value in json.loads((root / name / 'result.json').read_text()).items()
            if not key.endswith('_contract')
        }
        for name in conditions
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--scenario', choices=(*SCENARIOS, 'all'), default='all')
    parser.add_argument('--condition', choices=('nominal', 'robust', 'both'), default='both',
                        help='Run nominal, robust, or both conditions (default: both).')
    parser.add_argument('--robust-profile', choices=('historical', 'checkpoint'),
                        default='historical',
                        help='Historical stress envelope or the checkpoint training disturbance distribution.')
    parser.add_argument('--visualize', action='store_true',
                        help='Open the single-env nominal scenario in Isaac Sim; requires one scenario and --condition nominal.')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--episode-mode', choices=('continuous', 'two_hop'),
                        default='continuous',
                        help='Continuous route or V38 two-attempt episode with terminal settle check.')
    parser.add_argument('--episode-length-s', type=float,
                        help='Timeout in seconds (default: 301 continuous, 12 two-hop).')
    parser.add_argument('--nominal-seed', type=int, default=181,
                        help='Nominal evaluation seed (default: 181, preserving earlier runs).')
    parser.add_argument('--robust-seed', type=int, default=183,
                        help='Robust evaluation seed (default: 183, preserving earlier runs).')
    parser.add_argument('--short-distance-min', type=float)
    parser.add_argument('--short-distance-max', type=float)
    parser.add_argument('--long-distance-min', type=float)
    parser.add_argument('--long-distance-max', type=float)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--failure-trace', action='store_true',
                        help='Record first-death traces without changing vector env count.')
    parser.add_argument('--takeoff-velocity-audit', action='store_true',
                        help='Record all-hop liftoff velocity and P1 landing audit; requires --failure-trace.')
    parser.add_argument('--contact-phase-audit', action='store_true',
                        help='Add touchdown and stance proxies to all-hop audit; requires --takeoff-velocity-audit.')
    parser.add_argument('--closed-loop-p2-env-id', type=int,
                        help='Selected vector environment for a one-touchdown closed-loop P2 branch.')
    parser.add_argument('--closed-loop-p2-step', type=int,
                        help='Exact touchdown control step for the P2 branch.')
    parser.add_argument('--closed-loop-p2-mode', choices=('original', 'left90', 'right90', 'reverse'))
    parser.add_argument('--closed-loop-p2-expected-sha256',
                        help='Require the same pre-branch physical/policy state as the original arm.')
    parser.add_argument('--failure-trace-window-s', type=float, default=0.5,
                        help='Seconds before/after first death to retain (0 < value <= 8; default: 0.5).')
    parser.add_argument('--next-hop-probe-env-ids',
                        help='Read-only next-target actor probe, comma-separated vector environment IDs.')
    parser.add_argument('--next-hop-probe-stride', type=int, default=10)
    parser.add_argument('--route-recovery-mode', choices=('advance', 'retry_large'),
                        help='Frozen-policy V3 post-miss route A/B, with separate first/retry metrics.')
    parser.add_argument('--route-recovery-severe-miss-m', type=float, default=0.20,
                        help='Foot miss threshold for retry_large; recorded in both A/B arms.')
    parser.add_argument('--diagnostic-contact-floor-pwm', type=float,
                        help='Simulation-only frozen-policy contact collective floor; use 0 for the matched control.')
    parser.add_argument('--diagnostic-ascent-floor-pwm', type=float,
                        help='Simulation-only frozen-policy ascent collective floor; use 0 for the matched control.')
    parser.add_argument('--diagnostic-ascent-pulse-pwm', type=float,
                        help='Evaluation-only 0.20 s additive ascent pulse; use 0 for the matched control.')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Missing checkpoint: {checkpoint}')
    if args.steps <= 0:
        parser.error('--steps must be positive')
    episode_length_s = args.episode_length_s
    if episode_length_s is None:
        episode_length_s = 12.0 if args.episode_mode == 'two_hop' else 301.0
    if episode_length_s <= 0:
        parser.error('--episode-length-s must be positive')
    if args.episode_mode == 'two_hop' and args.scenario != 'all':
        parser.error('--episode-mode two_hop requires --scenario all (fixed V38 scenario mix)')
    if args.visualize and (args.condition != 'nominal' or args.scenario == 'all'):
        parser.error('--visualize requires --condition nominal and one --scenario')
    if not 0 < args.failure_trace_window_s <= 8:
        parser.error('--failure-trace-window-s must be in (0, 8]')
    if not 0.10 < args.route_recovery_severe_miss_m <= 0.50:
        parser.error('--route-recovery-severe-miss-m must be in (0.10, 0.50]')
    if args.failure_trace_window_s != 0.5 and not args.failure_trace:
        parser.error('--failure-trace-window-s requires --failure-trace')
    if args.takeoff_velocity_audit and not args.failure_trace:
        parser.error('--takeoff-velocity-audit requires --failure-trace')
    if args.contact_phase_audit and not args.takeoff_velocity_audit:
        parser.error('--contact-phase-audit requires --takeoff-velocity-audit')
    if args.closed_loop_p2_mode is not None:
        if args.closed_loop_p2_env_id is None or args.closed_loop_p2_step is None:
            parser.error('--closed-loop-p2-mode requires --closed-loop-p2-env-id and --closed-loop-p2-step')
        if not args.failure_trace or not args.takeoff_velocity_audit:
            parser.error('--closed-loop-p2-mode requires --failure-trace --takeoff-velocity-audit')
        if args.condition != 'robust' or args.scenario != 'uniform_turn180':
            parser.error('--closed-loop-p2-mode requires --condition robust --scenario uniform_turn180')
        if not 0 <= args.closed_loop_p2_env_id < 16 or not 1 <= args.closed_loop_p2_step < args.steps:
            parser.error('P2 branch env must be in [0,15], step must be within evaluation')
    elif any(value is not None for value in (args.closed_loop_p2_env_id,
                                             args.closed_loop_p2_step,
                                             args.closed_loop_p2_expected_sha256)):
        parser.error('P2 branch options require --closed-loop-p2-mode')
    if args.nominal_seed < 0 or args.robust_seed < 0:
        parser.error('evaluation seeds must be non-negative')
    if args.next_hop_probe_stride < 1:
        parser.error('--next-hop-probe-stride must be positive')
    if args.next_hop_probe_env_ids:
        try:
            probe_ids = [int(part) for part in args.next_hop_probe_env_ids.split(',')]
        except ValueError:
            parser.error('--next-hop-probe-env-ids must contain comma-separated integers')
        if not probe_ids or len(set(probe_ids)) != len(probe_ids) or any(
                env_id < 0 or env_id >= 16 for env_id in probe_ids):
            parser.error('--next-hop-probe-env-ids must be distinct IDs in [0, 15]')
        if args.condition != 'robust' and any(env_id != 0 for env_id in probe_ids):
            parser.error('nominal probe has only environment 0; use --condition robust')
    if (args.diagnostic_contact_floor_pwm is not None
            and not 0.0 <= args.diagnostic_contact_floor_pwm <= 1000.0):
        parser.error('--diagnostic-contact-floor-pwm must be in [0, 1000]')
    if (args.diagnostic_ascent_floor_pwm is not None
            and not 0.0 <= args.diagnostic_ascent_floor_pwm <= 1000.0):
        parser.error('--diagnostic-ascent-floor-pwm must be in [0, 1000]')
    if (args.diagnostic_contact_floor_pwm is not None
            and args.diagnostic_ascent_floor_pwm is not None):
        parser.error('select only one diagnostic phase floor')
    if args.diagnostic_ascent_pulse_pwm is not None and (
            not 0.0 <= args.diagnostic_ascent_pulse_pwm <= 100.0
            or args.diagnostic_contact_floor_pwm is not None
            or args.diagnostic_ascent_floor_pwm is not None
            or args.route_recovery_mode is not None):
        parser.error('--diagnostic-ascent-pulse-pwm must be in [0, 100] and used alone')
    manifest_path = checkpoint.parent / 'manifest.json'
    if not manifest_path.is_file():
        parser.error(f'Missing checkpoint manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if args.robust_profile == 'checkpoint':
        required = (
            'hardware_randomization_probability', 'observation_noise_std',
            'initial_yaw_perturb_deg', 'initial_angular_velocity_perturb_rad_s',
            'comprehensive_global_thrust_scale_min',
            'comprehensive_global_thrust_scale_max',
            'comprehensive_com_offset_x_max_mm',
            'comprehensive_com_offset_y_max_mm',
            'comprehensive_com_offset_z_max_mm',
            'comprehensive_yaw_effectiveness_min',
            'comprehensive_yaw_effectiveness_max',
            'comprehensive_yaw_bias_torque_max_mnm',
        )
        training_cli = manifest.get('cli') or {}
        missing = [name for name in required if training_cli.get(name) is None]
        if missing or not manifest.get('comprehensive_hardware_randomization'):
            parser.error(f'Checkpoint lacks complete hardware profile: {missing}')
    if args.route_recovery_mode and not manifest.get('v3_accounting_fix'):
        parser.error('--route-recovery-mode requires a V3 accounting-fix checkpoint')
    if manifest.get('stability_profile') != 'two_circle_v18':
        parser.error('Expected a two-circle v18/v19/V20 checkpoint')
    checkpoint_route = manifest['two_circle_route']
    override_values = (
        args.short_distance_min, args.short_distance_max,
        args.long_distance_min, args.long_distance_max,
    )
    if any(value is not None for value in override_values):
        if any(value is None for value in override_values):
            parser.error('all four short/long distance override values are required together')
        if not (
            0.0 <= args.short_distance_min <= args.short_distance_max
            and 0.0 <= args.long_distance_min <= args.long_distance_max
            and args.short_distance_max > 0.0
            and args.long_distance_max > 0.0
        ):
            parser.error('distance override ranges must satisfy 0 <= min <= max and max > 0')
        route = dict(checkpoint_route)
        route['short_hop_radius_m'] = [args.short_distance_min, args.short_distance_max]
        route['long_hop_radius_m'] = [args.long_distance_min, args.long_distance_max]
        route_source = 'command_line_override'
    else:
        route = checkpoint_route
        route_source = 'checkpoint_manifest'
    short, long = route['short_hop_radius_m'], route['long_hop_radius_m']
    layout = manifest.get('observation_layout', 'spatial52')
    if args.closed_loop_p2_mode is not None and layout != 'history_foot_v19':
        parser.error('--closed-loop-p2-mode requires history_foot_v19 checkpoint')
    if args.next_hop_probe_env_ids and layout != 'history_foot_v19':
        parser.error('--next-hop-probe-env-ids requires history_foot_v19 checkpoint')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = args.root or ROOT / 'outputs/tracking_recovery' / f'{checkpoint.parent.name}_directional_eval_{stamp}'
    root = root.resolve()
    if root.exists():
        parser.error(f'Refusing to overwrite: {root}')
    selected = (('two_hop_mixture',) if args.episode_mode == 'two_hop'
                else tuple(SCENARIOS) if args.scenario == 'all' else (args.scenario,))
    conditions = ('nominal', 'robust') if args.condition == 'both' else (args.condition,)

    common = [
        sys.executable, str(ROOT / 'run_tracking_recovery.py'), 'eval',
        '--checkpoint', str(checkpoint), '--steps', str(args.steps),
        '--episode-length-s', str(episode_length_s),
        '--successful-hops-per-episode', '10000', '--hard-sample-probability', '0',
        '--actuator-mode', 'parity', '--action-parameterization', 'direct_motor',
        '--stability-profile', 'two_circle_v18', '--observation-layout', layout,
        '--distance-min', str(min(short[0], long[0])), '--distance-max', str(max(short[1], long[1])),
        '--short-distance-min', str(short[0]), '--short-distance-max', str(short[1]),
        '--long-distance-min', str(long[0]), '--long-distance-max', str(long[1]),
        '--reverse-turn-halfwidth-deg', str(manifest['reverse_turn_halfwidth_deg']),
        '--target-height', str(manifest['policy_target_root_z_m']),
        '--target-tolerance', str(manifest['target_tolerance_m']),
    ]
    # Load the CLI-only task contract by file path.  Importing it through the
    # package would execute Quadhopper_Planner_Random/__init__.py, which imports
    # gymnasium even though this outer launcher intentionally runs with the
    # system Python and delegates Isaac work to Isaac Sim's python.sh.
    stage2_contract = runpy.run_path(
        str(ROOT / 'Quadhopper_Planner_Random/path_free_stage2.py'))
    common += stage2_contract['manifest_flags'](manifest)
    if args.episode_mode == 'two_hop':
        if not manifest.get('sysid_20261011_plant') or not manifest.get('foot_pair_v33'):
            parser.error('Two-hop evaluation requires a V38 SysID foot-pair checkpoint')
        common.append('--two-hop-episode-v38')
    if manifest.get('two_hop_episode_v38') or args.episode_mode == 'two_hop':
        # V38 changes training episode termination, while directional eval
        # can measure the same actor on the other task. The V38 hardware ramp
        # also advances the contract version without changing actor inputs.
        common.append('--allow-compatible-checkpoint')
    if 'height_progress_reward_scale' in manifest:
        common += ['--height-progress-reward-scale',
                   str(manifest['height_progress_reward_scale'])]
    if 'v30_liftoff_cross_track' in manifest:
        common += ['--liftoff-cross-track-penalty-scale',
                   str(manifest['v30_liftoff_cross_track']['penalty_scale'])]
    if args.diagnostic_contact_floor_pwm is not None or args.diagnostic_ascent_floor_pwm is not None:
        common += [
            '--simulation-only-phase-collective-baseline',
            '--phase-contact-collective-floor-pwm',
            str(args.diagnostic_contact_floor_pwm if args.diagnostic_contact_floor_pwm is not None else 0),
            '--phase-ascent-collective-floor-pwm',
            str(args.diagnostic_ascent_floor_pwm if args.diagnostic_ascent_floor_pwm is not None else 0),
        ]
    if args.diagnostic_ascent_pulse_pwm is not None:
        common += ['--diagnostic-ascent-pulse-pwm', str(args.diagnostic_ascent_pulse_pwm)]
    if args.route_recovery_mode:
        common += ['--route-recovery-mode', args.route_recovery_mode,
                   '--route-recovery-severe-miss-m', str(args.route_recovery_severe_miss_m)]
    if args.failure_trace:
        common += ['--failure-trace', '--failure-trace-window-s',
                   str(args.failure_trace_window_s)]
    if args.takeoff_velocity_audit:
        common.append('--takeoff-velocity-audit')
    if args.contact_phase_audit:
        common.append('--contact-phase-audit')
    if args.closed_loop_p2_mode is not None:
        common += ['--closed-loop-p2-env-id', str(args.closed_loop_p2_env_id),
                   '--closed-loop-p2-step', str(args.closed_loop_p2_step),
                   '--closed-loop-p2-mode', args.closed_loop_p2_mode]
        if args.closed_loop_p2_expected_sha256 is not None:
            common += ['--closed-loop-p2-expected-sha256', args.closed_loop_p2_expected_sha256]
    if args.next_hop_probe_env_ids:
        common += ['--next-hop-probe-env-ids', args.next_hop_probe_env_ids,
                   '--next-hop-probe-stride', str(args.next_hop_probe_stride)]
    guidance = manifest.get('guidance_planner')
    if guidance and guidance.get('type') == 'goal_anchored_raibert_foothold_plus_ballistic_hermite':
        common += [
            '--raibert-hermite-guidance',
            '--raibert-stance-time-s', str(guidance['stance_time_s']),
            '--raibert-velocity-gain-s', str(guidance['velocity_error_gain_s']),
            '--raibert-max-foothold-shift-m', str(guidance['max_foothold_shift_m']),
            '--raibert-preview-blend',
            str(guidance['touchdown_next_hop_preview_blend']),
            '--raibert-max-horizontal-speed-m-s',
            str(guidance['max_horizontal_speed_m_s']),
        ]
    if args.dry_run:
        common.append('--dry-run')
    if args.visualize:
        common.append('--visualize')

    summaries: dict[str, object] = {}
    for name in selected:
        scenario = (SCENARIOS[name] if name in SCENARIOS else
                    dict(max_turn=180.0, zero_probability=0.12,
                         reverse_probability=0.65,
                         sampling='V38 fixed 50% uniform, 25% reverse, 25% zero patterns'))
        scenario_root = root / name
        scenario_args = [
            '--max-turn-angle-deg', str(scenario['max_turn']),
            '--zero-hop-probability', str(scenario['zero_probability']),
            '--reverse-turn-probability', str(scenario['reverse_probability']),
        ]
        nominal = common + scenario_args + [
            '--root', str(scenario_root / 'nominal'), '--num-envs', '1',
            '--seed', str(args.nominal_seed),
            '--observation-noise-std', '0', '--initial-yaw-perturb-deg', '0',
            '--initial-angular-velocity-perturb-rad-s', '0',
        ]
        stress_profile = dict(
            observation_noise_std=.002,
            initial_yaw_perturb_deg=30,
            initial_angular_velocity_perturb_rad_s=.15,
            hardware_randomization_probability=1,
            comprehensive_global_thrust_scale_min=.90,
            comprehensive_global_thrust_scale_max=1.04,
            comprehensive_com_offset_x_max_mm=2,
            comprehensive_com_offset_y_max_mm=2,
            comprehensive_com_offset_z_max_mm=1,
            comprehensive_yaw_effectiveness_min=.80,
            comprehensive_yaw_effectiveness_max=1.20,
            comprehensive_yaw_bias_torque_max_mnm=4,
        )
        robust_profile = (training_cli if args.robust_profile == 'checkpoint'
                          else stress_profile)
        robust = common + scenario_args + [
            '--root', str(scenario_root / 'robust'), '--num-envs', '16',
            '--seed', str(args.robust_seed),
            '--comprehensive-hardware-randomization',
        ]
        for key in stress_profile:
            robust += ['--' + key.replace('_', '-'), str(robust_profile[key])]
        if 'nominal' in conditions:
            print(f'[DIRECTIONAL-EVAL] scenario={name} nominal', flush=True)
            subprocess.run(nominal, cwd=ROOT, check=True)
        if 'robust' in conditions:
            print(f'[DIRECTIONAL-EVAL] scenario={name} robust', flush=True)
            subprocess.run(robust, cwd=ROOT, check=True)
        if not args.dry_run:
            summaries[name] = {
                'definition': scenario,
                'results': stage_results(scenario_root, conditions),
            }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        summary = dict(
            checkpoint=str(checkpoint),
            route=route,
            checkpoint_route=checkpoint_route,
            route_source=route_source,
            observation_layout=layout,
            conditions=conditions,
            robust_profile=args.robust_profile,
            seeds=dict(nominal=args.nominal_seed, robust=args.robust_seed),
            steps=args.steps,
            episode_mode=args.episode_mode,
            episode_length_s=episode_length_s,
            guidance_planner=guidance,
            reverse_turn_halfwidth_deg=manifest['reverse_turn_halfwidth_deg'],
            route_recovery_mode=args.route_recovery_mode,
            route_recovery_severe_miss_m=(
                args.route_recovery_severe_miss_m if args.route_recovery_mode else None),
            diagnostic_contact_floor_pwm=args.diagnostic_contact_floor_pwm,
            diagnostic_ascent_floor_pwm=args.diagnostic_ascent_floor_pwm,
            diagnostic_ascent_pulse_pwm=args.diagnostic_ascent_pulse_pwm,
            scenarios=summaries,
        )
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'[DIRECTIONAL-EVAL] {root / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
