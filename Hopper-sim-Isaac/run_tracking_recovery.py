"""Launch continuous spatial-guidance PPO; retains the recovery entry name."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1/model_119.pt'
parser = argparse.ArgumentParser()
parser.add_argument('mode', nargs='?', choices=('train', 'eval'), default='train')
parser.add_argument('--root', type=Path, default=ROOT / 'outputs/tracking_recovery/working_run')
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=40)
parser.add_argument('--distance-min', type=float, default=0.20)
parser.add_argument('--distance-max', type=float, default=0.30)
parser.add_argument('--short-distance-min', type=float,
                    help=('Optional short-hop radius lower bound. Supply all four '
                          'short/long distance flags to enable an alternating two-circle route.'))
parser.add_argument('--short-distance-max', type=float)
parser.add_argument('--long-distance-min', type=float)
parser.add_argument('--long-distance-max', type=float)
parser.add_argument('--distance-curriculum-iterations', type=float, default=0.0,
                    help='PPO updates over which two-circle radius ranges expand from explicit start ranges.')
parser.add_argument('--distance-curriculum-start-short-min', type=float)
parser.add_argument('--distance-curriculum-start-short-max', type=float)
parser.add_argument('--distance-curriculum-start-long-min', type=float)
parser.add_argument('--distance-curriculum-start-long-max', type=float)
parser.add_argument('--distance-curriculum-start-turn-angle-deg', type=float,
                    help='Initial turn limit used by the distance curriculum; set equal to the final limit to vary only distance.')
parser.add_argument('--hard-distance-min', type=float, default=0.25)
parser.add_argument('--hard-sample-probability', type=float,
                    help='Defaults to 0.75 for training and 0 for uniform evaluation.')
parser.add_argument('--max-turn-angle-deg', type=float, default=30.0,
                    help='Maximum signed heading change between consecutive waypoints.')
parser.add_argument('--zero-hop-probability', type=float, default=0.0,
                    help='Probability that an appended waypoint is exactly at the current landing point.')
parser.add_argument('--reverse-turn-probability', type=float, default=0.0,
                    help='Probability of a heading reversal around 180 degrees after a hop.')
parser.add_argument('--reverse-turn-halfwidth-deg', type=float, default=30.0,
                    help='Half-width of the reversal band around +/- 180 degrees.')
parser.add_argument('--fixed-initial-heading-deg', type=float,
                    help='Evaluation diagnostic only: initial world heading; 0 gives +X.')
parser.add_argument('--direction-curriculum-iterations', type=float, default=0.0)
parser.add_argument('--direction-curriculum-start-turn-angle-deg', type=float)
parser.add_argument('--direction-curriculum-start-zero-hop-probability', type=float)
parser.add_argument('--landing-precision-width', type=float)
parser.add_argument('--stage2-foot-precision-scale', type=float,
                    help='V3 foot touchdown Gaussian reward scale; default 60.')
parser.add_argument('--stage2-foot-precision-width-m', type=float,
                    help='V3 foot touchdown Gaussian width in metres; default 0.06.')
parser.add_argument('--stage2-foot-hit-10cm-scale', type=float,
                    help='Opt-in per-hop strict-apex foot hit bonus below 10 cm; default 0.')
parser.add_argument('--pair-xy-quality-weight-v36', type=float,
                    help='Within strict V33 foot pairs, weight smooth two-foot XY quality in [0, 1].')
parser.add_argument('--sysid-20261011-plant', action='store_true',
                    help='Use the 2026-10-11 mounted SysID spring-leg proxy plant.')
parser.add_argument('--target-hit-reward-scale', type=float)
parser.add_argument('--landing-precision-reward-scale', type=float)
parser.add_argument('--landing-error-penalty-scale', type=float)
parser.add_argument('--landing-lateral-error-penalty-scale', type=float)
parser.add_argument('--robust-dynamics', action='store_true',
                    help='Randomize mass, inertia, action delay, motor bandwidth, and per-motor thrust.')
parser.add_argument('--hardware-robust', action='store_true',
                    help='Use the measured low-thrust/slow-motor sim-to-real envelope.')
parser.add_argument('--comprehensive-hardware-randomization', action='store_true',
                    help=('Also randomize common thrust, roll/pitch effectiveness cross-coupling, '
                          'and spring stiffness/damping; implies --hardware-robust.'))
parser.add_argument('--hardware-randomization-probability', type=float, default=1.0,
                    help='Fraction of reset environments receiving hardware perturbations.')
parser.add_argument('--hardware-randomization-probability-start', type=float,
                    help='Initial perturbation fraction for a nominal-to-robust curriculum.')
parser.add_argument('--hardware-randomization-curriculum-iterations', type=float,
                    help='PPO iterations over which hardware perturbation fraction reaches its final value.')
parser.add_argument('--comprehensive-global-thrust-scale-min', type=float)
parser.add_argument('--comprehensive-global-thrust-scale-max', type=float)
parser.add_argument('--comprehensive-com-offset-x-max-mm', type=float)
parser.add_argument('--comprehensive-com-offset-y-max-mm', type=float)
parser.add_argument('--comprehensive-com-offset-z-max-mm', type=float)
parser.add_argument('--comprehensive-yaw-effectiveness-min', type=float)
parser.add_argument('--comprehensive-yaw-effectiveness-max', type=float)
parser.add_argument('--comprehensive-yaw-bias-torque-max-mnm', type=float)
parser.add_argument('--observation-noise-std', type=float)
parser.add_argument('--initial-yaw-perturb-deg', type=float)
parser.add_argument('--initial-angular-velocity-perturb-rad-s', type=float)
parser.add_argument('--touchdown-wrench-probability', type=float)
parser.add_argument('--touchdown-roll-wrench-max-nm', type=float)
parser.add_argument('--touchdown-pitch-wrench-max-nm', type=float)
parser.add_argument('--touchdown-yaw-wrench-max-nm', type=float)
parser.add_argument('--touchdown-wrench-window-s', type=float)
parser.add_argument('--touchdown-wrench-persistence', type=float)
parser.add_argument('--touchdown-wrench-on-moving-long-hop', action='store_true')
parser.add_argument('--moving-long-recovery-window-s', type=float)
parser.add_argument('--moving-long-recovery-angle-deadband-deg', type=float)
parser.add_argument('--moving-long-recovery-rate-deadband-rad-s', type=float)
parser.add_argument('--moving-long-recovery-angle-penalty-scale', type=float)
parser.add_argument('--moving-long-recovery-rate-penalty-scale', type=float)
parser.add_argument('--airborne-attitude-penalty-scale', type=float)
parser.add_argument('--airborne-angvel-xy-penalty-scale', type=float)
parser.add_argument('--airborne-yaw-penalty-scale', type=float)
parser.add_argument('--airborne-yaw-rate-penalty-scale', type=float)
parser.add_argument('--airborne-action-spread-penalty-scale', type=float)
parser.add_argument('--airborne-cross-track-tilt-penalty-scale', type=float)
parser.add_argument('--airborne-cross-track-tilt-penalty-start', type=float)
parser.add_argument('--airborne-cross-track-velocity-penalty-scale', type=float)
parser.add_argument('--airborne-cross-track-velocity-penalty-start-mps', type=float)
parser.add_argument('--airborne-tilt-penalty-start', type=float)
parser.add_argument('--airborne-angvel-xy-penalty-start', type=float)
parser.add_argument('--airborne-yaw-deadband-deg', type=float)
parser.add_argument('--airborne-yaw-rate-penalty-start', type=float)
parser.add_argument('--airborne-action-spread-penalty-start', type=float)
parser.add_argument('--stationary-hold-radius', type=float)
parser.add_argument('--stationary-tilt-penalty-scale', type=float)
parser.add_argument('--stationary-yaw-rate-penalty-scale', type=float)
parser.add_argument('--stationary-action-spread-penalty-scale', type=float)
parser.add_argument('--stationary-pitch-deadband-deg', type=float)
parser.add_argument('--stationary-pitch-penalty-scale', type=float)
parser.add_argument('--stationary-pitch-rate-deadband-rad-s', type=float)
parser.add_argument('--stationary-pitch-rate-penalty-scale', type=float)
parser.add_argument('--stationary-airborne-differential-action-rate-deadband', type=float)
parser.add_argument('--stationary-airborne-differential-action-rate-penalty-scale', type=float)
parser.add_argument('--stationary-contact-recovery-window-s', type=float)
parser.add_argument('--stationary-contact-recovery-spread-deadband', type=float)
parser.add_argument('--stationary-contact-recovery-spread-penalty-scale', type=float)
parser.add_argument('--stationary-contact-recovery-delta-deadband', type=float)
parser.add_argument('--stationary-contact-recovery-delta-penalty-scale', type=float)
parser.add_argument('--stationary-apex-width', type=float)
parser.add_argument('--stationary-apex-reward-scale', type=float)
parser.add_argument('--stationary-apex-error-penalty-scale', type=float)
parser.add_argument('--stationary-apex-shortfall-penalty-scale', type=float)
parser.add_argument('--stationary-apex-progress-scale', type=float)
parser.add_argument('--stationary-apex-progress-width', type=float)
parser.add_argument('--height-progress-reward-scale', type=float,
                    help='Measured symmetric apex-progress reward per metre; baseline is 3.')
parser.add_argument('--stationary-ascent-support-scale', type=float)
parser.add_argument('--stationary-ascent-collective-scale', type=float)
parser.add_argument('--stationary-predicted-apex-width', type=float)
parser.add_argument('--stationary-predicted-apex-reward-scale', type=float)
parser.add_argument('--ascent-predicted-apex-width', type=float)
parser.add_argument('--ascent-predicted-apex-reward-scale', type=float)
parser.add_argument('--apex-event-height-width', type=float)
parser.add_argument('--apex-event-height-reward-scale', type=float)
parser.add_argument('--apex-event-shortfall-penalty-scale', type=float)
parser.add_argument('--simulation-only-phase-collective-baseline', action='store_true')
parser.add_argument('--phase-contact-collective-floor-pwm', type=float)
parser.add_argument('--phase-ascent-collective-floor-pwm', type=float)
parser.add_argument('--diagnostic-ascent-pulse-pwm', type=float)
parser.add_argument('--simulation-only-phase-collective-controller', action='store_true')
parser.add_argument('--phase-contact-collective-pwm', type=float)
parser.add_argument('--phase-ascent-collective-pwm', type=float)
parser.add_argument('--phase-descent-collective-pwm', type=float)
parser.add_argument('--phase-contact-differential-limit-pwm', type=float)
parser.add_argument('--phase-ascent-differential-limit-pwm', type=float)
parser.add_argument('--phase-descent-differential-limit-pwm', type=float)
parser.add_argument('--simulation-only-apex-bias-controller', action='store_true')
parser.add_argument('--apex-bias-initial-pwm', type=float)
parser.add_argument('--apex-bias-gain-pwm-per-m', type=float)
parser.add_argument('--apex-bias-max-pwm', type=float)
parser.add_argument('--apex-bias-feedback', choices=('apex', 'liftoff_vz'), default='apex')
parser.add_argument('--apex-bias-liftoff-vz-target', type=float)
parser.add_argument('--learning-rate', type=float, default=5e-6)
parser.add_argument('--target-tolerance', type=float, default=0.05)
parser.add_argument('--apex-tolerance', type=float,
                    help='Apex tolerance used by target-relative height gates.')
parser.add_argument('--target-height', type=float, default=1.000,
                    help='Absolute simulation/policy root-Z apex command.')
parser.add_argument('--raibert-hermite-guidance', action='store_true',
                    help='Enable bounded Raibert footholds and ballistic Hermite tangents.')
parser.add_argument('--path-free-stage2', action='store_true',
                    help='Endpoint/energy refinement with zero curve inputs and no curve rewards.')
parser.add_argument('--path-free-stage2-v2', action='store_true',
                    help='Foot progress and staged nominal/robust/energy refinement.')
parser.add_argument('--path-free-stage2-v3', action='store_true',
                    help='Strict-apex success refinement without added energy cost.')
parser.add_argument('--raibert-stance-time-s', type=float, default=0.045)
parser.add_argument('--raibert-velocity-gain-s', type=float, default=0.08)
parser.add_argument('--raibert-max-foothold-shift-m', type=float, default=0.04)
parser.add_argument('--raibert-preview-blend', type=float, default=0.5)
parser.add_argument('--raibert-max-horizontal-speed-m-s', type=float, default=2.5)
parser.add_argument('--disable-initial-drop', action='store_true',
                    help='Reset on the ground instead of training-only random aerial drops.')
parser.add_argument('--episode-length-s', type=float,
                    help='Override episode duration; use a value longer than the eval horizon for continuous playback.')
parser.add_argument('--successful-hops-per-episode', type=int,
                    help='Override success-reset hop count; continuous eval must exceed the expected number of hops.')
parser.add_argument('--play-motor-time-constant', type=float,
                    help='Fixed nominal motor time constant for deterministic plant identification.')
parser.add_argument('--play-thrust-scale', type=float,
                    help='Fixed nominal per-motor thrust multiplier for plant identification.')
parser.add_argument('--height-curriculum-start', type=float)
parser.add_argument('--height-curriculum-iterations', type=float)
parser.add_argument('--termination-penalty-scale', type=float)
parser.add_argument('--target-relative-apex-gate', action='store_true')
parser.add_argument('--actuator-mode',
                    choices=('parity', 'corrected_history', 'baseline', 'zero_delay_raw_history'),
                    default='parity')
parser.add_argument('--action-parameterization', choices=('direct_motor', 'collective_residual_v1'),
                    default='direct_motor')
parser.add_argument('--observation-layout', choices=('spatial52', 'history_foot_v19'), default='spatial52')
parser.add_argument('--collective-preserving-spread-limit-pwm', type=float, default=0.0,
                    help='Limit only zero-mean four-motor differential; 0 disables the constraint.')
parser.add_argument('--legacy-v11-contract', action='store_true')
parser.add_argument('--stability-profile',
                    choices=('baseline', 'takeoff_v9', 'launch_robust_v10', 'yaw_event_v11',
                             'touchdown_robust_v12', 'height_robust_v13',
                             'cross_hop_v14', 'cross_hop_v15', 'cross_hop_v15_1',
                             'cross_hop_v16', 'stationary_corridor_v17',
                             'two_circle_v18'), default='baseline',
                    help='Opt-in versioned stability training; v10-v18 preserve direct four-motor control.')
parser.add_argument('--allow-compatible-checkpoint', action='store_true')
parser.add_argument('--resume-runner-state', action='store_true',
                    help='Resume PPO optimizer and iteration counter from --checkpoint during training.')
parser.add_argument('--visualize', action='store_true',
                    help='Run an interactive one-environment evaluation with debug visualization.')
parser.add_argument('--control-diagnostics', action='store_true',
                    help='Record policy/RNN and pre/post-step plant state during one-environment evaluation.')
parser.add_argument('--failure-trace', action='store_true',
                    help='Evaluation-only first-death time windows; supports vector environments.')
parser.add_argument('--takeoff-velocity-audit', action='store_true',
                    help='Evaluation-only all-hop kinematics; requires --failure-trace.')
parser.add_argument('--contact-phase-audit', action='store_true',
                    help='Evaluation-only touchdown/stance proxies; requires --takeoff-velocity-audit.')
parser.add_argument('--closed-loop-p2-env-id', type=int)
parser.add_argument('--closed-loop-p2-step', type=int)
parser.add_argument('--closed-loop-p2-mode', choices=('original', 'left90', 'right90', 'reverse'))
parser.add_argument('--closed-loop-p2-expected-sha256')
parser.add_argument('--failure-trace-window-s', type=float, default=0.5,
                    help='Seconds before/after first death to retain (0 < value <= 8).')
parser.add_argument('--next-hop-probe-env-ids',
                    help='Read-only next-target actor probe, comma-separated vector environment IDs.')
parser.add_argument('--next-hop-probe-stride', type=int, default=10)
parser.add_argument('--moving-handoff-v11', action='store_true',
                    help='Opt-in moving short/long contact and early-launch reward shaping.')
parser.add_argument('--landing-recovery-v12', action='store_true',
                    help='Opt-in prelanding safety and near-ground recovery reward extension.')
parser.add_argument('--contact-readiness-v13', action='store_true',
                    help='Opt-in earlier danger forecast and low-altitude handoff reward.')
parser.add_argument('--apex-event-fix-v15', action='store_true',
                    help='Opt-in v12 apex velocity snapshot and once-per-hop event repair.')
parser.add_argument('--terminal-penalty-fix-v16', action='store_true',
                    help='Opt-in v15 repair aligning terminal reward with physical death.')
parser.add_argument('--v3-accounting-fix', action='store_true',
                    help='V3-only apex event and physical-death reward accounting repair.')
parser.add_argument('--foot-pair-v33', action='store_true',
                    help='Replace root pair bonus with strict foot-pair and touchdown posture quality.')
parser.add_argument('--two-hop-episode-v38', action='store_true',
                    help='Train two attempted hops per reset with scenario buckets and landing settle check.')
parser.add_argument('--moving-launch-rate-v34', action='store_true',
                    help='V33 pair task: penalize moving-hop launch roll/pitch rate during early flight.')
parser.add_argument('--liftoff-cross-track-penalty-scale', type=float,
                    help='V30 V3-accounting event penalty for lateral Body velocity at liftoff; 0 control, negative experiment.')
parser.add_argument('--hardware-severity-curriculum', action='store_true',
                    help='V3 accounting training: hard hardware offsets use 0.5/0.75/1.0 strength.')
parser.add_argument('--hardware-severity-stage-iterations', type=int, default=30,
                    help='PPO updates at each partial hardware-strength stage.')
parser.add_argument('--route-recovery-mode', choices=('advance', 'retry_large'),
                    help='Evaluation-only frozen-policy post-miss route A/B.')
parser.add_argument('--route-recovery-severe-miss-m', type=float, default=0.20,
                    help='Foot miss threshold used by retry_large and recorded for both arms.')
parser.add_argument('--deferred-landing-v17', action='store_true',
                    help='Opt-in v16 experiment deferring half the foot bonus to the next measured apex.')
parser.add_argument('--low-hop-recovery-v18', action='store_true',
                    help='Opt-in v16 measured-height debt and repayment after a low moving hop.')
parser.add_argument('--attitude-recovery-v14', action='store_true',
                    help='Opt-in quaternion risk forecast and attitude-gated recovery release.')
args = parser.parse_args()
if args.route_recovery_mode and (args.mode != 'eval' or not args.v3_accounting_fix):
    parser.error('--route-recovery-mode requires eval and --v3-accounting-fix')
if args.diagnostic_ascent_pulse_pwm is not None and (
    args.mode != 'eval' or not args.v3_accounting_fix or args.route_recovery_mode
    or args.simulation_only_phase_collective_baseline
    or args.simulation_only_phase_collective_controller
):
    parser.error('--diagnostic-ascent-pulse-pwm requires standalone V3 accounting eval without phase floors')
if args.diagnostic_ascent_pulse_pwm is not None and not 0 <= args.diagnostic_ascent_pulse_pwm <= 100:
    parser.error('--diagnostic-ascent-pulse-pwm must be in [0, 100]')
if args.hardware_severity_curriculum and (
    args.mode != 'train' or not args.v3_accounting_fix
    or not args.comprehensive_hardware_randomization or args.resume_runner_state
):
    parser.error('--hardware-severity-curriculum requires train, V3 accounting fix, comprehensive hardware randomization, and weights-only initialization')
if args.hardware_severity_stage_iterations <= 0:
    parser.error('--hardware-severity-stage-iterations must be positive')
if not 0.10 < args.route_recovery_severe_miss_m <= 0.50:
    parser.error('--route-recovery-severe-miss-m must be in (0.10, 0.50]')
if args.v3_accounting_fix and (
    not args.path_free_stage2_v3 or args.moving_handoff_v11 or args.landing_recovery_v12
    or args.contact_readiness_v13 or args.attitude_recovery_v14
    or args.apex_event_fix_v15 or args.terminal_penalty_fix_v16
    or args.deferred_landing_v17 or args.low_hop_recovery_v18
    or args.simulation_only_apex_bias_controller
):
    parser.error('--v3-accounting-fix requires standalone v3 and no other recovery tasks')
if args.foot_pair_v33 and (
    not args.v3_accounting_fix or args.hardware_severity_curriculum
    or args.route_recovery_mode or args.diagnostic_ascent_pulse_pwm is not None
):
    parser.error('--foot-pair-v33 requires standalone V3 accounting without diagnostic task overrides')
if args.two_hop_episode_v38 and (
    not args.foot_pair_v33 or not args.sysid_20261011_plant
    or args.resume_runner_state
):
    parser.error('--two-hop-episode-v38 requires SysID foot-pair weights without runner-state resume')
if args.moving_launch_rate_v34 and not args.foot_pair_v33:
    parser.error('--moving-launch-rate-v34 requires --foot-pair-v33')
if args.sysid_20261011_plant and not (
    args.path_free_stage2_v3 and args.v3_accounting_fix and args.foot_pair_v33
    and args.observation_layout == 'history_foot_v19'
    and args.action_parameterization == 'direct_motor'
):
    parser.error('--sysid-20261011-plant requires V3 accounting, foot pair, history_foot_v19, direct_motor')
if args.stage2_foot_precision_scale is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.stage2_foot_precision_scale)
    or not 0.0 <= args.stage2_foot_precision_scale <= 200.0
):
    parser.error('--stage2-foot-precision-scale requires --foot-pair-v33 and a finite value in [0, 200]')
if args.stage2_foot_precision_width_m is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.stage2_foot_precision_width_m)
    or not 0.0 < args.stage2_foot_precision_width_m <= 0.20
):
    parser.error('--stage2-foot-precision-width-m requires --foot-pair-v33 and a finite value in (0, 0.20]')
if args.stage2_foot_hit_10cm_scale is not None and (
    not args.path_free_stage2_v3 or not args.foot_pair_v33
    or not math.isfinite(args.stage2_foot_hit_10cm_scale)
    or not 0.0 <= args.stage2_foot_hit_10cm_scale <= 60.0
):
    parser.error('--stage2-foot-hit-10cm-scale requires V3 foot pair and a finite value in [0, 60]')
if args.pair_xy_quality_weight_v36 is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.pair_xy_quality_weight_v36)
    or not 0.0 <= args.pair_xy_quality_weight_v36 <= 1.0
):
    parser.error('--pair-xy-quality-weight-v36 requires --foot-pair-v33 and a finite value in [0, 1]')
if args.liftoff_cross_track_penalty_scale is not None and (
    not args.v3_accounting_fix
    or not math.isfinite(args.liftoff_cross_track_penalty_scale)
    or not -20.0 <= args.liftoff_cross_track_penalty_scale <= 0.0
):
    parser.error('--liftoff-cross-track-penalty-scale requires V3 accounting and a finite value in [-20, 0]')
if args.deferred_landing_v17 and args.low_hop_recovery_v18:
    parser.error('--low-hop-recovery-v18 is a separate experiment from --deferred-landing-v17')
if args.low_hop_recovery_v18 and not args.terminal_penalty_fix_v16:
    parser.error('--low-hop-recovery-v18 requires --terminal-penalty-fix-v16')
if args.deferred_landing_v17 and not args.terminal_penalty_fix_v16:
    parser.error('--deferred-landing-v17 requires --terminal-penalty-fix-v16')
if args.terminal_penalty_fix_v16 and not args.apex_event_fix_v15:
    parser.error('--terminal-penalty-fix-v16 requires --apex-event-fix-v15')
if args.apex_event_fix_v15:
    if not args.landing_recovery_v12 or args.contact_readiness_v13 or args.attitude_recovery_v14:
        parser.error('--apex-event-fix-v15 requires v12 and excludes v13/v14')
    if args.simulation_only_apex_bias_controller:
        parser.error('--apex-event-fix-v15 excludes the simulation-only apex controller')
if args.attitude_recovery_v14 and not args.contact_readiness_v13:
    parser.error('--attitude-recovery-v14 requires --contact-readiness-v13')
if args.contact_readiness_v13 and not args.landing_recovery_v12:
    parser.error('--contact-readiness-v13 requires --landing-recovery-v12')
if args.landing_recovery_v12 and not (args.moving_handoff_v11 and args.path_free_stage2_v3):
    parser.error('--landing-recovery-v12 requires --moving-handoff-v11 and --path-free-stage2-v3')
if args.moving_handoff_v11 and not args.path_free_stage2_v3:
    parser.error('--moving-handoff-v11 requires --path-free-stage2-v3')
if args.path_free_stage2_v2:
    args.path_free_stage2 = True
    if args.resume_runner_state:
        parser.error('Stage2 v2 uses weights-only initialization; exact curriculum resume is unsupported')
if args.path_free_stage2_v3:
    args.path_free_stage2 = True
    if args.resume_runner_state:
        parser.error('Stage2 v3 uses weights-only initialization; exact curriculum resume is unsupported')
if args.path_free_stage2 and (args.raibert_hermite_guidance
                            or args.observation_layout != 'history_foot_v19'):
    parser.error('path-free-stage2 requires history_foot_v19 and no Raibert guidance')
if args.observation_layout == 'history_foot_v19' and args.stability_profile != 'two_circle_v18':
    parser.error('history_foot_v19 requires the two_circle_v18 task profile')
if args.raibert_hermite_guidance and args.observation_layout != 'history_foot_v19':
    parser.error('Raibert-Hermite guidance requires --observation-layout history_foot_v19')
if args.raibert_stance_time_s < 0.0 or args.raibert_velocity_gain_s < 0.0:
    parser.error('Raibert stance time and velocity gain must be non-negative')
if not 0.0 <= args.raibert_preview_blend <= 1.0:
    parser.error('raibert-preview-blend must be in [0, 1]')
if (args.raibert_hermite_guidance
        and not 0.0 <= args.raibert_max_foothold_shift_m < args.target_tolerance):
    parser.error('Raibert foothold shift must be non-negative and below target tolerance')
if args.raibert_max_horizontal_speed_m_s <= 0.0:
    parser.error('raibert-max-horizontal-speed-m-s must be positive')
if args.stability_profile != 'baseline' and (
    args.actuator_mode != 'parity' or args.legacy_v11_contract
    or args.action_parameterization != 'direct_motor'
    or args.collective_preserving_spread_limit_pwm != 0.0
):
    parser.error('stability profiles require parity/direct_motor and no extra spread limiter')
if args.comprehensive_hardware_randomization:
    args.hardware_robust = True
if args.hard_sample_probability is None:
    args.hard_sample_probability = 0.75 if args.mode == 'train' else 0.0
root = args.root.resolve()
if args.mode == 'eval' and args.checkpoint is None:
    parser.error('eval requires --checkpoint and a new --root for its result')
if args.visualize and args.mode != 'eval':
    parser.error('--visualize is only supported in eval mode')
if args.control_diagnostics and (args.mode != 'eval' or args.num_envs != 1):
    parser.error('--control-diagnostics requires eval mode with --num-envs 1')
if args.failure_trace and args.mode != 'eval':
    parser.error('--failure-trace requires eval mode')
if args.takeoff_velocity_audit and not args.failure_trace:
    parser.error('--takeoff-velocity-audit requires --failure-trace')
if args.contact_phase_audit and not args.takeoff_velocity_audit:
    parser.error('--contact-phase-audit requires --takeoff-velocity-audit')
if args.closed_loop_p2_mode is not None:
    if args.mode != 'eval' or not args.failure_trace or not args.takeoff_velocity_audit:
        parser.error('--closed-loop-p2-mode requires eval with failure trace and takeoff audit')
    if args.closed_loop_p2_env_id is None or args.closed_loop_p2_step is None:
        parser.error('--closed-loop-p2-mode requires environment ID and step')
    if not 0 <= args.closed_loop_p2_env_id < args.num_envs or not 1 <= args.closed_loop_p2_step < args.steps:
        parser.error('P2 branch environment or step out of range')
elif any(value is not None for value in (args.closed_loop_p2_env_id,
                                         args.closed_loop_p2_step,
                                         args.closed_loop_p2_expected_sha256)):
    parser.error('P2 branch options require --closed-loop-p2-mode')
if args.next_hop_probe_env_ids and args.mode != 'eval':
    parser.error('--next-hop-probe-env-ids requires eval mode')
if args.next_hop_probe_stride < 1:
    parser.error('--next-hop-probe-stride must be positive')
if not 0 < args.failure_trace_window_s <= 8:
    parser.error('--failure-trace-window-s must be in (0, 8]')
if args.failure_trace_window_s != 0.5 and not args.failure_trace:
    parser.error('--failure-trace-window-s requires --failure-trace')
if args.checkpoint is not None:
    SOURCE = args.checkpoint.expanduser().resolve()
if min(args.steps, args.num_envs, args.iterations) <= 0:
    parser.error('steps, num-envs and iterations must be positive')
if not (0 <= args.distance_min <= args.distance_max and args.distance_max > 0.0) or args.target_tolerance <= 0:
    parser.error('Require 0 <= distance-min <= distance-max, distance-max > 0, and positive target-tolerance')
two_circle_range_values = (
    args.short_distance_min, args.short_distance_max,
    args.long_distance_min, args.long_distance_max,
)
custom_two_circle_ranges = any(value is not None for value in two_circle_range_values)
if custom_two_circle_ranges:
    if any(value is None for value in two_circle_range_values):
        parser.error('short-distance-min/max and long-distance-min/max must be supplied together')
    if not (
        0.0 <= args.short_distance_min <= args.short_distance_max
        and 0.0 <= args.long_distance_min <= args.long_distance_max
        and args.short_distance_max > 0.0
        and args.long_distance_max > 0.0
    ):
        parser.error('Each short/long distance range must satisfy 0 <= min <= max and max > 0')
if args.stability_profile == 'two_circle_v18' and not custom_two_circle_ranges:
    parser.error('two_circle_v18 requires explicit short/long distance ranges')
distance_curriculum_values = (
    args.distance_curriculum_start_short_min,
    args.distance_curriculum_start_short_max,
    args.distance_curriculum_start_long_min,
    args.distance_curriculum_start_long_max,
    args.distance_curriculum_start_turn_angle_deg,
)
if args.distance_curriculum_iterations < 0.0:
    parser.error('distance-curriculum-iterations must be non-negative')
if args.distance_curriculum_iterations == 0.0:
    if any(value is not None for value in distance_curriculum_values):
        parser.error('distance curriculum start values require positive distance-curriculum-iterations')
else:
    if args.mode != 'train':
        parser.error('distance curriculum is training-only; evaluate the final radius ranges directly')
    if not custom_two_circle_ranges:
        parser.error('distance curriculum requires explicit final short/long distance ranges')
    if any(value is None for value in distance_curriculum_values):
        parser.error('distance curriculum requires all four start ranges and the start turn angle')
    if not (
        0.0 <= args.distance_curriculum_start_short_min
        <= args.distance_curriculum_start_short_max
        and 0.0 <= args.distance_curriculum_start_long_min
        <= args.distance_curriculum_start_long_max
        and args.distance_curriculum_start_short_max > 0.0
        and args.distance_curriculum_start_long_max > 0.0
    ):
        parser.error('distance curriculum start ranges must satisfy 0 <= min <= max and max > 0')
    if not (0.0 < args.distance_curriculum_start_turn_angle_deg <= args.max_turn_angle_deg):
        parser.error('distance curriculum start turn must lie in (0, max-turn-angle-deg]')
if args.learning_rate <= 0.0:
    parser.error('learning-rate must be positive')
if args.target_height <= 0.38:
    parser.error('target-height must exceed the 0.380 m simulation landing root height')
if args.episode_length_s is not None and args.episode_length_s <= 0.0:
    parser.error('episode-length-s must be positive')
if args.successful_hops_per_episode is not None and args.successful_hops_per_episode <= 0:
    parser.error('successful-hops-per-episode must be positive')
if args.apex_tolerance is not None and not (0.01 <= args.apex_tolerance <= 0.30):
    parser.error('apex-tolerance must be in [0.01, 0.30] metres')
if args.play_motor_time_constant is not None and not (0.01 <= args.play_motor_time_constant <= 0.20):
    parser.error('play-motor-time-constant must be in [0.01, 0.20] seconds')
if args.play_thrust_scale is not None and not (0.50 <= args.play_thrust_scale <= 1.20):
    parser.error('play-thrust-scale must be in [0.50, 1.20]')
if (args.height_curriculum_start is None) != (args.height_curriculum_iterations is None):
    parser.error('height-curriculum-start and height-curriculum-iterations must be supplied together')
if args.height_curriculum_start is not None:
    if not (0.38 < args.height_curriculum_start <= args.target_height):
        parser.error('height-curriculum-start must be in (0.380, target-height]')
    if args.height_curriculum_iterations <= 0.0:
        parser.error('height-curriculum-iterations must be positive')
if not 0.0 <= args.hard_sample_probability <= 1.0:
    parser.error('hard-sample-probability must be in [0, 1]')
if not 0.0 <= args.hardware_randomization_probability <= 1.0:
    parser.error('hardware-randomization-probability must be in [0, 1]')
if ((args.hardware_randomization_probability_start is None)
        != (args.hardware_randomization_curriculum_iterations is None)):
    parser.error(
        'hardware-randomization-probability-start and '
        'hardware-randomization-curriculum-iterations must be supplied together'
    )
if args.hardware_randomization_probability_start is not None:
    if not 0.0 <= args.hardware_randomization_probability_start <= 1.0:
        parser.error('hardware-randomization-probability-start must be in [0, 1]')
    if args.hardware_randomization_curriculum_iterations <= 0.0:
        parser.error('hardware-randomization-curriculum-iterations must be positive')
if (args.comprehensive_global_thrust_scale_min is None) != (
    args.comprehensive_global_thrust_scale_max is None
):
    parser.error(
        'comprehensive-global-thrust-scale-min and '
        'comprehensive-global-thrust-scale-max must be supplied together'
    )
if args.comprehensive_global_thrust_scale_min is not None:
    if not args.comprehensive_hardware_randomization:
        parser.error('common thrust scaling overrides require --comprehensive-hardware-randomization')
    if not (
        0.60 <= args.comprehensive_global_thrust_scale_min
        <= args.comprehensive_global_thrust_scale_max <= 1.20
    ):
        parser.error('comprehensive global thrust scale range must lie in [0.60, 1.20]')
com_offset_values = (
    args.comprehensive_com_offset_x_max_mm,
    args.comprehensive_com_offset_y_max_mm,
    args.comprehensive_com_offset_z_max_mm,
)
if any(value is not None for value in com_offset_values):
    if not args.comprehensive_hardware_randomization:
        parser.error('CoM offset overrides require --comprehensive-hardware-randomization')
    if any(value is None for value in com_offset_values):
        parser.error('all three comprehensive CoM offset maxima must be supplied together')
    if any(not 0.0 <= value <= 30.0 for value in com_offset_values):
        parser.error('comprehensive CoM offset maxima must lie in [0, 30] mm')
if not 0.0 < args.max_turn_angle_deg <= 180.0:
    parser.error('max-turn-angle-deg must be in (0, 180]')
if not 0.0 <= args.zero_hop_probability <= 1.0:
    parser.error('zero-hop-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_probability <= 1.0:
    parser.error('reverse-turn-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_halfwidth_deg <= 180.0:
    parser.error('reverse-turn-halfwidth-deg must be in [0, 180]')
if args.fixed_initial_heading_deg is not None and not math.isfinite(args.fixed_initial_heading_deg):
    parser.error('fixed-initial-heading-deg must be finite')
if args.reverse_turn_probability > 0.0 and (
    args.max_turn_angle_deg < 180.0 - args.reverse_turn_halfwidth_deg
):
    parser.error('max-turn-angle-deg must include the configured reverse-turn band')
if args.direction_curriculum_iterations < 0.0:
    parser.error('direction-curriculum-iterations must be non-negative')
if args.direction_curriculum_iterations == 0.0:
    if (args.direction_curriculum_start_turn_angle_deg is not None
            or args.direction_curriculum_start_zero_hop_probability is not None):
        parser.error('direction curriculum start values require positive direction-curriculum-iterations')
else:
    if args.direction_curriculum_start_turn_angle_deg is None:
        parser.error('direction curriculum requires direction-curriculum-start-turn-angle-deg')
    if not 0.0 < args.direction_curriculum_start_turn_angle_deg <= args.max_turn_angle_deg:
        parser.error('direction curriculum start turn must lie in (0, max-turn-angle-deg]')
    if args.direction_curriculum_start_zero_hop_probability is None:
        args.direction_curriculum_start_zero_hop_probability = 0.0
    if not (0.0 <= args.direction_curriculum_start_zero_hop_probability
            <= args.zero_hop_probability):
        parser.error('direction curriculum zero-hop start must lie in [0, zero-hop-probability]')
if args.landing_precision_width is not None and args.landing_precision_width <= 0.0:
    parser.error('landing-precision-width must be positive')
if args.observation_noise_std is not None and args.observation_noise_std < 0.0:
    parser.error('observation-noise-std must be non-negative')
if args.stationary_hold_radius is not None and args.stationary_hold_radius <= 0.0:
    parser.error('stationary-hold-radius must be positive')
if (args.stationary_pitch_deadband_deg is not None
        and not 0.0 <= args.stationary_pitch_deadband_deg <= 30.0):
    parser.error('stationary-pitch-deadband-deg must be in [0, 30]')
if (args.stationary_pitch_rate_deadband_rad_s is not None
        and not 0.0 <= args.stationary_pitch_rate_deadband_rad_s <= 10.0):
    parser.error('stationary-pitch-rate-deadband-rad-s must be in [0, 10]')
if (args.stationary_airborne_differential_action_rate_deadband is not None
        and not 0.0 <= args.stationary_airborne_differential_action_rate_deadband <= 2.0):
    parser.error('stationary airborne differential action-rate deadband must be in [0, 2]')
for flag, value in (
    ('stationary-pitch-penalty-scale', args.stationary_pitch_penalty_scale),
    ('stationary-pitch-rate-penalty-scale', args.stationary_pitch_rate_penalty_scale),
    ('stationary-airborne-differential-action-rate-penalty-scale',
     args.stationary_airborne_differential_action_rate_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
if (args.stationary_contact_recovery_window_s is not None
        and not 0.0 <= args.stationary_contact_recovery_window_s <= 0.50):
    parser.error('stationary-contact-recovery-window-s must be in [0, 0.50] seconds')
for flag, value in (
    ('stationary-contact-recovery-spread-deadband', args.stationary_contact_recovery_spread_deadband),
    ('stationary-contact-recovery-delta-deadband', args.stationary_contact_recovery_delta_deadband),
):
    if value is not None and not 0.0 <= value <= 1.0:
        parser.error(f'{flag} must be in [0, 1]')
for flag, value in (
    ('stationary-contact-recovery-spread-penalty-scale', args.stationary_contact_recovery_spread_penalty_scale),
    ('stationary-contact-recovery-delta-penalty-scale', args.stationary_contact_recovery_delta_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
if args.stationary_apex_width is not None and args.stationary_apex_width <= 0.0:
    parser.error('stationary-apex-width must be positive')
if args.stationary_apex_progress_width is not None and args.stationary_apex_progress_width <= 0.0:
    parser.error('stationary-apex-progress-width must be positive')
if (args.height_progress_reward_scale is not None
        and not 0.0 <= args.height_progress_reward_scale <= 100.0):
    parser.error('height-progress-reward-scale must be in [0, 100]')
if args.stationary_predicted_apex_width is not None and args.stationary_predicted_apex_width <= 0.0:
    parser.error('stationary-predicted-apex-width must be positive')
if args.ascent_predicted_apex_width is not None and args.ascent_predicted_apex_width <= 0.0:
    parser.error('ascent-predicted-apex-width must be positive')
if (args.stationary_predicted_apex_reward_scale is not None
        and args.stationary_predicted_apex_reward_scale < 0.0):
    parser.error('stationary-predicted-apex-reward-scale must be non-negative')
if (args.ascent_predicted_apex_reward_scale is not None
        and args.ascent_predicted_apex_reward_scale < 0.0):
    parser.error('ascent-predicted-apex-reward-scale must be non-negative')
if args.apex_event_height_width is not None and args.apex_event_height_width <= 0.0:
    parser.error('apex-event-height-width must be positive')
if (args.apex_event_height_reward_scale is not None
        and args.apex_event_height_reward_scale < 0.0):
    parser.error('apex-event-height-reward-scale must be non-negative')
if (args.apex_event_shortfall_penalty_scale is not None
        and args.apex_event_shortfall_penalty_scale > 0.0):
    parser.error('apex-event-shortfall-penalty-scale must be non-positive')
if args.legacy_v11_contract and args.actuator_mode != 'baseline':
    parser.error('--legacy-v11-contract requires --actuator-mode baseline')
if args.action_parameterization == 'collective_residual_v1' and args.actuator_mode != 'baseline':
    parser.error('collective_residual_v1 requires --actuator-mode baseline')
if not 0.0 <= args.collective_preserving_spread_limit_pwm <= 1000.0:
    parser.error('collective-preserving-spread-limit-pwm must be in [0, 1000]')
if args.simulation_only_phase_collective_baseline:
    if args.phase_contact_collective_floor_pwm is None or args.phase_ascent_collective_floor_pwm is None:
        parser.error('simulation-only phase baseline requires both phase collective floor PWM values')
    if not (0.0 <= args.phase_contact_collective_floor_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_floor_pwm <= 1000.0):
        parser.error('phase collective floor PWM values must be in [0, 1000]')
elif (args.phase_contact_collective_floor_pwm is not None
      or args.phase_ascent_collective_floor_pwm is not None):
    parser.error('phase collective floor PWM values require --simulation-only-phase-collective-baseline')
if args.simulation_only_phase_collective_baseline and args.simulation_only_phase_collective_controller:
    parser.error('choose either the phase collective floor or the fixed collective controller, not both')
controller_values = (
    args.phase_contact_collective_pwm,
    args.phase_ascent_collective_pwm,
    args.phase_descent_collective_pwm,
    args.phase_contact_differential_limit_pwm,
    args.phase_ascent_differential_limit_pwm,
    args.phase_descent_differential_limit_pwm,
)
if args.simulation_only_phase_collective_controller:
    if any(value is None for value in controller_values):
        parser.error('phase collective controller requires both collective and both differential-limit PWM values')
    if not (0.0 <= args.phase_contact_collective_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_descent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_contact_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_ascent_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_descent_differential_limit_pwm <= 500.0):
        parser.error('phase controller PWM values must be collective [0,1000], differential limit [0,500]')
elif any(value is not None for value in controller_values):
    parser.error('phase controller PWM values require --simulation-only-phase-collective-controller')
apex_bias_values = (args.apex_bias_initial_pwm, args.apex_bias_gain_pwm_per_m, args.apex_bias_max_pwm)
if args.simulation_only_apex_bias_controller:
    if any(value is None for value in apex_bias_values):
        parser.error('apex bias controller requires initial, gain, and max PWM values')
    if not (0.0 <= args.apex_bias_initial_pwm <= args.apex_bias_max_pwm <= 400.0
            and 0.0 < args.apex_bias_gain_pwm_per_m <= 2000.0):
        parser.error('apex bias requires 0 <= initial <= max <= 400 and gain in (0, 2000]')
    if args.apex_bias_feedback == 'liftoff_vz':
        if args.apex_bias_liftoff_vz_target is None or args.apex_bias_liftoff_vz_target <= 0.0:
            parser.error('liftoff_vz feedback requires a positive --apex-bias-liftoff-vz-target')
    elif args.apex_bias_liftoff_vz_target is not None:
        parser.error('apex-bias-liftoff-vz-target requires --apex-bias-feedback liftoff_vz')
elif any(value is not None for value in apex_bias_values):
    parser.error('apex bias values require --simulation-only-apex-bias-controller')
elif args.apex_bias_liftoff_vz_target is not None:
    parser.error('apex-bias-liftoff-vz-target requires --simulation-only-apex-bias-controller')
if args.hard_sample_probability > 0.0 and not (
    args.distance_min < args.hard_distance_min < args.distance_max
):
    parser.error('hard-distance-min must lie strictly inside the distance range')
if custom_two_circle_ranges and args.hard_sample_probability > 0.0:
    parser.error(
        'Alternating two-circle ranges currently require hard-sample-probability 0; '
        'one hard-distance-min cannot safely split two independent radius ranges'
    )
if args.mode == 'eval' and args.hard_sample_probability != 0.0:
    parser.error('Evaluation must use uniform distances; set hard-sample-probability to 0')
if root.exists() and not args.dry_run:
    raise RuntimeError(f'Output already exists: {root}')
isaac = Path(os.environ.get('ISAAC_SIM_ROOT',
    '/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64')) / 'python.sh'
command = [str(isaac), str(ROOT / 'train_tracking_recovery.py')]
if not args.visualize:
    command.append('--headless')
command += [
           '--checkpoint', str(SOURCE), '--output', str(root), '--seed', str(args.seed),
           '--mode', args.mode, '--steps', str(args.steps),
           '--num_envs', str(args.num_envs), '--iterations', str(args.iterations),
           '--distance-min', str(args.distance_min), '--distance-max', str(args.distance_max),
           '--hard-distance-min', str(args.hard_distance_min),
           '--hard-sample-probability', str(args.hard_sample_probability),
           '--max-turn-angle-deg', str(args.max_turn_angle_deg),
           '--zero-hop-probability', str(args.zero_hop_probability),
           '--reverse-turn-probability', str(args.reverse_turn_probability),
           '--reverse-turn-halfwidth-deg', str(args.reverse_turn_halfwidth_deg),
           '--distance-curriculum-iterations', str(args.distance_curriculum_iterations),
           '--direction-curriculum-iterations', str(args.direction_curriculum_iterations),
           '--learning-rate', str(args.learning_rate),
           '--target-tolerance', str(args.target_tolerance),
           '--target-height', str(args.target_height),
           '--actuator-mode', args.actuator_mode,
           '--stability-profile', args.stability_profile,
           '--action-parameterization', args.action_parameterization,
           '--observation-layout', args.observation_layout,
           '--collective-preserving-spread-limit-pwm',
           str(args.collective_preserving_spread_limit_pwm)]
if args.fixed_initial_heading_deg is not None:
    command += ['--fixed-initial-heading-deg', str(args.fixed_initial_heading_deg)]
if args.path_free_stage2_v3:
    command.append('--path-free-stage2-v3')
elif args.path_free_stage2_v2:
    command.append('--path-free-stage2-v2')
elif args.path_free_stage2:
    command.append('--path-free-stage2')
if args.moving_handoff_v11:
    command.append('--moving-handoff-v11')
if args.landing_recovery_v12:
    command.append('--landing-recovery-v12')
if args.contact_readiness_v13:
    command.append('--contact-readiness-v13')
if args.attitude_recovery_v14:
    command.append('--attitude-recovery-v14')
if args.apex_event_fix_v15:
    command.append('--apex-event-fix-v15')
if args.terminal_penalty_fix_v16:
    command.append('--terminal-penalty-fix-v16')
if args.v3_accounting_fix:
    command.append('--v3-accounting-fix')
if args.foot_pair_v33:
    command.append('--foot-pair-v33')
if args.two_hop_episode_v38:
    command.append('--two-hop-episode-v38')
if args.sysid_20261011_plant:
    command.append('--sysid-20261011-plant')
if args.stage2_foot_precision_scale is not None:
    command += ['--stage2-foot-precision-scale', str(args.stage2_foot_precision_scale)]
if args.stage2_foot_precision_width_m is not None:
    command += ['--stage2-foot-precision-width-m', str(args.stage2_foot_precision_width_m)]
if args.stage2_foot_hit_10cm_scale is not None:
    command += ['--stage2-foot-hit-10cm-scale', str(args.stage2_foot_hit_10cm_scale)]
if args.pair_xy_quality_weight_v36 is not None:
    command += ['--pair-xy-quality-weight-v36', str(args.pair_xy_quality_weight_v36)]
if args.moving_launch_rate_v34:
    command.append('--moving-launch-rate-v34')
if args.liftoff_cross_track_penalty_scale is not None:
    command += ['--liftoff-cross-track-penalty-scale',
                str(args.liftoff_cross_track_penalty_scale)]
if args.hardware_severity_curriculum:
    command += ['--hardware-severity-curriculum',
                '--hardware-severity-stage-iterations',
                str(args.hardware_severity_stage_iterations)]
if args.route_recovery_mode:
    command += ['--route-recovery-mode', args.route_recovery_mode,
                '--route-recovery-severe-miss-m', str(args.route_recovery_severe_miss_m)]
if args.deferred_landing_v17:
    command.append('--deferred-landing-v17')
if args.low_hop_recovery_v18:
    command.append('--low-hop-recovery-v18')
if args.raibert_hermite_guidance:
    command += [
        '--raibert-hermite-guidance',
        '--raibert-stance-time-s', str(args.raibert_stance_time_s),
        '--raibert-velocity-gain-s', str(args.raibert_velocity_gain_s),
        '--raibert-max-foothold-shift-m', str(args.raibert_max_foothold_shift_m),
        '--raibert-preview-blend', str(args.raibert_preview_blend),
        '--raibert-max-horizontal-speed-m-s',
        str(args.raibert_max_horizontal_speed_m_s),
    ]
if custom_two_circle_ranges:
    command += [
        '--short-distance-min', str(args.short_distance_min),
        '--short-distance-max', str(args.short_distance_max),
        '--long-distance-min', str(args.long_distance_min),
        '--long-distance-max', str(args.long_distance_max),
    ]
if args.distance_curriculum_iterations > 0.0:
    command += [
        '--distance-curriculum-start-short-min',
        str(args.distance_curriculum_start_short_min),
        '--distance-curriculum-start-short-max',
        str(args.distance_curriculum_start_short_max),
        '--distance-curriculum-start-long-min',
        str(args.distance_curriculum_start_long_min),
        '--distance-curriculum-start-long-max',
        str(args.distance_curriculum_start_long_max),
        '--distance-curriculum-start-turn-angle-deg',
        str(args.distance_curriculum_start_turn_angle_deg),
    ]
if args.direction_curriculum_iterations > 0.0:
    command += [
        '--direction-curriculum-start-turn-angle-deg',
        str(args.direction_curriculum_start_turn_angle_deg),
        '--direction-curriculum-start-zero-hop-probability',
        str(args.direction_curriculum_start_zero_hop_probability),
    ]
if args.disable_initial_drop:
    command.append('--disable-initial-drop')
if args.episode_length_s is not None:
    command += ['--episode-length-s', str(args.episode_length_s)]
if args.successful_hops_per_episode is not None:
    command += ['--successful-hops-per-episode', str(args.successful_hops_per_episode)]
if args.apex_tolerance is not None:
    command += ['--apex-tolerance', str(args.apex_tolerance)]
for flag, value in (
    ('--play-motor-time-constant', args.play_motor_time_constant),
    ('--play-thrust-scale', args.play_thrust_scale),
):
    if value is not None:
        command += [flag, str(value)]
for flag, value in (
    ('--landing-precision-width', args.landing_precision_width),
    ('--target-hit-reward-scale', args.target_hit_reward_scale),
    ('--landing-precision-reward-scale', args.landing_precision_reward_scale),
    ('--landing-error-penalty-scale', args.landing_error_penalty_scale),
    ('--landing-lateral-error-penalty-scale', args.landing_lateral_error_penalty_scale),
    ('--observation-noise-std', args.observation_noise_std),
    ('--hardware-randomization-probability', args.hardware_randomization_probability),
    ('--hardware-randomization-probability-start', args.hardware_randomization_probability_start),
    ('--hardware-randomization-curriculum-iterations',
     args.hardware_randomization_curriculum_iterations),
    ('--comprehensive-global-thrust-scale-min', args.comprehensive_global_thrust_scale_min),
    ('--comprehensive-global-thrust-scale-max', args.comprehensive_global_thrust_scale_max),
    ('--comprehensive-com-offset-x-max-mm', args.comprehensive_com_offset_x_max_mm),
    ('--comprehensive-com-offset-y-max-mm', args.comprehensive_com_offset_y_max_mm),
    ('--comprehensive-com-offset-z-max-mm', args.comprehensive_com_offset_z_max_mm),
    ('--stationary-hold-radius', args.stationary_hold_radius),
    ('--stationary-tilt-penalty-scale', args.stationary_tilt_penalty_scale),
    ('--stationary-yaw-rate-penalty-scale', args.stationary_yaw_rate_penalty_scale),
    ('--stationary-action-spread-penalty-scale', args.stationary_action_spread_penalty_scale),
    ('--stationary-pitch-deadband-deg', args.stationary_pitch_deadband_deg),
    ('--comprehensive-yaw-effectiveness-min', args.comprehensive_yaw_effectiveness_min),
    ('--comprehensive-yaw-effectiveness-max', args.comprehensive_yaw_effectiveness_max),
    ('--comprehensive-yaw-bias-torque-max-mnm', args.comprehensive_yaw_bias_torque_max_mnm),
    ('--initial-yaw-perturb-deg', args.initial_yaw_perturb_deg),
    ('--initial-angular-velocity-perturb-rad-s', args.initial_angular_velocity_perturb_rad_s),
    ('--touchdown-wrench-probability', args.touchdown_wrench_probability),
    ('--touchdown-roll-wrench-max-nm', args.touchdown_roll_wrench_max_nm),
    ('--touchdown-pitch-wrench-max-nm', args.touchdown_pitch_wrench_max_nm),
    ('--touchdown-yaw-wrench-max-nm', args.touchdown_yaw_wrench_max_nm),
    ('--touchdown-wrench-window-s', args.touchdown_wrench_window_s),
    ('--touchdown-wrench-persistence', args.touchdown_wrench_persistence),
    ('--moving-long-recovery-window-s', args.moving_long_recovery_window_s),
    ('--moving-long-recovery-angle-deadband-deg', args.moving_long_recovery_angle_deadband_deg),
    ('--moving-long-recovery-rate-deadband-rad-s', args.moving_long_recovery_rate_deadband_rad_s),
    ('--moving-long-recovery-angle-penalty-scale', args.moving_long_recovery_angle_penalty_scale),
    ('--moving-long-recovery-rate-penalty-scale', args.moving_long_recovery_rate_penalty_scale),
    ('--airborne-attitude-penalty-scale', args.airborne_attitude_penalty_scale),
    ('--airborne-angvel-xy-penalty-scale', args.airborne_angvel_xy_penalty_scale),
    ('--airborne-yaw-penalty-scale', args.airborne_yaw_penalty_scale),
    ('--airborne-yaw-rate-penalty-scale', args.airborne_yaw_rate_penalty_scale),
    ('--airborne-action-spread-penalty-scale', args.airborne_action_spread_penalty_scale),
    ('--airborne-cross-track-tilt-penalty-scale', args.airborne_cross_track_tilt_penalty_scale),
    ('--airborne-cross-track-tilt-penalty-start', args.airborne_cross_track_tilt_penalty_start),
    ('--airborne-cross-track-velocity-penalty-scale', args.airborne_cross_track_velocity_penalty_scale),
    ('--airborne-cross-track-velocity-penalty-start-mps', args.airborne_cross_track_velocity_penalty_start_mps),
    ('--airborne-tilt-penalty-start', args.airborne_tilt_penalty_start),
    ('--airborne-angvel-xy-penalty-start', args.airborne_angvel_xy_penalty_start),
    ('--airborne-yaw-deadband-deg', args.airborne_yaw_deadband_deg),
    ('--airborne-yaw-rate-penalty-start', args.airborne_yaw_rate_penalty_start),
    ('--airborne-action-spread-penalty-start', args.airborne_action_spread_penalty_start),
    ('--stationary-pitch-penalty-scale', args.stationary_pitch_penalty_scale),
    ('--stationary-pitch-rate-deadband-rad-s', args.stationary_pitch_rate_deadband_rad_s),
    ('--stationary-pitch-rate-penalty-scale', args.stationary_pitch_rate_penalty_scale),
    ('--stationary-airborne-differential-action-rate-deadband',
     args.stationary_airborne_differential_action_rate_deadband),
    ('--stationary-airborne-differential-action-rate-penalty-scale',
     args.stationary_airborne_differential_action_rate_penalty_scale),
    ('--stationary-contact-recovery-window-s', args.stationary_contact_recovery_window_s),
    ('--stationary-contact-recovery-spread-deadband', args.stationary_contact_recovery_spread_deadband),
    ('--stationary-contact-recovery-spread-penalty-scale', args.stationary_contact_recovery_spread_penalty_scale),
    ('--stationary-contact-recovery-delta-deadband', args.stationary_contact_recovery_delta_deadband),
    ('--stationary-contact-recovery-delta-penalty-scale', args.stationary_contact_recovery_delta_penalty_scale),
    ('--stationary-apex-width', args.stationary_apex_width),
    ('--stationary-apex-reward-scale', args.stationary_apex_reward_scale),
    ('--stationary-apex-error-penalty-scale', args.stationary_apex_error_penalty_scale),
    ('--stationary-apex-shortfall-penalty-scale', args.stationary_apex_shortfall_penalty_scale),
    ('--stationary-apex-progress-scale', args.stationary_apex_progress_scale),
    ('--stationary-apex-progress-width', args.stationary_apex_progress_width),
    ('--height-progress-reward-scale', args.height_progress_reward_scale),
    ('--stationary-ascent-support-scale', args.stationary_ascent_support_scale),
    ('--stationary-ascent-collective-scale', args.stationary_ascent_collective_scale),
    ('--stationary-predicted-apex-width', args.stationary_predicted_apex_width),
    ('--stationary-predicted-apex-reward-scale', args.stationary_predicted_apex_reward_scale),
    ('--ascent-predicted-apex-width', args.ascent_predicted_apex_width),
    ('--ascent-predicted-apex-reward-scale', args.ascent_predicted_apex_reward_scale),
    ('--apex-event-height-width', args.apex_event_height_width),
    ('--apex-event-height-reward-scale', args.apex_event_height_reward_scale),
    ('--apex-event-shortfall-penalty-scale', args.apex_event_shortfall_penalty_scale),
    ('--phase-contact-collective-floor-pwm', args.phase_contact_collective_floor_pwm),
    ('--phase-ascent-collective-floor-pwm', args.phase_ascent_collective_floor_pwm),
    ('--diagnostic-ascent-pulse-pwm', args.diagnostic_ascent_pulse_pwm),
    ('--phase-contact-collective-pwm', args.phase_contact_collective_pwm),
    ('--phase-ascent-collective-pwm', args.phase_ascent_collective_pwm),
    ('--phase-descent-collective-pwm', args.phase_descent_collective_pwm),
    ('--phase-contact-differential-limit-pwm', args.phase_contact_differential_limit_pwm),
    ('--phase-ascent-differential-limit-pwm', args.phase_ascent_differential_limit_pwm),
    ('--phase-descent-differential-limit-pwm', args.phase_descent_differential_limit_pwm),
    ('--apex-bias-initial-pwm', args.apex_bias_initial_pwm),
    ('--apex-bias-gain-pwm-per-m', args.apex_bias_gain_pwm_per_m),
    ('--apex-bias-max-pwm', args.apex_bias_max_pwm),
    ('--apex-bias-liftoff-vz-target', args.apex_bias_liftoff_vz_target),
    ('--height-curriculum-start', args.height_curriculum_start),
    ('--height-curriculum-iterations', args.height_curriculum_iterations),
    ('--termination-penalty-scale', args.termination_penalty_scale),
):
    if value is not None:
        command += [flag, str(value)]
if args.robust_dynamics:
    command.append('--robust-dynamics')
if args.hardware_robust:
    command.append('--hardware-robust')
if args.comprehensive_hardware_randomization:
    command.append('--comprehensive-hardware-randomization')
if args.touchdown_wrench_on_moving_long_hop:
    command.append('--touchdown-wrench-on-moving-long-hop')
if args.target_relative_apex_gate:
    command.append('--target-relative-apex-gate')
if args.legacy_v11_contract:
    command.append('--legacy-v11-contract')
if args.simulation_only_phase_collective_baseline:
    command.append('--simulation-only-phase-collective-baseline')
if args.simulation_only_phase_collective_controller:
    command.append('--simulation-only-phase-collective-controller')
if args.simulation_only_apex_bias_controller:
    command.append('--simulation-only-apex-bias-controller')
    command += ['--apex-bias-feedback', args.apex_bias_feedback]
if args.allow_compatible_checkpoint:
    command.append('--allow-compatible-checkpoint')
if args.resume_runner_state:
    command.append('--resume-runner-state')
if args.visualize:
    command.append('--visualize')
if args.control_diagnostics:
    command.append('--control-diagnostics')
if args.failure_trace:
    command += ['--failure-trace', '--failure-trace-window-s',
                str(args.failure_trace_window_s)]
if args.takeoff_velocity_audit:
    command.append('--takeoff-velocity-audit')
if args.contact_phase_audit:
    command.append('--contact-phase-audit')
if args.closed_loop_p2_mode is not None:
    command += ['--closed-loop-p2-env-id', str(args.closed_loop_p2_env_id),
                '--closed-loop-p2-step', str(args.closed_loop_p2_step),
                '--closed-loop-p2-mode', args.closed_loop_p2_mode]
    if args.closed_loop_p2_expected_sha256 is not None:
        command += ['--closed-loop-p2-expected-sha256', args.closed_loop_p2_expected_sha256]
if args.next_hop_probe_env_ids:
    command += ['--next-hop-probe-env-ids', args.next_hop_probe_env_ids,
                '--next-hop-probe-stride', str(args.next_hop_probe_stride)]
if args.dry_run:
    import shlex
    print(shlex.join(command))
    raise SystemExit(0)
if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
log = root.with_suffix('.console.log')
log.parent.mkdir(parents=True, exist_ok=True)
child_env = os.environ.copy()
child_env['PYTHONUNBUFFERED'] = '1'
print(f'[RECOVERY] source={SOURCE}\n[RECOVERY] output={root}', flush=True)
with log.open('w') as stream:
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=child_env)
    for line in proc.stdout:
        stream.write(line)
        stream.flush()
        print(line, end='', flush=True)
    status = proc.wait()
marker = root / ('train_complete.json' if args.mode == 'train' else 'result.json')
if status != 0 or not marker.is_file():
    # Isaac teardown can prevent the traceback from reaching stdout. The inner
    # trainer saves it before closing the application; surface it here too.
    for failure_path in (root / 'startup_exception.txt', root.with_suffix('.startup_exception.txt')):
        if failure_path.is_file():
            print(f'[RECOVERY] saved exception: {failure_path}', flush=True)
            print(failure_path.read_text(encoding='utf-8', errors='replace'), flush=True)
            break
    reason = f'exit status {status}' if status != 0 else f'missing completion marker {marker}'
    raise RuntimeError(f'Recovery failed ({reason}); inspect {log}')
print('[RECOVERY] complete ' + json.dumps(json.loads(marker.read_text())), flush=True)
