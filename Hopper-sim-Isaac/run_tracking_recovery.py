"""Launch continuous spatial-guidance PPO; retains the recovery entry name."""
import argparse
import json
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
parser.add_argument('--landing-precision-width', type=float)
parser.add_argument('--target-hit-reward-scale', type=float)
parser.add_argument('--landing-precision-reward-scale', type=float)
parser.add_argument('--landing-error-penalty-scale', type=float)
parser.add_argument('--landing-lateral-error-penalty-scale', type=float)
parser.add_argument('--robust-dynamics', action='store_true',
                    help='Randomize mass, inertia, action delay, motor bandwidth, and per-motor thrust.')
parser.add_argument('--hardware-robust', action='store_true',
                    help='Use the measured low-thrust/slow-motor sim-to-real envelope.')
parser.add_argument('--hardware-randomization-probability', type=float, default=1.0,
                    help='Fraction of reset environments receiving hardware perturbations.')
parser.add_argument('--observation-noise-std', type=float)
parser.add_argument('--stationary-hold-radius', type=float)
parser.add_argument('--stationary-tilt-penalty-scale', type=float)
parser.add_argument('--stationary-yaw-rate-penalty-scale', type=float)
parser.add_argument('--stationary-action-spread-penalty-scale', type=float)
parser.add_argument('--stationary-apex-width', type=float)
parser.add_argument('--stationary-apex-reward-scale', type=float)
parser.add_argument('--stationary-apex-error-penalty-scale', type=float)
parser.add_argument('--stationary-apex-shortfall-penalty-scale', type=float)
parser.add_argument('--stationary-apex-progress-scale', type=float)
parser.add_argument('--stationary-apex-progress-width', type=float)
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
parser.add_argument('--target-height', type=float, default=1.095,
                    help='Absolute simulation/policy root-Z apex command.')
parser.add_argument('--disable-initial-drop', action='store_true',
                    help='Reset on the ground instead of training-only random aerial drops.')
parser.add_argument('--episode-length-s', type=float,
                    help='Override episode duration; use a value longer than the eval horizon for continuous playback.')
parser.add_argument('--play-motor-time-constant', type=float,
                    help='Fixed nominal motor time constant for deterministic plant identification.')
parser.add_argument('--play-thrust-scale', type=float,
                    help='Fixed nominal per-motor thrust multiplier for plant identification.')
parser.add_argument('--height-curriculum-start', type=float)
parser.add_argument('--height-curriculum-iterations', type=float)
parser.add_argument('--termination-penalty-scale', type=float)
parser.add_argument('--target-relative-apex-gate', action='store_true')
parser.add_argument('--actuator-mode', choices=('parity', 'corrected_history', 'baseline'),
                    default='parity')
parser.add_argument('--action-parameterization', choices=('direct_motor', 'collective_residual_v1'),
                    default='direct_motor')
parser.add_argument('--legacy-v11-contract', action='store_true')
parser.add_argument('--allow-compatible-checkpoint', action='store_true')
parser.add_argument('--visualize', action='store_true',
                    help='Run an interactive one-environment evaluation with debug visualization.')
args = parser.parse_args()
if args.hard_sample_probability is None:
    args.hard_sample_probability = 0.75 if args.mode == 'train' else 0.0
root = args.root.resolve()
if args.mode == 'eval' and args.checkpoint is None:
    parser.error('eval requires --checkpoint and a new --root for its result')
if args.visualize and args.mode != 'eval':
    parser.error('--visualize is only supported in eval mode')
if args.checkpoint is not None:
    SOURCE = args.checkpoint.expanduser().resolve()
if min(args.steps, args.num_envs, args.iterations) <= 0:
    parser.error('steps, num-envs and iterations must be positive')
if not (0 <= args.distance_min <= args.distance_max and args.distance_max > 0.0) or args.target_tolerance <= 0:
    parser.error('Require 0 <= distance-min <= distance-max, distance-max > 0, and positive target-tolerance')
if args.learning_rate <= 0.0:
    parser.error('learning-rate must be positive')
if args.target_height <= 0.38:
    parser.error('target-height must exceed the 0.380 m simulation landing root height')
if args.episode_length_s is not None and args.episode_length_s <= 0.0:
    parser.error('episode-length-s must be positive')
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
if not 0.0 < args.max_turn_angle_deg <= 180.0:
    parser.error('max-turn-angle-deg must be in (0, 180]')
if not 0.0 <= args.zero_hop_probability <= 1.0:
    parser.error('zero-hop-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_probability <= 1.0:
    parser.error('reverse-turn-probability must be in [0, 1]')
if not 0.0 < args.reverse_turn_halfwidth_deg <= 180.0:
    parser.error('reverse-turn-halfwidth-deg must be in (0, 180]')
if args.reverse_turn_probability > 0.0 and (
    args.max_turn_angle_deg < 180.0 - args.reverse_turn_halfwidth_deg
):
    parser.error('max-turn-angle-deg must include the configured reverse-turn band')
if args.landing_precision_width is not None and args.landing_precision_width <= 0.0:
    parser.error('landing-precision-width must be positive')
if args.observation_noise_std is not None and args.observation_noise_std < 0.0:
    parser.error('observation-noise-std must be non-negative')
if args.stationary_hold_radius is not None and args.stationary_hold_radius <= 0.0:
    parser.error('stationary-hold-radius must be positive')
if args.stationary_apex_width is not None and args.stationary_apex_width <= 0.0:
    parser.error('stationary-apex-width must be positive')
if args.stationary_apex_progress_width is not None and args.stationary_apex_progress_width <= 0.0:
    parser.error('stationary-apex-progress-width must be positive')
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
           '--learning-rate', str(args.learning_rate),
           '--target-tolerance', str(args.target_tolerance),
           '--target-height', str(args.target_height),
           '--actuator-mode', args.actuator_mode,
           '--action-parameterization', args.action_parameterization]
if args.disable_initial_drop:
    command.append('--disable-initial-drop')
if args.episode_length_s is not None:
    command += ['--episode-length-s', str(args.episode_length_s)]
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
    ('--stationary-hold-radius', args.stationary_hold_radius),
    ('--stationary-tilt-penalty-scale', args.stationary_tilt_penalty_scale),
    ('--stationary-yaw-rate-penalty-scale', args.stationary_yaw_rate_penalty_scale),
    ('--stationary-action-spread-penalty-scale', args.stationary_action_spread_penalty_scale),
    ('--stationary-apex-width', args.stationary_apex_width),
    ('--stationary-apex-reward-scale', args.stationary_apex_reward_scale),
    ('--stationary-apex-error-penalty-scale', args.stationary_apex_error_penalty_scale),
    ('--stationary-apex-shortfall-penalty-scale', args.stationary_apex_shortfall_penalty_scale),
    ('--stationary-apex-progress-scale', args.stationary_apex_progress_scale),
    ('--stationary-apex-progress-width', args.stationary_apex_progress_width),
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
if args.visualize:
    command.append('--visualize')
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
    raise RuntimeError(f'Recovery failed; inspect {log}')
print('[RECOVERY] complete ' + json.dumps(json.loads(marker.read_text())), flush=True)
