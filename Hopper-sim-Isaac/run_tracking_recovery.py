"""Launch continuous spatial-guidance PPO; retains the recovery entry name."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/tracking_recovery/curriculum_20_30_hard75_pair_precision_v1/model_59.pt'
parser = argparse.ArgumentParser()
parser.add_argument('mode', nargs='?', choices=('train', 'eval'), default='train')
parser.add_argument('--root', type=Path, default=ROOT / 'outputs/tracking_recovery/curriculum_20_30_hard75_convergence_v1')
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
parser.add_argument('--observation-noise-std', type=float)
parser.add_argument('--learning-rate', type=float, default=5e-6)
parser.add_argument('--target-tolerance', type=float, default=0.05)
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
if not 0.0 <= args.hard_sample_probability <= 1.0:
    parser.error('hard-sample-probability must be in [0, 1]')
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
           '--target-tolerance', str(args.target_tolerance)]
for flag, value in (
    ('--landing-precision-width', args.landing_precision_width),
    ('--target-hit-reward-scale', args.target_hit_reward_scale),
    ('--landing-precision-reward-scale', args.landing_precision_reward_scale),
    ('--landing-error-penalty-scale', args.landing_error_penalty_scale),
    ('--landing-lateral-error-penalty-scale', args.landing_lateral_error_penalty_scale),
    ('--observation-noise-std', args.observation_noise_std),
):
    if value is not None:
        command += [flag, str(value)]
if args.robust_dynamics:
    command.append('--robust-dynamics')
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
