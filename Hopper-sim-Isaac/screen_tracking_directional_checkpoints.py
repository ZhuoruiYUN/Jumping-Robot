"""Screen checkpoints on the actual zero-hop and arbitrary-turn curriculum."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--checkpoints', nargs='+', default=['60', '80', '100', '119'])
parser.add_argument('--seed', type=int, default=342)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--target-tolerance', type=float, default=0.05)
args = parser.parse_args()

root = args.root.resolve()
checkpoint_dir = args.checkpoint_dir.resolve()
if root.exists():
    parser.error(f'Output already exists: {root}')
if min(args.num_envs, args.steps) <= 0 or args.target_tolerance <= 0.0:
    parser.error('num-envs, steps, and target-tolerance must be positive')

# `stationary_exact` uses a point mass at radius zero, rather than a tiny
# continuous-radius interval. The other bands never inject zero hops so their
# error measures only moving-target tracking.
# Values are (distance_min, distance_max, max_turn, zero_probability,
# reverse_probability).  The exact-reversal scenario makes every turn lie in
# +/-[150, 180] degrees; it isolates the capability introduced by the latest
# curriculum from ordinary full-circle turns.
scenarios = {
    'mixed_0_40_turn180': (0.0, 0.40, 180.0, 0.10, 0.25),
    'near_0_20_turn180': (0.0, 0.20, 180.0, 0.0, 0.0),
    'far_30_40_turn180': (0.30, 0.40, 180.0, 0.0, 0.0),
    'far_30_40_reverse': (0.30, 0.40, 180.0, 0.0, 1.0),
    'stationary_exact': (0.0, 0.001, 180.0, 1.0, 0.0),
}

root.mkdir(parents=True)
launcher = Path(__file__).with_name('run_tracking_recovery.py')
rows: list[dict[str, object]] = []
for checkpoint_id in args.checkpoints:
    checkpoint = checkpoint_dir / f'model_{checkpoint_id}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for scenario, (distance_min, distance_max, turn, zero_probability, reverse_probability) in scenarios.items():
        output = root / f'model_{checkpoint_id}_{scenario}'
        command = [
            sys.executable, str(launcher), 'eval',
            '--checkpoint', str(checkpoint), '--root', str(output),
            '--num-envs', str(args.num_envs), '--seed', str(args.seed),
            '--steps', str(args.steps),
            '--distance-min', str(distance_min), '--distance-max', str(distance_max),
            '--target-tolerance', str(args.target_tolerance),
            '--max-turn-angle-deg', str(turn),
            '--zero-hop-probability', str(zero_probability),
            '--reverse-turn-probability', str(reverse_probability),
            '--reverse-turn-halfwidth-deg', '30',
        ]
        print('[SCREEN] ' + ' '.join(command), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((output / 'result.json').read_text())
        rows.append({
            'checkpoint': checkpoint.name,
            'scenario': scenario,
            'xy_error_m': result.get('xy_error_m'),
            'xy_hit_5cm': result.get('xy_hit_5cm'),
            'pair_valid_5cm': result.get('pair_valid_5cm'),
            'first_xy_error_m': result.get('first_xy_error_m'),
            'second_xy_error_m': result.get('second_xy_error_m'),
            'apex_error_m': result.get('apex_error_m'),
            'death_rate': result.get('death_rate'),
            'mean_path_distance_m': result.get('mean_path_distance_m'),
        })

with (root / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('[SCREEN] complete: ' + str(root / 'comparison.csv'))
