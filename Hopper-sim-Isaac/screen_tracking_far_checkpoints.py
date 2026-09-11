"""Evaluate far-course checkpoints under fixed, uniform distance bands."""
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
parser.add_argument('--checkpoints', nargs='+', default=['0', '40', '50', '60', '70', '79'],
                    help='Iteration ids; model_0 is the pre-exploration baseline.')
parser.add_argument('--seed', type=int, default=242)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=1200)
parser.add_argument('--target-tolerance', type=float, default=0.05)
args = parser.parse_args()

root = args.root.resolve()
checkpoint_dir = args.checkpoint_dir.resolve()
if root.exists():
    parser.error(f'Output already exists: {root}')
if min(args.num_envs, args.steps) <= 0 or args.target_tolerance <= 0:
    parser.error('num-envs, steps and target-tolerance must be positive')

bands = {'20_30': (0.20, 0.30), '30_40': (0.30, 0.40), 'fixed_40': (0.40, 0.40)}
root.mkdir(parents=True)
rows = []
launcher = Path(__file__).with_name('run_tracking_recovery.py')

for checkpoint_id in args.checkpoints:
    checkpoint = checkpoint_dir / f'model_{checkpoint_id}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for band, (distance_min, distance_max) in bands.items():
        output = root / f'model_{checkpoint_id}_{band}'
        command = [
            sys.executable, str(launcher), 'eval',
            '--checkpoint', str(checkpoint), '--root', str(output),
            '--num-envs', str(args.num_envs), '--seed', str(args.seed),
            '--steps', str(args.steps), '--distance-min', str(distance_min),
            '--distance-max', str(distance_max),
            '--target-tolerance', str(args.target_tolerance),
            '--max-turn-angle-deg', '30',
        ]
        print('[SCREEN] ' + ' '.join(command), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((output / 'result.json').read_text())
        rows.append({
            'checkpoint': checkpoint.name,
            'band': band,
            'xy_error_m': result.get('xy_error_m'),
            'xy_hit_5cm': result.get('xy_hit_5cm'),
            'pair_valid_5cm': result.get('pair_valid_5cm'),
            'first_xy_error_m': result.get('first_xy_error_m'),
            'second_xy_error_m': result.get('second_xy_error_m'),
            'apex_error_m': result.get('apex_error_m'),
            'death_rate': result.get('death_rate'),
            'mean_path_distance_m': result.get('mean_path_distance_m'),
        })

fields = list(rows[0])
with (root / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('[SCREEN] complete: ' + str(root / 'comparison.csv'))
