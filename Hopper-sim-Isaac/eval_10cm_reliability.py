"""Repeated fixed-distance evaluation for a 10 cm hopping reliability claim."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--seeds', nargs='+', type=int, default=[501, 502, 503, 504])
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=6000,
                    help='Long horizon; default samples roughly 50 hops per environment.')
parser.add_argument('--target-tolerance', type=float, default=0.05)
args = parser.parse_args()

root = args.root.resolve()
checkpoint = args.checkpoint.resolve()
if root.exists():
    parser.error(f'Output already exists: {root}')
if not checkpoint.is_file():
    parser.error(f'Missing checkpoint: {checkpoint}')
if min(args.num_envs, args.steps, len(args.seeds)) <= 0 or args.target_tolerance <= 0.0:
    parser.error('num-envs, steps, seeds, and target-tolerance must be positive')

# A narrow-turn sequence represents ordinary repeated forward hopping.  The
# all-direction and reverse tests keep the same 10 cm distance while exposing
# the heading changes relevant to deployment.
scenarios = {
    'nominal_turn30': (30.0, 0.0),
    'all_turn180': (180.0, 0.0),
    'reverse_turn': (180.0, 1.0),
}

root.mkdir(parents=True)
launcher = Path(__file__).with_name('run_tracking_recovery.py')
rows: list[dict[str, object]] = []
for scenario, (turn, reverse_probability) in scenarios.items():
    for seed in args.seeds:
        output = root / f'{scenario}_seed{seed}'
        command = [
            sys.executable, str(launcher), 'eval',
            '--checkpoint', str(checkpoint), '--root', str(output),
            '--num-envs', str(args.num_envs), '--seed', str(seed),
            '--steps', str(args.steps),
            '--distance-min', '0.10', '--distance-max', '0.10',
            '--target-tolerance', str(args.target_tolerance),
            '--max-turn-angle-deg', str(turn),
            '--zero-hop-probability', '0.0',
            '--reverse-turn-probability', str(reverse_probability),
            '--reverse-turn-halfwidth-deg', '30',
        ]
        print('[10CM] ' + ' '.join(command), flush=True)
        subprocess.run(command, check=True)
        result = json.loads((output / 'result.json').read_text())
        rows.append({
            'scenario': scenario,
            'seed': seed,
            'touchdowns': result.get('touchdowns'),
            'valid_hit_5cm': result.get('valid_hit_5cm'),
            'pair_valid_5cm': result.get('pair_valid_5cm'),
            'valid_misses_5cm': result.get('valid_misses_5cm'),
            'zero_miss_env_rate_5cm': result.get('zero_miss_env_rate_5cm'),
            'max_consecutive_valid_5cm_min': result.get('max_consecutive_valid_5cm_min'),
            'max_consecutive_valid_5cm_mean': result.get('max_consecutive_valid_5cm_mean'),
            'xy_error_m': result.get('xy_error_m'),
            'apex_error_m': result.get('apex_error_m'),
            'death_rate': result.get('death_rate'),
        })

with (root / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# A literal 100% claim requires no valid miss across all seeds and all three
# scenarios. This compact summary avoids confusing a high mean with zero
# failures.
all_zero_miss = all(row['valid_misses_5cm'] == 0 for row in rows)
summary = {
    'checkpoint': str(checkpoint),
    'target_distance_m': 0.10,
    'target_tolerance_m': args.target_tolerance,
    'seeds': args.seeds,
    'scenarios': list(scenarios),
    'total_touchdowns': sum(int(row['touchdowns'] or 0) for row in rows),
    'total_valid_misses_5cm': sum(int(row['valid_misses_5cm'] or 0) for row in rows),
    'all_touchdowns_valid_5cm': all_zero_miss,
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print('[10CM] ' + json.dumps(summary))
