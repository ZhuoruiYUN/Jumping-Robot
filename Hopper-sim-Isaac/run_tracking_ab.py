"""Standard-library launcher: python run_tracking_ab.py train|eval [options]."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / ('logs/rsl_rl/quadhopper_planner_random_two_hop_v67_custom_relative_next_'
    'baseline_warmstart_turn_030_tol_15_spd_030_tilt_06_arw_100_lcg_000_blend_000_'
    'landingxy_pairrestart_air_100_yawair_spring_fixed_100/2026-09-07_22-13-09/model_10.pt')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('train', 'eval'))
parser.add_argument('--root', type=Path)
parser.add_argument('--seeds', nargs='+', type=int, default=[42])
parser.add_argument('--screen-seed', type=int, default=142)
parser.add_argument('--eval-seeds', nargs='+', type=int, default=[242, 342])
parser.add_argument('--iterations', type=int, default=120)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
if args.mode == 'eval' and args.root is None:
    parser.error('eval requires --root from the training command')
if args.iterations <= 0 or args.steps <= 0 or args.num_envs <= 0:
    parser.error('iterations, steps and num-envs must be positive')
root = (args.root or ROOT / 'outputs/tracking_ab' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
isaac = Path(os.environ.get('ISAAC_SIM_ROOT',
    '/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64')) / 'python.sh'


def launch(mode, variant, checkpoint, directory, seed, distance='range'):
    command = [str(isaac), str(ROOT / 'train_tracking_ab.py'), '--headless',
        '--mode', mode, '--variant', variant, '--checkpoint', str(checkpoint),
        '--output', str(directory), '--seed', str(seed), '--distance', distance,
        '--num_envs', str(args.num_envs), '--iterations', str(args.iterations),
        '--steps', str(args.steps)]
    marker = directory / ('train_complete.json' if mode == 'train' else 'result.json')
    if args.dry_run:
        print(shlex.join(command))
        return
    if marker.exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        expected = dict(mode=mode, variant=variant, seed=seed, distance=distance,
                        num_envs=args.num_envs)
        expected['iterations' if mode == 'train' else 'steps'] = (
            args.iterations if mode == 'train' else args.steps
        )
        if (manifest['source'] != str(checkpoint.resolve()) or
                any(manifest['cli'].get(k) != v for k, v in expected.items())):
            raise RuntimeError(f'Existing run uses different settings: {directory}; choose a new root')
        print(f'[SKIP] {marker}')
        return json.loads(marker.read_text()) if mode == 'eval' else None
    if directory.exists():
        raise RuntimeError(f'Incomplete output: {directory}; inspect it and choose a new root')
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    directory.parent.mkdir(parents=True, exist_ok=True)
    log = directory.with_suffix('.console.log')
    print(f'[AB] {mode} {variant} seed={seed} distance={distance} log={log}', flush=True)
    child_env = os.environ.copy()
    child_env['PYTHONUNBUFFERED'] = '1'
    progress = {}
    progress_fields = {
        'Mean action noise std': 'std',
        'Mean value_function loss': 'value_loss',
        'Mean surrogate loss': 'surrogate',
        'Mean reward': 'reward',
        'Mean episode length': 'length',
        'Metrics/target_hit_rate_ema': 'hit_ema',
        'Metrics/touchdown_error_ema_m': 'xy_ema',
        'Diagnostics/reset_tilt_over_limit_count': 'tilt_resets',
    }
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=child_env)
        for line in proc.stdout:
            stream.write(line)
            plain = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
            match = re.search(r'Learning iteration\s+(\d+/\d+)', plain)
            if match:
                progress = {'iteration': match.group(1)}
            for label, key in progress_fields.items():
                if label in plain and ':' in plain:
                    progress[key] = plain.rsplit(':', 1)[-1].strip()
                    break
            if plain.startswith('Total timesteps:') and progress:
                ordered = ('iteration', 'reward', 'length', 'hit_ema', 'xy_ema',
                           'tilt_resets', 'value_loss', 'surrogate', 'std')
                print('[TRAIN] ' + ' '.join(
                    f'{key}={progress[key]}' for key in ordered if key in progress
                ), flush=True)
                progress = {}
            elif any(s in plain for s in ('[TRACKING-', 'Traceback')):
                print(line.rstrip(), flush=True)
        status = proc.wait()
    # Isaac's launcher can exit zero after a Python exception. Require output.
    if status != 0 or not marker.is_file():
        raise RuntimeError(f'Run failed or produced no result; inspect {log}')
    return json.loads(marker.read_text()) if mode == 'eval' else None


def checkpoint_iteration(path):
    return int(path.stem.split('_')[-1])


def main():
    print(f'[AB ROOT] {root}', flush=True)
    if args.mode == 'train':
        for seed in args.seeds:
            for variant in ('current', 'preview'):
                launch('train', variant, SOURCE, root / f'{variant}_seed{seed}', seed)
        print('Evaluate with: ' + shlex.join([
            'python', 'run_tracking_ab.py', 'eval', '--root', str(root),
            '--num-envs', str(args.num_envs), '--seeds', *map(str, args.seeds),
        ]))
    else:
        rows = []
        screen_rows = []
        selections = {}
        for training_seed in args.seeds:
            for variant in ('current', 'preview'):
                run = root / f'{variant}_seed{training_seed}'
                if not args.dry_run:
                    json.loads((run / 'train_complete.json').read_text())
                    models = sorted(run.glob('model_*.pt'), key=checkpoint_iteration)
                    if not models:
                        raise RuntimeError(f'No checkpoints found in {run}')
                else:
                    models = [run / f'model_{i}.pt' for i in range(0, args.iterations, 10)]
                    if not models or checkpoint_iteration(models[-1]) != args.iterations - 1:
                        models.append(run / f'model_{args.iterations - 1}.pt')
                candidates = [run / 'initial.pt', *models]
                for checkpoint in candidates:
                    directory = root / 'screen' / (
                        f'{variant}_train{training_seed}_{checkpoint.stem}_range_seed{args.screen_seed}'
                    )
                    result = launch('eval', variant, checkpoint, directory,
                                    args.screen_seed, 'range')
                    if not args.dry_run:
                        if not result.get('valid_evaluation', False):
                            raise RuntimeError(f'No touchdowns; screening is invalid: {directory}')
                        result.update(training_seed=training_seed, arm=variant,
                                      checkpoint_iteration=(-1 if checkpoint.name == 'initial.pt'
                                                            else checkpoint_iteration(checkpoint)))
                        screen_rows.append(result)
                if args.dry_run:
                    selections[(training_seed, variant)] = models[-1]
                    continue
                group = [r for r in screen_rows
                         if r['training_seed'] == training_seed and r['arm'] == variant]
                initial = next(r for r in group if r['checkpoint_iteration'] == -1)
                trained = [r for r in group if r['checkpoint_iteration'] >= 0]
                death_ceiling = initial['death_rate'] + 0.03
                safe = [r for r in trained if r['death_rate'] <= death_ceiling]
                pool = safe or trained
                best = max(pool, key=lambda r: (
                    r['valid_hit_10cm'], -abs(r['along_error_m']), -r['xy_error_m']
                ))
                selections[(training_seed, variant)] = Path(best['checkpoint'])
                print(f'[SELECT] {variant} train_seed={training_seed} '
                      f'{Path(best["checkpoint"]).name} valid10={best["valid_hit_10cm"]:.1%} '
                      f'along={100*best["along_error_m"]:.2f}cm death={best["death_rate"]:.1%}',
                      flush=True)

        if args.dry_run:
            print('[DRY-RUN] final evaluation assumes the last checkpoint; actual run selects by screening')
            return

        with (root / 'checkpoint_selection.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(screen_rows[0]))
            writer.writeheader()
            writer.writerows(screen_rows)

        for training_seed in args.seeds:
            for variant in ('current', 'preview'):
                run = root / f'{variant}_seed{training_seed}'
                checkpoints = [
                    (f'initial_{variant}', run / 'initial.pt'),
                    (variant, selections[(training_seed, variant)]),
                ]
                for arm, checkpoint in checkpoints:
                    for distance in ('range', '20cm'):
                        for seed in args.eval_seeds:
                            directory = root / 'eval' / (
                                f'{arm}_train{training_seed}_{checkpoint.stem}_{distance}_seed{seed}'
                            )
                            row = launch('eval', variant, checkpoint, directory, seed, distance)
                            if not row.get('valid_evaluation', False):
                                raise RuntimeError(f'No touchdowns; evaluation is invalid: {directory}')
                            row.update(training_seed=training_seed, arm=arm)
                            rows.append(row)
        if rows:
            with (root / 'comparison.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print('\narm             distance  valid10  pair10   xy(cm)  along(cm) apexerr(cm) death')
            for distance in ('range', '20cm'):
                for arm in ('initial_current', 'current', 'initial_preview', 'preview'):
                    group = [r for r in rows if r['arm'] == arm and r['distance'] == distance]
                    mean = lambda k: sum(r[k] for r in group) / len(group)
                    print(f'{arm:16} {distance:8} {mean("valid_hit_10cm"):7.1%} '
                          f'{mean("pair_valid_10cm"):7.1%} {100*mean("xy_error_m"):7.2f} '
                          f'{100*mean("along_error_m"):10.2f} '
                          f'{100*mean("apex_error_m"):11.2f} {mean("death_rate"):6.1%}')
            print(f'Per-seed results: {root / "comparison.csv"}')


if __name__ == '__main__':
    main()
