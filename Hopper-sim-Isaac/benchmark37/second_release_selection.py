"""Compare saved policies on v48's paired commands and full-omni routes."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def read(path):
    return json.loads(path.read_text())


def score(moving, paired):
    if (moving['episodes'] != 128 or paired['episodes'] != 128 or paired['direction'] != 'four-axis-rebound'
            or paired['landing_radius_m'] != .05 or not 0 <= paired['requested'] <= 256
            or paired['press_times_s'] != [14.,14.25] or paired['optimizer_updates'] != 0):
        raise ValueError('Expected the fixed 128-case omni and four-axis protocols')
    totals = [0,0]
    south = [0,0]
    for row in paired['cases']:
        step = row['steps'][1]
        if 'launch_along_m_s' in step:
            inside = ('foot_along_m' in step and
                      math.hypot(step['foot_along_m'],step['foot_cross_m']) <= paired['landing_radius_m'])
            totals[0] += inside
            totals[1] += 1
            if row['direction'] == '-Y':
                south[0] += inside
                south[1] += 1
    return dict(second=totals, south=south, both=paired['both_arrived'],
                peak=moving['flight_plane']['travel']['lean_peak_deg'],
                deaths=paired['deaths']+moving['counts']['physical_death'])


def completed_omni(series,checkpoint):
    entries = []
    for path in series.glob('*/manifest.json'):
        manifest = read(path)
        if (manifest.get('checkpoint') == str(checkpoint) and manifest.get('evaluation_scope') == 'moving'
                and manifest.get('status') == 'complete'):
            entries.append(path.parent)
    return entries


def evaluate(checkpoint, series, cases):
    entries = completed_omni(series,checkpoint)
    if not entries:
        subprocess.run(['bash','run_benchmark73_phase_motion.sh','eval','--headless','--checkpoint',str(checkpoint),
                        '--cases-file',str(cases)],check=True)
        entries = completed_omni(series,checkpoint)
    original = read(cases)['cases']
    run = next((p for p in entries if read(p/'cases.json')['cases']==original),None)
    if run is None:
        raise ValueError(f'No completed omni run with matching cases for {checkpoint}')
    report = run/'four_axis.json'
    if not report.exists():
        with TemporaryDirectory(dir=run) as tmp:
            new = Path(tmp)/report.name
            subprocess.run(['bash','run_benchmark73_phase_motion.sh','eval-guarded-keys','--checkpoint',str(checkpoint),
                            '--cases-file',str(cases),'--terminal-hold','four-axis-rebound','--hold-report',str(new)],check=True)
            new.replace(report)
    paired = read(report)
    if Path(paired['checkpoint'])!=checkpoint or Path(paired['cases_file']).resolve()!=cases:
        raise ValueError(f'Unexpected checkpoint or cases in {report}')
    return score(read(run/'summary.json'),paired)


def ratio(counts):
    return Fraction(*counts) if counts[1] else Fraction(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_dir',type=Path)
    args = parser.parse_args()
    run = args.train_dir.resolve()
    train = read(run/'manifest.json')
    if train.get('status')!='complete' or train.get('mode')!='train':
        parser.error('Training must finish before comparing frozen checkpoints')
    course = read(run/'ring_config.json')
    parent = Path(course['omni']['initial_checkpoint']).resolve()
    baseline = parent.parent.parent
    cases = baseline/'moving/cases.json'
    baseline_pair = read(baseline/'four_axis.json')
    if Path(baseline_pair['checkpoint'])!=parent or Path(baseline_pair['cases_file']).resolve()!=cases:
        raise ValueError('Parent reports do not match the initialization and comparison cases')
    reference = score(read(baseline/'moving/summary.json'),baseline_pair)
    checkpoints = sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))[-2:]
    if not checkpoints or checkpoints[-1].stem!=f"model_{train['iteration']}":
        raise ValueError('Final checkpoint missing')
    limit = course['flight_plane_limits']['lean_peak_deg']
    accepted = []
    for ckpt in checkpoints:
        row = evaluate(ckpt,Path('outputs')/course['output_series'],cases)
        rate = ratio(row['second'])
        peak = '—' if row['peak'] is None else f"{row['peak']:.2f}°"
        print(f"\n{ckpt.name}  第二跳入圈 {row['second'][0]}/{row['second'][1]}  "
              f"−Y {row['south'][0]}/{row['south'][1]}  两步到达 {row['both']}/128  "
              f"移动侧向峰值 {peak}  失稳 {row['deaths']}",flush=True)
        if (not row['deaths'] and row['peak'] is not None and row['peak']<=limit
                and row['both']>=reference['both'] and ratio(row['south'])>=ratio(reference['south'])
                and rate>ratio(reference['second'])):
            accepted.append((rate,ratio(row['south']),-row['peak'],ckpt))
    print(f"\n进入完整部署评估的候选权重  {max(accepted)[-1]}" if accepted else
          f"\n本轮无通过筛选的权重。第二跳入圈率须高于 {float(ratio(reference['second'])):.1%}，"
          f"−Y 第二跳至少 {float(ratio(reference['south'])):.1%}，两步到达至少 {reference['both']}/128，"
          f"移动侧向倾角峰值须在 {limit:g}° 以内，失稳次数须为零。",flush=True)


if __name__=='__main__':
    main()
