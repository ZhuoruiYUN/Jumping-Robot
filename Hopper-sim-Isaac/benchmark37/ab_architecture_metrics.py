"""Verify paired AB training/evaluation artifacts and summarize all original cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


SUITES = ('development', 'stream_straight60')
REFERENCE = 'ab_yaw_heading_v57/train_seed42_20261120_181912_898373Z'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def touchdown_stats(events):
    foot = sorted(event['foot_error_m'] for event in events)
    n = len(foot)
    p = .95 * (n - 1)
    p95 = foot[int(p)] + (p % 1) * (foot[math.ceil(p)] - foot[int(p)]) if n else None
    return dict(attempts=n, within_5cm=sum(error <= .05 for error in foot),
                strict=sum(event['strict'] for event in events), foot_p95_m=p95)


def score(folder, checkpoint, case_reference):
    manifest = read(folder / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['optimizer_updates'] == 0
    assert Path(manifest['checkpoint']).resolve() == checkpoint.resolve()
    assert manifest['seed'] == 7001 and read(folder / 'cases.json') == read(case_reference)
    rows = read(folder / 'episodes.json')
    assert [row['case_id'] for row in rows] == list(range(128))
    summary = read(folder / 'summary.json')
    events = [event for row in rows for event in row['events']]
    moving = [event for event in events if event['moving']]
    first = [event for event in moving if event['first_attempt_for_target']]
    long = [event for event in first if event['commanded_edge_m'] >= .4]
    retry = [event for event in moving if not event['first_attempt_for_target']]
    counts = summary['counts']
    return dict(cases=128, cases_sha256=digest(folder / 'cases.json'),
                deaths=counts['physical_death'], mean_seconds=mean(row['seconds'] for row in rows),
                median_seconds=median(row['seconds'] for row in rows), completed_jumps=len(events),
                cases_with_jump=sum(bool(row['events']) for row in rows),
                stationary_opening_strict=counts['first_stationary_strict'],
                first_moving_strict=counts['first_moving_strict'],
                cases_with_moving_landing=sum(any(e['moving'] for e in r['events']) for r in rows),
                three_moving_hits=counts['three_moving_hits'],
                first_moving_attempts=touchdown_stats(first), long_first=touchdown_stats(long),
                retries=touchdown_stats(retry), yaw=summary['all_phase_yaw'],
                clean10={k:v['clean'] for k,v in summary.get('precision10',{}).get('groups',{}).items()},
                cases_of_interest={r['case_id']:dict(physical_death=r['physical_death'],seconds=r['seconds'],
                    last_landed_target=max((e['target_index'] for e in r['events']),default=None),
                    focus_landings_cm=[100*e['foot_error_m'] for e in r['events'] if e['target_index']==target])
                    for target,r in [(19,rows[31]),(4,rows[55])]})


def compare(pair):
    reference = pair.parents[1] / REFERENCE
    configs = {}
    output = dict(arms={}, reference={}, settings_verified={})
    for arm in ('lstm','mlp'):
        folder = pair / arm
        manifest = read(folder / 'manifest.json')
        assert manifest['status'] == 'complete' and manifest['source'] is None
        assert manifest['iteration'] == 49 and manifest['ab_policy'] == arm and manifest['seed'] == 42
        assert manifest['steps_per_update'] == 256 and manifest['num_envs'] == 1024
        assert read(folder/'ring_config.json') == read(reference/'ring_config.json')
        rc = read(folder / 'runner_config.json')
        env = read(folder / 'env_config.json')
        env.pop('csv_log_path')
        architecture = rc.pop('policy')
        assert architecture['class_name'] == {'mlp':'ActorCritic', 'lstm':'ActorCriticRecurrent'}[arm]
        rc.pop('experiment_name')
        common_arch = {k:v for k,v in architecture.items() if k in
                       ('actor_obs_normalization','critic_obs_normalization','activation','noise_std_type','init_noise_std')}
        configs[arm] = (rc,env,common_arch,read(folder/'observation_contract.json'))
        checkpoint = folder / 'model_49.pt'
        suites = {suite:score(folder/suite,checkpoint,reference/suite/'cases.json') for suite in SUITES}
        trends = read(folder/'development/summary.json')['training_trend']['metrics']
        for tag in ('Train/mean_reward','Loss/value_function','Loss/surrogate','Policy/mean_noise_std'):
            assert sum(w['samples'] for w in trends[tag]) == 50 and all(math.isfinite(w['mean']) for w in trends[tag])
        output['arms'][arm] = dict(checkpoint_sha256=digest(checkpoint),parameters=manifest['policy_parameters'],
                                   policy=architecture, training=trends, evaluation=suites,
                                   valid_training_landings=int(manifest['vertical_valid_landings']),
                                   strict_training_landings=int(manifest['vertical_native_landings']))
    assert configs['lstm'] == configs['mlp']
    output['settings_verified'] = dict(training_seed=42, eval_seed=7001, num_envs=1024, updates=50,
                                       env_steps_per_update=262144, steps_per_arm=13107200,
                                       course_sha256=digest(reference/'ring_config.json'),
                                       ppo=configs['mlp'][0]['algorithm'])
    for suite in SUITES:
        output['reference'][suite] = score(reference/suite, reference/'model_49.pt', reference/suite/'cases.json')
    probe = pair/'v57_reset_memory'
    if probe.exists():
        output['v57_memory_reset_development'] = score(probe,reference/'model_49.pt',reference/'development/cases.json')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair',type=Path)
    pair = parser.parse_args().pair.resolve()
    path = pair / 'comparison.json'
    path.write_text(json.dumps(compare(pair), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(path)
