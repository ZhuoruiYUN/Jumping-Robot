"""Build the V68 command-drift probe from independent training-route rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np


DIRECTIONS = ('E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE')


def build(source, destination):
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('sample_use') != 'training replay only, excluded from evaluation and future holdout':
        raise ValueError('Expected an explicitly training-only rollout')
    cases = json.loads((source / 'cases.json').read_text())
    if cases['seed'] != 68193 or manifest['seed'] != cases['seed']:
        raise ValueError('Expected the reserved V68 training probe seed')
    rows = json.loads((source / 'episodes.json').read_text())
    record = np.load(source / 'record.npz')
    obs, actions = record['obs'], record['actions']
    labels = np.full(obs.shape[:2], -1, dtype=np.int16)
    for row in rows:
        lane = row['case_id']
        previous = None
        for event in row['events']:
            touchdown = event['touchdown_step']
            if previous is not None and event['moving'] and event.get('release_to_apex_sample_s') is not None:
                release = touchdown - round(100 * event['release_to_touch_s'])
                apex = release + round(100 * event['release_to_apex_sample_s'])
                group = DIRECTIONS.index(event['direction']) + 8 * (event['attempt_for_target'] > 1)
                labels[max(0, previous - 20):apex + 1, lane] = group
            previous = touchdown

    # Keep a compact bank with genuine first attempts and retries in every direction.
    # All selected histories start at reset, even if the first scored frame is much later.
    counts = np.array([[(labels[:, i] == g).sum() for g in range(16)] for i in range(len(rows))])
    selected = []
    coverage = np.zeros(16)
    for _ in range(min(48, len(rows))):
        scores = (np.minimum(counts, 160) / (160 + coverage)).sum(axis=1)
        scores[selected] = -1
        lane = int(scores.argmax())
        selected.append(lane)
        coverage += counts[lane]
    labels = labels[:, selected]
    actual = actions[:, selected]
    budgets = []
    audit = []
    for group in range(16):
        error = actual[labels == group]
        if len(error) < 100:
            raise ValueError(f'Insufficient frames for direction/attempt group {group}')
        diff = error[:, 1:] - error[:, :1]
        yaw = diff @ np.array([-1, 1, -1, 1], np.float32) / 4
        rms = np.stack([np.mean(diff**2, axis=(0, 2))**.5, np.mean(yaw**2, axis=0)**.5], axis=1)
        budgets.append(rms[0])
        audit.append(dict(group=group, frames=len(error), short=rms[0].tolist(), formal=rms[1].tolist()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, obs=obs[:, selected], labels=labels, reference=actual[:, :, 0],
                        short_rms=np.array(budgets), lanes=selected)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
