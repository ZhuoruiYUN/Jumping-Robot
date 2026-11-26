"""Summarize training scalars available up to an evaluated PPO checkpoint."""
import re
from statistics import mean


def checkpoint_trend(checkpoint):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
    if not match or not any(checkpoint.parent.glob('events.out.tfevents.*')):
        return {'status': 'checkpoint iteration or training event log unavailable'}
    iteration = int(match[1])
    events = EventAccumulator(str(checkpoint.parent), size_guidance={'scalars': 0}).Reload()
    tags = ('Train/mean_reward', 'Train/mean_episode_length', 'Loss/value_function',
            'Loss/surrogate', 'Policy/mean_noise_std', 'Episode_Reward/all_phase_yaw')
    metrics = {}
    for tag in tags:
        if tag not in events.Tags()['scalars']:
            continue
        # Repeated steps can occur after a training process restarts.
        values = {event.step: event.value for event in events.Scalars(tag) if event.step <= iteration}
        windows = {}
        for step, value in sorted(values.items()):
            windows.setdefault(step // 10, {})[step] = value
        metrics[tag] = [dict(first_step=min(block), samples=len(block), mean=mean(block.values()))
                        for block in windows.values()]
    return {'checkpoint_iteration': iteration, 'window_updates': 10, 'metrics': metrics}
