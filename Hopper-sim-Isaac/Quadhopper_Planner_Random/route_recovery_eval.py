"""Evaluation-only counts for first attempts and recovery after target misses."""
import torch


class RouteRecoveryMetrics:
    def __init__(self, num_envs, device, severe_miss_m):
        self.num_envs = num_envs
        self.device = device
        self.severe_miss_m = severe_miss_m
        self.retry_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.after_miss = torch.zeros_like(self.retry_pending)
        self.first_miss_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.first_miss_foot_error_m = torch.zeros(num_envs, device=device)
        names = (
            'touchdowns', 'first_attempts', 'first_task_hits', 'first_foot_hits_10cm',
            'first_strict_foot_hits_10cm', 'retry_attempts', 'retry_task_hits',
            'retry_foot_hits_10cm', 'retry_strict_foot_hits_10cm',
            'after_miss_attempts', 'after_miss_task_hits',
            'after_miss_strict_foot_hits_10cm', 'task_misses', 'severe_misses',
            'miss_route_advances', 'severe_miss_route_advances', 'severe_miss_retries',
            'route_advances',
        )
        self.counts = {name: torch.zeros(num_envs, dtype=torch.long, device=device)
                       for name in names}
        self.miss_next_distance_sum_m = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.severe_miss_next_distance_sum_m = torch.zeros_like(self.miss_next_distance_sum_m)

    def reset_pending(self, ids):
        # Totals and first-miss anchors span the full evaluation. Only the
        # per-episode target-attempt state is cleared by an environment reset.
        self.retry_pending[ids] = False
        self.after_miss[ids] = False

    def observe(self, ids, *, task_hit, foot_error_m, settled_apex_m,
                target_apex_m, route_before, route_after,
                next_target_distance_m, step):
        if len(ids) == 0:
            return
        task_hit = task_hit.bool()
        miss = ~task_hit
        retry = self.retry_pending[ids].clone()
        followup = self.after_miss[ids].clone()
        first = ~retry
        advanced = route_after != route_before
        foot_hit = foot_error_m < .10
        strict_foot_hit = foot_hit & (settled_apex_m >= target_apex_m)
        severe = miss & (foot_error_m > self.severe_miss_m)

        values = dict(
            touchdowns=torch.ones_like(miss),
            first_attempts=first,
            first_task_hits=first & task_hit,
            first_foot_hits_10cm=first & foot_hit,
            first_strict_foot_hits_10cm=first & strict_foot_hit,
            retry_attempts=retry,
            retry_task_hits=retry & task_hit,
            retry_foot_hits_10cm=retry & foot_hit,
            retry_strict_foot_hits_10cm=retry & strict_foot_hit,
            after_miss_attempts=followup,
            after_miss_task_hits=followup & task_hit,
            after_miss_strict_foot_hits_10cm=followup & strict_foot_hit,
            task_misses=miss,
            severe_misses=severe,
            miss_route_advances=miss & advanced,
            severe_miss_route_advances=severe & advanced,
            severe_miss_retries=severe & ~advanced,
            route_advances=advanced,
        )
        for name, value in values.items():
            self.counts[name][ids] += value.long()
        self.miss_next_distance_sum_m[ids] += torch.where(
            miss, next_target_distance_m.double(), 0.0)
        self.severe_miss_next_distance_sum_m[ids] += torch.where(
            severe, next_target_distance_m.double(), 0.0)
        first_miss = miss & (self.first_miss_step[ids] < 0)
        if torch.any(first_miss):
            first_ids = ids[first_miss]
            self.first_miss_step[first_ids] = step
            self.first_miss_foot_error_m[first_ids] = foot_error_m[first_miss]
        # A miss counts as a same-target retry only when route_index did not
        # advance. A hit always finishes the target, even on a retry.
        self.retry_pending[ids] = miss & ~advanced
        self.after_miss[ids] = miss

    def summary(self):
        counts = {name: int(value.sum().item()) for name, value in self.counts.items()}
        def rate(numerator, denominator):
            return counts[numerator] / counts[denominator] if counts[denominator] else None
        counts.update(
            first_task_hit_rate=rate('first_task_hits', 'first_attempts'),
            first_strict_foot_hit_10cm_rate=rate('first_strict_foot_hits_10cm', 'first_attempts'),
            retry_task_hit_rate=rate('retry_task_hits', 'retry_attempts'),
            retry_strict_foot_hit_10cm_rate=rate('retry_strict_foot_hits_10cm', 'retry_attempts'),
            after_miss_task_hit_rate=rate('after_miss_task_hits', 'after_miss_attempts'),
            after_miss_strict_foot_hit_10cm_rate=rate(
                'after_miss_strict_foot_hits_10cm', 'after_miss_attempts'),
            mean_actual_next_distance_after_miss_m=(
                self.miss_next_distance_sum_m.sum().item() / counts['task_misses']
                if counts['task_misses'] else None),
            mean_actual_next_distance_after_severe_miss_m=(
                self.severe_miss_next_distance_sum_m.sum().item() / counts['severe_misses']
                if counts['severe_misses'] else None),
            first_miss_step_by_env=[None if x < 0 else x
                                    for x in self.first_miss_step.tolist()],
            first_miss_foot_error_m_by_env=[
                None if step < 0 else error
                for step, error in zip(self.first_miss_step.tolist(),
                                       self.first_miss_foot_error_m.tolist())],
        )
        return counts
