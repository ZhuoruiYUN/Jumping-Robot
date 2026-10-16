"""Read-only evaluation of actual distance after a missed route touchdown.

Each queued long command is paired with its next touchdown in the same episode.
The waypoint queue and the robot's measured foot have different origins after
a miss, so the nominal command radius alone understates the physical demand.
"""
import torch


class NextHopDistanceAudit:
    def __init__(self, num_envs, device, *, long_radius_max_m):
        if num_envs < 1 or long_radius_max_m <= 0:
            raise ValueError('expected positive environment count and long radius')
        self.long_radius_max_m = float(long_radius_max_m)
        self.pending = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.pending_over_cap = torch.zeros_like(self.pending)
        self.exposure_count = torch.zeros(2, device=device, dtype=torch.float64)
        self.actual_distance_sum = torch.zeros_like(self.exposure_count)
        self.actual_distance_max = torch.zeros_like(self.exposure_count)
        self.followup_count = torch.zeros_like(self.exposure_count)
        self.followup_apex_sum = torch.zeros_like(self.exposure_count)
        self.followup_low_apex_count = torch.zeros_like(self.exposure_count)
        self.reset_before_followup_count = torch.zeros_like(self.exposure_count)

    def observe(self, *, ids, old_target_xy, next_target_xy, foot_xy,
                next_is_short, completed_apex_m, completed_target_m):
        if len(ids) == 0:
            return
        # Consume the prior touchdown's queued long hop first. The current
        # settled apex belongs to that completed hop, not the newly queued one.
        prior = self.pending[ids]
        for category, over_cap in enumerate((False, True)):
            mask = prior & (self.pending_over_cap[ids] == over_cap)
            apex = completed_apex_m[mask]
            self.followup_count[category] += mask.sum()
            self.followup_apex_sum[category] += apex.double().sum()
            self.followup_low_apex_count[category] += (
                apex < completed_target_m[mask]).sum()

        nominal = torch.linalg.norm(next_target_xy - old_target_xy, dim=-1)
        actual = torch.linalg.norm(next_target_xy - foot_xy, dim=-1)
        queued_long = ~next_is_short & (nominal > 1e-6)
        over_cap = actual > self.long_radius_max_m + 1e-6
        for category, is_over_cap in enumerate((False, True)):
            mask = queued_long & (over_cap == is_over_cap)
            distance = actual[mask]
            self.exposure_count[category] += mask.sum()
            self.actual_distance_sum[category] += distance.double().sum()
            if len(distance):
                self.actual_distance_max[category] = torch.maximum(
                    self.actual_distance_max[category], distance.max().double())
        self.pending[ids] = queued_long
        self.pending_over_cap[ids] = queued_long & over_cap

    def reset(self, ids):
        for category, over_cap in enumerate((False, True)):
            self.reset_before_followup_count[category] += (
                self.pending[ids] & (self.pending_over_cap[ids] == over_cap)
            ).sum()
        self.pending[ids] = False
        self.pending_over_cap[ids] = False

    def summary(self):
        out = {}
        for category, label in enumerate(('within_cap', 'over_cap')):
            prefix = f'next_long_{label}'
            count = self.exposure_count[category]
            followup = self.followup_count[category]
            out[f'{prefix}_count'] = int(count.item())
            out[f'{prefix}_actual_mean_m'] = float(
                (self.actual_distance_sum[category] / count.clamp_min(1)).item())
            out[f'{prefix}_actual_max_m'] = float(
                self.actual_distance_max[category].item())
            out[f'{prefix}_followup_count'] = int(followup.item())
            out[f'{prefix}_followup_apex_mean_m'] = float(
                (self.followup_apex_sum[category] / followup.clamp_min(1)).item())
            out[f'{prefix}_followup_low_apex_rate'] = float(
                (self.followup_low_apex_count[category] / followup.clamp_min(1)).item())
            out[f'{prefix}_reset_before_followup_count'] = int(
                self.reset_before_followup_count[category].item())
        total = self.exposure_count.sum()
        out['next_long_count'] = int(total.item())
        out['next_long_over_cap_rate'] = float(
            (self.exposure_count[1] / total.clamp_min(1)).item())
        out['next_long_pending_at_end'] = int(self.pending.sum().item())
        out['next_long_cap_m'] = self.long_radius_max_m
        return out
