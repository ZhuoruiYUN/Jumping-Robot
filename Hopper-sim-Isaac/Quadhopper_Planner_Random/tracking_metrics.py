"""Whole-evaluation touchdown counters; episode resets never erase totals."""
import torch


class TrackingMetrics:
    def __init__(self, num_envs, device):
        self.tolerances = torch.tensor([0.05, 0.10, 0.15], device=device)
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.first_hits = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.count = torch.zeros(2, device=device, dtype=torch.float64)
        self.error = torch.zeros_like(self.count)
        self.xy_hits = torch.zeros(2, 3, device=device, dtype=torch.float64)
        self.valid_hits = torch.zeros_like(self.xy_hits)
        self.strict_valid_hits = torch.zeros_like(self.xy_hits)
        self.pair_started = torch.zeros((), device=device, dtype=torch.float64)
        self.pair_completed = torch.zeros_like(self.pair_started)
        self.pair_hits = torch.zeros(3, device=device, dtype=torch.float64)
        self.strict_pair_hits = torch.zeros_like(self.pair_hits)
        self.strict_first_hits = torch.zeros_like(self.first_hits)
        self.apex_sum = torch.zeros_like(self.pair_started)
        self.apex_sq_sum = torch.zeros_like(self.pair_started)
        self.apex_min = torch.full_like(self.pair_started, torch.inf)
        self.apex_max = torch.full_like(self.pair_started, -torch.inf)
        self.apex_error_sum = torch.zeros_like(self.pair_started)
        self.height_hits = torch.zeros_like(self.pair_started)
        self.strict_height_hits = torch.zeros_like(self.pair_started)
        self.height_95cm_hits = torch.zeros_like(self.pair_started)
        self.landing_error_sq_sum = torch.zeros_like(self.pair_started)
        self.landing_error_min = torch.full_like(self.pair_started, torch.inf)
        self.landing_error_max = torch.zeros_like(self.pair_started)
        self.along_sum = torch.zeros_like(self.pair_started)
        self.lateral_sum = torch.zeros_like(self.pair_started)
        # Reliability statistics are independent of episodic reset counters:
        # they answer whether an environment made every observed touchdown
        # during a long evaluation, and how long its uninterrupted run was.
        self.env_touchdowns = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.env_valid_misses = torch.zeros(num_envs, 3, device=device, dtype=torch.long)
        self.current_valid_streak = torch.zeros(num_envs, 3, device=device, dtype=torch.long)
        self.max_valid_streak = torch.zeros_like(self.current_valid_streak)
        self.touchdown_tilt_deg = []
        self.env_touchdown_tilt_ge_15deg = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

    def observe(self, ids, short, error, along, lateral, apex, height,
                touchdown_tilt_deg=None):
        if len(ids) == 0:
            return
        if touchdown_tilt_deg is not None:
            tilt = touchdown_tilt_deg.detach()
            self.touchdown_tilt_deg.append(tilt)
            self.env_touchdown_tilt_ge_15deg[ids] |= tilt >= 15.0
        height_ok = torch.abs(apex - height) <= 0.15
        strict_height_ok = apex >= height
        xy = error[:, None] < self.tolerances
        valid = xy & height_ok[:, None]
        strict_valid = xy & strict_height_ok[:, None]
        for phase, mask in enumerate((short, ~short)):
            self.count[phase] += mask.sum()
            self.error[phase] += error[mask].sum()
            self.xy_hits[phase] += xy[mask].sum(0)
            self.valid_hits[phase] += valid[mask].sum(0)
            self.strict_valid_hits[phase] += strict_valid[mask].sum(0)
        self.apex_sum += apex.sum()
        self.apex_sq_sum += apex.square().sum()
        self.apex_min.copy_(torch.minimum(self.apex_min, apex.min().double()))
        self.apex_max.copy_(torch.maximum(self.apex_max, apex.max().double()))
        self.apex_error_sum += (apex - height).abs().sum()
        self.height_hits += height_ok.sum()
        self.strict_height_hits += strict_height_ok.sum()
        self.height_95cm_hits += (apex >= height - 0.05).sum()
        self.landing_error_sq_sum += error.square().sum()
        self.landing_error_min.copy_(torch.minimum(self.landing_error_min, error.min().double()))
        self.landing_error_max.copy_(torch.maximum(self.landing_error_max, error.max().double()))
        self.along_sum += along.sum()
        self.lateral_sum += lateral.sum()
        self.env_touchdowns[ids] += 1
        self.env_valid_misses[ids] += (~valid).long()
        next_streak = torch.where(
            valid,
            self.current_valid_streak[ids] + 1,
            torch.zeros_like(self.current_valid_streak[ids]),
        )
        self.current_valid_streak[ids] = next_streak
        self.max_valid_streak[ids] = torch.maximum(
            self.max_valid_streak[ids], next_streak
        )
        first = ids[short]
        self.pair_started += len(first)
        self.pending[first] = True
        self.first_hits[first] = valid[short]
        self.strict_first_hits[first] = strict_valid[short]
        second = ids[~short]
        eligible = self.pending[second]
        self.pair_completed += eligible.sum()
        self.pair_hits += (
            self.first_hits[second] & valid[~short] & eligible[:, None]
        ).sum(0)
        self.strict_pair_hits += (
            self.strict_first_hits[second] & strict_valid[~short] & eligible[:, None]
        ).sum(0)
        self.pending[second] = False

    def reset(self, ids):
        # A death between hops leaves a failed pair in the started denominator.
        self.pending[ids] = False
        self.first_hits[ids] = False
        self.strict_first_hits[ids] = False
        # A physical reset breaks a continuous sequence, but prior failures
        # and maximum streaks remain part of the full-evaluation totals.
        self.current_valid_streak[ids] = 0

    def summary(self):
        n = self.count.sum().clamp_min(1)
        apex_mean = self.apex_sum / n
        apex_std = torch.sqrt((self.apex_sq_sum / n - apex_mean.square()).clamp_min(0.0))
        landing_mean = self.error.sum() / n
        landing_std = torch.sqrt(
            (self.landing_error_sq_sum / n - landing_mean.square()).clamp_min(0.0)
        )
        has_samples = bool(self.count.sum().item())
        out = {
            'touchdowns': int(self.count.sum().item()),
            'apex_m': apex_mean.item(),
            'apex_std_m': apex_std.item(),
            'apex_min_m': self.apex_min.item() if has_samples else 0.0,
            'apex_max_m': self.apex_max.item() if has_samples else 0.0,
            'apex_error_m': (self.apex_error_sum / n).item(),
            'strict_apex_ge_target_rate': (self.strict_height_hits / n).item(),
            'apex_ge_target_minus_5cm_rate': (self.height_95cm_hits / n).item(),
            # Kept for compatibility; this is the historical +/-15 cm gate.
            'height_ok_rate': (self.height_hits / n).item(),
            'xy_error_m': landing_mean.item(),
            'xy_error_std_m': landing_std.item(),
            'xy_error_min_m': self.landing_error_min.item() if has_samples else 0.0,
            'xy_error_max_m': self.landing_error_max.item() if has_samples else 0.0,
            'along_error_m': (self.along_sum / n).item(),
            'lateral_error_m': (self.lateral_sum / n).item(),
            'pairs_started': int(self.pair_started.item()),
            'pairs_completed': int(self.pair_completed.item()),
            'pairs_pending_at_end': int(self.pending.sum().item()),
        }
        if self.touchdown_tilt_deg:
            touchdown_tilt = torch.cat(self.touchdown_tilt_deg)
            over_limit = touchdown_tilt >= 15.0
            out.update(
                touchdown_tilt_p90_deg=float(torch.quantile(touchdown_tilt, .9).item()),
                touchdown_tilt_max_deg=float(touchdown_tilt.max().item()),
                touchdown_tilt_ge_15deg_count=int(over_limit.sum().item()),
                touchdown_tilt_ge_15deg_rate=float(over_limit.float().mean().item()),
                env_any_touchdown_tilt_ge_15deg_rate=float(
                    self.env_touchdown_tilt_ge_15deg.float().mean().item()
                ),
            )
        for i, cm in enumerate((5, 10, 15)):
            out[f'xy_hit_{cm}cm'] = (self.xy_hits[:, i].sum() / n).item()
            out[f'valid_hit_{cm}cm'] = (self.valid_hits[:, i].sum() / n).item()
            out[f'strict_valid_hit_{cm}cm'] = (self.strict_valid_hits[:, i].sum() / n).item()
            out[f'pair_valid_{cm}cm'] = (
                self.pair_hits[i] / self.pair_started.clamp_min(1)
            ).item()
            out[f'strict_pair_valid_{cm}cm'] = (
                self.strict_pair_hits[i] / self.pair_started.clamp_min(1)
            ).item()
            evaluated = self.env_touchdowns > 0
            counts = self.env_touchdowns[evaluated]
            max_streaks = self.max_valid_streak[evaluated, i]
            misses = self.env_valid_misses[evaluated, i]
            out[f'valid_misses_{cm}cm'] = int(misses.sum().item())
            out[f'zero_miss_env_rate_{cm}cm'] = (
                (misses == 0).float().mean().item() if len(misses) else 0.0
            )
            out[f'max_consecutive_valid_{cm}cm_mean'] = (
                max_streaks.float().mean().item() if len(max_streaks) else 0.0
            )
            out[f'max_consecutive_valid_{cm}cm_min'] = (
                int(max_streaks.min().item()) if len(max_streaks) else 0
            )
            out[f'max_consecutive_valid_{cm}cm_max'] = (
                int(max_streaks.max().item()) if len(max_streaks) else 0
            )
            out[f'min_touchdowns_per_env'] = (
                int(counts.min().item()) if len(counts) else 0
            )
        for i, phase in enumerate(('first', 'second')):
            out[f'{phase}_touchdowns'] = int(self.count[i].item())
            out[f'{phase}_xy_error_m'] = (self.error[i] / self.count[i].clamp_min(1)).item()
            for tolerance_index, cm in enumerate((5, 10, 15)):
                denominator = self.count[i].clamp_min(1)
                out[f'{phase}_xy_hit_{cm}cm'] = (
                    self.xy_hits[i, tolerance_index] / denominator
                ).item()
                out[f'{phase}_valid_hit_{cm}cm'] = (
                    self.valid_hits[i, tolerance_index] / denominator
                ).item()
                out[f'{phase}_strict_valid_hit_{cm}cm'] = (
                    self.strict_valid_hits[i, tolerance_index] / denominator
                ).item()
        return out
