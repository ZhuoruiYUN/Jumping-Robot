"""Cross-hop height-debt shaping for a low moving touchdown.

The debt is charged first and can only be repaid by measured next-hop height.
This keeps a deliberate low previous hop from creating positive undiscounted
return. No simulator or command-generation dependency.
"""
import torch


SETTINGS_V18 = {
    'shortfall_debt_per_m': 60.0,
    'maximum_debt': 30.0,
    'repayment': 'measured_next_hop_root_z_gain_to_next_target',
    'source': 'every_non_initial_command_distance_over_0.01m_touchdown_below_its_apex_target',
    'minimum_command_distance_m': 0.01,
    'terminal_policy': 'unrepaid_debt_is_forfeited',
}


class LowHopRecoveryDebt:
    def __init__(self, num_envs, device):
        self.debt = torch.zeros(num_envs, device=device)
        self.origin_z = torch.zeros_like(self.debt)
        self.target_z = torch.ones_like(self.debt)
        self.peak_z = torch.zeros_like(self.debt)
        self.paid = torch.zeros_like(self.debt)
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.totals = {key: torch.zeros((), device=device, dtype=torch.float64) for key in (
            'charged', 'repaid', 'forfeited', 'death_forfeited',
            'low_touchdowns', 'recovery_hops', 'fully_recovered',
        )}

    def _forfeit(self, ids, physical_death):
        remaining = (self.debt - self.paid).clamp_min(0.0)
        amount = remaining[ids].double().sum()
        self.totals['forfeited'] += amount
        self.totals['death_forfeited'] += remaining[ids & physical_death].double().sum()
        self.debt[ids] = 0.0
        self.paid[ids] = 0.0
        self.active[ids] = False

    def reset(self, ids):
        mask = torch.zeros_like(self.active)
        mask[ids] = self.active[ids]
        self._forfeit(mask, torch.zeros_like(mask))

    def update(self, *, touchdown, previous_moving, previous_apex,
               previous_target, previous_valid, next_target, root_z,
               cycle_active, initial_drop, episode_end, physical_death):
        old_end = self.active & (touchdown | initial_drop | episode_end)
        self._forfeit(old_end, physical_death)

        eligible = self.active & cycle_active & ~touchdown & ~initial_drop & ~episode_end
        old_peak = self.peak_z.clone()
        new_peak = torch.where(eligible, torch.maximum(old_peak, root_z), old_peak)
        gain = (self.target_z - self.origin_z).clamp_min(1.0e-6)
        old_quality = ((old_peak - self.origin_z) / gain).clamp(0.0, 1.0)
        new_quality = ((new_peak - self.origin_z) / gain).clamp(0.0, 1.0)
        credit = torch.where(eligible, self.debt * (new_quality - old_quality).clamp_min(0.0), 0.0)
        self.peak_z.copy_(new_peak)
        self.paid += credit
        self.totals['repaid'] += credit.double().sum()

        source = (touchdown & previous_moving & previous_valid
                  & ~initial_drop & ~episode_end)
        shortfall = (previous_target - previous_apex).clamp_min(0.0)
        charged = torch.where(source, (shortfall * SETTINGS_V18['shortfall_debt_per_m'])
                              .clamp_max(SETTINGS_V18['maximum_debt']), 0.0)
        opening = charged > 0.0
        self.debt[opening] = charged[opening]
        self.paid[opening] = 0.0
        self.origin_z[opening] = root_z[opening]
        self.target_z[opening] = next_target[opening]
        self.peak_z[opening] = root_z[opening]
        self.active[opening] = True
        self.totals['charged'] += charged.double().sum()
        self.totals['low_touchdowns'] += opening.sum()
        # Count the first measured progress, even when the next apex is low.
        started = eligible & (old_peak == self.origin_z) & (new_peak > old_peak)
        self.totals['recovery_hops'] += started.sum()
        full = eligible & (old_quality < 1.0) & (new_quality >= 1.0)
        self.totals['fully_recovered'] += full.sum()
        return credit - charged

    def summary(self):
        result = {f'v18_{key}': value.item() for key, value in self.totals.items()}
        result['v18_active_debt'] = (self.debt - self.paid).clamp_min(0.0).double().sum().item()
        result['v18_active_count'] = self.active.sum().item()
        result['v18_accounting_residual'] = (
            result['v18_charged'] - result['v18_repaid']
            - result['v18_forfeited'] - result['v18_active_debt'])
        return result
