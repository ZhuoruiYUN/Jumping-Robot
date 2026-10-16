"""Reward-only escrow from a scored touchdown to the next airborne apex.

No simulator imports. A spring liftoff alone cannot release the escrow.
"""
import torch


SETTINGS_V17 = {
    'deferred_fraction': 0.5,
    'ascent_clearance_m': 0.035,
    'ascent_speed_m_s': 0.10,
    'ascent_confirmation_steps': 2,
    'quality': 'clamp((next_measured_apex-touchdown_root_z)/(next_target_root_z-touchdown_root_z),0,1)',
    'settlement': 'first_airborne_apex_after_confirmed_ascent',
    'cancellation': 'next_touchdown_without_settlement_or_episode_end_or_reset',
    'discount_compensation': False,
}


class DeferredLandingReward:
    def __init__(self, num_envs, device):
        self.pending = torch.zeros(num_envs, device=device)
        self.origin_z = torch.zeros_like(self.pending)
        self.target_z = torch.ones_like(self.pending)
        self.peak_z = torch.zeros_like(self.pending)
        self.ascent_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.ascent_seen = torch.zeros(num_envs, device=device, dtype=torch.bool)
        # Each reset starts a new sequence; pending never survives a reset.
        self.touchdown_sequence = torch.zeros_like(self.ascent_count)
        self.pending_sequence = torch.full_like(self.ascent_count, -1)
        self.totals = {key: torch.zeros((), device=device, dtype=torch.float64) for key in (
            'gross_reward', 'withheld_reward', 'paid_reward', 'quality_forfeit_reward',
            'cancelled_reward', 'death_cancelled_reward', 'reset_cancelled_reward',
            'reservations', 'settlements', 'full_quality_settlements',
        )}

    def _clear(self, ids):
        self.pending[ids] = 0
        self.ascent_count[ids] = 0
        self.ascent_seen[ids] = False
        self.pending_sequence[ids] = -1

    def reset(self, ids):
        amount = self.pending[ids].double().sum()
        self.totals['cancelled_reward'] += amount
        self.totals['reset_cancelled_reward'] += amount
        self._clear(ids)
        self.touchdown_sequence[ids] = 0

    def update(self, *, landing_reward, touchdown, apex, cycle_active,
               initial_drop, episode_end, physical_death, root_z, vertical_speed,
               clearance, next_target_z):
        """Resolve old escrow before opening new escrow on this touchdown.

        A touchdown-apex fallback belongs to the ending hop and never pays.
        Episode ends take precedence over all rewards held in escrow, including
        a new reservation on that frame. The immediate half is unchanged.
        """
        existing = self.pending > 0
        cancelled_mask = existing & (touchdown | initial_drop | episode_end)
        cancelled = torch.where(cancelled_mask, self.pending, 0.0)
        self.totals['cancelled_reward'] += cancelled.double().sum()
        self.totals['death_cancelled_reward'] += cancelled[physical_death].double().sum()
        self._clear(cancelled_mask)

        active = ((self.pending > 0) & cycle_active & ~initial_drop
                  & ~touchdown & ~episode_end)
        rising = (active & (clearance > SETTINGS_V17['ascent_clearance_m'])
                  & (vertical_speed > SETTINGS_V17['ascent_speed_m_s']))
        self.ascent_count = torch.where(rising, self.ascent_count + 1, 0)
        self.ascent_seen |= self.ascent_count >= SETTINGS_V17['ascent_confirmation_steps']
        self.peak_z = torch.where(active, torch.maximum(self.peak_z, root_z), self.peak_z)
        settle = active & apex & self.ascent_seen
        gain = self.target_z - self.origin_z
        quality = torch.where(
            gain > 1.0e-6,
            ((self.peak_z - self.origin_z) / gain.clamp_min(1.0e-6)).clamp(0.0, 1.0),
            0.0)
        paid = torch.where(settle, self.pending * quality, 0.0)
        forfeited = torch.where(settle, self.pending - paid, 0.0)
        self.totals['paid_reward'] += paid.double().sum()
        self.totals['quality_forfeit_reward'] += forfeited.double().sum()
        self.totals['settlements'] += settle.sum()
        self.totals['full_quality_settlements'] += (settle & (quality >= 1.0)).sum()
        settled_sequence = torch.where(settle, self.pending_sequence, -1)
        self._clear(settle)

        self.touchdown_sequence += touchdown.long()
        withheld = landing_reward * SETTINGS_V17['deferred_fraction']
        self.totals['gross_reward'] += landing_reward.double().sum()
        self.totals['withheld_reward'] += withheld.double().sum()
        reserve = touchdown & ~initial_drop & ~episode_end & (withheld > 0)
        self.pending[reserve] = withheld[reserve]
        self.origin_z[reserve] = root_z[reserve]
        self.peak_z[reserve] = root_z[reserve]
        self.target_z[reserve] = next_target_z[reserve]
        self.pending_sequence[reserve] = self.touchdown_sequence[reserve]
        self.totals['reservations'] += reserve.sum()
        # This also accounts for withheld reward on a terminal touchdown.
        rejected = torch.where(reserve, 0.0, withheld)
        self.totals['cancelled_reward'] += rejected.double().sum()
        self.totals['death_cancelled_reward'] += rejected[physical_death].double().sum()
        return {
            'correction': paid - withheld,
            'immediate': landing_reward - withheld,
            'paid': paid,
            'settled': settle,
            'settled_sequence': settled_sequence,
        }

    def summary(self):
        result = {f'v17_{key}': value.item() for key, value in self.totals.items()}
        result['v17_pending_reward'] = self.pending.double().sum().item()
        result['v17_pending_count'] = (self.pending > 0).sum().item()
        result['v17_accounting_residual'] = (
            result['v17_withheld_reward'] - result['v17_paid_reward']
            - result['v17_quality_forfeit_reward'] - result['v17_cancelled_reward']
            - result['v17_pending_reward'])
        return result
