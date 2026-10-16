"""Opt-in v15 continuation: repair missing physical-death reward only."""
import torch
from .path_free_stage2_v15_env import PathFreeStage2V15Env
from .terminal_penalty_fix_v16 import terminal_reward_correction


class PathFreeStage2V16Env(PathFreeStage2V15Env):
    def _get_dones(self):
        before = self._physical_death_total.clone()
        result = super()._get_dones()
        # _get_dones runs before rewards in DirectRLEnv.step. Use the actual
        # physical mask; returned done also includes success and time limits.
        self._v16_physical_death = self._physical_death_total > before
        return result

    def _get_rewards(self):
        correction, legacy = terminal_reward_correction(
            self._v16_physical_death, self._robot.data.root_pos_w[:, 2],
            self._robot.data.root_quat_w, self.cfg.termination_penalty_scale)
        reward = super()._get_rewards()
        if 'termination_p' in self._episode_sums:
            self._episode_sums['termination_p'] += correction
        if not hasattr(self, '_v16_terminal_totals'):
            self._v16_terminal_totals = torch.zeros(3, device=self.device, dtype=torch.long)
        self._v16_terminal_totals += torch.stack((
            self._v16_physical_death.sum(),
            (self._v16_physical_death & ~legacy).sum(),
            (legacy & ~self._v16_physical_death).sum()))
        self.extras['log'].update({
            'V16/terminal_reward_correction': correction.mean(),
            'V16/physical_death_events_total': self._v16_terminal_totals[0].float(),
            'V16/missing_terminal_penalties_added_total': self._v16_terminal_totals[1].float(),
            'V16/nonphysical_terminal_penalties_removed_total': self._v16_terminal_totals[2].float(),
        })
        return reward + correction

    def _reset_idx(self, env_ids):
        if hasattr(self, '_v16_physical_death'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._v16_physical_death[ids] = False
        super()._reset_idx(env_ids)

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, '_v16_terminal_totals'):
            result.update(dict(zip(
                ('v16_physical_death_events', 'v16_missing_terminal_penalties_added',
                 'v16_nonphysical_terminal_penalties_removed'),
                self._v16_terminal_totals.tolist())))
        return result
