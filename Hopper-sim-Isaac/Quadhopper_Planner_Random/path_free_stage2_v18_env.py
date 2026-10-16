"""Opt-in v16 continuation: measured height recovery after low moving hops."""
import torch
from .path_free_stage2_v16_env import PathFreeStage2V16Env
from .low_hop_recovery_v18 import LowHopRecoveryDebt, SETTINGS_V18


class PathFreeStage2V18Env(PathFreeStage2V16Env):
    def _recovery_debt(self):
        if not hasattr(self, '_v18_recovery_debt'):
            self._v18_recovery_debt = LowHopRecoveryDebt(self.num_envs, self.device)
        return self._v18_recovery_debt

    def _update_cycle_events(self):
        # Capture the completed command before touchdown advances the queue.
        # The generic stationary radius is 0.15 m and includes some prescribed
        # 0.12-0.15 m short hops. This reward distinguishes them from true zero.
        target = self.commands.lookahead()[0]
        distance = torch.linalg.norm(target - self.commands.anchor_w, dim=1)
        super()._update_cycle_events()
        if not hasattr(self, '_v18_completed_nonzero_command'):
            self._v18_completed_nonzero_command = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool)
        self._v18_completed_nonzero_command = torch.where(
            self._touchdown_event,
            distance > SETTINGS_V18['minimum_command_distance_m'],
            self._v18_completed_nonzero_command)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self._v18_episode_end = terminated | truncated
        return terminated, truncated

    def _get_rewards(self):
        reward = super()._get_rewards()
        debt = self._recovery_debt()
        height_reward = debt.update(
            touchdown=self._touchdown_event,
            previous_moving=self._v18_completed_nonzero_command,
            previous_apex=self._settled_apex_height,
            previous_target=self._settled_apex_target,
            previous_valid=self._settled_apex_valid,
            # Parent has advanced the command on touchdown.
            next_target=self._active_target_height,
            root_z=self._robot.data.root_pos_w[:, 2],
            cycle_active=self._cycle_active,
            initial_drop=self._initial_drop_pending,
            episode_end=self._v18_episode_end,
            physical_death=self._v16_physical_death)
        self.extras['log'].update({
            'V18/height_recovery_reward': height_reward.mean(),
            'V18/active_debt': (debt.debt - debt.paid).clamp_min(0.0).mean(),
            'V18/active_count': debt.active.sum().float(),
        })
        for key, value in debt.totals.items():
            self.extras['log'][f'V18/{key}_total'] = value.clone()
        return reward + height_reward

    def _reset_idx(self, env_ids):
        if hasattr(self, '_v18_recovery_debt'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._v18_recovery_debt.reset(ids)
            self._v18_completed_nonzero_command[ids] = False
        super()._reset_idx(env_ids)

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, '_v18_recovery_debt'):
            result.update(self._v18_recovery_debt.summary())
        return result
