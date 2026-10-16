"""Replace V3's root-based pair bonus with strict foot-pair/posture quality."""
import torch

from .foot_pair_reward_v33 import FootPairRewardState
from .path_free_stage2_v3_accounting_env import PathFreeStage2V3AccountingEnv
from .tracking_progress import two_hop_pair_success


class PathFreeStage2V33Env(PathFreeStage2V3AccountingEnv):
    def _update_cycle_events(self):
        if not hasattr(self, '_v33_pair_state'):
            self._v33_pair_state = FootPairRewardState(self.num_envs, self.device)
            self._v33_pair_totals = torch.zeros(3, device=self.device, dtype=torch.float64)
        phase_was_short = (self.commands.route_index % 2) == 0
        super()._update_cycle_events()
        self._v33_pair_bonus, strict_foot, tilt = self._v33_pair_state.observe(
            touchdown=self._touchdown_event,
            initial_drop=self._initial_drop_pending,
            phase_was_short=phase_was_short,
            foot_error_m=self._stage2_foot_error,
            apex_m=self._settled_apex_height,
            target_apex_m=self._settled_apex_target,
            quat_wxyz=self._robot.data.root_quat_w,
            pair_reward_scale=self.cfg.two_hop_pair_reward_scale,
            pair_xy_quality_weight=self.cfg.stage2_pair_xy_quality_weight,
        )
        self._v33_strict_foot_touchdown = strict_foot & self._touchdown_event
        self._v33_touchdown_tilt = tilt

    def _get_rewards(self):
        reward = super()._get_rewards()
        root_pair = two_hop_pair_success(
            self._final_touchdown_event,
            self._target_hit_event,
            self._consecutive_hits,
        ).float() * self.cfg.two_hop_pair_reward_scale
        correction = self._v33_pair_bonus - root_pair
        reward = reward + correction
        self._episode_sums['two_hop_pair_hit'] += correction
        self._v33_pair_totals += torch.stack((
            (self._v33_pair_bonus > 0).sum(),
            self._v33_pair_bonus.sum(),
            root_pair.sum(),
        )).to(torch.float64)
        self.extras['log'].update({
            'Guidance/two_hop_pair_reward': self._v33_pair_bonus.mean(),
            'V33/strict_foot_touchdowns': self._v33_strict_foot_touchdown.sum().float(),
            'V33/touchdown_tilt_deg': (
                (self._v33_touchdown_tilt * self._touchdown_event).sum()
                / self._touchdown_event.sum().clamp_min(1)
            ) * (180.0 / torch.pi),
            'V33/foot_pair_events_total': self._v33_pair_totals[0].float(),
            'V33/foot_pair_reward_total': self._v33_pair_totals[1].float(),
            'V33/replaced_root_pair_reward_total': self._v33_pair_totals[2].float(),
        })
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, '_v33_pair_state'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._v33_pair_state.reset(ids)

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, '_v33_pair_totals'):
            result.update(dict(zip(
                ('v33_foot_pair_events', 'v33_foot_pair_reward_total',
                 'v33_replaced_root_pair_reward_total'),
                self._v33_pair_totals.tolist(),
            )))
        return result
