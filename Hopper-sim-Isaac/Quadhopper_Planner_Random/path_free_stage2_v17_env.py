"""Opt-in v16 continuation: split the existing foot reward across two hops."""
from .path_free_stage2_v16_env import PathFreeStage2V16Env
from .path_free_stage2 import endpoint_terms
from .deferred_landing_v17 import DeferredLandingReward


class PathFreeStage2V17Env(PathFreeStage2V16Env):
    def _deferred_state(self):
        if not hasattr(self, '_v17_deferred'):
            self._v17_deferred = DeferredLandingReward(self.num_envs, self.device)
        return self._v17_deferred

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self._v17_episode_end = terminated | truncated
        return terminated, truncated

    def _get_rewards(self):
        # Parent computes cycle events, the original foot bonus and v16 death
        # correction exactly once. Re-evaluate only the pure endpoint formula.
        reward = super()._get_rewards()
        landing, _, _ = endpoint_terms(
            self._stage2_foot_error, self._stage2_valid_touchdown,
            self._stage2_power_w, self.step_dt, self.cfg)
        data = self._robot.data
        state = self._deferred_state()
        result = state.update(
            landing_reward=landing, touchdown=self._touchdown_event,
            apex=self._apex_event, cycle_active=self._cycle_active,
            initial_drop=self._initial_drop_pending,
            episode_end=self._v17_episode_end, physical_death=self._v16_physical_death,
            root_z=data.root_pos_w[:, 2], vertical_speed=data.root_lin_vel_w[:, 2],
            clearance=self.foot_position_w()[:, 2] - self.foot_geometry['sphere_radius_m'],
            # Parent has already advanced/replanned the command at touchdown.
            next_target_z=self._active_target_height)
        correction = result['correction']
        # Existing summary must report the reward actually paid, not gross.
        self._stage2_totals[2] += correction.double().sum()
        log = self.extras['log']
        log.update({
            'Stage2/foot_touchdown_reward': result['immediate'].mean(),
            'V17/foot_gross_reward': landing.mean(),
            'V17/deferred_paid_reward': result['paid'].mean(),
            'V17/reward_correction': correction.mean(),
            'V17/pending_reward': state.pending.mean(),
            'V17/pending_count': (state.pending > 0).sum().float(),
        })
        for key, value in state.totals.items():
            log[f'V17/{key}_total'] = value.clone()
        return reward + correction

    def _reset_idx(self, env_ids):
        if hasattr(self, '_v17_deferred'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._v17_deferred.reset(ids)
        super()._reset_idx(env_ids)

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, '_v17_deferred'):
            result.update(self._v17_deferred.summary())
        return result
