"""Opt-in v12 continuation repairing apex event bookkeeping only."""
import torch
from .path_free_stage2_v12_env import PathFreeStage2V12Env
from .apex_event_fix_v15 import ApexEventState


class PathFreeStage2V15Env(PathFreeStage2V12Env):
    def _apex_state(self):
        if not hasattr(self, '_v15_apex_state'):
            self._v15_apex_state = ApexEventState(self.num_envs, self.device)
            self._v15_apex_totals = torch.zeros(3, device=self.device, dtype=torch.long)
        return self._v15_apex_state

    def _update_cycle_events(self):
        state = self._apex_state()
        # The inherited event detector must read a snapshot of the preceding
        # step, not a view of the refreshed simulator velocity.
        self._previous_vz = state.previous_vz
        super()._update_cycle_events()
        # The inherited method assigns a live view at its end. Restore owned
        # storage before reset or the next simulation step can use it.
        self._previous_vz = state.previous_vz
        event, fallback, suppressed = state.update(
            self._apex_event, self._liftoff_event, self._touchdown_event,
            self._initial_drop_pending, self._robot.data.root_lin_vel_w[:, 2])
        self._apex_event.copy_(event)
        self._v15_apex_totals += torch.stack((
            (event & ~self._touchdown_event).sum(), fallback.sum(), suppressed.sum()))

    def _reset_idx(self, env_ids):
        state = getattr(self, '_v15_apex_state', None)
        if state is not None:
            self._previous_vz = state.previous_vz
        super()._reset_idx(env_ids)
        if state is not None:
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            state.reset(ids)

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras['log'].update({
            'V15/airborne_apex_event_rate':
                (self._apex_event & ~self._touchdown_event).float().mean(),
            'V15/touchdown_apex_fallback_rate':
                (self._apex_event & self._touchdown_event).float().mean(),
            'V15/airborne_apex_events_total': self._v15_apex_totals[0].float(),
            'V15/touchdown_apex_fallback_events_total': self._v15_apex_totals[1].float(),
            'V15/suppressed_duplicate_apex_events_total': self._v15_apex_totals[2].float(),
        })
        return reward

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, '_v15_apex_totals'):
            result.update(dict(zip(
                ('v15_airborne_apex_events', 'v15_touchdown_apex_fallback_events',
                 'v15_suppressed_duplicate_apex_events'),
                self._v15_apex_totals.tolist())))
        return result
