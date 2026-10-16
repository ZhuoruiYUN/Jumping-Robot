"""V3-only apex and physical-death reward accounting repairs.

This class deliberately inherits V3 directly. The earlier V15/V16 wrappers
inherit V12 and would also enable its recovery rewards.
"""
import torch

from .apex_event_fix_v15 import ApexEventState
from .liftoff_cross_track_v30 import liftoff_cross_track_term
from .path_free_stage2_v3_env import PathFreeStage2V3Env
from .terminal_penalty_fix_v16 import terminal_reward_correction


class PathFreeStage2V3AccountingEnv(PathFreeStage2V3Env):
    def _apex_state(self):
        if not hasattr(self, "_accounting_apex_state"):
            self._accounting_apex_state = ApexEventState(self.num_envs, self.device)
            self._accounting_apex_totals = torch.zeros(3, device=self.device, dtype=torch.long)
        return self._accounting_apex_state

    def _update_cycle_events(self):
        state = self._apex_state()
        self._previous_vz = state.previous_vz
        super()._update_cycle_events()
        # The parent retains a view of the refreshed simulator velocity.
        self._previous_vz = state.previous_vz
        event, fallback, suppressed = state.update(
            self._apex_event, self._liftoff_event, self._touchdown_event,
            self._initial_drop_pending, self._robot.data.root_lin_vel_w[:, 2],
        )
        self._apex_event.copy_(event)
        self._accounting_apex_totals += torch.stack((
            (event & ~self._touchdown_event).sum(), fallback.sum(), suppressed.sum(),
        ))

    def _get_dones(self):
        before = self._physical_death_total.clone()
        result = super()._get_dones()
        self._accounting_physical_death = self._physical_death_total > before
        return result

    def _get_rewards(self):
        correction, legacy = terminal_reward_correction(
            self._accounting_physical_death, self._robot.data.root_pos_w[:, 2],
            self._robot.data.root_quat_w, self.cfg.termination_penalty_scale,
        )
        reward = super()._get_rewards()
        if "termination_p" in self._episode_sums:
            self._episode_sums["termination_p"] += correction
        if not hasattr(self, "_accounting_terminal_totals"):
            self._accounting_terminal_totals = torch.zeros(3, device=self.device, dtype=torch.long)
        self._accounting_terminal_totals += torch.stack((
            self._accounting_physical_death.sum(),
            (self._accounting_physical_death & ~legacy).sum(),
            (legacy & ~self._accounting_physical_death).sum(),
        ))
        self.extras["log"].update({
            "V3Fix/airborne_apex_events_total": self._accounting_apex_totals[0].float(),
            "V3Fix/touchdown_apex_fallback_events_total": self._accounting_apex_totals[1].float(),
            "V3Fix/suppressed_duplicate_apex_events_total": self._accounting_apex_totals[2].float(),
            "V3Fix/terminal_reward_correction": correction.mean(),
            "V3Fix/physical_death_events_total": self._accounting_terminal_totals[0].float(),
            "V3Fix/missing_terminal_penalties_added_total": self._accounting_terminal_totals[1].float(),
            "V3Fix/nonphysical_terminal_penalties_removed_total": self._accounting_terminal_totals[2].float(),
        })
        result = reward + correction
        cross_scale = getattr(self.cfg, 'liftoff_cross_track_penalty_scale', 0.0)
        if cross_scale < 0.0:
            target, _ = self.commands.lookahead()
            term, lateral_speed, valid = liftoff_cross_track_term(
                self.foot_position_w()[:, :2], target,
                self._robot.data.root_lin_vel_w[:, :2],
                self._liftoff_event & ~self._initial_drop_pending,
                cross_scale,
            )
            result = result + term
            if not hasattr(self, '_v30_liftoff_totals'):
                self._v30_liftoff_totals = torch.zeros(3, device=self.device, dtype=torch.float64)
            self._v30_liftoff_totals += torch.stack((
                valid.sum().to(term.dtype), (term < 0).sum().to(term.dtype), term.sum(),
            )).to(torch.float64)
            self.extras['log'].update({
                'V30/liftoff_cross_track_penalty': term.mean(),
                'V30/liftoff_cross_track_speed_m_s': (
                    (lateral_speed * valid).sum() / valid.sum().clamp_min(1)),
                'V30/liftoff_events_total': self._v30_liftoff_totals[0].float(),
                'V30/penalized_liftoff_events_total': self._v30_liftoff_totals[1].float(),
                'V30/liftoff_cross_track_penalty_total': self._v30_liftoff_totals[2].float(),
                'V30/liftoff_cross_track_penalty_per_liftoff': (
                    self._v30_liftoff_totals[2] /
                    self._v30_liftoff_totals[0].clamp_min(1)).float(),
            })
        return result

    def _reset_idx(self, env_ids):
        state = getattr(self, "_accounting_apex_state", None)
        if state is not None:
            self._previous_vz = state.previous_vz
        super()._reset_idx(env_ids)
        ids = self._robot._ALL_INDICES if env_ids is None else env_ids
        if state is not None:
            state.reset(ids)
        if hasattr(self, "_accounting_physical_death"):
            self._accounting_physical_death[ids] = False

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, "_accounting_apex_totals"):
            result.update(dict(zip(
                ("v3fix_airborne_apex_events", "v3fix_touchdown_apex_fallback_events",
                 "v3fix_suppressed_duplicate_apex_events"),
                self._accounting_apex_totals.tolist(),
            )))
        if hasattr(self, "_accounting_terminal_totals"):
            result.update(dict(zip(
                ("v3fix_physical_death_events", "v3fix_missing_terminal_penalties_added",
                 "v3fix_nonphysical_terminal_penalties_removed"),
                self._accounting_terminal_totals.tolist(),
            )))
        if hasattr(self, '_v30_liftoff_totals'):
            result.update(dict(zip(
                ('v30_liftoff_events', 'v30_penalized_liftoff_events',
                 'v30_liftoff_cross_track_penalty_total'),
                self._v30_liftoff_totals.tolist(),
            )))
        return result
