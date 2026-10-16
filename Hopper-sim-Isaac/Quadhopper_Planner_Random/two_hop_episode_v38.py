"""Two attempted hops per episode, with a short post-landing safety window.

The initial random drop is setup, not a commanded hop. A miss consumes its
waypoint through the existing advance-on-miss route rule. This wrapper changes
only task termination and the command shown during the final landing window.
"""

import math

import torch

from .path_free_stage2_v33_env import PathFreeStage2V33Env
from .two_hop_episode_v38_config import (
    SETTLE_TIME_S, SETTLE_TILT_DEG, SETTLE_XY_RATE_RAD_S,
)


class TwoHopEpisodeV38Env(PathFreeStage2V33Env):
    def _ensure_pair_state(self):
        if hasattr(self, "_pair_settle_remaining"):
            return
        self._pair_settle_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._pair_settle_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pair_settle_bad = torch.zeros_like(self._pair_settle_active)
        self._pair_settle_penalized = torch.zeros_like(self._pair_settle_active)
        self._pair_finish_event = torch.zeros_like(self._pair_settle_active)
        self._pair_total = torch.zeros(3, dtype=torch.long, device=self.device)
        # Overlapping reasons on the first bad sample: tilt, XY rate, liftoff.
        self._pair_bad_reasons = torch.zeros(3, dtype=torch.long, device=self.device)
        self._pair_first_strict = torch.zeros_like(self._pair_settle_active)
        self._pair_strict_current = torch.zeros_like(self._pair_settle_active)
        # Rows: uniform, reversal, stationary transitions. Columns: first
        # attempts/hits, second attempts/hits, strict pairs, stable strict pairs.
        self._pair_strict_totals = torch.zeros(
            3, 6, dtype=torch.long, device=self.device
        )
        self._episode_sums["v38_settle_penalty"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _advance_route(self, env_ids):
        # Scoring snapshots the old target before this call. Keep the final
        # landing as a zero-displacement hold command until the terminal sample
        # instead of immediately asking for an untrained third jump.
        second = ~self.commands.current_hop_is_short(env_ids)
        super()._advance_route(env_ids)
        second_ids = env_ids[second]
        if len(second_ids) > 0:
            landing_xy = self._robot.data.root_pos_w[second_ids, :2]
            self.commands.anchor_w[second_ids] = landing_xy
            self.commands.targets_w[second_ids] = landing_xy[:, None, :]

    def _update_cycle_events(self):
        self._pair_phase_was_short = (self.commands.route_index % 2) == 0
        super()._update_cycle_events()

    def _get_dones(self):
        died, timeout = super()._get_dones()
        if not hasattr(self, "_pair_settle_remaining"):
            return died, timeout
        self._pair_finish_event.zero_()
        active = self._pair_settle_active
        self._pair_settle_remaining[active] -= 1
        self._pair_finish_event.copy_(active & (self._pair_settle_remaining <= 0))
        return died | self._pair_finish_event, timeout

    def _get_rewards(self):
        self._ensure_pair_state()
        reward = super()._get_rewards()
        if hasattr(self.commands, "pair_scenario"):
            valid = (
                self._touchdown_event & ~self._initial_drop_pending
                & ~self._pair_settle_active
            )
            first = valid & self._pair_phase_was_short
            second = valid & ~self._pair_phase_was_short
            strict = self._v33_strict_foot_touchdown
            group = self.commands.pair_scenario
            pair = second & strict & self._pair_first_strict
            for column, event in enumerate((first, first & strict,
                                            second, second & strict, pair)):
                self._pair_strict_totals[:, column] += torch.bincount(
                    group[event], minlength=3
                )
            self._pair_first_strict[first] = strict[first]
            self._pair_strict_current[second] = pair[second]
            self._pair_first_strict[second] = False
        final_touchdown = (
            self._final_touchdown_event & ~self._initial_drop_pending
            & ~self._pair_settle_active
        )
        if final_touchdown.any():
            self._pair_settle_active[final_touchdown] = True
            self._pair_settle_remaining[final_touchdown] = max(
                1, math.ceil(SETTLE_TIME_S / self.step_dt)
            )
        active = self._pair_settle_active
        q = self._robot.data.root_quat_w
        excessive_tilt = (
            q[:, 1].square() + q[:, 2].square()
            > math.sin(math.radians(SETTLE_TILT_DEG) / 2.0) ** 2
        )
        excessive_rate = (
            torch.linalg.norm(self._robot.data.root_ang_vel_b[:, :2], dim=1)
            > SETTLE_XY_RATE_RAD_S
        )
        newly_bad = active & (
            excessive_tilt | excessive_rate | self._liftoff_event
        ) & ~self._pair_settle_penalized
        self._pair_settle_bad |= newly_bad
        self._pair_settle_penalized |= newly_bad
        self._pair_bad_reasons += torch.stack((
            (newly_bad & excessive_tilt).sum(),
            (newly_bad & excessive_rate).sum(),
            (newly_bad & self._liftoff_event).sum(),
        ))
        # Cancel at least one full pair bonus if the landing is not controllable.
        penalty = newly_bad.float() * -self.cfg.two_hop_pair_reward_scale
        self._episode_sums["v38_settle_penalty"] += penalty
        reward = reward + penalty
        ended = self._pair_finish_event | (
            active & self._accounting_physical_death
        )
        stable = ended & ~self._pair_settle_bad & ~self._accounting_physical_death
        self._pair_strict_totals[:, 5] += torch.bincount(
            self.commands.pair_scenario[stable & self._pair_strict_current], minlength=3
        )
        self._pair_total += torch.stack((
            ended.sum(), stable.sum(),
            (ended & self._pair_settle_bad).sum(),
        ))
        self.extras["log"].update({
            "V38/two_hop_episodes_total": self._pair_total[0].float(),
            "V38/stable_pair_landings_total": self._pair_total[1].float(),
            "V38/unstable_pair_landings_total": self._pair_total[2].float(),
            "V38/terminal_tilt_failures_total": self._pair_bad_reasons[0].float(),
            "V38/terminal_xy_rate_failures_total": self._pair_bad_reasons[1].float(),
            "V38/terminal_liftoff_failures_total": self._pair_bad_reasons[2].float(),
            "V38/settle_penalty": penalty.mean(),
            "V38/strict_first_attempts_total": self._pair_strict_totals[:, 0].sum().float(),
            "V38/strict_first_hits_total": self._pair_strict_totals[:, 1].sum().float(),
            "V38/strict_second_attempts_total": self._pair_strict_totals[:, 2].sum().float(),
            "V38/strict_second_hits_total": self._pair_strict_totals[:, 3].sum().float(),
            "V38/strict_pairs_total": self._pair_strict_totals[:, 4].sum().float(),
            "V38/stable_strict_pairs_total": self._pair_strict_totals[:, 5].sum().float(),
        })
        for index, name in enumerate(("uniform", "reverse", "zero_patterns")):
            row = self._pair_strict_totals[index].float()
            self.extras["log"].update({
                f"V38/{name}_first_attempts_total": row[0],
                f"V38/{name}_second_attempts_total": row[2],
                f"V38/{name}_first_strict_rate": row[1] / row[0].clamp_min(1),
                f"V38/{name}_second_strict_rate": row[3] / row[2].clamp_min(1),
                f"V38/{name}_pair_strict_rate": row[4] / row[2].clamp_min(1),
                f"V38/{name}_stable_pair_strict_rate": row[5] / row[2].clamp_min(1),
            })
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "_pair_settle_remaining"):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._pair_settle_remaining[ids] = 0
            self._pair_settle_active[ids] = False
            self._pair_settle_bad[ids] = False
            self._pair_settle_penalized[ids] = False
            self._pair_finish_event[ids] = False
            self._pair_first_strict[ids] = False
            self._pair_strict_current[ids] = False

    def stage2_summary(self):
        result = super().stage2_summary()
        if hasattr(self, "_pair_total"):
            result.update(dict(zip(
                ("v38_two_hop_episodes", "v38_stable_pair_landings",
                 "v38_unstable_pair_landings"), self._pair_total.tolist()
            )))
            result["v38_strict_by_scenario"] = {
                name: dict(zip(
                    ("first_attempts", "first_hits", "second_attempts",
                     "second_hits", "pair_hits", "stable_pair_hits"), row.tolist()
                ))
                for name, row in zip(
                    ("uniform", "reverse", "zero_patterns"),
                    self._pair_strict_totals
                )
            }
            result["v38_terminal_failure_reasons"] = dict(zip(
                ("tilt", "xy_rate", "liftoff"), self._pair_bad_reasons.tolist()
            ))
        return result
