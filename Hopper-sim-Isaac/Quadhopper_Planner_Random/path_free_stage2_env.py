"""Endpoint/energy refinement of an already trained hopping policy."""
import torch
from .history_foot_env import HistoryFootEnv
from .next_hop_distance_audit import NextHopDistanceAudit
from .path_free_stage2 import endpoint_terms


class PathFreeStage2Env(HistoryFootEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._stage2_totals = torch.zeros(5, device=self.device, dtype=torch.float64)
        self._next_hop_distance_audit = (
            NextHopDistanceAudit(
                self.num_envs, self.device,
                long_radius_max_m=self.cfg.long_hop_radius_max,
            ) if self.cfg.collect_tracking_metrics else None
        )

    def _reset_idx(self, env_ids):
        audit = getattr(self, '_next_hop_distance_audit', None)
        if audit is not None:
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            audit.reset(ids)
        super()._reset_idx(env_ids)

    def _replan(self, env_ids):
        """Keep inherited event/command bookkeeping, without constructing a curve."""
        if len(env_ids) == 0:
            return
        p, q = self.commands.lookahead(env_ids)
        start = self._robot.data.root_pos_w[env_ids].clone()
        height, _ = self._height_commands(env_ids)
        self._active_target_height[env_ids] = height
        self._cycle_takeoff_direction_w[env_ids] = self._normalize_xy(p - start[:, :2])
        self._cycle_next_direction_w[env_ids] = self._normalize_xy(q - p)
        # Only the first point is read for inherited touchdown direction
        # diagnostics. These are inert storage, never projected or observed.
        self.planner.positions_w[env_ids] = start[:, None, :]
        self.planner.next_positions_w[env_ids] = start[:, None, :]
        self.planner.velocities_w[env_ids] = 0.0
        self.planner.next_velocities_w[env_ids] = 0.0
        self._cycle_time[env_ids] = 0.0
        if hasattr(self, '_descending_path'):
            self._descending_path[env_ids] = False
            self._path_arc_fraction[env_ids] = 0.0
            self._spatial_progress[env_ids] = 0.0
            self._xy_progress.reanchor(env_ids, (start[:, :2] - p).norm(dim=1))

    def _spatial_guidance(self):
        # SpatialTrackingEnv appends [carrot-position, tangent, hint] in the
        # body frame. All nine columns are exactly zero, before history stacking.
        pos = self._robot.data.root_pos_w
        zero = torch.zeros_like(pos[:, 0])
        return pos, torch.zeros_like(pos), zero, torch.zeros_like(pos), zero

    def _update_cycle_events(self):
        # Snapshot the scored target BEFORE the parent's touchdown queue advance.
        target = self.commands.lookahead()[0].clone()
        super()._update_cycle_events()
        foot_xy = self.foot_position_w()[:, :2]
        self._stage2_foot_error = (foot_xy - target).norm(dim=1)
        audit = getattr(self, '_next_hop_distance_audit', None)
        if audit is not None:
            ids = self._touchdown_event.nonzero(as_tuple=False).flatten()
            next_target = self.commands.lookahead()[0]
            audit.observe(
                ids=ids,
                old_target_xy=target[ids],
                next_target_xy=next_target[ids],
                foot_xy=foot_xy[ids],
                next_is_short=self.commands.current_hop_is_short(ids),
                completed_apex_m=self._settled_apex_height[ids],
                completed_target_m=self._settled_apex_target[ids],
            )
        strict_apex = (
            self._settled_apex_height >= self._settled_apex_target
            if getattr(self.cfg, 'stage2_strict_apex_for_landing_reward', False)
            else self._settled_apex_height >= (
                self._settled_apex_target - self.cfg.apex_tolerance
            )
        )
        self._stage2_valid_touchdown = (
            self._touchdown_event & ~self._initial_drop_pending
            & strict_apex
        )

    def _append_csv_row(self, predicted_power):
        # Reuse the one baseline power-model prediction; no second inference.
        self._stage2_power_w = predicted_power.detach()
        super()._append_csv_row(predicted_power)

    def _get_rewards(self):
        reward = super()._get_rewards()
        landing, energy, energy_j = endpoint_terms(
            self._stage2_foot_error, self._stage2_valid_touchdown,
            self._stage2_power_w, self.step_dt, self.cfg)
        self._stage2_totals += torch.stack((
            energy_j.sum(), self._touchdown_event.sum(), landing.sum(), energy.sum(),
            energy_j.new_tensor(self.num_envs * self.step_dt)))
        self.extras.setdefault('log', {}).update({
            'Stage2/foot_touchdown_reward': landing.mean(),
            'Stage2/energy_penalty': energy.mean(),
            'Stage2/estimated_power_w': self._stage2_power_w.clamp_min(0).mean(),
        })
        # No path has been evaluated, so path distance/corridor logs are invalid.
        for name in ('path_distance_m', 'path_penalty', 'path_tracking_reward',
                     'spatial_progress', 'progress_reward', 'landing_hint_penalty'):
            self.extras['log'].pop('Guidance/' + name, None)
        return reward + landing + energy

    def stage2_summary(self):
        energy, hops, landing_reward, cost, seconds = self._stage2_totals.tolist()
        result = {
            'estimated_energy_j_per_env': energy / self.num_envs,
            'estimated_energy_j_per_touchdown': energy / hops if hops else None,
            'estimated_mean_power_w': energy / seconds if seconds else None,
            'stage2_foot_reward_per_env': landing_reward / self.num_envs,
            'stage2_energy_penalty_per_env': cost / self.num_envs,
            'energy_scope': 'whole_evaluation_including_initial_drop_and_stance',
        }
        if self._next_hop_distance_audit is not None:
            result.update({f'stage2_{key}': value for key, value in
                           self._next_hop_distance_audit.summary().items()})
        return result

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        for name in ('_path_visualizer', '_next_path_visualizer', 'goal_pos_visualizer'):
            if hasattr(self, name):
                getattr(self, name).set_visibility(False)
