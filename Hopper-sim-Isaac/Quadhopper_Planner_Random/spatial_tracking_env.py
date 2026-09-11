"""Continuous two-waypoint PPO with weak, untimed geometric guidance."""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .tracking_ab_env import TrackingABEnv, TrackingABEnvCfg
from .spatial_path import hop_curve, project_path
from .tracking_progress import LandingXYProgress, two_hop_pair_success


@configclass
class SpatialTrackingEnvCfg(TrackingABEnvCfg):
    preview = True
    observation_space = 52
    episode_length_s = 15.0
    planner_nodes = 33
    # Geometric guidance is an optional corridor, not a position-vs-time target.
    path_corridor_m = 0.08
    path_penalty_width_m = 0.15
    path_penalty_scale = -1.0
    path_lookahead_m = 0.10
    # Untimed, continuous airborne guidance.  This rewards staying near the
    # geometric hop itself; it never names a target position at a time.
    path_tracking_width_m = 0.08
    path_tracking_reward_scale = 16.0
    # The final-target terms are intentionally weaker than the curve term
    # while airborne.  Actual touchdown remains the dominant accuracy signal.
    distance_to_xy_reward_scale = 10.0
    planner_position_reward_scale = 4.0
    planner_xy_reward_scale = 10.0
    projected_landing_reward_scale = 30.0
    projected_landing_penalty_scale = -30.0
    target_hit_reward_scale = 120.0
    landing_precision_reward_scale = 60.0
    landing_error_penalty_scale = -120.0
    landing_lateral_error_penalty_scale = -60.0
    spatial_progress_reward_scale = 12.0
    # Independent of the legacy stable-environment XY progress term (kept at 0).
    # Net approach of 20 cm adds 1 point, well below a physical target hit (120).
    spatial_xy_progress_reward_scale = 5.0
    # Sparse bonus for the second touchdown only when both points in the
    # alternating pair meet the unchanged target tolerance.
    two_hop_pair_reward_scale = 30.0
    # Retain stable jumping, but do not let target-independent height rewards
    # outweigh a missed landing for an entire episode.
    survival_reward_scale = 2.0
    distance_to_z_reward_scale = 1.0
    energy_tracking_reward_scale = 3.0
    apex_reward_scale = 3.0
    # Start each episode by falling into a real contact.  The pending-drop
    # guard suppresses task rewards/events until that contact is observed.
    # This exposes first-hop recovery states comparable to later hop landings.
    start_from_random_drop = True
    initial_drop_height_min = 0.65
    initial_drop_height_max = 1.05
    # A touchdown is scored from its physical XY error alone.  Height remains
    # a separate objective, so the policy may keep correcting thrust in the
    # air instead of having to satisfy a pre-landing apex window first.
    require_minimum_apex_for_hit = False
    require_apex_tolerance_for_hit = False
    gate_touchdown_rewards_by_apex = False
    landing_hint_tilt_rad = math.radians(4)
    landing_hint_deadband_rad = math.radians(8)
    # This vector stays in the observation for checkpoint compatibility, but
    # does not prescribe a descent/next-hop attitude.
    landing_hint_penalty_scale = 0.0
    # Disable inherited timed/velocity/attitude objectives. Endpoint and apex
    # objectives retain the v2 weights; the two additions have bounded cost.
    touchdown_attitude_penalty_scale = 0.0
    takeoff_tilt_reward_scale = 0.0
    takeoff_tilt_penalty_scale = 0.0
    takeoff_velocity_reward_scale = 0.0
    takeoff_velocity_penalty_scale = 0.0
    anticipatory_attitude_reward_scale = 0.0
    anticipatory_attitude_penalty_scale = 0.0
    anticipatory_velocity_penalty_scale = 0.0
    planner_velocity_reward_scale = 0.0
    descent_velocity_penalty_scale = 0.0
    planner_velocity_observation = False


class SpatialTrackingEnv(TrackingABEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._descending_path = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_arc_fraction = torch.zeros(self.num_envs, device=self.device)
        self._spatial_progress = torch.zeros(self.num_envs, device=self.device)
        self._xy_progress = LandingXYProgress(self.num_envs, self.device)
        self._episode_sums['tracking_xy_progress'] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_sums['two_hop_pair_hit'] = torch.zeros(
            self.num_envs, device=self.device
        )
        # distance sum, inside-corridor count, active count, progress sum
        self._guide_sum = torch.zeros(4, device=self.device, dtype=torch.float64)
        # Signed, approach-only, and retreat-only reward totals for evaluation.
        self._xy_progress_sum = torch.zeros(3, device=self.device, dtype=torch.float64)

    def _replan(self, env_ids):
        if len(env_ids) == 0:
            return
        p, q = self.commands.lookahead(env_ids)
        start = self._robot.data.root_pos_w[env_ids].clone()
        landing = torch.cat((p, torch.full_like(p[:, :1], self.cfg.landing_root_height)), -1)
        following = torch.cat((q, landing[:, 2:3]), -1)
        height, next_height = self._height_commands(env_ids)
        self._active_target_height[env_ids] = height
        self._cycle_takeoff_direction_w[env_ids] = self._normalize_xy(p - start[:, :2])
        self._cycle_next_direction_w[env_ids] = self._normalize_xy(q - p)
        # Reuse the parent's geometry buffers for landing metrics/visualization;
        # never solve or sample its time-parametrized collocation trajectory.
        self.planner.positions_w[env_ids] = hop_curve(start, landing, following, height, self.cfg.planner_nodes)
        beyond = following + (following - landing)
        self.planner.next_positions_w[env_ids] = hop_curve(landing, following, beyond, next_height, self.cfg.planner_nodes)
        self.planner.velocities_w[env_ids] = 0
        self.planner.next_velocities_w[env_ids] = 0
        self._cycle_time[env_ids] = 0  # Legacy bookkeeping only; guidance never reads it.
        if hasattr(self, '_descending_path'):
            self._descending_path[env_ids] = False
            self._path_arc_fraction[env_ids] = 0.0
            self._spatial_progress[env_ids] = 0.0
            # The target changed at this physical event.  Re-anchor the
            # potential, so queue advancement cannot look like backwards
            # motion or create a free progress reward.
            self._xy_progress.reanchor(
                env_ids, torch.linalg.norm(start[:, :2] - p, dim=1)
            )

    def _update_reference(self):
        # Keep the source policy's physical landing/apex command unchanged.
        # Spatial guidance gets its own appended inputs, not a moving old goal.
        p, _ = self.commands.lookahead()
        self._desired_pos_w[:, :2] = p
        self._desired_pos_w[:, 2] = self._active_target_height
        self._planned_velocity_w.zero_()

    def _spatial_guidance(self):
        pos = self._robot.data.root_pos_w
        minimum = self._path_arc_fraction if hasattr(self, '_path_arc_fraction') else None
        carrot, tangent, distance, arc_fraction = project_path(
            pos, self.planner.positions_w, self._descending_path,
            self.cfg.path_lookahead_m, minimum)
        next_direction = self._cycle_next_direction_w
        axis = torch.cat((math.sin(self.cfg.landing_hint_tilt_rad) * next_direction,
                          torch.full_like(pos[:, :1], math.cos(self.cfg.landing_hint_tilt_rad))), -1)
        return carrot, tangent, distance, axis, arc_fraction

    def _get_observations(self):
        obs = super()._get_observations()
        carrot, tangent, _, axis, _ = self._spatial_guidance()
        quat = self._robot.data.root_quat_w
        error = quat_apply_inverse(quat, carrot - self._robot.data.root_pos_w).clamp(-1, 1)
        extra = torch.cat((error, quat_apply_inverse(quat, tangent), quat_apply_inverse(quat, axis)), -1)
        obs['policy'] = torch.cat((obs['policy'], extra), -1)
        return obs

    def _update_cycle_events(self):
        super()._update_cycle_events()
        # Hysteresis: once a measured airborne apex is crossed, stay on the
        # descending branch until touchdown/replan; motor latency shifts timing freely.
        if hasattr(self, '_descending_path'):
            self._descending_path |= self._apex_event & self._cycle_active & ~self._touchdown_event

    def _get_rewards(self):
        reward = super()._get_rewards()
        _, _, distance, axis, arc_fraction = self._spatial_guidance()
        active = self._cycle_active & ~self._touchdown_event
        start_xy = self.planner.positions_w[:, 0, :2]
        landing_xy = self.planner.positions_w[:, -1, :2]
        hop_xy = landing_xy - start_xy
        horizontal_fraction = (((self._robot.data.root_pos_w[:, :2] - start_xy) * hop_xy).sum(-1)
                               / hop_xy.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
        raw_progress = torch.minimum(arc_fraction, horizontal_fraction)
        progress = torch.maximum(self._spatial_progress, raw_progress)
        progress_delta = (progress - self._spatial_progress) * active
        progress_reward = progress_delta * self.cfg.spatial_progress_reward_scale
        self._spatial_progress.copy_(progress)
        self._path_arc_fraction.copy_(torch.maximum(self._path_arc_fraction, arc_fraction))
        outside = ((distance - self.cfg.path_corridor_m).clamp_min(0)
                   / self.cfg.path_penalty_width_m).clamp_max(1).square()
        path_cost = outside * active * self.cfg.path_penalty_scale * self.step_dt
        path_tracking = (
            torch.exp(-torch.square(distance / self.cfg.path_tracking_width_m))
            * active * self.cfg.path_tracking_reward_scale * self.step_dt
        )
        if self.cfg.landing_hint_penalty_scale:
            alignment = (self._body_z_axis_w() * axis).sum(-1).clamp(-1, 1)
            angle = torch.acos(alignment)
            attitude_cost = ((angle - self.cfg.landing_hint_deadband_rad).clamp_min(0)
                             / math.radians(25)).clamp_max(1).square()
            # Only near the ground on measured descent. No deadline or phase clock.
            near = ((0.65 - self._robot.data.root_pos_w[:, 2]) / 0.27).clamp(0, 1)
            hint_cost = (attitude_cost * near * active * self._descending_path
                         * self.cfg.landing_hint_penalty_scale * self.step_dt)
        else:
            hint_cost = torch.zeros_like(path_cost)
        target, _ = self.commands.lookahead()
        target_distance = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - target, dim=1
        )
        # Parent event handling already re-anchors on touchdown/queue advance,
        # liftoff and reset. Score consecutive airborne samples only; initial
        # drop and stance movement cannot leak into the next flight's reward.
        xy_delta = self._xy_progress.update(
            target_distance, active & ~self._initial_drop_pending
        )
        xy_progress_reward = xy_delta * self.cfg.spatial_xy_progress_reward_scale
        self._episode_sums['tracking_xy_progress'] += xy_progress_reward
        pair_success = two_hop_pair_success(
            self._final_touchdown_event,
            self._target_hit_event,
            self._consecutive_hits,
        )
        pair_reward = pair_success.float() * self.cfg.two_hop_pair_reward_scale
        self._episode_sums['two_hop_pair_hit'] += pair_reward
        self._xy_progress_sum += torch.stack((xy_progress_reward.sum(),
                                             xy_progress_reward.clamp_min(0).sum(),
                                             xy_progress_reward.clamp_max(0).sum()))
        self._guide_sum += torch.stack(((distance * active).sum(),
                                       ((distance <= self.cfg.path_corridor_m) & active).sum(),
                                       active.sum(), (progress * active).sum()))
        self.extras.setdefault('log', {}).update({
            'Guidance/path_distance_m': (distance * active).sum() / active.sum().clamp_min(1),
            'Guidance/path_penalty': path_cost.mean(),
            'Guidance/path_tracking_reward': path_tracking.mean(),
            'Guidance/landing_hint_penalty': hint_cost.mean(),
            'Guidance/spatial_progress': (progress * active).sum() / active.sum().clamp_min(1),
            'Guidance/progress_reward': progress_reward.mean(),
            'Guidance/target_distance_m': (target_distance * active).sum() / active.sum().clamp_min(1),
            'Guidance/xy_progress_reward': xy_progress_reward.mean(),
            'Guidance/two_hop_pair_reward': pair_reward.mean(),
        })
        return (reward + path_cost + path_tracking + hint_cost + progress_reward
                + xy_progress_reward + pair_reward)
