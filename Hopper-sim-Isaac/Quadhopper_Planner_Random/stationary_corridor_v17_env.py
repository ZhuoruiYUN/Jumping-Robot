"""Stationary corridor v17: 15 cm tolerance + attitude priority + slow recenter.

This task isolates exact zero-hop stability for 40_v1 deployment. Unlike the
broad spatial tracking task, it enforces a clear priority order:

1. No death/reset, stable pitch/roll/yaw, smooth touchdown
2. Apex root Z near 1.000 m
3. Touchdown inside R_outer = 0.15 m
4. Slow drift back toward center when outside R_inner = 0.05 m

The stationary hold gate no longer suppresses attitude costs when the robot
drifts beyond 5 cm. Instead, stationary mission commands generate a virtual
landing target that recedes slowly (2-3 cm per hop) toward the global circle
center, preventing one-hop full corrections that destabilize pitch/yaw/thrust.

Key differences from spatial_tracking_env:
- Landing inside 0.15 m is a success; no sparse bonus for exact center hit
- Weak quartic potential inside the circle; stronger Huber barrier outside
- Virtual target logic replaces direct circle-center observation for stationary
- Attitude/yaw/touchdown penalties active throughout stationary flight, not
  gated by a 5 cm hold radius
-姿态优先 checkpoint selection: no reset → attitude → height → corridor
"""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .spatial_tracking_env import SpatialTrackingEnv, SpatialTrackingPhysical1mEnvCfg


@configclass
class StationaryCorridorV17EnvCfg(SpatialTrackingPhysical1mEnvCfg):
    """15 cm corridor + attitude priority + slow recenter for v17."""

    # Corridor radii
    stationary_inner_radius_m = 0.05
    stationary_outer_radius_m = 0.15

    # Slow recenter parameters
    stationary_recenter_gain = 0.25  # fraction of error per hop
    stationary_recenter_max_step_m = 0.025  # 2.5 cm max correction per hop

    # Landing rewards: uniform inside outer circle, weak quartic + Huber barrier
    target_hit_reward_scale = 120.0  # event reward for landing inside R_outer
    landing_precision_reward_scale = 0.0  # disable dense Gaussian center bonus
    landing_error_penalty_scale = 0.0  # replaced by corridor logic below
    landing_lateral_error_penalty_scale = 0.0

    # New corridor-specific landing terms
    corridor_center_potential_scale = -2.0  # weak (e/R_outer)^4 inside circle
    corridor_barrier_scale = -30.0  # Huber barrier outside R_outer
    corridor_barrier_width_m = 0.05  # Huber transition width

    # Decouple attitude from the 5 cm hold gate: stationary attitude costs now
    # apply whenever stationary_command is true, regardless of target_distance.
    # The 5 cm radius is only used for deciding when to stop issuing virtual
    # recenter corrections (i.e., treating it as truly zero-hop).
    stationary_hold_radius_m = 0.05  # still used for action_spread gate

    # Attitude penalties: active for all stationary flight, not hold-gated
    stationary_tilt_penalty_scale = -8.0
    stationary_tilt_deadband_rad = math.radians(3.0)
    stationary_yaw_rate_penalty_scale = -3.0
    stationary_yaw_rate_deadband_rad_s = 0.18
    stationary_pitch_penalty_scale = -6.0
    stationary_pitch_deadband_rad = math.radians(3.0)
    stationary_pitch_rate_penalty_scale = -4.0
    stationary_pitch_rate_deadband_rad_s = 0.25

    # Touchdown attitude (stationary missions only)
    stationary_touchdown_angle_penalty_scale = -8.0
    stationary_touchdown_angle_deadband_rad = math.radians(3.0)
    stationary_touchdown_rate_penalty_scale = -6.0
    stationary_touchdown_rate_deadband_rad_s = 0.25

    # Yaw event penalties: apex and touchdown, for accumulated drift
    stationary_yaw_event_penalty_scale = -4.0
    stationary_yaw_event_deadband_rad = math.radians(1.5)
    stationary_yaw_hop_change_penalty_scale = -3.0
    stationary_yaw_hop_change_deadband_rad = math.radians(1.0)

    # Height: symmetric error, no single-sided shortfall boost
    stationary_apex_width_m = 0.05  # high-quality zone: ±5 cm
    stationary_apex_reward_scale = 15.0
    stationary_apex_error_penalty_scale = -30.0
    stationary_apex_shortfall_penalty_scale = -30.0  # symmetric with overshoot

    # Causal ascent-phase predicted apex, weaker than attitude safety
    stationary_predicted_apex_width_m = 0.10
    stationary_predicted_apex_reward_scale = 8.0

    # Potential-based apex progress (bounded per hop)
    stationary_apex_progress_scale = 12.0
    stationary_apex_progress_width_m = 0.25

    # Disable moving-hop spatial tracking terms
    path_tracking_reward_scale = 0.0
    path_penalty_scale = 0.0
    spatial_progress_reward_scale = 0.0
    spatial_xy_progress_reward_scale = 0.0
    projected_landing_reward_scale = 0.0
    projected_landing_penalty_scale = 0.0
    distance_to_xy_reward_scale = 0.0
    planner_position_reward_scale = 0.0
    planner_xy_reward_scale = 0.0
    two_hop_pair_reward_scale = 0.0

    # Action spread gate remains at hold radius (inside inner circle)
    stationary_action_spread_penalty_scale = -4.0
    stationary_action_spread_deadband = 0.15

    # Deployment actuator contract v17 (already in parent)
    deployment_launch_recovery_window_s = 0.16
    deployment_launch_recovery_spread_limit_pwm = 450.0
    deployment_launch_recovery_hold_s = 0.08
    deployment_launch_recovery_spread_limit_end_pwm = 600.0


class StationaryCorridorV17Env(SpatialTrackingEnv):
    cfg: StationaryCorridorV17EnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Global circle center for stationary missions: remains fixed, never
        # moves. The virtual target (what the policy sees) may differ.
        self._stationary_circle_center_w = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        # Virtual landing target for the next stationary hop
        self._stationary_virtual_target_w = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        # Last touchdown XY for recenter logic
        self._stationary_last_touchdown_xy = torch.zeros(
            self.num_envs, 2, device=self.device
        )

    def _update_stationary_virtual_target(self, env_ids: torch.Tensor):
        """Generate virtual landing target for slow recenter after touchdown."""
        if len(env_ids) == 0:
            return

        last_xy = self._stationary_last_touchdown_xy[env_ids]
        center_xy = self._stationary_circle_center_w[env_ids]
        error_vec = center_xy - last_xy
        error_norm = torch.linalg.norm(error_vec, dim=1, keepdim=True).clamp_min(1e-8)

        # Inside inner radius: treat as zero-hop, no correction
        inside_inner = error_norm[:, 0] <= self.cfg.stationary_inner_radius_m

        # Outside inner: apply limited recenter step
        correction_magnitude = (
            self.cfg.stationary_recenter_gain * error_norm
        ).clamp_max(self.cfg.stationary_recenter_max_step_m)
        correction_vec = (error_vec / error_norm) * correction_magnitude

        virtual_target = torch.where(
            inside_inner.unsqueeze(-1),
            last_xy,  # truly zero-hop
            last_xy + correction_vec  # slow drift toward center
        )

        self._stationary_virtual_target_w[env_ids] = virtual_target

    def _update_cycle_events(self):
        super()._update_cycle_events()

        # Capture touchdown XY for stationary missions
        stationary_touchdown = (
            self._touchdown_event
            & self._completed_stationary_command
            & ~self._initial_drop_pending
        )
        if stationary_touchdown.any():
            td_ids = stationary_touchdown.nonzero(as_tuple=False).flatten()
            self._stationary_last_touchdown_xy[td_ids] = (
                self._robot.data.root_pos_w[td_ids, :2]
            )
            # Update virtual target after this touchdown for the next hop
            self._update_stationary_virtual_target(td_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Initialize circle center and virtual target at env origin
        self._stationary_circle_center_w[env_ids] = self._scene.env_origins[env_ids, :2]
        self._stationary_virtual_target_w[env_ids] = self._scene.env_origins[env_ids, :2]
        self._stationary_last_touchdown_xy[env_ids] = self._scene.env_origins[env_ids, :2]

    def _desired_landing_xy_w(self):
        """Override parent to return virtual target for stationary missions."""
        # Parent's _desired_landing_xy_w uses self.planner.positions_w[:, -1, :2]
        # or commands.lookahead(). For stationary commands, replace with virtual.
        target, _ = self.commands.lookahead()
        commanded_hop_distance = torch.linalg.norm(
            target - self.commands.anchor_w, dim=1
        )
        stationary_command = (
            commanded_hop_distance <= self.cfg.stationary_command_radius_m
        )

        # Moving hops use the actual commanded target
        # Stationary hops use the virtual recenter target
        landing_xy = torch.where(
            stationary_command.unsqueeze(-1),
            self._stationary_virtual_target_w,
            target
        )
        return landing_xy

    def _get_rewards(self):
        reward_dict = super()._get_rewards()

        # Override landing rewards with corridor logic
        target, _ = self.commands.lookahead()
        commanded_hop_distance = torch.linalg.norm(
            target - self.commands.anchor_w, dim=1
        )
        stationary_command = (
            commanded_hop_distance <= self.cfg.stationary_command_radius_m
        )

        # Radial error to global circle center (not virtual target)
        touchdown_xy = self._robot.data.root_pos_w[:, :2]
        center_error_vec = touchdown_xy - self._stationary_circle_center_w
        center_error_norm = torch.linalg.norm(center_error_vec, dim=1)

        # Corridor touchdown logic (only for stationary missions)
        stationary_touchdown = (
            self._final_touchdown_event
            & stationary_command
            & ~self._initial_drop_pending
        )

        # Inside outer radius: success event + weak quartic potential
        inside_outer = center_error_norm <= self.cfg.stationary_outer_radius_m
        corridor_hit = stationary_touchdown & inside_outer
        corridor_hit_reward = corridor_hit.float() * self.cfg.target_hit_reward_scale

        # Quartic potential inside circle: (e/R_outer)^4, weak near center
        normalized_error = center_error_norm / self.cfg.stationary_outer_radius_m
        quartic_cost = torch.where(
            inside_outer,
            torch.pow(normalized_error, 4.0),
            torch.zeros_like(normalized_error)
        )
        corridor_potential = (
            stationary_touchdown.float() * quartic_cost
            * self.cfg.corridor_center_potential_scale
        )

        # Huber barrier outside outer radius
        outside_excess = (center_error_norm - self.cfg.stationary_outer_radius_m).clamp_min(0)
        huber_input = outside_excess / self.cfg.corridor_barrier_width_m
        huber = torch.where(
            huber_input <= 1.0,
            0.5 * huber_input.square(),
            huber_input - 0.5
        )
        corridor_barrier = (
            stationary_touchdown.float() * huber
            * self.cfg.corridor_barrier_scale
        )

        # Replace parent's landing precision/error with corridor terms
        # (parent already computed target_hit based on old tolerance; we override)
        reward_dict['landing_precision'] = corridor_potential
        reward_dict['landing_radial'] = corridor_barrier
        reward_dict['target_hit'] = corridor_hit_reward

        # Log corridor diagnostics
        if stationary_touchdown.any():
            td_count = stationary_touchdown.float().sum().clamp_min(1)
            self.extras.setdefault('log', {}).update({
                'Corridor/landing_error_m': (
                    center_error_norm * stationary_touchdown.float()
                ).sum() / td_count,
                'Corridor/inside_inner_rate': (
                    (center_error_norm <= self.cfg.stationary_inner_radius_m)
                    & stationary_touchdown
                ).float().mean(),
                'Corridor/inside_outer_rate': corridor_hit.float().mean(),
                'Corridor/ring_5_to_15cm_rate': (
                    (center_error_norm > self.cfg.stationary_inner_radius_m)
                    & (center_error_norm <= self.cfg.stationary_outer_radius_m)
                    & stationary_touchdown
                ).float().mean(),
                'Corridor/outside_outer_rate': (
                    (center_error_norm > self.cfg.stationary_outer_radius_m)
                    & stationary_touchdown
                ).float().mean(),
            })

        return reward_dict