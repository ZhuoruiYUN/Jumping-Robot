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
    # When enabled by the height-refinement launcher, validity follows the
    # *current* commanded height minus ``apex_tolerance``.  This matters for a
    # moving fixed-height curriculum: a static minimum either makes the first
    # stage impossible or leaves the final 1 m task under-constrained.
    target_relative_valid_apex = False
    landing_hint_tilt_rad = math.radians(4)
    landing_hint_deadband_rad = math.radians(8)
    # This vector stays in the observation for checkpoint compatibility, but
    # does not prescribe a descent/next-hop attitude.
    landing_hint_penalty_scale = 0.0
    # Exact in-place hops need a different last-flight preference from moving
    # hops: once the body is already close to its fixed target, stop spending
    # differential thrust on centimetre-scale corrections.  These terms are
    # deliberately command-conditioned, so they never constrain a genuine
    # horizontal hop or a recovery when the stationary target has been missed.
    stationary_command_radius_m = 0.002
    # Match the 5 cm touchdown-success tolerance: do not suppress a recovery
    # until the stationary target is already within its accepted landing band.
    stationary_hold_radius_m = 0.05
    stationary_tilt_deadband_rad = math.radians(3.0)
    stationary_tilt_penalty_scale = -6.0
    stationary_yaw_rate_deadband_rad_s = 0.18
    stationary_yaw_rate_penalty_scale = -2.0
    stationary_action_spread_deadband = 0.15
    stationary_action_spread_penalty_scale = -3.0
    # Disabled by default so ordinary spatial tracking keeps its current apex
    # objective.  The stationary-apex refinement enables these physical-event
    # terms explicitly through its launcher.
    stationary_apex_width_m = 0.05
    stationary_apex_reward_scale = 0.0
    stationary_apex_error_penalty_scale = 0.0
    stationary_apex_shortfall_penalty_scale = 0.0
    # Potential-difference reward for measured maximum height.  Unlike a
    # fixed-time trajectory this emits reward only when the robot actually
    # gains height, and its total return per hop is bounded.
    stationary_apex_progress_scale = 0.0
    # This is intentionally much wider than the terminal apex-quality window:
    # the potential must distinguish a 0.90 m hop from a 0.40 m hop, otherwise
    # it supplies no gradient until the final 5 cm.
    stationary_apex_progress_width_m = 0.25
    # Vertical-specialist terms are disabled in ordinary tracking.  They give
    # credit only during the physically observed ascent, below the requested
    # apex, so a policy cannot earn them by hovering or applying thrust after
    # it has already met the height command.
    stationary_ascent_support_scale = 0.0
    stationary_ascent_collective_scale = 0.0
    # Adapted from the reference full-reward task. During measured ascent,
    # reward an energy state only when its ballistic apex approaches the
    # commanded apex. It remains off for all existing checkpoints.
    stationary_predicted_apex_width_m = 0.10
    stationary_predicted_apex_reward_scale = 0.0
    # All-hop counterpart of the stationary predicted-apex term.  It gives
    # the direct-motor policy a causal ascent-phase signal before touchdown;
    # leave disabled unless touchdown rewards are height-gated.
    ascent_predicted_apex_width_m = 0.12
    ascent_predicted_apex_reward_scale = 0.0
    # Small all-hop event term. Unlike the stationary refinement terms, this
    # covers moving hops too, but it is deliberately sparse and bounded so a
    # correct landing objective remains dominant.
    apex_event_height_width_m = 0.06
    apex_event_height_reward_scale = 0.0
    apex_event_shortfall_penalty_scale = 0.0
    stationary_ascent_vz_min_mps = 0.10
    stationary_ascent_height_margin_m = 0.03
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
    # When below one, only that fraction of reset environments receives the
    # task-level hardware randomization.  The remaining environments use the
    # calibrated plant exactly, preserving a strong nominal hopping signal.
    hardware_randomization_probability = 1.0


@configclass
class SpatialTrackingPhysical1mEnvCfg(SpatialTrackingEnvCfg):
    """52-D physical-1m task with the measured deployment actuator contract."""

    # Sim ground root-Z is 0.380 m while the Vicon-frame physical root is
    # 0.285 m. A physical 1.000 m apex maps to 1.095 m in policy/sim space.
    target_height = 1.095
    deployment_action_target_height = target_height
    corrected_action_delay_indexing = True
    record_executed_action_history = True
    apply_deployment_action_shaping = True


class SpatialTrackingEnv(TrackingABEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._descending_path = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_arc_fraction = torch.zeros(self.num_envs, device=self.device)
        self._spatial_progress = torch.zeros(self.num_envs, device=self.device)
        self._xy_progress = LandingXYProgress(self.num_envs, self.device)
        # Parent touchdown handling advances the waypoint queue before rewards
        # are assembled.  Preserve the completed command class so event rewards
        # for that touchdown are never attributed to the following random hop.
        self._completed_stationary_command = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stationary_apex_potential = torch.zeros(self.num_envs, device=self.device)
        self._stationary_apex_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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

    def _reset_idx(self, env_ids):
        """Mix calibrated and perturbed resets without changing the base plant."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        probability = self.cfg.hardware_randomization_probability
        if (
            self.num_envs == 1
            or probability >= 1.0
            or not self.cfg.randomize_dynamics
        ):
            return
        hard_mask = torch.rand(len(env_ids), device=self.device) < probability
        nominal_ids = env_ids[~hard_mask]
        if len(nominal_ids) == 0:
            return
        # Parent reset has already sampled the perturbations.  Restore the
        # calibrated values for nominal curriculum samples only.  This is
        # task-local and leaves the stable baseline physics unchanged.
        self.dr_mass_multi[nominal_ids] = 1.0
        self.dr_inertia_multi[nominal_ids] = 1.0
        self.dr_tm[nominal_ids] = self.cfg.play_motor_time_constant
        self.dr_thrust_scale[nominal_ids] = 1.0
        self.dr_thrust_curve_shape[nominal_ids] = 1.0

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

    def _advance_route(self, env_ids):
        """Latch the completed command before the parent replaces it."""
        if len(env_ids) > 0:
            target, _ = self.commands.lookahead(env_ids)
            self._completed_stationary_command[env_ids] = (
                torch.linalg.norm(
                    target - self.commands.anchor_w[env_ids], dim=1
                ) <= self.cfg.stationary_command_radius_m
            )
        super()._advance_route(env_ids)

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
        if self.cfg.simulation_only_apex_bias_controller:
            if self.cfg.apex_bias_feedback_mode == "liftoff_vz":
                update_ids = self._liftoff_event & ~self._initial_drop_pending
                signed_error = (
                    self.cfg.apex_bias_liftoff_vz_target_mps
                    - self._robot.data.root_lin_vel_w[:, 2]
                )
            else:
                # The parent also resolves a missing apex at touchdown, after
                # it has cleared _cycle_active.  That remains a valid
                # measured-hop sample for apex-feedback mode.
                update_ids = self._apex_event & ~self._initial_drop_pending
                signed_error = self._active_target_height - self._cycle_max_z
            update = torch.where(
                signed_error.abs() > self.cfg.apex_bias_deadband_m,
                signed_error * self.cfg.apex_bias_gain_pwm_per_m,
                torch.zeros_like(signed_error),
            )
            self._apex_bias_pwm = torch.where(
                update_ids,
                (self._apex_bias_pwm + update).clamp(0.0, self.cfg.apex_bias_max_pwm),
                self._apex_bias_pwm,
            )
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
        # ``anchor_w -> target`` is the commanded displacement, unlike the
        # physical start-to-target vector, which includes accumulated landing
        # error.  This makes zero-hop detection robust to the very drift the
        # policy is allowed to correct outside the hold radius.
        commanded_hop_distance = torch.linalg.norm(
            target - self.commands.anchor_w, dim=1
        )
        stationary_command = (
            commanded_hop_distance <= self.cfg.stationary_command_radius_m
        )
        # Use the measured vertical velocity rather than the legacy apex-event
        # latch.  The latter is not populated on every compatible recovery
        # checkpoint, whereas negative Vz is available on hardware and exactly
        # identifies the descending part of an untimed hop.
        stationary_descent = (
            (self._robot.data.root_lin_vel_w[:, 2] < -0.10)
            & ~self._initial_drop_pending
        )
        stationary_hold = (
            stationary_command
            & (target_distance <= self.cfg.stationary_hold_radius_m)
            & stationary_descent
            & active
        )
        body_z = self._body_z_axis_w()[:, 2].clamp(-1.0, 1.0)
        stationary_tilt = torch.acos(body_z)
        stationary_tilt_cost = torch.square(
            (stationary_tilt - self.cfg.stationary_tilt_deadband_rad).clamp_min(0)
            / math.radians(12.0)
        )
        stationary_yaw_rate_cost = torch.square(
            (torch.abs(self._robot.data.root_ang_vel_b[:, 2])
             - self.cfg.stationary_yaw_rate_deadband_rad_s).clamp_min(0)
            / 0.5
        )
        motor_spread = (
            torch.max(self._motor_u, dim=1).values
            - torch.min(self._motor_u, dim=1).values
        )
        stationary_action_spread_cost = torch.square(
            (motor_spread - self.cfg.stationary_action_spread_deadband).clamp_min(0)
            / 0.15
        )
        stationary_tilt_reward = (
            stationary_hold.float() * stationary_tilt_cost
            * self.cfg.stationary_tilt_penalty_scale * self.step_dt
        )
        stationary_yaw_reward = (
            stationary_hold.float() * stationary_yaw_rate_cost
            * self.cfg.stationary_yaw_rate_penalty_scale * self.step_dt
        )
        stationary_action_reward = (
            stationary_hold.float() * stationary_action_spread_cost
            * self.cfg.stationary_action_spread_penalty_scale * self.step_dt
        )
        # Height is rewarded as a potential difference of *measured maximum
        # height*.  Thus a stationary hop earns return only as it rises toward
        # the physical apex target; descent, hover, and a time schedule earn
        # nothing.  Clipping makes the complete-hop contribution bounded.
        stationary_apex_active = (
            stationary_command & active & ~self._initial_drop_pending
        )
        stationary_apex_potential = -torch.square(
            (self._active_target_height - self._cycle_max_z).clamp_min(0.0)
            / self.cfg.stationary_apex_progress_width_m
        ).clamp_max(1.0)
        stationary_apex_progress = torch.where(
            stationary_apex_active & self._stationary_apex_active,
            (stationary_apex_potential - self._stationary_apex_potential).clamp_min(0.0),
            torch.zeros_like(stationary_apex_potential),
        )
        stationary_apex_progress_reward = (
            stationary_apex_progress * self.cfg.stationary_apex_progress_scale
        )
        self._stationary_apex_potential.copy_(torch.where(
            stationary_apex_active, stationary_apex_potential,
            torch.zeros_like(stationary_apex_potential),
        ))
        self._stationary_apex_active.copy_(stationary_apex_active)
        # This terminal check is also measured at a physical velocity apex,
        # never at a prescribed time.
        stationary_apex_command = stationary_command | (
            self._touchdown_event & self._completed_stationary_command
        )
        stationary_apex_event = stationary_apex_command & self._apex_event
        stationary_apex_error = self._apex_error
        stationary_apex_shortfall = torch.relu(
            self._active_target_height - self._cycle_max_z
        )
        stationary_apex_quality = torch.exp(-torch.square(
            stationary_apex_error / self.cfg.stationary_apex_width_m
        ))
        stationary_apex_reward = stationary_apex_event.float() * (
            stationary_apex_quality * self.cfg.stationary_apex_reward_scale
            + stationary_apex_error * self.cfg.stationary_apex_error_penalty_scale
            + stationary_apex_shortfall * self.cfg.stationary_apex_shortfall_penalty_scale
        )
        stationary_ascent = (
            stationary_command
            & active
            & ~self._initial_drop_pending
            & (self._robot.data.root_lin_vel_w[:, 2] > self.cfg.stationary_ascent_vz_min_mps)
            & (self._robot.data.root_pos_w[:, 2] > self.cfg.landing_root_height + 0.02)
            & (self._robot.data.root_pos_w[:, 2] < (
                self._active_target_height - self.cfg.stationary_ascent_height_margin_m
            ))
        )
        stationary_ascent_support_reward = (
            stationary_ascent.float()
            * self.cfg.stationary_ascent_support_scale
            * self.step_dt
        )
        stationary_ascent_collective_reward = (
            stationary_ascent.float()
            * self._motor_u.mean(dim=1)
            * self.cfg.stationary_ascent_collective_scale
            * self.step_dt
        )
        gravity = abs(self.sim.cfg.gravity[2]) if self.sim.cfg.gravity is not None else 9.81
        predicted_apex = self._robot.data.root_pos_w[:, 2] + (
            torch.clamp(self._robot.data.root_lin_vel_w[:, 2], min=0.0).square()
            / (2.0 * gravity)
        )
        stationary_predicted_apex_reward = (
            stationary_ascent.float()
            * torch.exp(-torch.abs(predicted_apex - self._active_target_height)
                        / self.cfg.stationary_predicted_apex_width_m)
            * self.cfg.stationary_predicted_apex_reward_scale
            * self.step_dt
        )
        spring_contact = (
            self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
        )
        ascent_predicted_apex = (
            active
            & ~self._initial_drop_pending
            & ~spring_contact
            & (self._robot.data.root_lin_vel_w[:, 2] > self.cfg.stationary_ascent_vz_min_mps)
            & (self._robot.data.root_pos_w[:, 2] < (
                self._active_target_height - self.cfg.stationary_ascent_height_margin_m
            ))
        )
        ascent_predicted_apex_reward = (
            ascent_predicted_apex.float()
            * torch.exp(-torch.abs(predicted_apex - self._active_target_height)
                        / self.cfg.ascent_predicted_apex_width_m)
            * self.cfg.ascent_predicted_apex_reward_scale
            * self.step_dt
        )
        apex_event = self._apex_event & ~self._initial_drop_pending
        apex_shortfall = torch.relu(self._active_target_height - self._cycle_max_z)
        apex_quality = torch.exp(-torch.square(
            (self._cycle_max_z - self._active_target_height)
            / self.cfg.apex_event_height_width_m
        ))
        apex_event_height_reward = apex_event.float() * (
            apex_quality * self.cfg.apex_event_height_reward_scale
            + apex_shortfall * self.cfg.apex_event_shortfall_penalty_scale
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
            'Guidance/stationary_hold_rate': stationary_hold.float().mean(),
            'Guidance/stationary_command_rate': stationary_command.float().mean(),
            'Guidance/stationary_descent_rate': stationary_descent.float().mean(),
            'Guidance/stationary_tilt_penalty': stationary_tilt_reward.mean(),
            'Guidance/stationary_yaw_rate_penalty': stationary_yaw_reward.mean(),
            'Guidance/stationary_action_spread_penalty': stationary_action_reward.mean(),
            'Guidance/stationary_apex_reward': stationary_apex_reward.mean(),
            'Guidance/stationary_apex_progress_reward': stationary_apex_progress_reward.mean(),
            'Guidance/stationary_apex_event_rate': stationary_apex_event.float().mean(),
            'Guidance/stationary_apex_error_m': (
                stationary_apex_error * stationary_apex_event.float()
            ).sum() / stationary_apex_event.float().sum().clamp_min(1),
            'Guidance/stationary_ascent_rate': stationary_ascent.float().mean(),
            'Guidance/stationary_ascent_support_reward': stationary_ascent_support_reward.mean(),
            'Guidance/stationary_ascent_collective_reward': stationary_ascent_collective_reward.mean(),
            'Guidance/stationary_predicted_apex_reward': stationary_predicted_apex_reward.mean(),
            'Guidance/ascent_predicted_apex_rate': ascent_predicted_apex.float().mean(),
            'Guidance/ascent_predicted_apex_reward': ascent_predicted_apex_reward.mean(),
            'Guidance/apex_event_height_reward': apex_event_height_reward.mean(),
            'Guidance/apex_event_shortfall_m': (
                apex_shortfall * apex_event.float()
            ).sum() / apex_event.float().sum().clamp_min(1),
        })
        return (reward + path_cost + path_tracking + hint_cost + progress_reward
                + xy_progress_reward + pair_reward + stationary_tilt_reward
                + stationary_yaw_reward + stationary_action_reward
                + stationary_apex_reward + stationary_apex_progress_reward
                + stationary_ascent_support_reward + stationary_ascent_collective_reward
                + stationary_predicted_apex_reward + ascent_predicted_apex_reward
                + apex_event_height_reward)
