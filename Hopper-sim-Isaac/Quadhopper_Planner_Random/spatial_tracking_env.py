"""Continuous two-waypoint PPO with weak, untimed geometric guidance."""
import math
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from .tracking_ab_env import TrackingABEnv, TrackingABEnvCfg
from .contact_recovery import advance_recovery_window, differential_recovery_costs, recovery_window_steps
from .flight_stability import (correlated_launch_wrench, differential_action_rate_cost, launch_pitch_command_cost,
                               rotor_wrench_torque_from_com_offset, stability_phase_masks,
                               angular_rate_excess_cost, touchdown_roll_pitch_cost)
from .spatial_path import hop_curve, project_path
from .raibert_hermite import raibert_hermite_curve
from .tracking_progress import LandingXYProgress, two_hop_pair_success
from .moving_handoff import moving_contact_event, moving_launch_mask, moving_rate_cost


@configclass
class SpatialTrackingEnvCfg(TrackingABEnvCfg):
    preview = True
    observation_space = 52
    episode_length_s = 15.0
    planner_nodes = 33
    # Opt-in step-to-step guidance.  Defaults retain every pre-Raibert
    # checkpoint's exact geometry and observation semantics.
    use_raibert_hermite_guidance = False
    raibert_gravity_m_s2 = 9.81
    # Bai et al. report a stance phase shorter than roughly 45 ms.  This is a
    # planner parameter only; it does not change Isaac contact dynamics.
    raibert_stance_time_s = 0.045
    raibert_velocity_gain_s = 0.08
    # Keep the guidance endpoint inside the configured success circle (the
    # V22/Raibert launcher uses a 10 cm target tolerance).
    raibert_max_foothold_shift_m = 0.04
    # Blend this hop's desired speed with the following hop: reverse/in-place
    # commands brake before touchdown, straight commands preserve momentum.
    raibert_preview_blend = 0.5
    raibert_max_horizontal_speed_m_s = 2.5
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
    # Event terms target accumulated heading drift.  Unlike a per-timestep
    # yaw-rate cost, an apex/touchdown cost remains visible when a small
    # release bias repeats over hundreds of hops.
    stationary_yaw_event_deadband_rad = math.radians(1.5)
    stationary_yaw_event_penalty_scale = 0.0
    stationary_yaw_hop_change_deadband_rad = math.radians(1.0)
    stationary_yaw_hop_change_penalty_scale = 0.0
    stationary_action_spread_deadband = 0.15
    stationary_action_spread_penalty_scale = -3.0
    # Optional pitch-specific stability shaping for exact zero-hop training.
    # The inherited tilt term remains symmetric; these terms give the hardware
    # pitch axis extra weight without changing observations or motor mixing.
    stationary_pitch_deadband_rad = math.radians(3.0)
    stationary_pitch_penalty_scale = 0.0
    stationary_pitch_rate_deadband_rad_s = 0.25
    stationary_pitch_rate_penalty_scale = 0.0
    # Opt-in phase-specific angular-rate costs. No added torque/controller,
    # collective floor, or changed observation/action dimensions.
    launch_stability_window_s = 0.0
    launch_rate_deadband_rad_s = 0.18
    launch_roll_rate_penalty_scale = 0.0
    launch_pitch_rate_penalty_scale = 0.0
    launch_pitch_command_deadband_pwm = 150.0
    launch_pitch_command_penalty_scale = 0.0
    launch_pitch_command_guard_angle_rad = math.radians(7.0)
    launch_pitch_command_guard_rate_rad_s = 1.0
    # Training-only per-hop torque uncertainty. Nominal and single-env eval
    # remain untouched; no new deployed controller or changed motor interface.
    train_launch_pitch_impulse_max_nm = 0.0
    train_launch_wrench_probability = 0.0
    train_launch_roll_impulse_max_nm = 0.0
    train_launch_pitch_wrench_max_nm = 0.0
    train_launch_yaw_impulse_max_nm = 0.0
    # Correlation of the training-only launch disturbance across successive
    # physical hops.  Zero preserves v12/v13's independent samples exactly.
    # This is a task uncertainty model, not a deployment controller.
    train_launch_wrench_persistence = 0.0
    # Training-only touchdown wrench.  Unlike the launch wrench above, this
    # acts only while the foot is physically in contact immediately after a
    # landing.  It models the measured cross-hop failure mode: a small,
    # persistent contact asymmetry leaves roll/pitch/yaw state for the next
    # liftoff.  Defaults preserve all earlier profiles exactly.
    train_touchdown_wrench_probability = 0.0
    train_touchdown_roll_wrench_max_nm = 0.0
    train_touchdown_pitch_wrench_max_nm = 0.0
    train_touchdown_yaw_wrench_max_nm = 0.0
    train_touchdown_wrench_window_s = 0.0
    train_touchdown_wrench_persistence = 0.0
    train_touchdown_wrench_on_moving_long_hop = False
    # A short physical-contact window after a moving long-hop landing. The
    # 10-degree angle deadband leaves normal directional takeoff lean alone;
    # only excessive roll/pitch and angular rate receive a recovery cost.
    moving_long_recovery_window_s = 0.0
    moving_long_recovery_angle_deadband_rad = math.radians(10.0)
    moving_long_recovery_rate_deadband_rad_s = 0.8
    moving_long_recovery_angle_penalty_scale = 0.0
    moving_long_recovery_rate_penalty_scale = 0.0
    # Task-only, opt-in v11. Keep disturbance timers and stationary shaping separate.
    moving_contact_recovery_all_hops = False
    moving_launch_stability_window_s = 0.0
    moving_launch_rate_deadband_rad_s = 0.8
    moving_launch_rate_penalty_scale = 0.0
    # Deployment-matched direct-motor envelope.  A stationary task may limit
    # only the zero-mean motor component immediately after touchdown, while
    # preserving takeoff collective.  Both zero defaults retain v14 exactly.
    deployment_launch_recovery_window_s = 0.0
    deployment_launch_recovery_spread_limit_pwm = 0.0
    # Zero end value retains a fixed cap.  A non-zero end value enables a
    # deployment-matched post-touchdown ramp back to recovery authority.
    deployment_launch_recovery_hold_s = 0.0
    deployment_launch_recovery_spread_limit_end_pwm = 0.0
    launch_recovery_raw_spread_penalty_scale = 0.0
    stationary_touchdown_angle_penalty_scale = 0.0
    stationary_touchdown_rate_penalty_scale = 0.0
    stationary_touchdown_angle_deadband_rad = math.radians(3.0)
    stationary_touchdown_rate_deadband_rad_s = 0.25
    preapex_yaw_braking_vz_mps = 1.0
    preapex_yaw_rate_deadband_rad_s = 0.20
    preapex_yaw_rate_penalty_scale = 0.0
    # Additional exact-zero-hop shaping for airborne motor chatter.  This
    # measures only the zero-mean four-motor component, preserving collective
    # take-off/ascent authority and the 100 Hz policy contract.
    stationary_airborne_differential_action_rate_deadband = 0.025
    stationary_airborne_differential_action_rate_penalty_scale = 0.0
    # Disabled by default.  This window starts at a physical stationary
    # touchdown and scores the *delayed target* which actually enters the
    # motor model.  It never constrains common-mode/collective thrust.
    stationary_contact_recovery_window_s = 0.0
    stationary_contact_recovery_spread_deadband = 0.35
    stationary_contact_recovery_spread_penalty_scale = 0.0
    stationary_contact_recovery_delta_deadband = 0.20
    stationary_contact_recovery_delta_penalty_scale = 0.0
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
    # Evaluation-only V26 probe; zero leaves the trained action path unchanged.
    diagnostic_ascent_pulse_pwm = 0.0
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
    # V30 path-free task experiment. Default zero preserves every older task.
    liftoff_cross_track_penalty_scale = 0.0
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
    # Optional nominal-to-robust curriculum.  When enabled, the probability
    # rises linearly from ``hardware_randomization_probability_start`` to the
    # final probability above.  It is measured in PPO iterations, preserving
    # its meaning across vectorized environment counts.
    hardware_randomization_probability_start = 1.0
    hardware_randomization_curriculum_iterations = 0.0
    # Optional task-local sim-to-real layer derived from the audited
    # smooth160/model360 training task.  It deliberately leaves the direct
    # four-motor policy interface, motor order, 100 Hz control rate, and
    # deployed actuator contract unchanged.  It models common thrust loss,
    # roll/pitch effectiveness and spring variation *during training only*.
    # Do not randomize motor ordering or axis signs: those are calibration
    # faults to identify, not uncertainty for a policy to average away.
    comprehensive_hardware_randomization = False
    train_global_thrust_scale_min = 0.94
    train_global_thrust_scale_max = 1.04
    train_rp_effectiveness_diag_min = 0.90
    train_rp_effectiveness_diag_max = 1.10
    train_rp_effectiveness_rotation_deg = 5.0
    train_rp_effectiveness_cross_max = 0.05
    # The real rotor drag coefficients are not perfectly paired.  A common
    # yaw-authority scale alone cannot reproduce the observed release-time
    # heading drift, because equal diagonal commands still cancel exactly.
    # Sample both yaw authority and a collective-dependent residual reaction
    # torque.  The latter is zero with stopped motors and reaches the sampled
    # value only at full collective command.
    train_yaw_effectiveness_min = 1.0
    train_yaw_effectiveness_max = 1.0
    train_yaw_bias_torque_max_nm = 0.0
    # Per-rotor reaction-torque coefficient uncertainty.  This is distinct
    # from the common collective-dependent yaw bias: unequal rotor drag makes
    # the residual yaw torque depend on the individual motor commands.  It is
    # sampled once per hard episode and is identity by default.
    train_yaw_reaction_imbalance_max = 0.0
    train_spring_stiffness_scale_min = 0.92
    train_spring_stiffness_scale_max = 1.08
    train_spring_damping_scale_min = 0.90
    train_spring_damping_scale_max = 1.10
    # Battery placement changes the full-system Body CoM.  Ranges are offsets
    # from the calibrated USD CoM, sampled per hard environment reset.
    train_body_com_offset_x_max_m = 0.0
    train_body_com_offset_y_max_m = 0.0
    train_body_com_offset_z_max_m = 0.0


@configclass
class SpatialTrackingPhysical1mEnvCfg(SpatialTrackingEnvCfg):
    """52-D physical-1m task with the measured deployment actuator contract."""

    # Policy, Isaac, and externally translated Vicon data all use absolute
    # root-Z.  The 0.380 m value elsewhere is only the legacy spatial path's
    # landing endpoint; it is not a deployment coordinate offset.
    target_height = 1.000
    deployment_action_target_height = target_height
    corrected_action_delay_indexing = True
    record_executed_action_history = True
    apply_deployment_action_shaping = True


class SpatialTrackingEnv(TrackingABEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # These task-local factors are identity unless the explicit
        # comprehensive training switch is enabled.  They are kept outside
        # Quadhopper_Stable so the proven nominal plant remains untouched.
        self._hardware_global_thrust_scale = torch.ones(
            self.num_envs, 1, device=self.device
        )
        self._hardware_rp_effectiveness = torch.eye(
            2, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._hardware_yaw_effectiveness = torch.ones(
            self.num_envs, device=self.device
        )
        self._hardware_yaw_bias_torque_nm = torch.zeros(
            self.num_envs, device=self.device
        )
        self._hardware_yaw_reaction_scale = torch.ones(
            self.num_envs, 4, device=self.device
        )
        self._sysid_loaded_voltage_v = torch.full(
            (self.num_envs, 1), 3.7, device=self.device
        )
        self._sysid_voltage_beta = torch.full(
            (self.num_envs, 1), 1.44630397739629, device=self.device
        )
        self._sysid_yaw_kqf_m = torch.full(
            (self.num_envs,), 0.010, device=self.device
        )
        self._hardware_spring_stiffness_scale = torch.ones(
            self.num_envs, 1, device=self.device
        )
        self._hardware_spring_damping_scale = torch.ones(
            self.num_envs, 1, device=self.device
        )
        self._hardware_body_com_offset_b = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        # PhysX stores CoM poses on CPU.  Preserve the calibrated pose so an
        # environment returning to the nominal curriculum cannot retain an old
        # battery-placement perturbation.
        self._default_body_coms = self._robot.root_physx_view.get_coms().clone()
        body_ids = torch.as_tensor(self._body_id, dtype=torch.long).flatten()
        if body_ids.numel() != 1:
            raise ValueError("expected exactly one Body rigid link")
        self._body_com_id = int(body_ids.item())
        self._default_spring_stiffness = self._robot.data.default_joint_stiffness[
            :, self._spring_joint_id : self._spring_joint_id + 1
        ].clone()
        self._default_spring_damping = self._robot.data.default_joint_damping[
            :, self._spring_joint_id : self._spring_joint_id + 1
        ].clone()
        self._descending_path = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_arc_fraction = torch.zeros(self.num_envs, device=self.device)
        self._spatial_progress = torch.zeros(self.num_envs, device=self.device)
        self._raibert_foothold_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._raibert_shift_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._raibert_velocity_error_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._raibert_desired_velocity_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._raibert_touchdown_velocity_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._raibert_flight_time_s = torch.zeros(self.num_envs, device=self.device)
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
        self._stationary_contact_recovery_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._moving_long_recovery_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._moving_launch_remaining = torch.zeros_like(self._moving_long_recovery_remaining)
        self._stationary_previous_target_u = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._launch_stability_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._deployment_launch_recovery_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stationary_liftoff_yaw = torch.zeros(self.num_envs, device=self.device)
        self._launch_pitch_impulse_nm = torch.zeros(self.num_envs, device=self.device)
        self._launch_wrench_impulse_nm = torch.zeros(self.num_envs, 3, device=self.device)
        self._persistent_launch_wrench_nm = torch.zeros_like(self._launch_wrench_impulse_nm)
        self._persistent_launch_wrench_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._touchdown_wrench_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._touchdown_wrench_nm = torch.zeros_like(self._launch_wrench_impulse_nm)
        self._persistent_touchdown_wrench_nm = torch.zeros_like(self._launch_wrench_impulse_nm)
        self._persistent_touchdown_wrench_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if not 0.0 <= self.cfg.train_launch_wrench_persistence < 1.0:
            raise ValueError('train_launch_wrench_persistence must be in [0, 1)')
        if not 0.0 <= self.cfg.train_touchdown_wrench_probability <= 1.0:
            raise ValueError('train_touchdown_wrench_probability must be in [0, 1]')
        if not 0.0 <= self.cfg.train_touchdown_wrench_persistence < 1.0:
            raise ValueError('train_touchdown_wrench_persistence must be in [0, 1)')
        if self.cfg.raibert_stance_time_s < 0.0 or self.cfg.raibert_velocity_gain_s < 0.0:
            raise ValueError('Raibert stance time and velocity gain must be non-negative')
        if not 0.0 <= self.cfg.raibert_preview_blend <= 1.0:
            raise ValueError('raibert_preview_blend must be in [0, 1]')
        if (self.cfg.use_raibert_hermite_guidance
                and not 0.0 <= self.cfg.raibert_max_foothold_shift_m < self.cfg.target_tolerance):
            raise ValueError('Raibert foothold shift must stay inside target tolerance')
        if self.cfg.train_touchdown_wrench_window_s < 0.0:
            raise ValueError('train_touchdown_wrench_window_s must be non-negative')
        if not 0.0 <= self.cfg.moving_long_recovery_window_s <= 0.20:
            raise ValueError('moving_long_recovery_window_s must be in [0, 0.20]')
        if not 0.0 <= self.cfg.moving_launch_stability_window_s <= 0.30:
            raise ValueError('moving_launch_stability_window_s must be in [0, 0.30]')
        if (self.cfg.moving_launch_rate_deadband_rad_s < 0.0
                or self.cfg.moving_launch_rate_penalty_scale > 0.0):
            raise ValueError('moving launch needs a non-negative deadband and non-positive scale')
        if (self.cfg.moving_long_recovery_angle_deadband_rad < 0.0
                or self.cfg.moving_long_recovery_rate_deadband_rad_s < 0.0):
            raise ValueError('moving long recovery deadbands must be non-negative')
        if (self.cfg.moving_long_recovery_angle_penalty_scale > 0.0
                or self.cfg.moving_long_recovery_rate_penalty_scale > 0.0):
            raise ValueError('moving long recovery penalties must be non-positive')
        if min(self.cfg.train_touchdown_roll_wrench_max_nm,
               self.cfg.train_touchdown_pitch_wrench_max_nm,
               self.cfg.train_touchdown_yaw_wrench_max_nm) < 0.0:
            raise ValueError('train touchdown wrench bounds must be non-negative')
        if not 0.0 <= self.cfg.train_yaw_reaction_imbalance_max <= 0.5:
            raise ValueError('train_yaw_reaction_imbalance_max must be in [0, 0.5]')
        if self.cfg.deployment_launch_recovery_window_s < 0.0:
            raise ValueError('deployment_launch_recovery_window_s must be non-negative')
        if not 0.0 <= self.cfg.deployment_launch_recovery_spread_limit_pwm <= 1000.0:
            raise ValueError('deployment_launch_recovery_spread_limit_pwm must be in [0, 1000]')
        if not 0.0 <= self.cfg.deployment_launch_recovery_hold_s <= self.cfg.deployment_launch_recovery_window_s:
            raise ValueError('deployment_launch_recovery_hold_s must lie in [0, window]')
        if not 0.0 <= self.cfg.deployment_launch_recovery_spread_limit_end_pwm <= 1000.0:
            raise ValueError('deployment_launch_recovery_spread_limit_end_pwm must be in [0, 1000]')
        self._hardware_hard_mask = torch.zeros(
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
        # foothold shift, velocity error, desired speed, touchdown speed,
        # predicted flight time, active samples
        self._raibert_sum = torch.zeros(6, device=self.device, dtype=torch.float64)
        # Signed, approach-only, and retreat-only reward totals for evaluation.
        self._xy_progress_sum = torch.zeros(3, device=self.device, dtype=torch.float64)

    def _write_body_com_offsets(self, env_ids, offsets_b: torch.Tensor):
        """Write calibrated-plus-offset Body CoMs for selected environments."""
        env_ids_cpu = env_ids.detach().to(device='cpu', dtype=torch.long)
        offsets_cpu = offsets_b.detach().to(device='cpu')
        coms = self._robot.root_physx_view.get_coms().clone()
        coms[env_ids_cpu, self._body_com_id] = self._default_body_coms[
            env_ids_cpu, self._body_com_id
        ]
        coms[env_ids_cpu, self._body_com_id, :3] += offsets_cpu
        self._robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _current_hardware_randomization_probability(self) -> float:
        """Return the reset-time hardware-perturbation probability.

        ``common_step_counter`` advances once for every vectorized simulation
        step, so dividing by the rollout length makes this schedule invariant
        to the number of parallel environments.
        """
        duration = self.cfg.hardware_randomization_curriculum_iterations
        if duration <= 0.0:
            return self.cfg.hardware_randomization_probability
        iteration = float(getattr(self, 'common_step_counter', 0)) / max(
            self.cfg.curriculum_steps_per_iteration, 1.0
        )
        blend = max(0.0, min(1.0, iteration / duration))
        return (
            self.cfg.hardware_randomization_probability_start
            + blend * (
                self.cfg.hardware_randomization_probability
                - self.cfg.hardware_randomization_probability_start
            )
        )

    def _reset_idx(self, env_ids):
        """Mix calibrated and perturbed resets without changing the base plant."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        # DirectRLEnv may dispatch a reset while ``super().__init__`` is still
        # building this subclass, before these task-local buffers exist.
        if hasattr(self, '_stationary_contact_recovery_remaining'):
            self._stationary_contact_recovery_remaining[env_ids] = 0
            self._stationary_previous_target_u[env_ids] = 0.0
            self._moving_long_recovery_remaining[env_ids] = 0
            self._moving_launch_remaining[env_ids] = 0
        if hasattr(self, '_launch_stability_remaining'):
            self._launch_stability_remaining[env_ids] = 0
        if hasattr(self, '_deployment_launch_recovery_remaining'):
            self._deployment_launch_recovery_remaining[env_ids] = 0
            self._launch_pitch_impulse_nm[env_ids] = 0.0
            self._launch_wrench_impulse_nm[env_ids] = 0.0
            self._persistent_launch_wrench_nm[env_ids] = 0.0
            self._persistent_launch_wrench_initialized[env_ids] = False
            self._touchdown_wrench_remaining[env_ids] = 0
            self._touchdown_wrench_nm[env_ids] = 0.0
            self._persistent_touchdown_wrench_nm[env_ids] = 0.0
            self._persistent_touchdown_wrench_initialized[env_ids] = False
            self._hardware_hard_mask[env_ids] = False
            self._stationary_liftoff_yaw[env_ids] = 0.0
        if not hasattr(self, '_hardware_global_thrust_scale'):
            # DirectRLEnv can dispatch reset while the subclass is still being
            # constructed.  The parent has completed its safe reset already.
            return
        identity_rp = torch.eye(
            2, device=self.device, dtype=self._hardware_rp_effectiveness.dtype
        ).unsqueeze(0).expand(len(env_ids), -1, -1)
        self._hardware_global_thrust_scale[env_ids] = 1.0
        self._hardware_rp_effectiveness[env_ids] = identity_rp
        self._hardware_yaw_effectiveness[env_ids] = 1.0
        self._hardware_yaw_bias_torque_nm[env_ids] = 0.0
        self._hardware_yaw_reaction_scale[env_ids] = 1.0
        self._hardware_spring_stiffness_scale[env_ids] = 1.0
        self._hardware_spring_damping_scale[env_ids] = 1.0
        self._hardware_body_com_offset_b[env_ids] = 0.0
        if self.cfg.sysid_20261011_plant:
            self._sysid_loaded_voltage_v[env_ids] = 3.7
            self._sysid_voltage_beta[env_ids] = 1.44630397739629
            self._sysid_yaw_kqf_m[env_ids] = 0.010
        # Reset a previously perturbed spring before sampling this episode.
        # Otherwise an environment that moves from the hard to nominal subset
        # would silently retain the prior episode's physical coefficients.
        if self.cfg.comprehensive_hardware_randomization:
            self._robot.write_joint_stiffness_to_sim(
                self._default_spring_stiffness[env_ids],
                joint_ids=[self._spring_joint_id], env_ids=env_ids,
            )
            self._robot.write_joint_damping_to_sim(
                self._default_spring_damping[env_ids],
                joint_ids=[self._spring_joint_id], env_ids=env_ids,
            )
            self._write_body_com_offsets(
                env_ids, self._hardware_body_com_offset_b[env_ids]
            )

        randomized = self.num_envs > 1 and self.cfg.randomize_dynamics
        probability = self._current_hardware_randomization_probability()
        if not randomized:
            return
        if probability <= 0.0:
            # Parent reset already sampled dynamics. A nominal curriculum
            # phase must restore them below, not return with random values.
            hard_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        elif probability >= 1.0:
            hard_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        else:
            hard_mask = torch.rand(len(env_ids), device=self.device) < probability
        hard_ids = env_ids[hard_mask]
        nominal_ids = env_ids[~hard_mask]
        self._hardware_hard_mask[hard_ids] = True
        # Parent reset has already sampled the perturbations.  Restore the
        # calibrated values for nominal curriculum samples only.  This is
        # task-local and leaves the stable baseline physics unchanged.
        if len(nominal_ids) > 0:
            self.dr_mass_multi[nominal_ids] = 1.0
            self.dr_inertia_multi[nominal_ids] = 1.0
            self.dr_tm[nominal_ids] = self.cfg.play_motor_time_constant
            self.dr_thrust_scale[nominal_ids] = 1.0
            self.dr_thrust_curve_shape[nominal_ids] = 1.0

        if self.cfg.sysid_20261011_plant and len(hard_ids) > 0:
            # First curriculum stage from the source report: physical priors
            # near nominal, KQ/F in 0.008..0.012 m. The broad engineering
            # envelope is deliberately not combined into the first run.
            count = len(hard_ids)
            self.dr_mass_multi[hard_ids] = torch.empty(count, 1, device=self.device).uniform_(
                0.190 / 0.194, 0.198 / 0.194
            )
            relative = torch.empty(count, 1, device=self.device).uniform_(0.95, 1.05)
            self.dr_inertia_multi[hard_ids] = relative.expand(-1, 3)
            common_tau = torch.empty(count, 1, device=self.device).uniform_(0.06, 0.08)
            channel_tau = torch.empty(count, 4, device=self.device).uniform_(0.95, 1.05)
            self.dr_tm[hard_ids] = common_tau * channel_tau
            channel = torch.empty(count, 4, device=self.device).uniform_(-0.05, 0.05)
            channel -= channel.mean(dim=1, keepdim=True)
            channel *= (0.05 / channel.abs().amax(dim=1, keepdim=True).clamp_min(0.05))
            self.dr_thrust_scale[hard_ids] = 1.0 + channel
            self.dr_thrust_curve_shape[hard_ids] = 1.0
            self._sysid_loaded_voltage_v[hard_ids] = torch.empty(
                count, 1, device=self.device
            ).uniform_(3.5, 4.0)
            self._sysid_voltage_beta[hard_ids] = torch.empty(
                count, 1, device=self.device
            ).uniform_(1.3, 1.6)
            self._sysid_yaw_kqf_m[hard_ids] = torch.empty(
                count, device=self.device
            ).uniform_(math.log(0.008), math.log(0.012)).exp_()

        if not self.cfg.comprehensive_hardware_randomization or len(hard_ids) == 0:
            return

        count = len(hard_ids)
        self._hardware_global_thrust_scale[hard_ids] = torch.empty(
            count, 1, device=self.device
        ).uniform_(
            self.cfg.train_global_thrust_scale_min,
            self.cfg.train_global_thrust_scale_max,
        )
        diagonal = torch.empty(count, 2, device=self.device).uniform_(
            self.cfg.train_rp_effectiveness_diag_min,
            self.cfg.train_rp_effectiveness_diag_max,
        )
        angle = torch.empty(count, device=self.device).uniform_(
            -math.radians(self.cfg.train_rp_effectiveness_rotation_deg),
            math.radians(self.cfg.train_rp_effectiveness_rotation_deg),
        )
        cosine, sine = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=1).reshape(
            count, 2, 2
        )
        effectiveness = torch.bmm(
            torch.bmm(rotation, torch.diag_embed(diagonal)), rotation.transpose(1, 2)
        )
        cross = torch.empty(count, 2, device=self.device).uniform_(
            -self.cfg.train_rp_effectiveness_cross_max,
            self.cfg.train_rp_effectiveness_cross_max,
        )
        effectiveness[:, 0, 1].add_(cross[:, 0])
        effectiveness[:, 1, 0].add_(cross[:, 1])
        self._hardware_rp_effectiveness[hard_ids] = effectiveness
        self._hardware_yaw_effectiveness[hard_ids] = torch.empty(
            count, device=self.device
        ).uniform_(
            self.cfg.train_yaw_effectiveness_min,
            self.cfg.train_yaw_effectiveness_max,
        )
        self._hardware_yaw_bias_torque_nm[hard_ids] = torch.empty(
            count, device=self.device
        ).uniform_(
            -self.cfg.train_yaw_bias_torque_max_nm,
            self.cfg.train_yaw_bias_torque_max_nm,
        )
        reaction_limit = self.cfg.train_yaw_reaction_imbalance_max
        if reaction_limit > 0.0:
            # Keep the mean coefficient exactly one.  That preserves the
            # calibrated common yaw authority while creating an episode-long
            # diagonal/per-rotor residual under collective thrust.
            innovation = 2.0 * torch.rand(count, 4, device=self.device) - 1.0
            innovation -= innovation.mean(dim=1, keepdim=True)
            innovation /= innovation.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
            self._hardware_yaw_reaction_scale[hard_ids] = 1.0 + reaction_limit * innovation

        stiffness_scale = torch.empty(count, 1, device=self.device).uniform_(
            self.cfg.train_spring_stiffness_scale_min,
            self.cfg.train_spring_stiffness_scale_max,
        )
        damping_scale = torch.empty(count, 1, device=self.device).uniform_(
            self.cfg.train_spring_damping_scale_min,
            self.cfg.train_spring_damping_scale_max,
        )
        self._hardware_spring_stiffness_scale[hard_ids] = stiffness_scale
        self._hardware_spring_damping_scale[hard_ids] = damping_scale
        com_max = torch.tensor(
            [self.cfg.train_body_com_offset_x_max_m,
             self.cfg.train_body_com_offset_y_max_m,
             self.cfg.train_body_com_offset_z_max_m],
            device=self.device,
        )
        com_offset = (2.0 * torch.rand(count, 3, device=self.device) - 1.0) * com_max
        self._hardware_body_com_offset_b[hard_ids] = com_offset
        self._write_body_com_offsets(hard_ids, com_offset)
        self._robot.write_joint_stiffness_to_sim(
            self._default_spring_stiffness[hard_ids] * stiffness_scale,
            joint_ids=[self._spring_joint_id], env_ids=hard_ids,
        )
        self._robot.write_joint_damping_to_sim(
            self._default_spring_damping[hard_ids] * damping_scale,
            joint_ids=[self._spring_joint_id], env_ids=hard_ids,
        )
        self.extras.setdefault('log', {}).update({
            'Randomization/hard_env_fraction': hard_mask.float().mean(),
            'Randomization/hard_env_probability_target': probability,
            'Randomization/global_thrust_scale': (
                self._hardware_global_thrust_scale[env_ids].mean()
            ),
            'Randomization/rp_effectiveness_diag': (
                self._hardware_rp_effectiveness[env_ids].diagonal(dim1=1, dim2=2).mean()
            ),
            'Randomization/rp_effectiveness_cross_abs': (
                self._hardware_rp_effectiveness[env_ids, 0, 1].abs().mean()
                + self._hardware_rp_effectiveness[env_ids, 1, 0].abs().mean()
            ) / 2.0,
            'Randomization/yaw_effectiveness': (
                self._hardware_yaw_effectiveness[env_ids].mean()
            ),
            'Randomization/yaw_bias_torque_abs_mnm': (
                1000.0 * self._hardware_yaw_bias_torque_nm[env_ids].abs().mean()
            ),
            'Randomization/yaw_reaction_imbalance_abs_pct': (
                100.0 * (self._hardware_yaw_reaction_scale[env_ids] - 1.0).abs().mean()
            ),
            'Randomization/yaw_reaction_diagonal_abs_pct': (
                50.0 * (
                    self._hardware_yaw_reaction_scale[env_ids][:, (0, 2)].mean(dim=1)
                    - self._hardware_yaw_reaction_scale[env_ids][:, (1, 3)].mean(dim=1)
                ).abs().mean()
            ),
            'Randomization/spring_stiffness_scale': (
                self._hardware_spring_stiffness_scale[env_ids].mean()
            ),
            'Randomization/spring_damping_scale': (
                self._hardware_spring_damping_scale[env_ids].mean()
            ),
            'Randomization/body_com_xy_offset_mm': (
                1000.0 * torch.linalg.norm(
                    self._hardware_body_com_offset_b[env_ids, :2], dim=1
                ).mean()
            ),
            'Randomization/body_com_z_offset_abs_mm': (
                1000.0 * self._hardware_body_com_offset_b[env_ids, 2].abs().mean()
            ),
        })

    def _deployment_launch_recovery_spread_limit_pwm(self) -> torch.Tensor:
        """Return the active per-environment differential limit in PWM."""
        start = float(self.cfg.deployment_launch_recovery_spread_limit_pwm)
        end = float(self.cfg.deployment_launch_recovery_spread_limit_end_pwm) or start
        total = recovery_window_steps(
            self.cfg.deployment_launch_recovery_window_s, self.step_dt
        )
        hold = min(
            total,
            recovery_window_steps(self.cfg.deployment_launch_recovery_hold_s, self.step_dt),
        )
        elapsed = (total - self._deployment_launch_recovery_remaining).clamp_min(0)
        ramp_steps = max(total - hold, 1)
        # First ``hold`` policy updates use the strict cap; the remaining
        # samples increase linearly and reach ``end`` at window end.
        blend = ((elapsed - hold + 1).clamp_min(0).to(torch.float32) / ramp_steps)
        return start + (end - start) * blend.clamp(0.0, 1.0)

    def _task_shape_direct_motor_target_u(self, target_u: torch.Tensor) -> torch.Tensor:
        """Apply the deployment-matched post-touchdown spread envelope.

        This limits only the zero-mean direct-motor residual.  The action
        delay has already selected the command that reaches the motor model,
        and the common-mode/height authority is mathematically unchanged.
        """
        limit_pwm = self.cfg.deployment_launch_recovery_spread_limit_pwm
        if limit_pwm <= 0.0 or not hasattr(self, '_deployment_launch_recovery_remaining'):
            return target_u
        active = (
            (self._deployment_launch_recovery_remaining > 0)
            & ~self._initial_drop_pending
        )
        residual = target_u - target_u.mean(dim=-1, keepdim=True)
        half_limit = 0.5 * self._deployment_launch_recovery_spread_limit_pwm() / 1000.0
        limited = target_u.mean(dim=-1, keepdim=True) + residual.clamp(
            -half_limit.unsqueeze(-1), half_limit.unsqueeze(-1)
        )
        return torch.where(active.unsqueeze(-1), limited, target_u).clamp(0.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply task-local hardware effectiveness after the direct motor model.

        The parent computes the preserved F1--F4 thrust polynomial, lag and
        mixer.  This layer only changes the training plant: a common thrust
        factor affects all wrench components, while the 2x2 matrix captures
        roll/pitch authority rotation and cross-coupling seen on a real frame.
        """
        super()._pre_physics_step(actions)
        if not hasattr(self, '_hardware_global_thrust_scale'):
            return
        global_scale = self._hardware_global_thrust_scale
        self._thrust[:, 0, 2].mul_(global_scale[:, 0])
        self._moment[:, 0].mul_(global_scale)
        self._moment[:, 0, :2] = torch.bmm(
            self._hardware_rp_effectiveness,
            self._moment[:, 0, :2].unsqueeze(-1),
        ).squeeze(-1)
        self._moment[:, 0, 2].mul_(self._hardware_yaw_effectiveness)
        collective_u_sq = self._motor_u.square().mean(dim=1)
        self._moment[:, 0, 2].add_(
            self._hardware_yaw_bias_torque_nm
            * collective_u_sq
            / self.dr_inertia_multi[:, 2]
        )
        # The calibrated model has one reaction-torque coefficient.  In a
        # hard training environment, retain its mean but perturb individual
        # rotor coefficients.  A collective command can then generate a
        # persistent residual yaw torque, while a differential yaw command
        # can have asymmetric authority.  This is not a motor-map change: the
        # four-action F1--F4 interface is untouched and scale==1 is exact
        # nominal behavior.
        reaction_delta = self._hardware_yaw_reaction_scale - 1.0
        yaw_signs = torch.tensor(
            [-1.0, 1.0, -1.0, 1.0], device=self.device, dtype=self._motor_u.dtype
        )
        if self.cfg.sysid_20261011_plant:
            reaction_torque = self._sysid_yaw_kqf_m * torch.sum(
                yaw_signs.unsqueeze(0) * reaction_delta * self._sysid_last_motor_forces,
                dim=1,
            )
        else:
            reaction_torque = 0.0172 * torch.sum(
                yaw_signs.unsqueeze(0) * reaction_delta * self._motor_u.square(), dim=1
            )
        reaction_torque *= (
            global_scale[:, 0]
            * self._hardware_yaw_effectiveness
            / self.dr_inertia_multi[:, 2]
        )
        self._moment[:, 0, 2].add_(reaction_torque)
        # The base wrench API applies force at the PhysX CoM.  Add the moment
        # corresponding to the real rotor resultant at the Body link origin.
        com_torque = rotor_wrench_torque_from_com_offset(
            self._thrust[:, 0, 2], self._hardware_body_com_offset_b
        )
        # The base plant represents inertia randomization by scaling applied
        # moments, so apply the same sampled inertia to this added moment.
        self._moment[:, 0].add_(com_torque / self.dr_inertia_multi)
        # Only hard training environments receive a short unmeasured takeoff
        # pitch torque. Sampled afresh every hop below, it cannot be memorized
        # as a fixed episode bias. Keep the calibrated nominal plant exact.
        if self.cfg.train_launch_pitch_impulse_max_nm > 0.0:
            pulse = torch.where(
                self._launch_stability_remaining > 0,
                self._launch_pitch_impulse_nm,
                torch.zeros_like(self._launch_pitch_impulse_nm),
            )
            self._moment[:, 0, 1].add_(pulse / self.dr_inertia_multi[:, 1])
        if self.cfg.train_launch_wrench_probability > 0.0:
            pulse = torch.where(
                (self._launch_stability_remaining > 0).unsqueeze(-1),
                self._launch_wrench_impulse_nm,
                torch.zeros_like(self._launch_wrench_impulse_nm),
            )
            self._moment[:, 0].add_(pulse / self.dr_inertia_multi)
        if self.cfg.train_touchdown_wrench_probability > 0.0:
            contact = self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
            pulse = torch.where(
                ((self._touchdown_wrench_remaining > 0) & contact).unsqueeze(-1),
                self._touchdown_wrench_nm,
                torch.zeros_like(self._touchdown_wrench_nm),
            )
            self._moment[:, 0].add_(pulse / self.dr_inertia_multi)

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
        if self.cfg.use_raibert_hermite_guidance:
            curve, raibert = raibert_hermite_curve(
                start,
                landing,
                following,
                height,
                self._robot.data.root_lin_vel_w[env_ids, :2],
                nodes=self.cfg.planner_nodes,
                next_apex_height=next_height,
                gravity_m_s2=self.cfg.raibert_gravity_m_s2,
                stance_time_s=self.cfg.raibert_stance_time_s,
                velocity_gain_s=self.cfg.raibert_velocity_gain_s,
                max_foothold_shift_m=self.cfg.raibert_max_foothold_shift_m,
                preview_blend=self.cfg.raibert_preview_blend,
                max_horizontal_speed_m_s=self.cfg.raibert_max_horizontal_speed_m_s,
            )
            self.planner.positions_w[env_ids] = curve
            # DirectRLEnv may call _replan while this subclass is still inside
            # super().__init__, before task-local diagnostics are allocated.
            if hasattr(self, '_raibert_foothold_w'):
                self._raibert_foothold_w[env_ids] = raibert['guidance_foothold']
                self._raibert_shift_xy[env_ids] = raibert['foothold_shift_xy']
                self._raibert_velocity_error_xy[env_ids] = raibert['velocity_error_xy']
                self._raibert_desired_velocity_xy[env_ids] = (
                    raibert['desired_takeoff_velocity_xy']
                )
                self._raibert_touchdown_velocity_xy[env_ids] = (
                    raibert['touchdown_velocity_xy']
                )
                self._raibert_flight_time_s[env_ids] = raibert['flight_time_s']
        else:
            self.planner.positions_w[env_ids] = hop_curve(
                start, landing, following, height, self.cfg.planner_nodes
            )
            if hasattr(self, '_raibert_foothold_w'):
                self._raibert_foothold_w[env_ids] = landing
                self._raibert_shift_xy[env_ids] = 0.0
                self._raibert_velocity_error_xy[env_ids] = 0.0
                self._raibert_desired_velocity_xy[env_ids] = 0.0
                self._raibert_touchdown_velocity_xy[env_ids] = 0.0
                self._raibert_flight_time_s[env_ids] = 0.0
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
        # The parent advances the target queue on a hit. Capture the actual
        # completed displacement first, including missed touchdowns, so a
        # zero-distance final phase is never treated as a moving long hop.
        completed_target, _ = self.commands.lookahead()
        completed_stationary = (
            torch.linalg.norm(completed_target - self.commands.anchor_w, dim=1)
            <= self.cfg.stationary_command_radius_m
        )
        super()._update_cycle_events()
        self._completed_stationary_command = torch.where(
            self._touchdown_event,
            completed_stationary,
            self._completed_stationary_command,
        )
        # Capture heading at the physical event, before rewards score this
        # flight.  The body quaternion uses the established world frame, so
        # this measures exactly the yaw held by the policy/deployment contract.
        if hasattr(self, '_stationary_liftoff_yaw'):
            _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
            liftoff = self._liftoff_event & ~self._initial_drop_pending
            self._stationary_liftoff_yaw = torch.where(
                liftoff, yaw, self._stationary_liftoff_yaw
            )
        self._launch_stability_remaining.copy_(advance_recovery_window(
            self._launch_stability_remaining,
            self._liftoff_event & ~self._initial_drop_pending,
            recovery_window_steps(self.cfg.launch_stability_window_s, self.step_dt),
        ))
        self._launch_stability_remaining[self._touchdown_event] = 0
        recovery_touchdown = (
            self._touchdown_event
            & self._completed_stationary_command
            & ~self._initial_drop_pending
        )
        moving_long_touchdown = (
            self._final_touchdown_event
            & ~self._completed_stationary_command
            & ~self._initial_drop_pending
        )
        self._moving_long_recovery_remaining.copy_(advance_recovery_window(
            self._moving_long_recovery_remaining,
            moving_contact_event(
                self._touchdown_event, self._final_touchdown_event,
                self._completed_stationary_command, self._initial_drop_pending,
                all_hops=self.cfg.moving_contact_recovery_all_hops,
            ),
            recovery_window_steps(
                self.cfg.moving_long_recovery_window_s, self.step_dt
            ),
        ))
        self._moving_long_recovery_remaining[self._liftoff_event] = 0
        # This timer must NOT lengthen the existing launch-wrench exposure.
        self._moving_launch_remaining.copy_(advance_recovery_window(
            self._moving_launch_remaining,
            self._liftoff_event & ~self._initial_drop_pending,
            recovery_window_steps(self.cfg.moving_launch_stability_window_s, self.step_dt),
        ))
        self._moving_launch_remaining[self._touchdown_event] = 0
        self._deployment_launch_recovery_remaining.copy_(advance_recovery_window(
            self._deployment_launch_recovery_remaining,
            recovery_touchdown,
            recovery_window_steps(
                self.cfg.deployment_launch_recovery_window_s, self.step_dt
            ),
        ))
        if self.cfg.train_launch_pitch_impulse_max_nm > 0.0:
            launch = self._liftoff_event & ~self._initial_drop_pending
            sampled = (
                2.0 * torch.rand(self.num_envs, device=self.device) - 1.0
            ) * self.cfg.train_launch_pitch_impulse_max_nm
            self._launch_pitch_impulse_nm = torch.where(
                launch & self._hardware_hard_mask,
                sampled,
                self._launch_pitch_impulse_nm,
            )
            self._launch_pitch_impulse_nm[self._touchdown_event] = 0.0
        if self.cfg.train_launch_wrench_probability > 0.0:
            launch = self._liftoff_event & ~self._initial_drop_pending
            bounds = torch.tensor(
                [self.cfg.train_launch_roll_impulse_max_nm,
                 self.cfg.train_launch_pitch_wrench_max_nm,
                 self.cfg.train_launch_yaw_impulse_max_nm],
                device=self.device,
            )
            sampled = (2.0 * torch.rand_like(self._launch_wrench_impulse_nm) - 1.0) * bounds
            enabled = (self.num_envs > 1 and self.cfg.randomize_dynamics)
            selected = launch & (torch.rand(self.num_envs, device=self.device)
                                 < self.cfg.train_launch_wrench_probability) & enabled
            if self.cfg.train_launch_wrench_persistence > 0.0:
                correlated = correlated_launch_wrench(
                    self._persistent_launch_wrench_nm,
                    sampled,
                    self._persistent_launch_wrench_initialized,
                    self.cfg.train_launch_wrench_persistence,
                    bounds,
                )
                self._persistent_launch_wrench_nm = torch.where(
                    (launch & selected).unsqueeze(-1), correlated,
                    self._persistent_launch_wrench_nm,
                )
                self._persistent_launch_wrench_initialized |= launch & selected
                launch_wrench = torch.where(
                    selected.unsqueeze(-1), self._persistent_launch_wrench_nm,
                    torch.zeros_like(sampled),
                )
            else:
                launch_wrench = torch.where(
                    selected.unsqueeze(-1), sampled, torch.zeros_like(sampled)
                )
            self._launch_wrench_impulse_nm = torch.where(
                launch.unsqueeze(-1), launch_wrench, self._launch_wrench_impulse_nm,
            )
            self._launch_wrench_impulse_nm[self._touchdown_event] = 0.0
        if self.cfg.train_touchdown_wrench_probability > 0.0:
            wrench_touchdown = (
                recovery_touchdown | moving_long_touchdown
                if self.cfg.train_touchdown_wrench_on_moving_long_hop
                else recovery_touchdown
            )
            bounds = torch.tensor(
                [self.cfg.train_touchdown_roll_wrench_max_nm,
                 self.cfg.train_touchdown_pitch_wrench_max_nm,
                 self.cfg.train_touchdown_yaw_wrench_max_nm],
                device=self.device,
            )
            sampled = (
                2.0 * torch.rand_like(self._touchdown_wrench_nm) - 1.0
            ) * bounds
            enabled = self.num_envs > 1 and self.cfg.randomize_dynamics
            selected = (
                wrench_touchdown
                & self._hardware_hard_mask
                & (torch.rand(self.num_envs, device=self.device)
                   < self.cfg.train_touchdown_wrench_probability)
                & enabled
            )
            if self.cfg.train_touchdown_wrench_persistence > 0.0:
                correlated = correlated_launch_wrench(
                    self._persistent_touchdown_wrench_nm,
                    sampled,
                    self._persistent_touchdown_wrench_initialized,
                    self.cfg.train_touchdown_wrench_persistence,
                    bounds,
                )
                self._persistent_touchdown_wrench_nm = torch.where(
                    selected.unsqueeze(-1), correlated,
                    self._persistent_touchdown_wrench_nm,
                )
                self._persistent_touchdown_wrench_initialized |= selected
                wrench = torch.where(
                    selected.unsqueeze(-1), self._persistent_touchdown_wrench_nm,
                    torch.zeros_like(sampled),
                )
            else:
                wrench = torch.where(
                    selected.unsqueeze(-1), sampled, torch.zeros_like(sampled)
                )
            self._touchdown_wrench_nm = torch.where(
                wrench_touchdown.unsqueeze(-1), wrench, self._touchdown_wrench_nm
            )
            self._touchdown_wrench_remaining.copy_(advance_recovery_window(
                self._touchdown_wrench_remaining,
                selected,
                recovery_window_steps(
                    self.cfg.train_touchdown_wrench_window_s, self.step_dt
                ),
            ))
        # This is task logic only: it does not inject an impulse or modify the
        # contact model.  The window starts at observed physical touchdown and
        # lets PPO assign cost to the delayed recovery commands that follow.
        self._stationary_contact_recovery_remaining.copy_(advance_recovery_window(
            self._stationary_contact_recovery_remaining,
            recovery_touchdown,
            recovery_window_steps(
                self.cfg.stationary_contact_recovery_window_s, self.step_dt
            ),
        ))
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
        target, _ = self.commands.lookahead()
        start_xy = self.planner.positions_w[:, 0, :2]
        # Task progress stays anchored to the commanded circle centre.  The
        # bounded Raibert foothold changes guidance only, never the scoring
        # target or the touchdown success definition.
        landing_xy = target
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
        roll, pitch, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        touchdown_recovery = (
            self._final_touchdown_event & self._completed_stationary_command
            & ~self._initial_drop_pending
        )
        touchdown_angle_cost, touchdown_rate_cost = touchdown_roll_pitch_cost(
            torch.stack((roll, pitch), dim=-1),
            self._robot.data.root_ang_vel_b[:, :2],
            self.cfg.stationary_touchdown_angle_deadband_rad,
            self.cfg.stationary_touchdown_rate_deadband_rad_s,
        )
        touchdown_angle_reward = (
            touchdown_recovery.float() * touchdown_angle_cost
            * self.cfg.stationary_touchdown_angle_penalty_scale
        )
        touchdown_rate_reward = (
            touchdown_recovery.float() * touchdown_rate_cost
            * self.cfg.stationary_touchdown_rate_penalty_scale
        )
        moving_long_recovery = (
            (self._moving_long_recovery_remaining > 0)
            & (self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002)
            & ~self._initial_drop_pending
        )
        moving_angle_cost, moving_contact_rate_cost = touchdown_roll_pitch_cost(
            torch.stack((roll, pitch), dim=-1),
            self._robot.data.root_ang_vel_b[:, :2],
            self.cfg.moving_long_recovery_angle_deadband_rad,
            self.cfg.moving_long_recovery_rate_deadband_rad_s,
        )
        moving_angle_reward = (
            moving_long_recovery.float() * moving_angle_cost
            * self.cfg.moving_long_recovery_angle_penalty_scale * self.step_dt
        )
        moving_rate_reward = (
            moving_long_recovery.float() * moving_contact_rate_cost
            * self.cfg.moving_long_recovery_rate_penalty_scale * self.step_dt
        )
        yaw_hop_change = torch.atan2(
            torch.sin(yaw - self._stationary_liftoff_yaw),
            torch.cos(yaw - self._stationary_liftoff_yaw),
        )
        stationary_yaw_event = (
            (self._apex_event | self._touchdown_event)
            & (stationary_command | self._completed_stationary_command)
            & ~self._initial_drop_pending
        )
        stationary_yaw_event_cost = torch.square(
            (yaw.abs() - self.cfg.stationary_yaw_event_deadband_rad).clamp_min(0.0)
            / math.radians(10.0)
        )
        stationary_yaw_hop_change_cost = torch.square(
            (yaw_hop_change.abs() - self.cfg.stationary_yaw_hop_change_deadband_rad)
            .clamp_min(0.0) / math.radians(5.0)
        )
        stationary_yaw_event_reward = (
            stationary_yaw_event.float() * stationary_yaw_event_cost
            * self.cfg.stationary_yaw_event_penalty_scale
        )
        stationary_yaw_hop_change_reward = (
            stationary_yaw_event.float() * stationary_yaw_hop_change_cost
            * self.cfg.stationary_yaw_hop_change_penalty_scale
        )
        stationary_pitch_active = (
            stationary_command & active & ~self._initial_drop_pending
        )
        stationary_pitch_cost = torch.square(
            (torch.abs(pitch) - self.cfg.stationary_pitch_deadband_rad).clamp_min(0.0)
            / math.radians(10.0)
        )
        stationary_pitch_rate_cost = torch.square(
            (torch.abs(self._robot.data.root_ang_vel_b[:, 1])
             - self.cfg.stationary_pitch_rate_deadband_rad_s).clamp_min(0.0)
            / 1.0
        )
        stationary_pitch_reward = (
            stationary_pitch_active.float() * stationary_pitch_cost
            * self.cfg.stationary_pitch_penalty_scale * self.step_dt
        )
        stationary_pitch_rate_reward = (
            stationary_pitch_active.float() * stationary_pitch_rate_cost
            * self.cfg.stationary_pitch_rate_penalty_scale * self.step_dt
        )
        launch_active, yaw_braking_active = stability_phase_masks(
            self._robot.data.joint_pos[:, self._spring_joint_id],
            self._robot.data.joint_vel[:, self._spring_joint_id],
            active, self._launch_stability_remaining, self._initial_drop_pending,
            stationary_command, self._robot.data.root_lin_vel_w[:, 2],
            self.cfg.preapex_yaw_braking_vz_mps,
        )
        moving_launch_active = moving_launch_mask(
            self._robot.data.joint_pos[:, self._spring_joint_id],
            self._robot.data.joint_vel[:, self._spring_joint_id],
            active, self._moving_launch_remaining, self._initial_drop_pending,
            stationary_command, self._touchdown_event,
        ) & (self.cfg.moving_launch_stability_window_s > 0.0)
        moving_launch_reward = (
            moving_launch_active.float()
            * moving_rate_cost(self._robot.data.root_ang_vel_b[:, :2],
                               self.cfg.moving_launch_rate_deadband_rad_s)
            * self.cfg.moving_launch_rate_penalty_scale * self.step_dt
        )
        launch_cost = angular_rate_excess_cost(
            self._robot.data.root_ang_vel_b[:, :2], self.cfg.launch_rate_deadband_rad_s,
        )
        launch_roll_reward = (launch_active.float() * launch_cost[:, 0]
                              * self.cfg.launch_roll_rate_penalty_scale * self.step_dt)
        launch_pitch_reward = (launch_active.float() * launch_cost[:, 1]
                               * self.cfg.launch_pitch_rate_penalty_scale * self.step_dt)
        launch_pitch_command_cost_value, launch_pitch_command_pwm = (
            launch_pitch_command_cost(
                self._shaped_target_u,
                self.cfg.launch_pitch_command_deadband_pwm,
            )
        )
        launch_pitch_command_active = (
            launch_active
            & (pitch.abs() <= self.cfg.launch_pitch_command_guard_angle_rad)
            & (self._robot.data.root_ang_vel_b[:, 1].abs()
               <= self.cfg.launch_pitch_command_guard_rate_rad_s)
        )
        launch_pitch_command_reward = (
            launch_pitch_command_active.float() * launch_pitch_command_cost_value
            * self.cfg.launch_pitch_command_penalty_scale * self.step_dt
        )
        yaw_braking_reward = (
            yaw_braking_active.float() * angular_rate_excess_cost(
                self._robot.data.root_ang_vel_b[:, 2],
                self.cfg.preapex_yaw_rate_deadband_rad_s,
            ) * self.cfg.preapex_yaw_rate_penalty_scale * self.step_dt
        )
        airborne_action_rate_cost, airborne_action_rate = differential_action_rate_cost(
            self._actions, self._previous_actions,
            self.cfg.stationary_airborne_differential_action_rate_deadband,
        )
        airborne_action_rate_reward = (
            stationary_pitch_active.float() * airborne_action_rate_cost
            * self.cfg.stationary_airborne_differential_action_rate_penalty_scale
            * self.step_dt
        )
        recovery_active = self._stationary_contact_recovery_remaining > 0
        recovery_spread_cost, recovery_delta_cost, recovery_spread, recovery_delta = (
            differential_recovery_costs(
                self._shaped_target_u,
                self._stationary_previous_target_u,
                self.cfg.stationary_contact_recovery_spread_deadband,
                self.cfg.stationary_contact_recovery_delta_deadband,
            )
        )
        recovery_spread_reward = (
            recovery_active.float() * recovery_spread_cost
            * self.cfg.stationary_contact_recovery_spread_penalty_scale * self.step_dt
        )
        recovery_delta_reward = (
            recovery_active.float() * recovery_delta_cost
            * self.cfg.stationary_contact_recovery_delta_penalty_scale * self.step_dt
        )
        launch_envelope_active = self._deployment_launch_recovery_remaining > 0
        raw_launch_spread = (
            self._raw_target_u.max(dim=-1).values
            - self._raw_target_u.min(dim=-1).values
        )
        launch_spread_limit = self._deployment_launch_recovery_spread_limit_pwm() / 1000.0
        if self.cfg.deployment_launch_recovery_spread_limit_pwm > 0.0:
            launch_raw_spread_cost = torch.square(
                (raw_launch_spread - launch_spread_limit).clamp_min(0.0) / 0.15
            )
        else:
            launch_raw_spread_cost = torch.zeros_like(raw_launch_spread)
        launch_raw_spread_reward = (
            launch_envelope_active.float() * launch_raw_spread_cost
            * self.cfg.launch_recovery_raw_spread_penalty_scale * self.step_dt
        )
        # Save the post-delay target after scoring, so a common-mode change is
        # exactly invisible to the next differential-delta cost as well.
        self._stationary_previous_target_u.copy_(self._shaped_target_u)
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
        self._raibert_sum += torch.stack((
            (torch.linalg.vector_norm(self._raibert_shift_xy, dim=1) * active).sum(),
            (torch.linalg.vector_norm(self._raibert_velocity_error_xy, dim=1) * active).sum(),
            (torch.linalg.vector_norm(self._raibert_desired_velocity_xy, dim=1) * active).sum(),
            (torch.linalg.vector_norm(self._raibert_touchdown_velocity_xy, dim=1) * active).sum(),
            (self._raibert_flight_time_s * active).sum(),
            active.sum(),
        ))
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
            'Guidance/raibert_foothold_shift_m': (
                torch.linalg.vector_norm(self._raibert_shift_xy, dim=1) * active
            ).sum() / active.sum().clamp_min(1),
            'Guidance/raibert_velocity_error_m_s': (
                torch.linalg.vector_norm(self._raibert_velocity_error_xy, dim=1) * active
            ).sum() / active.sum().clamp_min(1),
            'Guidance/raibert_desired_speed_m_s': (
                torch.linalg.vector_norm(self._raibert_desired_velocity_xy, dim=1) * active
            ).sum() / active.sum().clamp_min(1),
            'Guidance/raibert_touchdown_speed_m_s': (
                torch.linalg.vector_norm(self._raibert_touchdown_velocity_xy, dim=1) * active
            ).sum() / active.sum().clamp_min(1),
            'Guidance/raibert_flight_time_s': (
                self._raibert_flight_time_s * active
            ).sum() / active.sum().clamp_min(1),
            'Guidance/stationary_hold_rate': stationary_hold.float().mean(),
            'Guidance/stationary_command_rate': stationary_command.float().mean(),
            'Guidance/stationary_descent_rate': stationary_descent.float().mean(),
            'Guidance/stationary_tilt_penalty': stationary_tilt_reward.mean(),
            'Guidance/stationary_yaw_rate_penalty': stationary_yaw_reward.mean(),
            'Guidance/stationary_action_spread_penalty': stationary_action_reward.mean(),
            'Guidance/stationary_pitch_abs_deg': (
                torch.rad2deg(torch.abs(pitch)) * stationary_pitch_active.float()
            ).sum() / stationary_pitch_active.float().sum().clamp_min(1),
            'Guidance/stationary_yaw_abs_deg': (
                torch.rad2deg(torch.abs(yaw)) * stationary_pitch_active.float()
            ).sum() / stationary_pitch_active.float().sum().clamp_min(1),
            'Guidance/stationary_yaw_event_rate': stationary_yaw_event.float().mean(),
            'Guidance/stationary_yaw_event_abs_deg': (
                torch.rad2deg(yaw.abs()) * stationary_yaw_event.float()
            ).sum() / stationary_yaw_event.float().sum().clamp_min(1),
            'Guidance/stationary_yaw_hop_change_abs_deg': (
                torch.rad2deg(yaw_hop_change.abs()) * stationary_yaw_event.float()
            ).sum() / stationary_yaw_event.float().sum().clamp_min(1),
            'Guidance/stationary_yaw_event_penalty': stationary_yaw_event_reward.mean(),
            'Guidance/stationary_yaw_hop_change_penalty': (
                stationary_yaw_hop_change_reward.mean()
            ),
            'Guidance/stationary_pitch_penalty': stationary_pitch_reward.mean(),
            'Guidance/touchdown_recovery_event_rate': touchdown_recovery.float().mean(),
            'Guidance/touchdown_angle_penalty': touchdown_angle_reward.mean(),
            'Guidance/touchdown_rate_penalty': touchdown_rate_reward.mean(),
            'Guidance/moving_long_recovery_active_rate': moving_long_recovery.float().mean(),
            'Guidance/moving_long_recovery_angle_penalty': moving_angle_reward.mean(),
            'Guidance/moving_long_recovery_rate_penalty': moving_rate_reward.mean(),
            'Guidance/moving_launch_active_rate': moving_launch_active.float().mean(),
            'Guidance/moving_launch_rate_penalty': moving_launch_reward.mean(),
            'Randomization/launch_wrench_active_rate': (
                (self._launch_wrench_impulse_nm.abs().sum(-1) > 0)
                & (self._launch_stability_remaining > 0)
            ).float().mean(),
            'Randomization/persistent_launch_wrench_abs_mnm': (
                1000.0 * self._persistent_launch_wrench_nm.abs().mean()
            ),
            'Randomization/launch_wrench_persistence': torch.full(
                (), self.cfg.train_launch_wrench_persistence, device=self.device
            ),
            'Randomization/touchdown_wrench_active_rate': (
                (self._touchdown_wrench_remaining > 0).float().mean()
            ),
            'Randomization/persistent_touchdown_wrench_abs_mnm': (
                1000.0 * self._persistent_touchdown_wrench_nm.abs().mean()
            ),
            'Randomization/touchdown_wrench_persistence': torch.full(
                (), self.cfg.train_touchdown_wrench_persistence, device=self.device
            ),
            'Guidance/stationary_pitch_rate_penalty': stationary_pitch_rate_reward.mean(),
            'Guidance/launch_stability_active_rate': launch_active.float().mean(),
            'Guidance/launch_roll_rate_penalty': launch_roll_reward.mean(),
            'Guidance/launch_pitch_rate_penalty': launch_pitch_reward.mean(),
            'Guidance/launch_pitch_command_penalty': launch_pitch_command_reward.mean(),
            'Guidance/launch_pitch_command_abs_pwm': (
                launch_pitch_command_pwm.abs() * launch_pitch_command_active.float()
            ).sum() / launch_pitch_command_active.float().sum().clamp_min(1),
            'Randomization/launch_pitch_impulse_abs_mnm': (
                1000.0 * self._launch_pitch_impulse_nm.abs()
                * self._hardware_hard_mask.float()
            ).sum() / self._hardware_hard_mask.float().sum().clamp_min(1),
            'Randomization/launch_pitch_impulse_active_rate': (
                (self._hardware_hard_mask
                 & (self._launch_stability_remaining > 0)).float().mean()
            ),
            'Guidance/launch_pitch_abs_rate_rad_s': (
                self._robot.data.root_ang_vel_b[:, 1].abs() * launch_active.float()
            ).sum() / launch_active.float().sum().clamp_min(1),
            'Guidance/preapex_yaw_braking_active_rate': yaw_braking_active.float().mean(),
            'Guidance/preapex_yaw_rate_penalty': yaw_braking_reward.mean(),
            'Guidance/airborne_differential_action_rate': (
                airborne_action_rate * stationary_pitch_active.float()
            ).sum() / stationary_pitch_active.float().sum().clamp_min(1),
            'Guidance/airborne_differential_action_rate_penalty': (
                airborne_action_rate_reward.mean()
            ),
            'Guidance/contact_recovery_active_rate': recovery_active.float().mean(),
            'Guidance/contact_recovery_target_spread': (
                recovery_spread * recovery_active.float()
            ).sum() / recovery_active.float().sum().clamp_min(1),
            'Guidance/contact_recovery_target_delta': (
                recovery_delta * recovery_active.float()
            ).sum() / recovery_active.float().sum().clamp_min(1),
            'Guidance/contact_recovery_spread_penalty': recovery_spread_reward.mean(),
            'Guidance/contact_recovery_delta_penalty': recovery_delta_reward.mean(),
            'Guidance/launch_envelope_active_rate': launch_envelope_active.float().mean(),
            'Guidance/launch_envelope_raw_spread_pwm': (
                1000.0 * raw_launch_spread * launch_envelope_active.float()
            ).sum() / launch_envelope_active.float().sum().clamp_min(1),
            'Guidance/launch_envelope_limit_pwm': (
                self._deployment_launch_recovery_spread_limit_pwm()
                * launch_envelope_active.float()
            ).sum() / launch_envelope_active.float().sum().clamp_min(1),
            'Guidance/launch_envelope_raw_spread_penalty': launch_raw_spread_reward.mean(),
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
                + stationary_yaw_event_reward + stationary_yaw_hop_change_reward
                + stationary_pitch_reward + stationary_pitch_rate_reward
                + touchdown_angle_reward + touchdown_rate_reward
                + moving_angle_reward + moving_rate_reward + moving_launch_reward
                + launch_roll_reward + launch_pitch_reward
                + launch_pitch_command_reward + yaw_braking_reward
                + airborne_action_rate_reward
                + recovery_spread_reward + recovery_delta_reward + launch_raw_spread_reward
                + stationary_apex_reward + stationary_apex_progress_reward
                + stationary_ascent_support_reward + stationary_ascent_collective_reward
                + stationary_predicted_apex_reward + ascent_predicted_apex_reward
                + apex_event_height_reward)
