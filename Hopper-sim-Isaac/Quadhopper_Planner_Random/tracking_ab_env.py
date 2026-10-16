"""Isolated direct-motor PPO contract for current/next waypoint ablations."""
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from .random_two_hop_env import PlannerRandomTwoHopEnv, PlannerRandomTwoHopEnvCfg
from .failure_diagnostics import FailureDiagnostics
from .tracking_metrics import TrackingMetrics


@configclass
class TrackingABEnvCfg(PlannerRandomTwoHopEnvCfg):
    preview = True
    collect_tracking_metrics = False
    observation_space = 43
    # Matches the 10--20 cm source checkpoint; never derives from eval distance.
    hop_distance = 0.20
    relative_next_hop_observation = True
    target_height = 1.0
    alternate_target_heights = False
    symmetric_height_tracking = True
    short_hop_radius_min = 0.10
    short_hop_radius_max = 0.20
    long_hop_radius_min = 0.10
    long_hop_radius_max = 0.20
    max_turn_angle_deg = 30.0
    # advance() consumes exactly the announced target, including pair boundaries.
    restart_two_hop_pair = False
    terminate_after_two_hop_pair = False
    advance_route_on_miss = True
    randomize_route_phase = False
    start_from_random_drop = False
    force_full_planner = False
    planner_reference_blend = 0.0
    flight_reference_uses_landing_xy = True
    stance_reference_uses_apex_height = True
    randomize_dynamics = False
    randomize_action_delay = False
    randomize_motor_time_constant = False
    observation_noise_std = 0.0
    # Keep the source actuator contract, including its legacy delay indexing.
    play_motor_time_constant = 0.0674
    play_action_delay = 0
    target_tolerance = 0.10
    apex_tolerance = 0.15
    minimum_valid_apex = 0.85
    require_minimum_apex_for_hit = True
    require_apex_tolerance_for_hit = True
    gate_touchdown_rewards_by_apex = True
    gate_dense_xy_rewards_by_apex = False
    goal_bonus_scale = 0.0
    xy_reward_width = 0.15
    distance_to_xy_reward_scale = 6.0
    # Avoid artificial progress jumps when the target advances.
    xy_progress_reward_scale = 0.0
    xy_error_penalty_scale = -0.10
    lateral_vel_penalty_scale = -0.25
    planner_position_reward_scale = 4.0
    planner_xy_reward_scale = 6.0
    # Static goal does not mean zero desired velocity throughout a jump.
    planner_velocity_reward_scale = 0.0
    descent_velocity_penalty_scale = 0.0
    projected_landing_reward_scale = 4.0
    projected_landing_penalty_scale = -3.0
    target_hit_reward_scale = 18.0
    target_miss_penalty_scale = 0.0
    landing_precision_reward_scale = 10.0
    landing_precision_width = 0.10
    landing_error_penalty_scale = -12.0
    prepared_landing_reward_scale = 0.0
    pair_hit_reward_scale = 0.0
    streak_progress_reward_scale = 0.0
    circle_complete_reward_scale = 0.0
    apex_event_reward_scale = 8.0
    apex_error_penalty_scale = -18.0
    apex_shortfall_penalty_scale = -18.0
    airborne_overshoot_penalty_scale = -8.0
    height_progress_reward_scale = 3.0
    touchdown_attitude_penalty_scale = -4.0
    attitude_penalty_scale = -2.0
    angular_vel_penalty_scale = -0.02
    yaw_penalty_scale = -0.4
    tilt_barrier_penalty_scale = -20.0
    tilt_barrier_start = 0.22
    tilt_barrier_ground_height = 1.15
    termination_penalty_scale = -10.0
    terminate_far_xy_distance = 2.0
    terminate_angvel_norm = 18.0


class TrackingABEnv(PlannerRandomTwoHopEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.tracking_metrics = TrackingMetrics(self.num_envs, self.device)
        self.failure_diagnostics = (
            FailureDiagnostics(self.num_envs, self.device)
            if cfg.collect_tracking_metrics else None
        )

    def _get_observations(self):
        obs = super()._get_observations()
        if not self.cfg.preview:
            obs['policy'][:, 39:41] = 0.0
        return obs

    def _update_cycle_events(self):
        # Latch the physical target BEFORE super advances the queue.
        target = self.commands.lookahead()[0].clone()
        direction = target - self.planner.positions_w[:, 0, :2]
        super()._update_cycle_events()
        diagnostics = getattr(self, 'failure_diagnostics', None)
        if diagnostics is not None:
            diagnostics.note_events(
                self.common_step_counter, self._touchdown_event,
                self._liftoff_event, self._setup_touchdown_event,
            )
        # Keep this diagnostic gate aligned with the configured hit contract.
        # Spatial tracking deliberately scores touchdown XY independently.
        if getattr(self.cfg, 'require_apex_tolerance_for_hit', True):
            self._valid_apex_touchdown_event &= (
                self._settled_apex_error <= self.cfg.apex_tolerance
            )
        if not self.cfg.collect_tracking_metrics or not hasattr(self, 'tracking_metrics'):
            return
        ids = self._touchdown_event.nonzero(as_tuple=False).flatten()
        delta = self._robot.data.root_pos_w[ids, :2] - target[ids]
        d = direction[ids]
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.tracking_metrics.observe(
            ids, self._setup_touchdown_event[ids], delta.norm(dim=1),
            (delta * d).sum(1), (delta[:, 0]*d[:, 1] - delta[:, 1]*d[:, 0]).abs(),
            self._settled_apex_height[ids], self._settled_apex_target[ids],
            torch.rad2deg(torch.acos((1.0 - 2.0 * (
                self._robot.data.root_quat_w[ids, 1].square()
                + self._robot.data.root_quat_w[ids, 2].square()
            )).clamp(-1.0, 1.0))),
        )

    def _get_dones(self):
        diagnostics = getattr(self, 'failure_diagnostics', None)
        if diagnostics is None:
            return super()._get_dones()
        before = {
            name: getattr(self, name).clone()
            for name in ('_physical_death_total', '_tilt_death_total',
                         '_low_height_death_total', '_far_xy_death_total',
                         '_high_angvel_death_total')
        }
        died, timeout = super()._get_dones()
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        root = self._robot.data.root_pos_w
        target = self.commands.lookahead()[0]
        spring = self._robot.data.joint_pos[:, self._spring_joint_id]
        zeros = torch.zeros(self.num_envs, device=self.device)
        diagnostics.capture(
            self._physical_death_total > before['_physical_death_total'],
            step=self.common_step_counter, step_dt=self.step_dt,
            values={
                'root_z_m': root[:, 2],
                'root_vz_m_s': self._robot.data.root_lin_vel_w[:, 2],
                'roll_deg': torch.rad2deg(roll),
                'pitch_deg': torch.rad2deg(pitch),
                'roll_rate_rad_s': self._robot.data.root_ang_vel_b[:, 0],
                'pitch_rate_rad_s': self._robot.data.root_ang_vel_b[:, 1],
                'spring_position_m': spring,
                'target_distance_m': torch.linalg.norm(root[:, :2] - target, dim=1),
                'in_contact': (spring > 0.002).float(),
                'cycle_active': self._cycle_active.float(),
                'moving_recovery_remaining_steps': getattr(
                    self, '_moving_long_recovery_remaining', zeros
                ),
                'hard_environment': getattr(self, '_hardware_hard_mask', zeros),
                'global_thrust_scale': getattr(
                    self, '_hardware_global_thrust_scale', torch.ones_like(zeros)
                ).reshape(-1),
                'tilt_death': self._tilt_death_total > before['_tilt_death_total'],
                'low_height_death': (
                    self._low_height_death_total > before['_low_height_death_total']
                ),
                'far_xy_death': self._far_xy_death_total > before['_far_xy_death_total'],
                'high_angvel_death': (
                    self._high_angvel_death_total > before['_high_angvel_death_total']
                ),
            },
        )
        return died, timeout

    def _reset_idx(self, env_ids):
        diagnostics = getattr(self, 'failure_diagnostics', None)
        if diagnostics is not None:
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            diagnostics.reset_episode(ids)
        if hasattr(self, 'tracking_metrics'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self.tracking_metrics.reset(ids)
        super()._reset_idx(env_ids)
        # Keep the training console small; eval metrics use independent totals.
        keep = ('Metrics/target_hit_rate_ema', 'Metrics/touchdown_error_ema_m',
                'Episode_Reward/apex_error', 'Episode_Reward/landing_error',
                'Episode_Reward/landing_lateral_error',
                'Episode_Reward/target_hit', 'Episode_Reward/takeoff_velocity',
                'Episode_Reward/tracking_xy_progress',
                'Episode_Reward/two_hop_pair_hit',
                'Episode_Reward/projected_landing_error',
                'Diagnostics/reset_tilt_over_limit_count')
        if 'log' in self.extras:
            self.extras['log'] = {k: v for k, v in self.extras['log'].items() if k in keep}
