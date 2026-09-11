"""Single-touchdown recovery curriculum for direct-motor horizontal tracking."""
import math

import torch
from isaaclab.utils import configclass

from .tracking_ab_env import TrackingABEnv, TrackingABEnvCfg


@configclass
class TrackingRecoveryEnvCfg(TrackingABEnvCfg):
    # Teach the current physical landing before asking for two-hop preparation.
    preview = False
    successful_hops_per_episode = 1
    short_hop_radius_min = 0.10
    short_hop_radius_max = 0.20
    long_hop_radius_min = 0.10
    long_hop_radius_max = 0.20
    distance_curriculum_iterations = 180.0
    curriculum_short_radius_min = 0.08
    curriculum_short_radius_max = 0.12
    curriculum_long_radius_min = 0.08
    curriculum_long_radius_max = 0.12
    curriculum_max_turn_angle_deg = 30.0
    curriculum_steps_per_iteration = 256.0
    target_tolerance = 0.06

    # Dense flight feedback plus a physically interpretable liftoff objective.
    gate_dense_xy_rewards_by_apex = True
    projected_landing_reward_scale = 8.0
    projected_landing_penalty_scale = -8.0
    landing_precision_reward_scale = 12.0
    landing_precision_width = 0.08
    landing_error_penalty_scale = -20.0
    target_hit_reward_scale = 25.0
    takeoff_velocity_target_mps = 0.22
    takeoff_velocity_width_mps = 0.12
    takeoff_velocity_from_planner = False
    takeoff_velocity_event_reward = True
    takeoff_velocity_reward_scale = 35.0
    takeoff_velocity_penalty_scale = -12.0
    takeoff_tilt_rad = math.radians(4.0)
    takeoff_tilt_reward_scale = 8.0
    takeoff_tilt_penalty_scale = -2.0
    takeoff_phase_tilt_barrier_relax = 0.5


class TrackingRecoveryEnv(TrackingABEnv):
    cfg: TrackingRecoveryEnvCfg

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._reset_after_touchdown = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _update_cycle_events(self):
        super()._update_cycle_events()
        if hasattr(self, '_reset_after_touchdown'):
            self._reset_after_touchdown |= self._touchdown_event

    def _get_dones(self):
        died, timeout = super()._get_dones()
        if hasattr(self, '_reset_after_touchdown'):
            died = died | self._reset_after_touchdown
        return died, timeout

    def _reset_idx(self, env_ids):
        if hasattr(self, '_reset_after_touchdown'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._reset_after_touchdown[ids] = False
        super()._reset_idx(env_ids)
