"""Opt-in v12: prelanding safety shaping and near-ground recovery extension.

No changes to observations, cycle events, route advancement, physics, wrench
timers, termination, success criteria, or motor commands. Old v11 is intact.
"""
import torch
from isaaclab.utils.math import euler_xyz_from_quat
from .path_free_stage2_v3_env import PathFreeStage2V3Env
from .flight_stability import touchdown_roll_pitch_cost
from .landing_recovery_v12 import (
    SETTINGS_V12, NearGroundRecovery, prelanding_mask, prelanding_costs,
)


class PathFreeStage2V12Env(PathFreeStage2V3Env):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._near_ground_recovery = self._create_recovery_state()

    def _create_recovery_state(self):
        return NearGroundRecovery(self.num_envs, self.device, self.step_dt)

    def _update_recovery_state(self, clearance, vz):
        return self._near_ground_recovery.update(
            self._touchdown_event, self._completed_stationary_command,
            self._initial_drop_pending, clearance, vz)

    def _reset_idx(self, env_ids):
        if hasattr(self, '_near_ground_recovery'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._near_ground_recovery.reset(ids)
        super()._reset_idx(env_ids)

    def _get_rewards(self):
        reward = super()._get_rewards()  # Exactly once: includes cycle events.
        data = self._robot.data
        # This task uses a global flat plane at world z=0 (not tiled Z terrain).
        clearance = self.foot_position_w()[:, 2] - self.foot_geometry['sphere_radius_m']
        vz = data.root_lin_vel_w[:, 2]
        recovery, released = self._update_recovery_state(clearance, vz)
        # Parent already charges these samples: use the union, never double count.
        legacy = ((self._moving_long_recovery_remaining > 0)
                  & (data.joint_pos[:, self._spring_joint_id] > 0.002)
                  & ~self._initial_drop_pending)
        extension = recovery & ~legacy
        roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w)
        angle_cost, rate_cost = touchdown_roll_pitch_cost(
            torch.stack((roll, pitch), -1), data.root_ang_vel_b[:, :2],
            self.cfg.moving_long_recovery_angle_deadband_rad,
            self.cfg.moving_long_recovery_rate_deadband_rad_s)
        extension_reward = extension.float() * self.step_dt * (
            angle_cost * self.cfg.moving_long_recovery_angle_penalty_scale
            + rate_cost * self.cfg.moving_long_recovery_rate_penalty_scale)
        target = self.commands.lookahead()[0]
        moving = (target - self.commands.anchor_w).norm(dim=-1) > self.cfg.stationary_command_radius_m
        prelanding = prelanding_mask(
            clearance, vz, moving, self._cycle_active, self._touchdown_event,
            self._initial_drop_pending, self._near_ground_recovery.remaining)
        tilt_cost, pre_rate_cost = prelanding_costs(data.root_quat_w, data.root_ang_vel_b)
        tilt_reward = (prelanding.float() * tilt_cost
                       * SETTINGS_V12['prelanding_tilt_scale'] * self.step_dt)
        rate_reward = (prelanding.float() * pre_rate_cost
                       * SETTINGS_V12['prelanding_rate_scale'] * self.step_dt)
        self.extras['log'].update({
            'V12/prelanding_active_rate': prelanding.float().mean(),
            'V12/prelanding_tilt_penalty': tilt_reward.mean(),
            'V12/prelanding_rate_penalty': rate_reward.mean(),
            'V12/recovery_union_active_rate': (legacy | recovery).float().mean(),
            'V12/recovery_extension_active_rate': extension.float().mean(),
            'V12/recovery_extension_penalty': extension_reward.mean(),
            'V12/recovery_release_rate': released.float().mean(),
        })
        return reward + extension_reward + tilt_reward + rate_reward
