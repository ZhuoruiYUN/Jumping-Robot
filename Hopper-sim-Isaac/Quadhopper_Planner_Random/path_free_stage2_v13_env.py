"""Opt-in v13 risk shaping layered on v12; all robot dynamics stay unchanged."""
import torch
from isaaclab.utils.math import euler_xyz_from_quat
from .flight_stability import touchdown_roll_pitch_cost
from .path_free_stage2_v12_env import PathFreeStage2V12Env
from .contact_readiness_v13 import (
    SETTINGS_V13, predicted_touchdown_tilt, danger_cost, approach_mask,
    low_altitude_recovery_mask, touchdown_rate_cost,
)


class PathFreeStage2V13Env(PathFreeStage2V12Env):
    def _forecast_tilt(self, quat, rates, clearance, vz):
        return predicted_touchdown_tilt(quat, rates, clearance, vz)

    def _recovery_coverage(self, clearance):
        return ((self._near_ground_recovery.remaining > 0) & (clearance <= 0.035))

    def _get_rewards(self):
        reward = super()._get_rewards()  # Includes v12 and advances cycle exactly once.
        data = self._robot.data
        foot_clearance = self.foot_position_w()[:, 2] - self.foot_geometry['sphere_radius_m']
        vz = data.root_lin_vel_w[:, 2]
        target = self.commands.lookahead()[0]
        moving = (target - self.commands.anchor_w).norm(dim=-1) > self.cfg.stationary_command_radius_m
        tilt, forecast, horizon = self._forecast_tilt(
            data.root_quat_w, data.root_ang_vel_b, foot_clearance, vz)
        approach = approach_mask(moving, self._cycle_active, self._touchdown_event,
                                 self._initial_drop_pending,
                                 self._near_ground_recovery.remaining, foot_clearance, vz)
        risk = danger_cost(tilt, forecast)
        approach_reward = (approach.float() * risk * SETTINGS_V13['approach_risk_scale']
                           * self.step_dt)
        # Direct event credit assignment: a foot hit with a dangerous attitude
        # is not a safe handoff, even if the target circle and apex were hit.
        moving_touchdown = (self._touchdown_event & ~self._completed_stationary_command
                            & ~self._initial_drop_pending)
        touchdown_cost = danger_cost(tilt, tilt)
        touchdown_reward = (moving_touchdown.float() * touchdown_cost
                            * SETTINGS_V13['touchdown_danger_scale'])
        touchdown_rate = touchdown_rate_cost(data.root_ang_vel_b)
        touchdown_rate_reward = (moving_touchdown.float() * touchdown_rate
                                 * SETTINGS_V13['touchdown_rate_scale'])
        v12_near_ground = self._recovery_coverage(foot_clearance)
        low_altitude = low_altitude_recovery_mask(
            self._near_ground_recovery.remaining, v12_near_ground,
            self._initial_drop_pending, data.root_pos_w[:, 2], vz,
            foot_clearance, tilt)
        roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w)
        angle_cost, rate_cost = touchdown_roll_pitch_cost(
            torch.stack((roll, pitch), -1), data.root_ang_vel_b[:, :2],
            self.cfg.moving_long_recovery_angle_deadband_rad,
            self.cfg.moving_long_recovery_rate_deadband_rad_s)
        recovery_reward = (low_altitude.float() * self.step_dt * (
            angle_cost * self.cfg.moving_long_recovery_angle_penalty_scale
            + rate_cost * self.cfg.moving_long_recovery_rate_penalty_scale))
        self.extras['log'].update({
            'V13/approach_active_rate': approach.float().mean(),
            'V13/approach_danger_rate': (approach & (risk > 0)).float().mean(),
            'V13/approach_risk_penalty': approach_reward.mean(),
            'V13/forecast_horizon_s': (horizon * approach.float()).sum()
            / approach.float().sum().clamp_min(1),
            'V13/touchdown_danger_rate': (moving_touchdown & (touchdown_cost > 0)).float().mean(),
            'V13/touchdown_danger_penalty': touchdown_reward.mean(),
            'V13/touchdown_rate_penalty': touchdown_rate_reward.mean(),
            'V13/low_altitude_recovery_active_rate': low_altitude.float().mean(),
            'V13/low_altitude_recovery_penalty': recovery_reward.mean(),
        })
        return (reward + approach_reward + touchdown_reward
                + touchdown_rate_reward + recovery_reward)
