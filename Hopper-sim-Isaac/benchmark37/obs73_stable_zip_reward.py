"""Task adapter for the stable ZIP reward, keeping direct motors and OBS73."""
import torch

from .stable_zip_reward_core import (
    ASCENT, CONTACT, DESCENT, LANDING, RewardInputs, RewardWeights, compute_reward,
)


class StableZipReward:
    def __init__(self, n, device, config):
        values = dict(config['weights'])
        values['action_rate_channels'] = tuple(values['action_rate_channels'])
        self.weights = RewardWeights(**values)
        self.previous_distance = torch.zeros(n, device=device)
        self.previous_action = torch.zeros(n, 4, device=device)
        self.action_ready = torch.zeros(n, device=device, dtype=torch.bool)
        self.hover_dwell = torch.zeros(n, device=device)

    def reset(self, ids, distance):
        self.previous_distance[ids] = distance
        self.previous_action[ids] = 0.
        self.action_ready[ids] = False
        self.hover_dwell[ids] = 0.

    def step(self, env, contact, physical, valid, qualified, peak):
        d = env._robot.data
        f = env._foot_event
        vz = d.root_lin_vel_w[:, 2]
        phase = torch.where(contact, CONTACT, torch.where(vz > 0., ASCENT, DESCENT))
        phase = torch.where(valid, LANDING, phase).long()
        # Existing detector latches apex_seen once per confirmed flight.
        apex = ((f.phase == f.FLIGHT) & (f.air_steps >= 5) & ~f.apex_seen
                & (vz <= 0.) & ~contact & ~physical)
        q = d.root_quat_w
        yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))
        target_yaw = env.yaw_config['target_heading_rad'] if env.yaw_config else 0.
        yaw_error = torch.atan2(torch.sin(yaw-target_yaw),torch.cos(yaw-target_yaw))
        ground = env._terrain.env_origins[:,2]
        output = compute_reward(RewardInputs(
            xy_error=env._desired_pos_w[:,:2]-d.root_pos_w[:,:2],
            height=d.root_pos_w[:,2]-ground,
            target_height=env._desired_pos_w[:,2]-ground,
            yaw_error=yaw_error, quaternion_xy=q[:,1:3],
            body_horizontal_velocity=d.root_lin_vel_b[:,:2],
            motor_commands=env._motor_u,
            spring_position=d.joint_pos[:,env._spring_joint_id],
            spring_velocity=d.joint_vel[:,env._spring_joint_id],
            vertical_velocity=vz, body_rates=d.root_ang_vel_b, phase=phase,
            previous_xy_distance=self.previous_distance,
            previous_executed_action=self.previous_action,
            executed_action=env._actions, action_rate_ready=self.action_ready,
            apex_height=peak-ground, apex_event=apex,
            qualified_landing_event=qualified, failure_event=physical,
            contact=contact, hover_dwell_time=self.hover_dwell,
        ), self.weights, env.step_dt)
        self.previous_distance.copy_(output.next_xy_distance)
        self.previous_action.copy_(env._actions)
        self.action_ready.copy_(output.next_action_rate_ready)
        self.hover_dwell.copy_(output.next_hover_dwell_time)
        for name, value in output.terms.items():
            env._episode_sums['stable_zip_'+name] += value
        env.extras.setdefault('log',{}).update({
            'StableZip/predicted_apex_error_m': output.diagnostics['predicted_apex_error'].mean(),
            'StableZip/hover_trigger_fraction': output.diagnostics['hover_trigger'].mean(),
            'StableZip/goal_events': output.diagnostics['goal_hit'].sum(),
            'StableZip/apex_events': apex.sum(),
        })
        return output.reward
