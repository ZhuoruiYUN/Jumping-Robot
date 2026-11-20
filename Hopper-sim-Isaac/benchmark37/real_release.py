"""Stage 2c release contract, separate from the frozen 37-D source and SysID physics."""

import torch


ROOT_HEIGHT_MIN_M = 0.8
ROOT_HEIGHT_MAX_M = 1.2
TARGET_ROOT_APEX_WORLD_M = 1.0
FIRST_TOUCH_SPRING_Q_M = 0.002


def reset_release(env, env_ids):
    """Keep the source's XY, attitude and DR; change absolute root Z and motor gate."""
    if not hasattr(env, "_release_pending"):
        env._release_pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._release_pending[env_ids] = True
    pose = torch.cat((env._robot.data.root_pos_w[env_ids],
                      env._robot.data.root_quat_w[env_ids]), dim=-1).clone()
    pose[:, 2] = torch.empty(len(env_ids), dtype=pose.dtype, device=env.device).uniform_(
        ROOT_HEIGHT_MIN_M, ROOT_HEIGHT_MAX_M)
    env._desired_pos_w[env_ids, 2] = TARGET_ROOT_APEX_WORLD_M
    env._robot.write_root_pose_to_sim(pose, env_ids)
    env._motor_u[env_ids] = 0.0
    env._thrust[env_ids] = 0.0
    env._moment[env_ids] = 0.0


def gate_actions_before_physics(env, actions):
    """Only a previously observed spring compression may unlock the motors."""
    if not hasattr(env, "_release_pending"):
        raise RuntimeError("Release gate was not initialized by reset")
    contact = env._robot.data.joint_pos[:, env._spring_joint_id] > FIRST_TOUCH_SPRING_Q_M
    env._release_pending &= ~contact
    return torch.where(env._release_pending[:, None], torch.full_like(actions, -1.0), actions)


def enforce_off_before_touch(env):
    """Zero actual motor state and wrench, including the thrust polynomial intercept."""
    pending = env._release_pending
    env._motor_u[pending] = 0.0
    env._thrust[pending] = 0.0
    env._moment[pending] = 0.0
    env._motor_u_history[pending, -1, :] = 0.0


def audit_reset(env):
    """Fail before a run if the live PhysX reset does not match the release contract."""
    z = env._robot.data.root_link_pos_w[:, 2]
    target_z = env._desired_pos_w[:, 2]
    joint_q = env._robot.data.joint_pos[:, env._spring_joint_id]
    joint_qdot = env._robot.data.joint_vel[:, env._spring_joint_id]
    pending = env._release_pending
    if (torch.any(z < ROOT_HEIGHT_MIN_M - 1e-5)
            or torch.any(z > ROOT_HEIGHT_MAX_M + 1e-5)
            or torch.any(torch.abs(target_z - TARGET_ROOT_APEX_WORLD_M) > 1e-5)
            or not torch.all(pending)
            or torch.any(torch.abs(joint_q) > 1e-6)
            or torch.any(torch.abs(joint_qdot) > 1e-6)
            or torch.any(env._motor_u != 0)
            or torch.any(env._thrust != 0)
            or torch.any(env._moment != 0)):
        raise RuntimeError("Live reset violates stage 2c release height, target or motor-off gate")
    return dict(root_link_z_min_m=float(z.min().item()), root_link_z_max_m=float(z.max().item()),
                target_root_apex_world_m=TARGET_ROOT_APEX_WORLD_M,
                first_touch_spring_q_m=FIRST_TOUCH_SPRING_Q_M,
                spring_reset_q_absmax_m=float(joint_q.abs().max().item()),
                spring_reset_qdot_absmax_m_s=float(joint_qdot.abs().max().item()),
                motor_off_envs=int(pending.sum().item()), num_envs=env.num_envs,
                note="Spring compression is a touchdown proxy, not a physical contact sensor")
