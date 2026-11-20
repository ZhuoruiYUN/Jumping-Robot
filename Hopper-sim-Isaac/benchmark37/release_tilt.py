"""Evaluation-only release attitude probe; no changes to the frozen task/physics."""
import math

import torch

LIMIT_DEG = 5.0
SEED_OFFSET = 520024


def contract(seed):
    return dict(name="release_roll_pitch_5deg", roll_deg=[-LIMIT_DEG, LIMIT_DEG],
                pitch_deg=[-LIMIT_DEG, LIMIT_DEG], yaw_deg=0.0,
                rotation="Body to world: Rz(0) Ry(pitch) Rx(roll); quaternion wxyz",
                rng="separate generator on env.device; does not consume base global RNG",
                rng_seed=int(seed) + SEED_OFFSET,
                scope="engineering diagnostic range, not a measured real-release distribution",
                initialization="existing checkpoint and normalization, no learning",
                unchanged="root position/velocity, motor gate, physics, 37D observations, rewards, PPO, metrics")


def quaternion(roll_pitch):
    """Euler XYZ with yaw=0, body->world, wxyz (radians)."""
    roll, pitch = (roll_pitch / 2).unbind(-1)
    cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
    return torch.stack((cr * cp, sr * cp, cr * sp, -sr * sp), dim=-1)


def reset_tilt(env, env_ids, seed):
    if not len(env_ids):
        return
    if not hasattr(env, "_release_tilt_generator"):
        env._release_tilt_generator = torch.Generator(device=env.device)
        env._release_tilt_generator.manual_seed(int(seed) + SEED_OFFSET)
        env._release_tilt_rp = torch.zeros(env.num_envs, 2, device=env.device)
    d = env._robot.data
    # This probe replaces the zero-attitude reset. Reject an active source curriculum
    # rather than silently changing a different attitude distribution.
    if torch.any(d.root_quat_w[env_ids, 1:].abs() > 1e-6):
        raise RuntimeError("Release tilt expects the source reset attitude to be identity")
    rp = torch.empty(len(env_ids), 2, device=env.device, dtype=d.root_quat_w.dtype)
    rp.uniform_(-math.radians(LIMIT_DEG), math.radians(LIMIT_DEG),
                generator=env._release_tilt_generator)
    pose = torch.cat((d.root_link_pos_w[env_ids], quaternion(rp)), dim=-1)
    env._robot.write_root_pose_to_sim(pose, env_ids)
    env._release_tilt_rp[env_ids] = rp


def audit_tilt(env, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    rp = env._release_tilt_rp[ids]
    actual = env._robot.data.root_quat_w[ids]
    expected = quaternion(rp)
    error = torch.minimum((actual - expected).abs().amax(-1), (actual + expected).abs().amax(-1))
    if (not torch.all(torch.isfinite(actual)) or torch.any(error > 1e-5)
            or torch.any(rp.abs() > math.radians(LIMIT_DEG) + 1e-6)):
        raise RuntimeError("Live release quaternion does not match requested roll/pitch")
    tilt = torch.rad2deg(torch.acos((1 - 2 * actual[:, 1:3].square().sum(-1)).clamp(-1, 1)))
    return dict(envs=len(actual), max_quaternion_error=float(error.max().item()),
                roll_absmax_deg=float(torch.rad2deg(rp[:, 0]).abs().max().item()),
                pitch_absmax_deg=float(torch.rad2deg(rp[:, 1]).abs().max().item()),
                tilt_max_deg=float(tilt.max().item()))
