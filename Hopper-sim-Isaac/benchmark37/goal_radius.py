"""Isolated goal-bonus radius test on the completed attitude policy."""
import json
import math

import torch

from .artifacts import ROOT, sha
from .attitude_training import contract as attitude_contract
from .release_tilt import quaternion

SOURCE = ROOT / "outputs/benchmark37_stage2d_attitude/train_seed42_20261024_022328_716528Z/model_299.pt"
SOURCE_SHA256 = "c425692433a2605161c48a0fb3f648103760137e94e330675fe1df7c08a2210d"
OLD_RADIUS_M = 0.35
NEW_RADIUS_M = 0.10
ITERATIONS = 200
ZERO_FRACTION = 0.20


def contract():
    return dict(name="attitude_goal_radius_10cm_v1", source_checkpoint=str(SOURCE),
                source_checkpoint_sha256=SOURCE_SHA256, source_attitude_contract=attitude_contract(),
                changed_term="goal_bonus", old_root_xy_radius_m=OLD_RADIUS_M,
                new_root_xy_radius_m=NEW_RADIUS_M, goal_bonus_scale=8.0,
                training_release="roll/pitch independent uniform +/-5deg, 20% upright, yaw zero",
                upright_probability=ZERO_FRACTION, training_iterations=ITERATIONS,
                code_sha256=sha(ROOT / "benchmark37/goal_radius.py"),
                note="Only the goal_bonus root-XY threshold changes; fixed training release matches prior candidate's final exposure. Foot success still uses sphere center.")


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    m = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or m.get("mode") != "train" or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != "real_release_attitude" or m.get("source_lock") != source_lock
            or m.get("physics_lock") != physics_lock
            or m.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or m.get("reward_intervention", {}).get("code_sha256") != sha(ROOT / "benchmark37/energy_margin.py")
            or m.get("task_intervention") != attitude_contract()):
        raise ValueError("Goal-radius training requires the pinned, completed attitude model and matching contracts")
    return m


def goal_bonus_delta(root_xy, target_xy, scale, dt):
    distance = torch.linalg.vector_norm(root_xy - target_xy, dim=-1)
    return ((distance < NEW_RADIUS_M).to(distance.dtype)
            - (distance < OLD_RADIUS_M).to(distance.dtype)) * scale * dt


def reset_fixed_attitude(env, env_ids, seed):
    """Continue with the predecessor's final release mixture, without its ramp."""
    if not len(env_ids):
        return
    if not hasattr(env, "_goal_radius_rng"):
        env._goal_radius_rng = torch.Generator(device=env.device).manual_seed(int(seed) + 720024)
        env._release_tilt_rp = torch.zeros(env.num_envs, 2, device=env.device)
        env._attitude_resets = env._attitude_zero_resets = 0
    d = env._robot.data
    if torch.any(d.root_quat_w[env_ids, 1:].abs() > 1e-6):
        raise RuntimeError("Source release attitude unexpectedly nonzero before fixed-5deg reset")
    rp = torch.empty(len(env_ids), 2, device=env.device, dtype=d.root_quat_w.dtype)
    rp.uniform_(-math.radians(5), math.radians(5), generator=env._goal_radius_rng)
    upright = torch.rand(len(env_ids), device=env.device, generator=env._goal_radius_rng) < ZERO_FRACTION
    rp[upright] = 0.0
    env._robot.write_root_pose_to_sim(torch.cat((d.root_link_pos_w[env_ids], quaternion(rp)), -1), env_ids)
    env._release_tilt_rp[env_ids] = rp
    env._attitude_resets += len(env_ids)
    env._attitude_zero_resets += int(upright.sum().item())
    env._attitude_current_limit_deg = 5.0
