"""Isolated release-after-unlock policy experiment on the completed 37-D goal10 candidate."""
import json

import torch

from .artifacts import ROOT, sha
from .goal_radius import contract as goal10_contract
from .real_release import FIRST_TOUCH_SPRING_Q_M

SOURCE = ROOT / "outputs/benchmark37_stage2d_goal10/train_seed42_20261024_033414_809380Z/model_199.pt"
SOURCE_SHA256 = "4038b72408d83d27ec3af9dfc9a085cb88d2382c7c678869d802bbf1e15a52f5"
ITERATIONS = 200
OFF_ACTION = -1.0


def contract():
    return dict(
        name="release_pre_touch_policy_37d_v1",
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        source_goal10_contract=goal10_contract(),
        before_release="motor state and wrench zero; delayed action queue primed to four OFF actions",
        after_release="policy commands accepted from first control step; existing 2-4 step delay and motor lag retained",
        off_wrench="when all four motor states remain zero, suppress the thrust polynomial's nonzero intercept",
        touchdown_proxy_spring_q_m=FIRST_TOUCH_SPRING_Q_M,
        observation="unchanged 37D: current state/target and five four-motor actions; no next-waypoint observation",
        task="stationary fixed target; pre-touch control timing only, not directional two-hop training",
        release_training="fixed +/-5deg roll/pitch, 20% upright, same as goal10 terminal distribution",
        training_iterations=ITERATIONS,
        code_sha256=sha(ROOT / "benchmark37/pre_touch_control.py"),
    )


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or manifest.get("mode") != "train" or manifest.get("status") != "complete"
            or manifest.get("smoke") or manifest.get("variant") != "real_release_attitude_goal10"
            or manifest.get("source_lock") != source_lock or manifest.get("physics_lock") != physics_lock
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or manifest.get("task_intervention") != goal10_contract()
            or manifest.get("reward_intervention", {}).get("code_sha256")
            != sha(ROOT / "benchmark37/energy_margin.py")):
        raise ValueError("Pre-touch candidate requires the pinned complete goal10 model and contracts")
    return manifest


def prime_off_history(env, env_ids):
    """A zero-valued action requests half PWM. OFF is -1 on all four channels."""
    if not len(env_ids):
        return
    for history in env._action_history:
        history[env_ids] = OFF_ACTION
    env._actions[env_ids] = OFF_ACTION
    env._previous_actions[env_ids] = OFF_ACTION
    if not hasattr(env, "_pretouch_motor_samples"):
        env._pretouch_motor_samples = torch.zeros((), dtype=torch.long, device=env.device)
        env._pretouch_nonzero_motor_samples = torch.zeros((), dtype=torch.long, device=env.device)
        env._pretouch_max_motor_u = torch.zeros((), device=env.device)


def update_release_phase(env):
    """Track first compression for measurement only; never mask the policy action."""
    contact = env._robot.data.joint_pos[:, env._spring_joint_id] > FIRST_TOUCH_SPRING_Q_M
    env._release_pending &= ~contact
    return env._release_pending.clone()


def record_pre_touch_motor(env, pending):
    mask = pending[:, None]
    env._pretouch_motor_samples += pending.sum() * env._motor_u.shape[1]
    env._pretouch_nonzero_motor_samples += ((env._motor_u.abs() > 1e-6) & mask).sum()
    env._pretouch_max_motor_u = torch.maximum(
        env._pretouch_max_motor_u, torch.where(mask, env._motor_u.abs(), 0.0).amax())


def suppress_off_wrench(env):
    """The identified polynomial has F(0)>0; a truly stopped vehicle has no motor wrench."""
    stopped = env._release_pending & (env._motor_u.abs().amax(dim=-1) <= 1e-8)
    env._thrust[stopped] = 0.0
    env._moment[stopped] = 0.0
