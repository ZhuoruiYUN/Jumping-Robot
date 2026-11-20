"""Isolated stationary release-attitude curriculum and explicit policy warm start."""
import json
import math

import torch

from .artifacts import ROOT, sha
from .release_tilt import quaternion

VARIANT = "real_release_attitude"
ITERATIONS = 300
RAMP_ITERATIONS = 100
STEPS_PER_ITERATION = 256
ZERO_FRACTION = 0.2


def limit_deg(control_steps):
    return 1.0 + 4.0 * min(max(float(control_steps), 0.0) / (RAMP_ITERATIONS * STEPS_PER_ITERATION), 1.0)


def contract():
    return dict(name="release_attitude_curriculum_v1", start_roll_pitch_limit_deg=1.0,
                final_roll_pitch_limit_deg=5.0, ramp_control_steps=RAMP_ITERATIONS * STEPS_PER_ITERATION,
                schedule="1 + 4 * clip(common_step_counter / 25600, 0, 1); no num_envs divisor",
                zero_attitude_probability=ZERO_FRACTION, yaw_deg=0.0,
                training_iterations=ITERATIONS, steps_per_iteration=STEPS_PER_ITERATION,
                code_sha256=sha(ROOT / "benchmark37/attitude_training.py"),
                quaternion_helper_sha256=sha(ROOT / "benchmark37/release_tilt.py"),
                initialization="actor, critic, action std and empirical normalization from pinned baseline; fresh optimizer and iteration 0",
                evaluation="default: original upright release; --release-tilt-5deg: fixed +/-5 deg with no upright mixture",
                note="engineering training distribution, not measured release statistics; no XY/yaw curriculum changes")


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    pin = json.loads((ROOT / "benchmark37/provisional_baseline.json").read_text())
    m = json.loads((path.parent / "manifest.json").read_text())
    if (path != (ROOT / pin["checkpoint"]).resolve() or sha(path) != pin["checkpoint_sha256"]
            or m.get("mode") != "train" or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != "real_release_energy2cm" or m.get("source_lock") != source_lock
            or m.get("physics_lock") != physics_lock
            or m.get("checkpoint_hashes", {}).get(path.name) != pin["checkpoint_sha256"]
            or m.get("reward_intervention", {}).get("code_sha256") != sha(ROOT / "benchmark37/energy_margin.py")):
        raise ValueError("Attitude warm start must use the pinned, completed energy baseline with matching locks")
    return m


def initialize_runner(runner, path):
    """Reuse installed runner's loader, explicitly discard old iteration/optimizer state."""
    if runner.alg.optimizer.state:
        raise RuntimeError("Warm start requires a newly constructed optimizer")
    runner.load(str(path), load_optimizer=False, map_location=runner.device)
    source_iteration = runner.current_learning_iteration
    runner.current_learning_iteration = 0
    if runner.alg.optimizer.state:
        raise RuntimeError("Unexpected optimizer state after weights-only initialization")
    return dict(source_iteration=source_iteration, new_iteration=0, optimizer_state_entries=0,
                normalization="loaded with policy state; continues updating during training",
                recurrent_state="fresh environment; no source rollout or LSTM hidden state resumed")


def reset_attitude(env, env_ids, seed):
    if not len(env_ids):
        return
    if not hasattr(env, "_attitude_rng"):
        env._attitude_rng = torch.Generator(device=env.device).manual_seed(int(seed) + 620024)
        env._release_tilt_rp = torch.zeros(env.num_envs, 2, device=env.device)
        env._attitude_resets = env._attitude_zero_resets = 0
        env._attitude_log_bucket = -1
    d = env._robot.data
    if torch.any(d.root_quat_w[env_ids, 1:].abs() > 1e-6):
        raise RuntimeError("Source attitude curriculum unexpectedly active; do not combine curricula")
    bound = limit_deg(env.common_step_counter)
    rp = torch.empty(len(env_ids), 2, device=env.device, dtype=d.root_quat_w.dtype)
    rp.uniform_(-math.radians(bound), math.radians(bound), generator=env._attitude_rng)
    upright = torch.rand(len(env_ids), device=env.device, generator=env._attitude_rng) < ZERO_FRACTION
    rp[upright] = 0.0
    env._robot.write_root_pose_to_sim(torch.cat((d.root_link_pos_w[env_ids], quaternion(rp)), -1), env_ids)
    env._release_tilt_rp[env_ids] = rp
    env._attitude_resets += len(env_ids)
    env._attitude_zero_resets += int(upright.sum().item())
    env._attitude_current_limit_deg = bound
    bucket = min(env.common_step_counter // (25 * STEPS_PER_ITERATION), 4)
    if bucket != env._attitude_log_bucket:
        env._attitude_log_bucket = bucket
        print(f"[ATTITUDE-CURRICULUM] control_step={env.common_step_counter} limit_deg={bound:.4f} "
              f"upright_probability={ZERO_FRACTION}", flush=True)
