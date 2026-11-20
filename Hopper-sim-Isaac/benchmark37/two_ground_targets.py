"""Stage 4a: two simultaneously visible ground landing points, short hops.

The robot and 37 original observation slots remain unchanged. The added four
slots are planar commands, not simulator-only link or COM truth.
"""

import json
import math

import torch

from .artifacts import ROOT, sha
from .first_two_focus import FirstTwoTracker, contract as source_contract
from .foot_touchdown_reward import APEX_MIN_M, FOOT_LIMIT_M, TILT_LIMIT_DEG


SOURCE = ROOT / "outputs/benchmark37_stage3b_first_two_focus/train_seed42_20261027_073609_769821Z/model_199.pt"
SOURCE_SHA256 = "7308e2a519e097e51d46593685aaa9c0812670863cd598014e177abe6514bbc0"
VARIANT = "real_release_two_ground_short"
SERIES = "benchmark37_stage4a_two_ground_short"
ITERATIONS = 200
HOP_MAX_M = 0.40
SCALE_M = 0.80
ZERO_PROBABILITY = 0.10
RNG_OFFSET = 1800427


def contract():
    return dict(
        name="two_ground_landing_points_short_v1", variant=VARIANT,
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_contract("focus"),
        targets="P_t and P_(t+1) are ground XY foot-center landing points, both generated at release; "
                "root-link world 1.0m apex H remains separate",
        sampling="each edge has exact-zero probability 0.10; otherwise uniform radius 0..0.4m "
                 "and uniform full-circle heading; independent seed offset",
        observation="41D = preserved 37D (including five past 4-motor actions) + "
                    "[R(root)^T(P_t.xy-root_link.xy,0)]_xy/0.8m + "
                    "[R(root)^T(P_(t+1).xy-P_t.xy,0)]_xy/0.8m; "
                    "both points valid from release, no masked phase or zero-as-absence",
        advance="only after a confirmed post-drop flight touchdown qualified by foot<=10cm, "
                "apex>=1.0m, tilt<=15deg and alive; miss retains both points for retry; "
                "qualified P_t becomes completed, old P_(t+1) becomes P_t, append new ground point",
        reset="clear route index and sample both points; reset original action and LSTM histories only at episode reset",
        initialization="copy all matching 37D focus tensors; pad actor/critic LSTM input weights with "
                       "four zero columns; copy first 37 normalizer statistics, extra mean=0/std=1/var=1; "
                       "fresh optimizer and iteration 0",
        unchanged="SysID robot physics, motor timing, release 0.8..1.2m, immediate post-release control, "
                  "world 1.0m apex, original reward terms, PPO hyperparameters and 15s horizon",
        training_iterations=ITERATIONS,
        code_sha256=sha(ROOT / "benchmark37/two_ground_targets.py"),
    )


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or manifest.get("variant") != "real_release_first_two_focus"
            or manifest.get("mode") != "train" or manifest.get("status") != "complete"
            or manifest.get("smoke") or manifest.get("source_lock") != source_lock
            or manifest.get("physics_lock") != physics_lock
            or manifest.get("task_intervention") != source_contract("focus")
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256):
        raise ValueError("Two-ground training requires the pinned completed Stage 3b-D focus checkpoint")
    return manifest


def _ensure(env, seed):
    if hasattr(env, "_two_ground_rng"):
        return
    env._two_ground_rng = torch.Generator(device=env.device).manual_seed(int(seed) + RNG_OFFSET)
    env._two_ground_pt_xy_w = torch.zeros(env.num_envs, 2, device=env.device)
    env._two_ground_pt1_xy_w = torch.zeros_like(env._two_ground_pt_xy_w)
    env._two_ground_route_index = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._two_ground_total_advances = torch.zeros((), device=env.device, dtype=torch.long)
    env._two_ground_total_resets = torch.zeros((), device=env.device, dtype=torch.long)


def _offset(env, n):
    radius = HOP_MAX_M * torch.rand(n, device=env.device, generator=env._two_ground_rng)
    angle = 2 * math.pi * torch.rand(n, device=env.device, generator=env._two_ground_rng)
    zero = torch.rand(n, device=env.device, generator=env._two_ground_rng) < ZERO_PROBABILITY
    radius = torch.where(zero, 0.0, radius)
    return radius[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def _set_active_goal(env, ids):
    env._desired_pos_w[ids, :2] = env._two_ground_pt_xy_w[ids]
    root_xy = env._robot.data.root_link_pos_w[ids, :2]
    displacement = env._two_ground_pt_xy_w[ids] - root_xy
    env._prev_distance_to_goal_xy[ids] = torch.linalg.vector_norm(displacement, dim=-1)
    env._point_target_displacement_w[ids] = displacement
    env._point_target_bucket[ids] = -3


def reset_pair(env, ids, seed):
    if len(ids) == 0:
        return
    _ensure(env, seed)
    start_xy = env._robot.data.root_link_pos_w[ids, :2]
    env._two_ground_pt_xy_w[ids] = start_xy + _offset(env, len(ids))
    env._two_ground_pt1_xy_w[ids] = env._two_ground_pt_xy_w[ids] + _offset(env, len(ids))
    env._two_ground_route_index[ids] = 0
    env._two_ground_total_resets += len(ids)
    _set_active_goal(env, ids)


def advance_on_hit(env, hit):
    ids = torch.nonzero(hit, as_tuple=False).flatten()
    if len(ids) == 0:
        return
    env._two_ground_pt_xy_w[ids] = env._two_ground_pt1_xy_w[ids].clone()
    env._two_ground_pt1_xy_w[ids] = env._two_ground_pt_xy_w[ids] + _offset(env, len(ids))
    env._two_ground_route_index[ids] += 1
    env._two_ground_total_advances += len(ids)
    _set_active_goal(env, ids)


def _rotate_world_to_body(q_wxyz, vectors_w):
    q_vec = -q_wxyz[:, 1:]
    t = 2 * torch.cross(q_vec, vectors_w, dim=-1)
    return vectors_w + q_wxyz[:, :1] * t + torch.cross(q_vec, t, dim=-1)


def extra_observation(env):
    root_xy = env._robot.data.root_link_pos_w[:, :2]
    current_w = torch.nn.functional.pad(env._two_ground_pt_xy_w - root_xy, (0, 1))
    lookahead_w = torch.nn.functional.pad(env._two_ground_pt1_xy_w - env._two_ground_pt_xy_w, (0, 1))
    q = env._robot.data.root_quat_w
    current_b = _rotate_world_to_body(q, current_w)[:, :2] / SCALE_M
    lookahead_b = _rotate_world_to_body(q, lookahead_w)[:, :2] / SCALE_M
    return torch.cat((current_b, lookahead_b), dim=-1)


class TwoGroundTracker(FirstTwoTracker):
    """Expose the per-env qualified hit mask without changing frozen reward code."""

    def __init__(self, num_envs, device):
        super().__init__(num_envs, device, "focus")
        self.last_qualified = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        super().reset(env_ids)
        self.last_qualified[env_ids] = False

    def step(self, q, z, vz, foot_xy, target_xy, tilt_deg, died):
        airborne = (self.phase == self.CANDIDATE) | (self.phase == self.FLIGHT)
        peak = torch.where(airborne, torch.maximum(self.peak, z), self.peak)
        apex_seen = self.apex_seen | (airborne & (vz <= 0))
        valid = ((self.phase == self.FLIGHT) & (q > 0.002) & (self.air_steps >= 5)
                 & apex_seen & ~died)
        foot_error = torch.linalg.vector_norm(foot_xy - target_xy, dim=-1)
        self.last_qualified = (valid & (peak >= APEX_MIN_M) & (tilt_deg <= TILT_LIMIT_DEG)
                               & (foot_error <= FOOT_LIMIT_M))
        reward = super().step(q, z, vz, foot_xy, target_xy, tilt_deg, died)
        if not torch.equal(self.last_qualified, reward > 0):
            raise RuntimeError("Two-ground route gate diverged from frozen foot reward")
        return reward


def initialize_runner(runner, path):
    """Pad the two LSTM input layers and normalizers; reset optimizer and iteration."""
    if runner.alg.optimizer.state:
        raise RuntimeError("Two-ground warm start requires a fresh optimizer")
    saved = torch.load(path, map_location=runner.device, weights_only=False)["model_state_dict"]
    target = runner.alg.policy.state_dict()
    if set(saved) != set(target):
        raise RuntimeError("Policy state keys changed; refuse automatic transfer")
    padded = []
    for key, dst in target.items():
        src = saved[key]
        if src.shape == dst.shape:
            target[key] = src.clone()
        elif key in ("memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0"):
            if src.shape != (1024, 37) or dst.shape != (1024, 41):
                raise RuntimeError(f"Unexpected LSTM input shape for {key}")
            dst.zero_()
            dst[:, :37] = src
            padded.append(key)
        elif key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer.")) and key.split(".")[-1] in ("_mean", "_var", "_std"):
            if src.shape != (1, 37) or dst.shape != (1, 41):
                raise RuntimeError(f"Unexpected normalizer shape for {key}")
            dst[:, :37] = src
            padded.append(key)
        else:
            raise RuntimeError(f"Unreviewed parameter shape change: {key} {tuple(src.shape)} -> {tuple(dst.shape)}")
    runner.alg.policy.load_state_dict(target, strict=True)
    runner.current_learning_iteration = 0
    if runner.alg.optimizer.state:
        raise RuntimeError("Optimizer state was populated during transfer")
    return dict(source_checkpoint=str(path), source_sha256=sha(path), new_iteration=0,
                padded_keys=padded, new_obs_columns_initially_zero=True,
                normalizer_extra="mean=0, variance=1, std=1; source count retained",
                optimizer_state_entries=0)


def audit_pair(env):
    current = env._two_ground_pt_xy_w
    next_point = env._two_ground_pt1_xy_w
    if not torch.allclose(env._desired_pos_w[:, :2], current, atol=1e-5):
        raise RuntimeError("Active apex XY and current ground point disagree")
    if not torch.allclose(env._desired_pos_w[:, 2], torch.ones_like(env._desired_pos_w[:, 2]), atol=1e-5):
        raise RuntimeError("World apex target must remain 1.0m")
    radius = torch.linalg.vector_norm(next_point - current, dim=-1)
    if torch.any(radius > HOP_MAX_M + 1e-5):
        raise RuntimeError("Second ground point escaped short-hop range")
    return dict(obs_dim=41, active_ground_point="P_t", next_ground_point="P_(t+1)",
                next_edge_radius_max_m=float(radius.max()), route_advances=int(env._two_ground_total_advances),
                route_resets=int(env._two_ground_total_resets), apex_world_m=1.0)
