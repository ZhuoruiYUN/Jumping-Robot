"""Stage4b: change only the two ground-edge radius sampler, 0.4m -> 0.8m."""

import json
import math

import torch

from .artifacts import ROOT, sha
from .calibration_20261028_v2 import REVISION, physics_contract
from .two_ground_targets import (_ensure, _set_active_goal, ZERO_PROBABILITY,
                                 contract as source_contract)

SOURCE = ROOT / "outputs/benchmark37_stage4a_calibrated_20261028_v2/train_seed42_20261028_085657_389015Z/model_199.pt"
SOURCE_SHA256 = "db9e64d9824b5050272458104bd2fd15aaa6f0cc79e2c22c0b3f0fcd9578ae2b"
VARIANT = "real_release_two_ground_range08"
SERIES = "benchmark37_stage4b_range08"
ITERATIONS = 200
RAMP_UPDATES = 120
STEPS_PER_UPDATE = 256
INITIAL_MAX_M = .4
FINAL_MAX_M = .8


def contract():
    return dict(name="two_ground_range08_curriculum_v1", variant=VARIANT,
                source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
                source_task_contract=source_contract(), physics_revision=REVISION,
                intervention="Only change both ground-edge radius samplers: max 0.4m -> 0.8m linearly over first 120 PPO updates, then hold 0.8m for 80 updates",
                sampling="Each edge retains 10% exact zero; otherwise radius uniform 0..current_max and heading uniform full-circle. Misses retain targets. Already queued targets stay fixed as curriculum grows",
                timing="Curriculum uses common policy-step counter / 256, not per-env episode age; policy period remains 10ms",
                observation="Same 41 slots, same root-Body transformations, same /0.8m scale and five 4-motor action history; no clipping or new state",
                evaluation="two_ground15: 16 environments x 17 15s episodes at full 0..0.8m; report both short/long edges and combinations, true second-point attempts separately",
                smoke="2-update diagnostics sample full 0..0.8m to exercise end-range wiring; never continue formal training from smoke weights",
                unchanged="v2 mass/spring/air/motor physics, release 0.8..1.2m, 1m apex, 10cm foot gate, 15deg tilt, rewards, target progression, termination, PPO, reset and observation semantics",
                initialization="All 41D actor/critic and normalizer tensors copied exactly from pinned v2 short candidate; fresh optimizer and iteration zero",
                training_iterations=ITERATIONS,
                code_sha256=sha(ROOT / "benchmark37/range08_curriculum.py"))


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    selected = json.loads((ROOT / "benchmark37/calibrated_v2_short_candidate.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or selected.get("checkpoint_sha256") != SOURCE_SHA256
            or selected.get("status") != "development_seed7001_passed_short_hop_candidate"
            or manifest.get("mode") != "train" or manifest.get("status") != "complete"
            or manifest.get("smoke") or manifest.get("variant") != "real_release_two_ground_short"
            or manifest.get("source_lock") != source_lock or manifest.get("physics_lock") != physics_lock
            or manifest.get("physics_revision") != REVISION
            or manifest.get("calibration_lock") != physics_contract()
            or manifest.get("task_intervention") != source_contract()
            or manifest.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256):
        raise ValueError("Range08 training requires the pinned completed calibrated-v2 short candidate")
    return manifest


def current_max(env):
    if not env._range08_training:
        return FINAL_MAX_M
    progress = min(max(env.common_step_counter / (RAMP_UPDATES * STEPS_PER_UPDATE), 0.), 1.)
    return INITIAL_MAX_M + (FINAL_MAX_M - INITIAL_MAX_M) * progress


def _offset(env, count):
    maximum = current_max(env)
    radius = maximum * torch.rand(count, device=env.device, generator=env._two_ground_rng)
    angle = 2 * math.pi * torch.rand(count, device=env.device, generator=env._two_ground_rng)
    zero = torch.rand(count, device=env.device, generator=env._two_ground_rng) < ZERO_PROBABILITY
    radius = torch.where(zero, 0., radius)
    previous = getattr(env, "_range08_max_sampled_edge", radius.new_zeros(()))
    env._range08_max_sampled_edge = torch.maximum(previous, radius.max())
    return radius[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def reset_pair(env, ids, seed):
    if len(ids) == 0:
        return
    _ensure(env, seed)
    env._two_ground_pt_xy_w[ids] = env._robot.data.root_link_pos_w[ids, :2] + _offset(env, len(ids))
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


def audit_pair(env):
    radius = torch.linalg.vector_norm(env._two_ground_pt1_xy_w - env._two_ground_pt_xy_w, dim=-1)
    if (torch.any(radius > current_max(env) + 1e-5)
            or not torch.allclose(env._desired_pos_w[:, :2], env._two_ground_pt_xy_w, atol=1e-5)
            or not torch.allclose(env._desired_pos_w[:, 2], torch.ones_like(env._desired_pos_w[:, 2]), atol=1e-5)):
        raise RuntimeError("Range08 target queue violates radius, active point or world-apex contract")
    return dict(obs_dim=41, active_ground_point="P_t", next_ground_point="P_(t+1)",
                current_sampling_max_m=current_max(env), configured_final_max_m=FINAL_MAX_M,
                next_edge_radius_max_m=float(radius.max()),
                max_edge_sampled_m=float(getattr(env, "_range08_max_sampled_edge", 0.)),
                curriculum_update=env.common_step_counter / STEPS_PER_UPDATE,
                ramp_updates=RAMP_UPDATES, training_curriculum=env._range08_training,
                route_advances=int(env._two_ground_total_advances),
                route_resets=int(env._two_ground_total_resets), apex_world_m=1.)
