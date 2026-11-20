"""Stage 3b-B: isolated edge-radius sampling ablation on the frozen 37-D task."""

import torch

from .artifacts import ROOT, sha
from .point_target_short import (SOURCE, SOURCE_SHA256, ITERATIONS, MAX_RADIUS_M,
                                 PointCollector, audit_target, contract as short_contract,
                                 reset_target, validate_source)


VARIANT = "real_release_point_edge"
SERIES = "benchmark37_stage3b_point_edge"
EDGE_RADIUS_MIN_M = 0.35
EDGE_FRACTION_OF_NONSTATIONARY = 0.50


def contract():
    return dict(
        name="first_hop_short_edge_sampling_v1",
        source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
        base_point_contract=short_contract(),
        intervention="training reset radius sampler only; same 37D observations, physics, rewards, PPO and evaluation",
        training_radius="10% stationary; otherwise original uniform 0..0.4m radius with 50% of nonstationary "
                        "samples replaced by uniform 0.35..0.4m; original uniform heading unchanged",
        evaluation="same point_short15 17 buckets and metrics as Stage 3b-A",
        training_iterations=ITERATIONS,
        initialization="all 37D weights and normalization from pinned Stage 3a; fresh optimizer, iteration 0; "
                       "same 200-update budget as Stage 3b-A",
        code_sha256=sha(ROOT / "benchmark37/point_target_edge.py"),
    )


def reset_target_edge(env, env_ids, seed, evaluation):
    """Run the frozen sampler, then change radius only for selected train resets."""
    reset_target(env, env_ids, seed, evaluation)
    if not len(env_ids):
        return
    if not hasattr(env, "_point_edge_selected"):
        env._point_edge_selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._point_edge_selected[env_ids] = False
    if evaluation:
        return
    disp = env._point_target_displacement_w[env_ids]
    old_radius = torch.linalg.vector_norm(disp, dim=-1)
    selected = ((old_radius > 0)
                & (torch.rand(len(env_ids), device=env.device, generator=env._point_target_rng)
                   < EDGE_FRACTION_OF_NONSTATIONARY))
    new_radius = EDGE_RADIUS_MIN_M + (MAX_RADIUS_M - EDGE_RADIUS_MIN_M) * torch.rand(
        len(env_ids), device=env.device, generator=env._point_target_rng)
    scale = torch.where(selected, new_radius / old_radius.clamp_min(1e-9), 1.0)
    new_disp = disp * scale[:, None]
    root_xy = env._robot.data.root_link_pos_w[env_ids, :2]
    env._desired_pos_w[env_ids, :2] = root_xy + new_disp
    env._prev_distance_to_goal_xy[env_ids] = torch.linalg.vector_norm(new_disp, dim=-1)
    env._point_target_displacement_w[env_ids] = new_disp
    env._point_edge_selected[env_ids] = selected


def audit_edge(env, evaluation):
    audit = audit_target(env, evaluation)
    selected = env._point_edge_selected
    radius = torch.linalg.vector_norm(env._point_target_displacement_w, dim=-1)
    if torch.any(selected & ((radius < EDGE_RADIUS_MIN_M - 1e-5) | (radius > MAX_RADIUS_M + 1e-5))):
        raise RuntimeError("Edge-selected training target escaped the 0.35..0.4m ring")
    if evaluation and torch.any(selected):
        raise RuntimeError("Evaluation must use the unchanged fixed-bucket target protocol")
    audit.update(edge_selected_initial_resets=int(selected.sum()), edge_radius_min_m=EDGE_RADIUS_MIN_M,
                 edge_fraction_of_nonstationary=EDGE_FRACTION_OF_NONSTATIONARY)
    return audit
