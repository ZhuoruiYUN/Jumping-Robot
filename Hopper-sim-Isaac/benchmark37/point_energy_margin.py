"""Stage 3b-C: edge-target training with one extra 2 cm energy-reward margin."""

import torch

from .artifacts import ROOT, sha
from .energy_margin import ENERGY_TARGET_MARGIN_M
from .point_target_edge import contract as edge_contract


VARIANT = "real_release_point_edge_energy4cm"
SERIES = "benchmark37_stage3b_point_energy4cm"
EXTRA_MARGIN_M = 0.02
TOTAL_MARGIN_M = ENERGY_TARGET_MARGIN_M + EXTRA_MARGIN_M


def reward_contract():
    return dict(
        term="energy_tracking", base_target_world_z_m=1.0,
        target_energy_height_shift_m=TOTAL_MARGIN_M,
        inherited_margin_m=ENERGY_TARGET_MARGIN_M, extra_margin_m=EXTRA_MARGIN_M,
        inherited_code_sha256=sha(ROOT / "benchmark37/energy_margin.py"),
        code_sha256=sha(ROOT / "benchmark37/point_energy_margin.py"),
        note="Only the vertical-energy reward target moves from 1.02 to 1.04m; "
             "actor target, apex success threshold and all other terms remain unchanged",
    )


def contract():
    return dict(
        name="first_hop_short_edge_energy4cm_v1",
        source_task_contract=edge_contract(),
        intervention="only shift training energy_tracking reward target from world 1.02m to 1.04m; "
                     "37D observation, edge target sampler, other rewards, physics, PPO and eval unchanged",
        old_energy_target_margin_m=ENERGY_TARGET_MARGIN_M,
        new_energy_target_margin_m=TOTAL_MARGIN_M,
        reward_intervention=reward_contract(),
        actor_target_world_z_m=1.0,
        success_apex_world_z_m=1.0,
        initialization="same pinned Stage 3a 37D weights and normalizers as Stage 3b-B; "
                       "fresh optimizer, iteration 0, 768 environments and 200 updates",
        code_sha256=sha(ROOT / "benchmark37/point_energy_margin.py"),
    )


def extra_energy_reward_delta(root_z, root_vz, target_z, gravity, scale, error_scale, step_dt):
    """Add after the existing source->1.02m delta; net reward target becomes 1.04m."""
    vertical_energy = gravity * torch.clamp(root_z, min=0.0) + 0.5 * torch.square(root_vz)
    old_target = gravity * torch.clamp(target_z + ENERGY_TARGET_MARGIN_M, min=0.0)
    new_target = gravity * torch.clamp(target_z + TOTAL_MARGIN_M, min=0.0)
    old = torch.exp(-torch.abs(vertical_energy - old_target) / error_scale)
    new = torch.exp(-torch.abs(vertical_energy - new_target) / error_scale)
    return (new - old) * scale * step_dt
