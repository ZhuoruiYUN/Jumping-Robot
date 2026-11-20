"""Separate stage 2c candidate: shift only vertical-energy tracking to 1.02 m."""

import torch


ENERGY_TARGET_MARGIN_M = 0.02


def energy_reward_delta(root_z, root_vz, target_z, gravity, scale, error_scale, step_dt):
    """Replace the source energy term's target; observation and 1.0 m success test stay fixed."""
    vertical_energy = gravity * torch.clamp(root_z, min=0.0) + 0.5 * torch.square(root_vz)
    old_target = gravity * torch.clamp(target_z, min=0.0)
    new_target = gravity * torch.clamp(target_z + ENERGY_TARGET_MARGIN_M, min=0.0)
    old = torch.exp(-torch.abs(vertical_energy - old_target) / error_scale)
    new = torch.exp(-torch.abs(vertical_energy - new_target) / error_scale)
    return (new - old) * scale * step_dt
