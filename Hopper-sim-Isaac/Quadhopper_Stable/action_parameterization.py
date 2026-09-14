"""Shared direct-motor and collective/residual action coordinates."""
from __future__ import annotations

import torch

# Columns are roll, pitch, yaw residuals.  Every column has zero mean and the
# columns are orthogonal, so the transform is invertible away from saturation.
COLLECTIVE_RESIDUAL_BASIS = (
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0),
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, 1.0),
)


def collective_residual_to_motor_u(actions: torch.Tensor) -> torch.Tensor:
    """Decode [collective, roll, pitch, yaw] to four motors in [0, 1].

    The residual is uniformly scaled when an actuator would saturate.  This
    preserves its zero mean and therefore preserves collective thrust exactly.
    """
    actions = actions.clamp(-1.0, 1.0)
    basis = torch.as_tensor(COLLECTIVE_RESIDUAL_BASIS, device=actions.device, dtype=actions.dtype)
    collective = (actions[:, :1] + 1.0) * 0.5
    delta = 0.5 * actions[:, 1:] @ basis.T
    positive = delta.clamp_min(0.0).amax(dim=1, keepdim=True)
    negative = (-delta).clamp_min(0.0).amax(dim=1, keepdim=True)
    alpha_pos = torch.where(positive > 1.0e-8, (1.0 - collective) / positive, torch.ones_like(positive))
    alpha_neg = torch.where(negative > 1.0e-8, collective / negative, torch.ones_like(negative))
    alpha = torch.minimum(torch.ones_like(collective), torch.minimum(alpha_pos, alpha_neg))
    return (collective + alpha * delta).clamp(0.0, 1.0)


def direct_motor_to_collective_residual_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Linear map from legacy motor-domain actions to new policy actions."""
    basis = torch.tensor(COLLECTIVE_RESIDUAL_BASIS, dtype=dtype)
    return torch.cat((torch.full((1, 4), 0.25, dtype=dtype), basis.T * 0.25), dim=0)
