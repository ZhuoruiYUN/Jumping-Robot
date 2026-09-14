"""Task-selectable actuator helpers for deployment-parity simulation."""

from __future__ import annotations

import torch


def corrected_delay_history_index(delay_steps: int, history_length: int) -> int:
    """Return the deque index where zero delay means the newest command."""
    if not isinstance(delay_steps, int):
        raise TypeError("delay_steps must be an integer")
    if delay_steps < 0 or delay_steps >= history_length:
        raise ValueError(
            f"delay_steps={delay_steps} is outside history length {history_length}"
        )
    return -1 - delay_steps


def shape_deployment_motor_commands(
    target_u: torch.Tensor,
    *,
    is_contact: torch.Tensor,
    root_z: torch.Tensor,
    root_vz: torch.Tensor,
    target_z: torch.Tensor,
    max_pwm: float,
    max_pwm_spread: float,
    max_airborne_yaw_diagonal_pwm_diff: float,
    max_grounded_yaw_diagonal_pwm_diff: float,
    height_brake_start_m: float,
    height_brake_full_m: float,
    height_brake_vz_min: float,
    height_brake_pwm_mean_soft: float,
    height_brake_pwm_mean_hard: float,
    projection_iterations: int = 2,
    quantize_pwm: bool = True,
) -> torch.Tensor:
    """Apply the deployed command envelope and return normalized commands.

    Two fixed alternating projections preserve the existing deployed final
    spread/yaw behavior while exposing it as one actuator-contract operation.
    """
    if target_u.ndim != 2 or target_u.shape[1] != 4:
        raise ValueError(f"Expected motor commands shaped (N, 4), got {target_u.shape}")
    if projection_iterations <= 0:
        raise ValueError("projection_iterations must be positive")

    pwm = target_u.clamp(0.0, 1.0) * max_pwm
    target_z = torch.as_tensor(target_z, device=pwm.device, dtype=pwm.dtype)
    if target_z.ndim == 0:
        target_z = target_z.expand(len(pwm))

    overshoot_span = max(height_brake_full_m - height_brake_start_m, 1.0e-6)
    brake = ((root_z - target_z - height_brake_start_m) / overshoot_span).clamp(0.0, 1.0)
    max_mean = (
        (1.0 - brake) * height_brake_pwm_mean_soft
        + brake * height_brake_pwm_mean_hard
    )
    brake_active = (root_z > target_z + height_brake_start_m) & (
        root_vz > height_brake_vz_min
    )
    collective_excess = (pwm.mean(dim=1) - max_mean).clamp_min(0.0)
    collective_excess = torch.where(
        brake_active, collective_excess, torch.zeros_like(collective_excess)
    )
    pwm = (pwm - collective_excess[:, None]).clamp(0.0, max_pwm)

    contact = is_contact.to(dtype=torch.bool)
    yaw_limit = torch.where(
        contact,
        torch.full_like(root_z, max_grounded_yaw_diagonal_pwm_diff),
        torch.full_like(root_z, max_airborne_yaw_diagonal_pwm_diff),
    )
    yaw_signs = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0], device=pwm.device, dtype=pwm.dtype
    )
    half_spread = 0.5 * max_pwm_spread
    for _ in range(projection_iterations):
        center = pwm.mean(dim=1, keepdim=True)
        pwm = torch.maximum(
            torch.minimum(pwm, center + half_spread), center - half_spread
        ).clamp(0.0, max_pwm)
        diag_13 = 0.5 * (pwm[:, 0] + pwm[:, 2])
        diag_24 = 0.5 * (pwm[:, 1] + pwm[:, 3])
        diff = diag_13 - diag_24
        correction = 0.5 * torch.sign(diff) * (torch.abs(diff) - yaw_limit).clamp_min(0.0)
        pwm = (pwm + correction[:, None] * yaw_signs).clamp(0.0, max_pwm)

    if quantize_pwm:
        pwm = torch.floor(pwm)
    return pwm / max_pwm
