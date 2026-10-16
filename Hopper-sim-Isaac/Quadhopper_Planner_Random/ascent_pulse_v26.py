"""Simulation-only early-ascent common-mode pulse for a frozen-policy probe."""
import torch


PULSE_DURATION_S = 0.20


def apply_ascent_pulse(target_u: torch.Tensor, *, spring_q: torch.Tensor,
                       cycle_active: torch.Tensor, initial_drop: torch.Tensor,
                       root_vz: torch.Tensor, cycle_time_s: torch.Tensor,
                       bias_pwm: float) -> torch.Tensor:
    """Add a bounded symmetric motor increment during the first 0.20 s of flight."""
    if bias_pwm == 0:
        return target_u
    active = (
        (spring_q <= 0.002) & cycle_active & ~initial_drop
        & (root_vz > 0.10)
        & (cycle_time_s > 0.0) & (cycle_time_s <= PULSE_DURATION_S)
    )
    return (target_u + active.to(target_u.dtype).unsqueeze(-1) * (bias_pwm / 1000.0)).clamp(0.0, 1.0)
