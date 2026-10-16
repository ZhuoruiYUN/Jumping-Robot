"""Pure reward helpers for the stationary post-touchdown recovery window.

The policy keeps its four direct-motor interface.  These helpers deliberately
separate collective command from the zero-mean differential so that height
authority is never penalized as a side effect of stabilizing recovery.
"""
import torch


def recovery_window_steps(window_s: float, step_dt: float) -> int:
    """Return the inclusive number of 100 Hz samples after touchdown."""
    if window_s <= 0.0:
        return 0
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")
    return max(1, round(window_s / step_dt))


def advance_recovery_window(
    remaining: torch.Tensor, touchdown: torch.Tensor, window_steps: int
) -> torch.Tensor:
    """Start a window on touchdown, otherwise advance an existing window."""
    if window_steps < 0:
        raise ValueError("window_steps must be non-negative")
    advanced = (remaining - 1).clamp_min(0)
    return torch.where(touchdown, torch.full_like(remaining, window_steps), advanced)


def differential_recovery_costs(
    target_u: torch.Tensor,
    previous_target_u: torch.Tensor,
    spread_deadband: float,
    delta_deadband: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return spread/delta costs and measurements for four motor targets.

    ``target_u`` is the delayed command actually entering the motor model.  A
    common-mode change cancels exactly from both costs, so no collective-height
    penalty is introduced.
    """
    if target_u.shape != previous_target_u.shape or target_u.shape[-1] != 4:
        raise ValueError("expected matching (..., 4) motor command tensors")
    if not (0.0 <= spread_deadband <= 1.0 and 0.0 <= delta_deadband <= 1.0):
        raise ValueError("differential deadbands must lie in [0, 1]")
    residual = target_u - target_u.mean(dim=-1, keepdim=True)
    previous_residual = previous_target_u - previous_target_u.mean(dim=-1, keepdim=True)
    spread = residual.max(dim=-1).values - residual.min(dim=-1).values
    delta = (residual - previous_residual).abs().max(dim=-1).values
    spread_cost = ((spread - spread_deadband).clamp_min(0.0) / 0.20).square()
    delta_cost = ((delta - delta_deadband).clamp_min(0.0) / 0.15).square()
    return spread_cost, delta_cost, spread, delta
