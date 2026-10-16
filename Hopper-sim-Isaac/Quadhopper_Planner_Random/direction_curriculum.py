"""Pure task-command curriculum math for V20 directional training."""
from __future__ import annotations


def directional_command_values(
    iteration: float,
    duration_iterations: float,
    start_turn_deg: float,
    final_turn_deg: float,
    start_zero_probability: float,
    final_zero_probability: float,
    final_reverse_probability: float,
    reverse_halfwidth_deg: float,
) -> tuple[float, float, float]:
    """Return current maximum turn, exact-zero rate, and reversal rate.

    The reverse sampler emits headings in the signed 180-degree band whose
    nearest edge is ``180 - reverse_halfwidth_deg``. Its probability remains
    zero until that full band can be represented by the current maximum turn.
    """
    if duration_iterations <= 0.0:
        return final_turn_deg, final_zero_probability, final_reverse_probability
    blend = max(0.0, min(1.0, iteration / duration_iterations))
    current_turn = start_turn_deg + blend * (final_turn_deg - start_turn_deg)
    current_zero = start_zero_probability + blend * (
        final_zero_probability - start_zero_probability
    )
    reverse_required_turn = 180.0 - reverse_halfwidth_deg
    if final_reverse_probability <= 0.0 or current_turn < reverse_required_turn:
        current_reverse = 0.0
    elif final_turn_deg <= reverse_required_turn:
        current_reverse = final_reverse_probability
    else:
        reverse_blend = min(
            1.0,
            (current_turn - reverse_required_turn)
            / (final_turn_deg - reverse_required_turn),
        )
        current_reverse = final_reverse_probability * reverse_blend
    return current_turn, current_zero, current_reverse
