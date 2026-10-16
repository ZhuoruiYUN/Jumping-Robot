"""Step-to-step Raibert foothold guidance with time-scaled Hermite arches.

The Raibert layer selects a bounded next foothold.  It does not prescribe a
time-indexed flight trajectory.  The Hermite arch is only an untimed spatial
guide, but its tangents are scaled from measured/desired horizontal velocity
and ballistic flight time instead of arbitrary fractions of hop distance.

This is a goal-anchored variant of the classic Raibert rule.  The commanded
landing point supplies the nominal foothold, so only the measured velocity
error is added.  Consequently, a hop already matching its desired step speed
is not biased away from the commanded circle centre.
"""

from __future__ import annotations

import torch


RAIBERT_HERMITE_PROTOCOL = "physical_1m_two_circle_history_foot_raibert_hermite_v1"


def _clip_vector_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    if maximum <= 0.0:
        return torch.zeros_like(vector)
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = torch.clamp(maximum / norm.clamp_min(1e-8), max=1.0)
    return vector * scale


def _hermite_segment(
    start: torch.Tensor,
    end: torch.Tensor,
    start_tangent: torch.Tensor,
    end_tangent: torch.Tensor,
    samples: torch.Tensor,
) -> torch.Tensor:
    u = samples
    return (
        (2 * u**3 - 3 * u**2 + 1) * start[:, None]
        + (u**3 - 2 * u**2 + u) * start_tangent[:, None]
        + (-2 * u**3 + 3 * u**2) * end[:, None]
        + (u**3 - u**2) * end_tangent[:, None]
    )


def raibert_hermite_curve(
    start: torch.Tensor,
    commanded_landing: torch.Tensor,
    following: torch.Tensor,
    apex_height: torch.Tensor,
    measured_velocity_xy: torch.Tensor,
    *,
    next_apex_height: torch.Tensor | None = None,
    nodes: int = 33,
    gravity_m_s2: float = 9.81,
    stance_time_s: float = 0.045,
    velocity_gain_s: float = 0.08,
    max_foothold_shift_m: float = 0.04,
    preview_blend: float = 0.5,
    max_horizontal_speed_m_s: float = 2.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build a Raibert-conditioned, C1-in-time two-segment Hermite arch.

    ``commanded_landing`` remains the scoring target.  The returned spatial
    curve ends at a bounded Raibert guidance foothold, allowing the policy to
    react to launch-speed error without silently redefining task success.

    The endpoint velocity previews the following hop.  A 180-degree reversal
    therefore asks for a low touchdown horizontal velocity, while a straight
    continuation preserves momentum and an in-place next hop brakes it.
    """
    if nodes < 9 or nodes % 2 == 0:
        raise ValueError("nodes must be odd and >= 9")
    if gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive")
    if stance_time_s < 0.0 or velocity_gain_s < 0.0:
        raise ValueError("Raibert timing/gain values must be non-negative")
    if not 0.0 <= preview_blend <= 1.0:
        raise ValueError("preview_blend must be in [0, 1]")
    if max_horizontal_speed_m_s <= 0.0:
        raise ValueError("max_horizontal_speed_m_s must be positive")

    apex = apex_height.reshape(-1, 1)
    ascent_height = (apex - start[:, 2:3]).clamp_min(0.0)
    descent_height = (apex - commanded_landing[:, 2:3]).clamp_min(0.0)
    ascent_time = torch.sqrt(2.0 * ascent_height / gravity_m_s2).clamp_min(1e-3)
    descent_time = torch.sqrt(2.0 * descent_height / gravity_m_s2).clamp_min(1e-3)
    flight_time = ascent_time + descent_time

    commanded_step_xy = commanded_landing[:, :2] - start[:, :2]
    desired_takeoff_velocity_xy = commanded_step_xy / flight_time
    desired_takeoff_velocity_xy = _clip_vector_norm(
        desired_takeoff_velocity_xy, max_horizontal_speed_m_s
    )
    measured_velocity_xy = _clip_vector_norm(
        measured_velocity_xy, max_horizontal_speed_m_s
    )

    # Goal-anchored Raibert residual.  The classical half-stance feed-forward
    # term is applied to velocity *error*, because the commanded circle centre
    # already supplies the nominal foothold for the desired velocity.
    velocity_error_xy = measured_velocity_xy - desired_takeoff_velocity_xy
    raw_shift_xy = (0.5 * stance_time_s + velocity_gain_s) * velocity_error_xy
    foothold_shift_xy = _clip_vector_norm(raw_shift_xy, max_foothold_shift_m)
    guidance_foothold = commanded_landing.clone()
    guidance_foothold[:, :2] += foothold_shift_xy

    next_step_xy = following[:, :2] - commanded_landing[:, :2]
    next_apex = apex if next_apex_height is None else next_apex_height.reshape(-1, 1)
    next_ascent_time = torch.sqrt(
        2.0 * (next_apex - commanded_landing[:, 2:3]).clamp_min(0.0) / gravity_m_s2
    ).clamp_min(1e-3)
    next_descent_time = torch.sqrt(
        2.0 * (next_apex - following[:, 2:3]).clamp_min(0.0) / gravity_m_s2
    ).clamp_min(1e-3)
    next_flight_time = next_ascent_time + next_descent_time
    desired_next_velocity_xy = _clip_vector_norm(
        next_step_xy / next_flight_time, max_horizontal_speed_m_s
    )
    touchdown_velocity_xy = (
        (1.0 - preview_blend) * desired_takeoff_velocity_xy
        + preview_blend * desired_next_velocity_xy
    )

    # A ballistic vertical launch reaches the specified apex.  Horizontal
    # apex locations are predictions from the measured launch state on ascent
    # and the previewed touchdown state on descent; averaging them avoids a
    # discontinuous position target when those predictions disagree.
    launch_velocity_z = torch.sqrt(2.0 * gravity_m_s2 * ascent_height)
    peak_from_start_xy = start[:, :2] + measured_velocity_xy * ascent_time
    peak_from_end_xy = guidance_foothold[:, :2] - touchdown_velocity_xy * descent_time
    peak = start.clone()
    peak[:, :2] = 0.5 * (peak_from_start_xy + peak_from_end_xy)
    peak[:, 2:3] = apex

    launch_velocity = torch.cat((measured_velocity_xy, launch_velocity_z), dim=-1)
    apex_velocity = torch.cat(
        (desired_takeoff_velocity_xy, torch.zeros_like(launch_velocity_z)), dim=-1
    )
    touchdown_velocity = torch.cat(
        (touchdown_velocity_xy, -gravity_m_s2 * descent_time), dim=-1
    )

    # Hermite parameters are local segment fractions, hence dp/du = T dp/dt.
    ascent_start_tangent = launch_velocity * ascent_time
    ascent_peak_tangent = apex_velocity * ascent_time
    descent_peak_tangent = apex_velocity * descent_time
    descent_end_tangent = touchdown_velocity * descent_time
    u = torch.linspace(0.0, 1.0, nodes // 2 + 1, device=start.device,
                       dtype=start.dtype)[None, :, None]
    curve = torch.cat(
        (
            _hermite_segment(start, peak, ascent_start_tangent, ascent_peak_tangent, u),
            _hermite_segment(peak, guidance_foothold, descent_peak_tangent,
                             descent_end_tangent, u)[:, 1:],
        ),
        dim=1,
    )
    diagnostics = {
        "guidance_foothold": guidance_foothold,
        "foothold_shift_xy": foothold_shift_xy,
        "velocity_error_xy": velocity_error_xy,
        "desired_takeoff_velocity_xy": desired_takeoff_velocity_xy,
        "desired_next_velocity_xy": desired_next_velocity_xy,
        "touchdown_velocity_xy": touchdown_velocity_xy,
        "ascent_time_s": ascent_time[:, 0],
        "descent_time_s": descent_time[:, 0],
        "flight_time_s": flight_time[:, 0],
        "next_flight_time_s": next_flight_time[:, 0],
    }
    return curve, diagnostics
