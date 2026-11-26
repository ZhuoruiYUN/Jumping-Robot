"""Bounded cruise-cost allowance for existing forward landing guidance."""

import torch

from .first_landing_cross_feedback import _smooth_ramp


VERSION = "confirmed_first_continuation_cruise_v2"


def validate(config):
    if config != {"version": VERSION}:
        raise ValueError("Invalid confirmed-continuation cruise configuration")


def forward_allowance(landing, velocity, edge, elapsed, remaining, coast, requested, eligible, radius):
    """Only excuse a forward request when an optimistic endpoint stays short."""
    axis = torch.nn.functional.normalize(edge, dim=-1)
    speed = (velocity[:, :2] * axis).sum(-1)
    # Already-attainable horizontal velocity plus generous additional travel.
    # This conservative screen is separate from the native reference magnitude.
    time = remaining.clamp_min(0.) + .08
    optimistic = landing["projected_along_m"] + .08 * speed.clamp_min(0.) + .275 * time.square()
    deficit = _smooth_ramp(-optimistic, radius, radius + .04)
    cross_margin = 1. - _smooth_ramp(landing["projected_cross_m"].abs(), radius, radius + .03)
    phase = _smooth_ramp(elapsed, .30, .36) * (1. - _smooth_ramp(elapsed, .47, .54))
    # Leave room for ascent work from later actor commands after the forecast.
    retain_height = (coast >= 1.005) & (requested >= 1.025) & (requested <= 1.08)
    active = eligible & retain_height & (speed > 0.) & (landing["undertravel_gate"] > 0.)
    reference = landing["reference_along_accel_m_s2"].clamp_min(0.)
    return torch.where(active, reference * deficit * cross_margin * phase, 0.)


def cruise_squared_relief(acceleration_xy, edge, allowance):
    """Restore only the squared forward effort covered by the allowance."""
    axis = torch.nn.functional.normalize(edge, dim=-1)
    forward_effort = (acceleration_xy * axis).sum(-1).clamp_min(0.)
    excused = torch.minimum(forward_effort, allowance)
    return excused * (2. * forward_effort - excused)
