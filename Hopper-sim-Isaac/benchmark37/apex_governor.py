"""Limit collective ascent thrust while preserving the nominal motor-force mix."""

import math

import torch

from .yaw_thrust_feedback import (
    _THRUST_CURVATURE, _THRUST_OFFSET, _THRUST_SLOPE, _duty_from_thrust, _thrust_from_duty,
)


def validate(config):
    bounds = {
        'target_apex_m': (1.03, 1.20),
        'lag_time_s': (.02, .18),
        'fade_height_start_m': (.40, .90),
        'fade_height_end_m': (.65, 1.),
        'max_collective_trim_n': (.01, .40),
        'up_fade_start': (.65, .99),
        'up_fade_end': (.80, 1.),
    }
    override_fields = {'override_start_apex_m', 'override_full_apex_m'}
    optional = config.keys() & ({'protected_height_m', 'protect_landing'} | override_fields)
    if (config.keys() != bounds.keys() | {'version'} | optional
            or config['version'] != 'ascent_collective_ceiling_v1'
            or any(not math.isfinite(config[k]) or not lo <= config[k] <= hi
                   for k, (lo, hi) in bounds.items())
            or config['fade_height_start_m'] >= config['fade_height_end_m']
            or config['up_fade_start'] >= config['up_fade_end']):
        raise ValueError('Invalid ascent thrust ceiling or fade band')
    if 'protected_height_m' in optional and not .98 <= config['protected_height_m'] < config['target_apex_m'] - .03:
        raise ValueError('Protected height must be below the ascent ceiling')
    if 'protect_landing' in optional and not isinstance(config['protect_landing'], bool):
        raise ValueError('Landing protection must be explicitly enabled or disabled')
    if optional & override_fields and (
        not override_fields <= optional or 'protected_height_m' not in optional
        or not config.get('protect_landing')
        or not config['target_apex_m'] < config['override_start_apex_m'] < config['override_full_apex_m'] <= 1.4
    ):
        raise ValueError('The height override requires a complete forecast band and the lower height guard')


def projected_landing_margin(edge, foot_minus_goal, horizontal_velocity, remaining):
    """Fade out altitude trimming when the projected travel falls short."""
    projected = foot_minus_goal + horizontal_velocity * remaining[:, None]
    axis = torch.nn.functional.normalize(edge, dim=-1)
    overshoot = (projected * axis).sum(-1)
    moving = ((overshoot - .03) / .06).clamp(0., 1.) * ((horizontal_velocity * axis).sum(-1) > 0.)
    hold = ((.08 - projected.norm(dim=-1)) / .04).clamp(0., 1.)
    return torch.where(edge.norm(dim=-1) > .10, moving, hold)


def overheight_debit(height, previous_peak, airborne, config):
    """Charge new measured overheight once, while it develops during the hop."""
    upper = config['upper_m']
    before = (previous_peak - upper).clamp_min(0.)
    after = torch.maximum(before, height - upper)
    width = config['soft_band_m']
    # Smooth at the ceiling, with bounded slope for an extreme rebound.
    growth = after.square() / (after + width) - before.square() / (before + width)
    return torch.where(airborne, -config['cost_per_m'] * growth.clamp_min(0.), 0.)


def limit_ascent_collective(actions, height, vz, up, flight_gate, mass, thrust_scale, voltage_gain, config,
                           *, ascent_drag_n_s2_m2=None, landing_margin=None, applied_thrust=None):
    """Limit the commanded collective to the remaining vertical work, with a lag allowance."""
    gravity = 9.81
    duty = .5 * (actions.clamp(-1., 1.) + 1.)
    gain = voltage_gain[:, None]
    effective_duty = (duty * gain).clamp_max(1.)
    commanded = thrust_scale * _thrust_from_duty(effective_duty).sum(-1)

    # Reserve the height reached during the actuator response window before
    # solving the remaining constant-acceleration ascent.
    speed = vz.clamp_min(0.)
    acceleration = commanded * up / mass - gravity
    delay = torch.minimum(torch.full_like(speed, config['lag_time_s']), speed / gravity)
    delayed_height = height + delay * (speed + .5 * delay * acceleration)
    delayed_speed = (speed + delay * acceleration).clamp_min(0.)
    coast_apex = delayed_height + delayed_speed.square() / (2. * gravity)
    remaining = (config['target_apex_m'] - delayed_height).clamp_min(.001)
    deceleration = delayed_speed.square() / (2. * remaining)
    allowable = (gravity - deceleration).clamp_min(0.) * mass / up.clamp_min(.65)

    height_gate = ((height - config['fade_height_start_m'])
                   / (config['fade_height_end_m'] - config['fade_height_start_m'])).clamp(0., 1.)
    up_gate = ((up - config['up_fade_start']) / (config['up_fade_end'] - config['up_fade_start'])).clamp(0., 1.)
    if landing_margin is not None and 'override_start_apex_m' not in config:
        flight_gate = flight_gate * landing_margin
        landing_margin = None
    if 'protected_height_m' in config:
        drag_per_mass = (ascent_drag_n_s2_m2 / mass).clamp_min(1.e-5)
        # Coast under gravity and quadratic drag. Current thrust and thrust
        # remaining in the motor are excluded from this lower-height gate.
        coast_with_drag = height + torch.log1p(drag_per_mass * speed.square() / gravity) / (2. * drag_per_mass)
        flight_gate = flight_gate * ((coast_with_drag - config['protected_height_m']) / .03).clamp(0., 1.)
    if landing_margin is not None:
        # Budget only the residual motor impulse during the response window.
        # Future commanded thrust can taper, so do not treat holding it all
        # the way to the apex as grounds to override landing protection.
        drag_deceleration = drag_per_mass * speed.square()
        lag_force = .5 * torch.minimum(applied_thrust, commanded)
        lag_acceleration = lag_force * up / mass - gravity - drag_deceleration
        lag_height = height + delay * (speed + .5 * delay * lag_acceleration)
        lag_speed = (speed + delay * lag_acceleration).clamp_min(0.)
        projected_apex = lag_height + torch.log1p(drag_per_mass * lag_speed.square() / gravity) / (2. * drag_per_mass)
        start, full = config['override_start_apex_m'], config['override_full_apex_m']
        urgency = ((projected_apex - start) / (full - start)).clamp(0., 1.)
        flight_gate = flight_gate * torch.maximum(landing_margin, urgency)
    request = (commanded - allowable).clamp(0., config['max_collective_trim_n'])
    request = request * height_gate * up_gate * flight_gate * (vz > 0.)
    # Convert the physical-force budget to a common nominal-force decrement.
    # An upper bound on the voltage-scaled slope keeps the applied trim within
    # the configured budget across the nonlinear duty-to-thrust curve.
    duty_slope = _THRUST_SLOPE - 2. * _THRUST_CURVATURE * duty
    effective_slope = _THRUST_SLOPE - 2. * _THRUST_CURVATURE * effective_duty
    gain_bound = voltage_gain * (effective_slope / duty_slope).amax(-1).clamp_min(1.)
    nominal = _thrust_from_duty(duty)
    headroom = (nominal.amin(-1) - _THRUST_OFFSET).clamp_min(0.)
    decrement = torch.minimum(request / (4. * thrust_scale * gain_bound), headroom)
    limited = _duty_from_thrust(nominal - decrement[:, None]).clamp(0., 1.)
    trimmed = commanded - thrust_scale * _thrust_from_duty((limited * gain).clamp_max(1.)).sum(-1)
    active = decrement > 0.
    return torch.where(active[:, None], 2. * limited - 1., actions), coast_apex, torch.where(active, trimmed.clamp_min(0.), 0.)
