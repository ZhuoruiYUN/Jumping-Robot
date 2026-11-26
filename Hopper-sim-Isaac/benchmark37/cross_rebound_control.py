"""Transverse attitude correction in the hopper's nominal thrust coordinates."""

import math

import torch

from .ring_geometry import rotate
from .yaw_thrust_feedback import _THRUST_OFFSET, _THRUST_SPAN, _duty_from_thrust, _thrust_from_duty


def validate(config):
    limits = {'position_gain_n_rad': (0., 1.), 'rate_gain_n_s_rad': (0., .12),
              'max_mode_n': (0., .08), 'activation_error_deg': (0., 8.),
              'full_error_deg': (0., 12.), 'lookahead_s': (0., .2)}
    if config.keys() != limits.keys() or any(
        not math.isfinite(config[key]) or not lo < config[key] <= hi
        for key, (lo, hi) in limits.items()
    ) or config['activation_error_deg'] >= config['full_error_deg']:
        raise ValueError('Invalid transverse rebound gains or thrust limit')


def correct_cross_rebound(actions, quaternion, omega_body, cross_axis, desired_cross_tilt_rad, gate, config):
    cross_world = torch.cat((cross_axis, torch.zeros_like(cross_axis[:, :1])), dim=-1)
    world_to_body = quaternion * quaternion.new_tensor([1., -1., -1., -1.])
    cross_body = rotate(world_to_body, cross_world)
    sine = cross_body[:, 2].clamp(-.99, .99)
    tilt = sine.asin()
    # d(asin(up · cross))/dt is the angular velocity about up × cross.
    axis_body_xy = torch.stack((-cross_body[:, 1], cross_body[:, 0]), dim=-1)
    axis_body_xy = axis_body_xy / (1. - sine.square()).sqrt()[:, None]
    tilt_rate = (omega_body[:, :2] * axis_body_xy).sum(-1)
    error = tilt - desired_cross_tilt_rad
    future_error = (error + config['lookahead_s'] * tilt_rate).abs().rad2deg()
    authority = ((future_error - config['activation_error_deg'])
                 / (config['full_error_deg'] - config['activation_error_deg'])).clamp(0., 1.)
    roll_pitch = -config['position_gain_n_rad'] * error - config['rate_gain_n_s_rad'] * tilt_rate
    requested = gate * authority * roll_pitch.clamp(-config['max_mode_n'], config['max_mode_n'])
    signs = actions.new_tensor([[1., -1., -1., 1.], [1., 1., -1., -1.]])
    delta = requested[:, None] * (axis_body_xy @ signs)

    thrust = _thrust_from_duty(.5 * (actions.clamp(-1., 1.) + 1.))
    hi = _THRUST_OFFSET + _THRUST_SPAN
    room = torch.where(delta >= 0., hi - thrust, thrust - _THRUST_OFFSET)
    magnitude = delta.abs()
    fraction = torch.where(magnitude > 1.e-9, room / magnitude.clamp_min(1.e-9), 1.)
    fraction = fraction.amin(-1).clamp(0., 1.)
    modified = 2. * _duty_from_thrust(thrust + (fraction[:, None] * delta)) - 1.
    return torch.where((fraction * requested != 0.)[:, None], modified.clamp(-1., 1.), actions), tilt, fraction * requested
