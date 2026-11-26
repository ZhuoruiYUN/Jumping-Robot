"""Allocate heading servo authority independently of the policy's lateral thrust."""

import math

import torch

from .moving_phase_reward import heading_motion
from .yaw_thrust_feedback import (
    _NOMINAL_YAW_INERTIA,
    _THRUST_OFFSET,
    _THRUST_SPAN,
    _YAW_NM_PER_NEWTON,
    _duty_from_thrust,
    _thrust_from_duty,
    validate_thrust_reserve,
)


def validate_heading_servo(config, reserve_n):
    validate_thrust_reserve(reserve_n)
    bounds = {
        'position_gain_s2': (0., 150.),
        'rate_gain_s': (0., 30.),
        'max_yaw_mode_n': (reserve_n, .2),
        'collective_shift_max_n': (0., .1),
    }
    if set(config) != set(bounds) or any(
        not math.isfinite(config[key]) or not low <= config[key] <= high
        for key, (low, high) in bounds.items()
    ):
        raise ValueError('Invalid heading servo gains or actuator envelope')


def allocate_heading_servo(quaternion, omega_body, actions, target_heading_rad, reserve_n, config):
    """Retain the policy's thrust and roll/pitch channels, subject to motor bounds.

    Collective moves only as needed to make the heading allocation feasible.
    The policy's nominal yaw channel cannot oppose the heading servo.
    """
    heading, rate = heading_motion(quaternion, omega_body)
    error = heading - target_heading_rad
    error = torch.atan2(error.sin(), error.cos())

    # World-heading rate becomes ill-conditioned as the forward axis tips
    # towards vertical. Give the policy full roll/pitch priority near a flip.
    up = 1.0 - 2.0 * quaternion[:, 1:3].square().sum(dim=-1)
    tilted, upright = math.cos(math.radians(50.)), math.cos(math.radians(24.))
    blend = ((up - tilted) / (upright - tilted)).clamp(0., 1.)
    fade = blend.square() * (3. - 2. * blend)
    yaw_target = -(config['position_gain_s2'] * error + config['rate_gain_s'] * rate)
    yaw_target *= _NOMINAL_YAW_INERTIA / (4. * _YAW_NM_PER_NEWTON)
    yaw_target = yaw_target.clamp(-config['max_yaw_mode_n'], config['max_yaw_mode_n'])

    duty = .5 * (actions.clamp(-1., 1.) + 1.)
    thrust = _thrust_from_duty(duty)
    thrust = (_THRUST_OFFSET + reserve_n
              + (thrust - _THRUST_OFFSET) * (1. - 2. * reserve_n / _THRUST_SPAN))
    signs = thrust.new_tensor([-1., 1., -1., 1.])
    actor_yaw = (thrust * signs).mean(dim=-1, keepdim=True)
    collective = thrust.mean(dim=-1)
    roll_pitch = thrust - collective[:, None] - actor_yaw * signs
    yaw_target = fade * yaw_target + (1. - fade) * actor_yaw[:, 0]

    # Bound collective adjustment as well as the four motor commands. The
    # two opposite-spin pairs impose lower and upper bounds on the yaw mode.
    lo = _THRUST_OFFSET
    hi = lo + _THRUST_SPAN
    c_shift = config['collective_shift_max_n'] * fade
    c_lo, c_hi = collective - c_shift, collective + c_shift
    minus, plus = roll_pitch[:, 0::2], roll_pitch[:, 1::2]
    nmin, nmax = minus.amin(dim=-1), minus.amax(dim=-1)
    pmin, pmax = plus.amin(dim=-1), plus.amax(dim=-1)
    y_lo = torch.maximum(lo - c_hi - pmin, c_lo + nmax - hi)
    y_lo = torch.maximum(y_lo, .5 * (nmax - pmin - _THRUST_SPAN))
    y_hi = torch.minimum(hi - c_lo - pmax, c_hi + nmin - lo)
    y_hi = torch.minimum(y_hi, .5 * (_THRUST_SPAN + nmin - pmax))
    yaw_target = yaw_target.clamp(y_lo, y_hi)
    offset = roll_pitch + yaw_target[:, None] * signs
    collective = collective.clamp(lo - offset.amin(dim=-1), hi - offset.amax(dim=-1))
    selected = collective[:, None] + offset
    return (2. * _duty_from_thrust(selected) - 1.).clamp(-1., 1.)
