"""Experimental OBS73 heading feedback using spare motor thrust."""

import math

import torch

from .moving_phase_reward import heading_motion


# Nominal motor fit and static yaw term from Quadhopper_PPO_Reference_SysID.
_THRUST_OFFSET = 0.0028762422544365922
_THRUST_SLOPE = 0.9884448461365533
_THRUST_CURVATURE = 0.3647607454907179
_YAW_NM_PER_NEWTON = 0.011069143101433988
_NOMINAL_YAW_INERTIA = 0.00055  # Calibrated whole q=0 yaw inertia, kg m².
_THRUST_LIMIT = 0.05
_THRUST_SPAN = _THRUST_SLOPE - _THRUST_CURVATURE


def _thrust_from_duty(duty):
    return _THRUST_OFFSET + duty * (_THRUST_SLOPE - _THRUST_CURVATURE * duty)


def _duty_from_thrust(thrust):
    offset = thrust - _THRUST_OFFSET
    # The lower quadratic root lies on the monotone branch for [0, 1].
    discriminant = (_THRUST_SLOPE**2 - 4.0 * _THRUST_CURVATURE * offset).clamp_min(0.0)
    return 2.0 * offset / (_THRUST_SLOPE + discriminant.sqrt())


def validate_thrust_reserve(reserve_n):
    if not 0.0 < reserve_n <= _THRUST_LIMIT:
        raise ValueError(f"Per-motor yaw thrust reserve must be in (0, {_THRUST_LIMIT}] N")


def apply_yaw_feedback(observation, actions, target_heading_rad=0.0, *, reserve_n=0.0,
                       priority=None):
    """Add a bounded yaw correction from measured heading and heading rate.

    Actions use the existing F1/F2/F3/F4 normalized motor interface. The
    correction preserves commanded total thrust and nominal roll/pitch
    moment even near saturation. Actual motor lag still applies downstream.
    """
    if observation.ndim != 2 or observation.shape[1] != 73 or actions.shape != (len(observation), 4):
        raise ValueError("Expected batched quaternion OBS73 and four motor commands")

    return allocate_yaw_thrust(observation[:, 9:13], observation[:, 13:16], actions,
                               target_heading_rad, reserve_n=reserve_n, priority=priority)


def allocate_yaw_thrust(quaternion, omega_body, actions, target_heading_rad, *, reserve_n,
                         priority=None):
    heading, rate = heading_motion(quaternion, omega_body)
    relative = heading - target_heading_rad
    error = torch.atan2(relative.sin(), relative.cos())
    position_error = error.sign() * (error.abs() - math.radians(1.5)).clamp_min(0.0)

    # Lose authority smoothly as the body leaves the upright flight envelope.
    up = 1.0 - 2.0 * quaternion[:, 1:3].square().sum(dim=-1)
    lower, upper = math.cos(math.radians(49.0)), math.cos(math.radians(26.0))
    blend = ((up - lower) / (upper - lower)).clamp(0.0, 1.0)
    fade = blend.square() * (3.0 - 2.0 * blend)
    requested_acceleration = -(20.0 * position_error + 7.0 * rate)
    requested = requested_acceleration * fade * (_NOMINAL_YAW_INERTIA / (4.0 * _YAW_NM_PER_NEWTON))

    native = actions.clamp(-1.0, 1.0)
    thrust = _thrust_from_duty(0.5 * (native + 1.0))
    if reserve_n:
        validate_thrust_reserve(reserve_n)
        # Make room in both directions, even when the policy asks for an endpoint.
        thrust = (_THRUST_OFFSET + reserve_n
                  + (thrust - _THRUST_OFFSET) * (1.0 - 2.0 * reserve_n / _THRUST_SPAN))
    sign = thrust.new_tensor([-1.0, 1.0, -1.0, 1.0])
    if priority is not None:
        # Blend out opposing actor yaw when the heading is predicted to drift
        # farther from target. The other three mixer channels are preserved.
        forecast = (error + priority['horizon_s'] * rate).abs()
        warning = math.radians(priority['warning_deg'])
        width = math.radians(priority['full_deg'] - priority['warning_deg'])
        risk = ((forecast - warning) / width).clamp(0.0, 1.0)
        risk = risk.square() * (3.0 - 2.0 * risk) * fade
        actor_yaw = (thrust * sign).mean(dim=-1)
        outward = (error * rate >= 0.0) & (actor_yaw * requested < 0.0)
        requested = requested - torch.where(outward, risk * actor_yaw, 0.0)
    up_space = _THRUST_OFFSET + _THRUST_SPAN - thrust
    down_space = thrust - _THRUST_OFFSET
    limit = _THRUST_LIMIT
    if priority is not None:
        limit = limit + torch.where(outward, risk * actor_yaw.abs(), 0.0)
    positive = torch.where(sign > 0.0, up_space, down_space).amin(dim=-1).clamp_min(0.0)
    negative = torch.where(sign > 0.0, down_space, up_space).amin(dim=-1).clamp_min(0.0)
    positive = positive.clamp_max(limit)
    negative = negative.clamp_max(limit)
    correction = torch.clamp(requested, min=-negative, max=positive)
    adjusted_thrust = thrust + correction[:, None] * sign
    adjusted = (2.0 * _duty_from_thrust(adjusted_thrust) - 1.0).clamp(-1.0, 1.0)
    if reserve_n:
        return adjusted
    return torch.where(correction[:, None] != 0.0, adjusted, native)
