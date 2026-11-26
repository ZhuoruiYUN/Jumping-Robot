"""Preview free ascent using the live collective motor state and pending duty."""

import torch

from .yaw_thrust_feedback import _thrust_from_duty


_CONTROL_TICKS = 16
_PULSE_TICKS = 8
_GRAVITY = 9.81
_DESCENT_ACCEL = 8.24
_TOUCH_ROOT_HEIGHT = .283


def _queued_thrust(motor_u, history, tau, delay, dt, voltage_gain, thrust_scale, candidate_duty,
                   control_ticks=_CONTROL_TICKS):
    """Use the same fractional delay, voltage and filter as ImuLinkPlant."""
    ticks = torch.arange(control_ticks, device=history.device)
    if candidate_duty is None:
        future = torch.zeros_like(motor_u[:, None]).expand(-1, control_ticks, -1)
    else:
        future = candidate_duty[:, None].clamp(0., 1.) * (ticks < _PULSE_TICKS)[None, :, None]
    commands = torch.cat((history, future), dim=1)
    rows = torch.arange(len(motor_u), device=history.device)[:, None]
    whole_delay = delay.floor().long()
    tap = history.shape[1] + ticks[None] - whole_delay[:, None]
    fraction = (delay - whole_delay)[:, None, None]
    delayed = commands[rows, tap].lerp(commands[rows, tap - 1], fraction)
    target = (delayed * voltage_gain[:, None, None]).clamp(0., 1.)
    log_decay = -dt / tau
    alpha = -log_decay.expm1()
    decay = torch.exp(log_decay[:, None] * (ticks + 1))
    filtered = decay[:, :, None] * (
        motor_u[:, None] + alpha[:, None, None] * (target / decay[:, :, None]).cumsum(dim=1)
    )
    return thrust_scale[:, None] * _thrust_from_duty(filtered).sum(dim=-1)


def queued_coast(height, vz, up, mass, drag, thrust_scale, voltage_gain, motor_u, history,
                 tau, delay, dt, candidate_duty=None, *, control_ticks=_CONTROL_TICKS):
    """Return a modeled apex and time to touch from a causal motor preview.

    History is chronological through the last completed control tick. Future
    input is zero except for an optional eight-tick diagnostic pulse. After
    the modeled window the remaining ascent uses zero thrust and quadratic
    drag. Descending lanes use only the descent estimate.
    """
    thrust = _queued_thrust(motor_u, history, tau, delay, dt, voltage_gain, thrust_scale, candidate_duty, control_ticks)
    powered_accel = thrust * (up.clamp(0., 1.) / mass)[:, None]
    drag_over_mass = drag.clamp_min(0.) / mass
    speed = vz.clamp_min(0.)
    altitude = height
    ascending_time = torch.zeros_like(height)
    for accel in powered_accel.unbind(dim=1):
        # Use the estimated acceleration over this control interval and
        # interpolate the zero crossing before accumulating peak height.
        next_speed = speed + dt * (accel - _GRAVITY - drag_over_mass * speed.square())
        active = speed > 0.
        partial = (speed / (speed - next_speed).clamp_min(1.e-8)).clamp(0., 1.)
        duration = dt * torch.where(active, torch.where(next_speed < 0., partial, 1.), 0.)
        next_speed = torch.where(active, next_speed.clamp_min(0.), 0.)
        altitude = altitude + .5 * duration * (speed + next_speed)
        ascending_time = ascending_time + duration
        speed = next_speed

    drag_ratio = drag_over_mass.clamp_min(1.e-8)
    scaled_speed = speed * (drag_ratio / _GRAVITY).sqrt()
    apex = altitude + torch.log1p(scaled_speed.square()) / (2. * drag_ratio)
    ascending_time = ascending_time + torch.atan(scaled_speed) / (_GRAVITY * drag_ratio).sqrt()
    falling_speed = (-vz).clamp_min(0.)
    fall_distance = (apex - _TOUCH_ROOT_HEIGHT).clamp_min(0.)
    fall_time = 2. * fall_distance / (
        (falling_speed.square() + 2. * _DESCENT_ACCEL * fall_distance).sqrt() + falling_speed
    ).clamp_min(1.e-8)
    return apex, ascending_time + fall_time
