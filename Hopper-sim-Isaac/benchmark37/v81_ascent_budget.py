"""Budget newly requested ascent thrust against the live actuator queue."""

import math

import torch

from .ascent_coast_preview import queued_coast
from .yaw_thrust_feedback import (
    _THRUST_CURVATURE, _THRUST_OFFSET, _THRUST_SLOPE, _duty_from_thrust, _thrust_from_duty,
)


VERSION = 'queued_ascent_budget_v1'


def validate(config):
    keys = {'version', 'protected_height_m', 'target_apex_m', 'max_collective_trim_n'}
    if config.keys() != keys or config['version'] != VERSION:
        raise ValueError('Invalid queued ascent budget fields')
    floor, ceiling, trim = (config[key] for key in ('protected_height_m', 'target_apex_m', 'max_collective_trim_n'))
    if (not all(math.isfinite(value) for value in (floor, ceiling, trim))
            or not 1.0 <= floor <= 1.02 or not floor + .03 <= ceiling <= 1.10 or not .02 <= trim <= .40):
        raise ValueError('Invalid queued ascent floor, ceiling or collective trim')


def validate_retained_pulse(config):
    limits = {'minimum_apex_m': (1.01, 1.04), 'max_collective_trim_n': (.02, .24),
              'min_removable_apex_m': (.001, .01)}
    if (config.keys() != limits.keys() | {'control_ticks'}
            or type(config['control_ticks']) is not int or not 16 <= config['control_ticks'] <= 32
            or any(not math.isfinite(config[key]) or not lo <= config[key] <= hi
                   for key, (lo, hi) in limits.items())):
        raise ValueError('Invalid retained ascent pulse forecast settings')


def _common_force_room(duty, thrust_scale, voltage_gain, cap_n):
    effective = (duty * voltage_gain[:, None]).clamp_max(1.)
    nominal = _thrust_from_duty(duty)
    nominal_slope = _THRUST_SLOPE - 2. * _THRUST_CURVATURE * duty
    effective_slope = _THRUST_SLOPE - 2. * _THRUST_CURVATURE * effective
    gain_bound = voltage_gain * (effective_slope / nominal_slope).amax(-1).clamp_min(1.)
    headroom = (nominal.amin(-1) - _THRUST_OFFSET).clamp_min(0.)
    return nominal, torch.minimum(headroom, cap_n / (4. * thrust_scale * gain_bound))


def limit_queued_ascent(actions, gate, height, vz, up, mass, drag, thrust_scale, voltage_gain,
                        applied_duty, queued_duty, tau, delay, dt, settings):
    """Return actions, queued coast apex, requested apex and collective trim (N).

    The live actuator state and queued commands are retained in every preview.
    Future commands beyond the preview pulse are zero. A trim is allowed only
    when even zero future input retains the modeled minimum height. Removing
    equal nominal thrust from F1–F4 preserves differential force commands.
    """
    coast = torch.zeros_like(height)
    preview = torch.zeros_like(height)
    trimmed = torch.zeros_like(height)
    ids = ((gate > 0.) & (vz > 0.)).nonzero().flatten()
    if not len(ids):
        return actions, coast, preview, trimmed

    duty = ((actions[ids] + 1.) * .5).clamp(0., 1.)
    gain = voltage_gain[ids, None]
    effective = (duty * gain).clamp_max(1.)
    nominal, max_decrement = _common_force_room(
        duty, thrust_scale[ids], voltage_gain[ids], settings['max_collective_trim_n'] * gate[ids].clamp(0., 1.))
    smallest_duty = _duty_from_thrust(nominal - max_decrement[:, None]).clamp(0., 1.)
    state = (height, vz, up, mass, drag, thrust_scale, voltage_gain, applied_duty, queued_duty, tau, delay)
    batched_state = tuple(torch.cat((value[ids],) * 3, dim=0) for value in state)
    candidate = torch.cat((torch.zeros_like(duty), duty, smallest_duty), dim=0)
    peaks, _ = queued_coast(*batched_state, dt, candidate_duty=candidate)
    coast_apex, requested_apex, min_apex = peaks.chunk(3)

    # Use the maximum feasible trim preview to estimate the portion of the
    # available authority that reaches the ceiling. Fade in safely as the
    # predicted coast clears the floor, without relying on new thrust.
    protected = ((coast_apex - settings['protected_height_m']) / .015).clamp(0., 1.)
    excess = (requested_apex - settings['target_apex_m']).clamp_min(0.)
    fraction = (excess / (requested_apex - min_apex).clamp_min(1.e-6)).clamp_max(1.)
    decrement = protected * fraction * max_decrement
    accepted = _duty_from_thrust(nominal - decrement[:, None]).clamp(0., 1.)
    active = decrement > 0.
    result = actions.clone()
    result[ids] = torch.where(active[:, None], 2. * accepted - 1., actions[ids])
    coast[ids] = coast_apex
    preview[ids] = requested_apex
    reduction = _thrust_from_duty(effective) - _thrust_from_duty((accepted * gain).clamp_max(1.))
    trimmed[ids] = torch.where(active, thrust_scale[ids] * reduction.sum(-1).clamp_min(0.), 0.)
    return result, coast, preview, trimmed


def retained_ascent_pulse_cost(actions, gate, height, vz, up, mass, drag, thrust_scale, voltage_gain,
                               applied_duty, queued_duty, tau, delay, dt, ceiling, weight, settings):
    """Score reducible overshoot under a bounded, shorter-than-ascent motor pulse.

    Both predictions retain the real queue and lag. The reduced pulse preserves
    motor differential force and still has to reach the protected lower height.
    The measured action and dynamics are left with the regular controller.
    """
    cost = torch.zeros_like(height)
    requested = torch.zeros_like(height)
    retained = torch.zeros_like(height)
    ids = ((gate > 0.) & (vz > 0.)).nonzero().flatten()
    if not len(ids):
        return cost, requested, retained

    duty = .5 * (actions[ids].clamp(-1., 1.) + 1.)
    nominal, decrement = _common_force_room(duty, thrust_scale[ids], voltage_gain[ids],
                                             settings['max_collective_trim_n'])
    reduced = _duty_from_thrust(nominal - decrement[:, None]).clamp(0., 1.)
    state = (height, vz, up, mass, drag, thrust_scale, voltage_gain, applied_duty, queued_duty, tau, delay)
    paired = tuple(torch.cat((value[ids], value[ids])) for value in state)
    peaks, _ = queued_coast(*paired, dt, candidate_duty=torch.cat((duty, reduced)),
                            control_ticks=settings['control_ticks'])
    predicted, protected = peaks.chunk(2)

    reserve = ((protected - settings['minimum_apex_m']) / .015).clamp(0., 1.)
    removable = ((predicted - protected - settings['min_removable_apex_m']) / .01).clamp(0., 1.)
    excess = ((predicted - ceiling) / .05).clamp(0., 1.)
    cost[ids] = -dt * weight * gate[ids] * reserve * removable * excess.square()
    requested[ids], retained[ids] = predicted, protected
    return cost, requested, retained
