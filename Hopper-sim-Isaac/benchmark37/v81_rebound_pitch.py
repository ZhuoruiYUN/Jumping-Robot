"""Protect onward speed during the next rebound of a straight continuation."""

import math

import torch

from .cross_rebound_control import validate as validate_attitude_controller
from .ring_geometry import rotate
from .yaw_thrust_feedback import _thrust_from_duty


def validate(config):
    limits = {
        'minimum_incoming_m_s': (.04, .30),
        'target_release_m_s': (.15, .45),
        'assumed_release_vz_m_s': (2., 3.4),
        'maximum_lean_deg': (2., 9.),
        'prediction_limit_s': (.04, .18),
        'trigger_release_m_s': (.01, .10),
        'full_reverse_m_s': (.05, .35),
    }
    if (config.keys() != limits.keys() | {'version', 'controller'}
            or config['version'] != 'same_direction_release_pitch_v1'
            or any(not math.isfinite(config[key]) or not lo <= config[key] <= hi
                   for key, (lo, hi) in limits.items())
            or config['minimum_incoming_m_s'] >= config['target_release_m_s']
            or config['trigger_release_m_s'] >= config['minimum_incoming_m_s']):
        raise ValueError('Invalid same-direction rebound pitch settings')
    validate_attitude_controller(config['controller'])


def rebound_pitch_reference(actions, quaternion, omega_body, axis, incoming_along, landing_speed,
                            time_to_touch, gate, config, *, opening_target=None):
    """Compare the actor's along-axis thrust mode with the restoring reference.

    The caller gates first-attempt straight continuations in free descent.
    The signed world XY axis points toward the current target, the quaternion
    is wxyz, and omega is body-frame. Landing speed is the positive contact
    speed magnitude. The target depends only on state, so motor saturation
    cannot lower the reference.
    """
    axis_world = torch.cat((axis, torch.zeros_like(axis[:, :1])), dim=-1)
    world_to_body = quaternion * quaternion.new_tensor([1., -1., -1., -1.])
    axis_body = rotate(world_to_body, axis_world)
    sine = axis_body[:, 2].clamp(-.99, .99)
    tilt = sine.asin()
    tilt_axis_body = torch.stack((-axis_body[:, 1], axis_body[:, 0]), dim=-1)
    tilt_axis_body = tilt_axis_body / (1. - sine.square()).sqrt()[:, None]
    tilt_rate = (omega_body[:, :2] * tilt_axis_body).sum(-1)
    contact_lean = tilt + time_to_touch.clamp(0., config['prediction_limit_s']) * tilt_rate
    # Use the measured downward velocity and nominal upward release velocity
    # to estimate the vertical rebound impulse per unit mass.
    vertical_delta = landing_speed.clamp_min(0.) + config['assumed_release_vz_m_s']
    predicted_release = incoming_along + vertical_delta * contact_lean.clamp(-.35, .35).tan()

    onset = config['trigger_release_m_s']
    risk = ((onset - predicted_release) / (onset + config['full_reverse_m_s'])).clamp(0., 1.)
    min_incoming, target_release = config['minimum_incoming_m_s'], config['target_release_m_s']
    incoming_gate = ((incoming_along - min_incoming) / (target_release - min_incoming)).clamp(0., 1.)
    # A correction may preserve available speed, while the raw policy still
    # owns forward acceleration and ordinary braking when reversal is unlikely.
    target_speed = incoming_along.clamp(0., target_release)
    if opening_target is not None:
        opening = opening_target > 0.
        opening_risk = ((opening_target - predicted_release) / opening_target.clamp_min(.05)).clamp(0., 1.)
        risk = torch.where(opening, opening_risk, risk)
        incoming_gate = torch.where(opening, 1., incoming_gate)
        target_speed = torch.where(opening, opening_target, target_speed)
    effective_gate = gate * risk * incoming_gate
    max_lean = math.radians(config['maximum_lean_deg'])
    desired_lean = torch.atan2(target_speed - incoming_along, vertical_delta).clamp(-max_lean, max_lean)

    controller = config['controller']
    error = tilt - desired_lean
    predicted_error = (error + controller['lookahead_s'] * tilt_rate).abs().rad2deg()
    start, full = controller['activation_error_deg'], controller['full_error_deg']
    effective_gate = effective_gate * ((predicted_error - start) / (full - start)).clamp(0., 1.)
    target_mode = (-controller['position_gain_n_rad'] * error
                   - controller['rate_gain_n_s_rad'] * tilt_rate).clamp(0., controller['max_mode_n'])
    signs = actions.new_tensor([[1., -1., -1., 1.], [1., 1., -1., -1.]])
    thrust = _thrust_from_duty(.5 * (actions.clamp(-1., 1.) + 1.))
    actual_mode = .25 * (thrust * (tilt_axis_body @ signs)).sum(-1)
    return actual_mode, target_mode, effective_gate, predicted_release, desired_lean
