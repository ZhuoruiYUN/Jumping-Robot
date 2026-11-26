"""Guidance for bounded fore-aft braking and cross-track correction before apex."""

import math

import torch

from .cross_rebound_control import validate as validate_attitude
from .first_landing_cross_feedback import _smooth_ramp
from .ring_geometry import rotate
from .yaw_thrust_feedback import _thrust_from_duty


DEFAULT_CONFIG = {
    'version': 'v81_first_landing_reference_v1',
    'minimum_step_m': .40,
    'start_elapsed_s': .25,
    'full_elapsed_s': .40,
    'minimum_remaining_s': .25,
    'full_remaining_s': .40,
    'axis_deadband_m': .025,
    'start_error_m': .05,
    'full_error_m': .10,
    'max_accel_m_s2': .55,
    'attitude': {
        'position_gain_n_rad': .24,
        'rate_gain_n_s_rad': .026,
        'max_mode_n': .020,
        'activation_error_deg': .6,
        'full_error_deg': 2.5,
        'lookahead_s': .075,
    },
}


def validate(config):
    if config.keys() - {'undertravel'} != DEFAULT_CONFIG.keys() or config.get('version') != DEFAULT_CONFIG['version']:
        raise ValueError('Invalid V81 first landing reference configuration')
    for name in DEFAULT_CONFIG.keys() - {'version', 'attitude'}:
        if not math.isfinite(config[name]) or config[name] <= 0.:
            raise ValueError(f'V81 first landing reference requires positive finite {name}')
    if not (.25 <= config['start_elapsed_s'] < config['full_elapsed_s']
            and config['minimum_remaining_s'] < config['full_remaining_s']
            and config['axis_deadband_m'] <= config['start_error_m'] < config['full_error_m']
            and config['max_accel_m_s2'] <= .8):
        raise ValueError('Invalid V81 first landing guidance envelope')
    validate_attitude(config['attitude'])
    short = config.get('undertravel')
    if short is not None:
        fields = {'early_elapsed_s', 'early_full_elapsed_s', 'minimum_shortfall_m', 'full_shortfall_m',
                  'clock_uncertainty_s', 'max_forward_accel_m_s2'}
        if (set(short) != fields
                or not (.10 <= short['early_elapsed_s'] < short['early_full_elapsed_s'] <= .40)
                or not (.05 <= short['minimum_shortfall_m'] < short['full_shortfall_m'] <= .25)
                or not (.02 <= short['clock_uncertainty_s'] <= .16)
                or not (0. < short['max_forward_accel_m_s2'] <= config['max_accel_m_s2'])):
            raise ValueError('Invalid first-landing shortfall reference')


def first_landing_reference(actions, quaternion, omega_body, foot_xy, velocity,
                            target_xy, edge, elapsed_s, remaining_s, eligible, config, *,
                            undertravel_allowed=None, undertravel_credit_m=None):
    """Use the supplied powered remaining-time estimate to form a motor-mode reference.

    The caller supplies the first-attempt and free-ascent eligibility. All
    positions, edge vectors and velocities use world coordinates, the
    quaternion is wxyz, and angular velocity is in body coordinates.
    """
    axis = torch.nn.functional.normalize(edge, dim=-1)
    cross = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    axes = torch.stack((axis, cross), dim=1)
    remaining_s = remaining_s.clamp_min(0.)
    forecast = foot_xy - target_xy + velocity[:, :2] * remaining_s[:, None]
    projected = (axes * forecast[:, None, :]).sum(-1)
    projected_along, projected_cross = projected.unbind(-1)
    along_speed = (axis * velocity[:, :2]).sum(-1)
    # Apply braking only to a forward-moving overshoot.
    relevant_along = torch.where(along_speed > 0., projected_along.clamp_min(0.), 0.)
    relevant_error = torch.stack((relevant_along, projected_cross), dim=-1)
    excess = relevant_error.sign() * (relevant_error.abs() - config['axis_deadband_m']).clamp_min(0.)
    timing_gate = _smooth_ramp(elapsed_s, config['start_elapsed_s'], config['full_elapsed_s'])
    timing_gate *= _smooth_ramp(remaining_s, config['minimum_remaining_s'], config['full_remaining_s'])
    timing_gate *= _smooth_ramp(velocity[:, 2], 0., .20)
    upright = 1. - 2. * quaternion[:, 1:3].square().sum(dim=-1)
    timing_gate *= _smooth_ramp(upright, math.cos(math.radians(24.)), math.cos(math.radians(16.)))
    long_free_ascent = eligible & (edge.norm(dim=-1) + 1.e-6 >= config['minimum_step_m'])
    gate = torch.where(
        long_free_ascent,
        timing_gate * _smooth_ramp(relevant_error.norm(dim=-1), config['start_error_m'], config['full_error_m']),
        0.,
    )
    along_gate = gate
    undertravel_gate = torch.zeros_like(gate)
    credit_audit = {}
    native_along_excess = None
    short = config.get('undertravel')
    if short is not None and undertravel_allowed is not None:
        # A valid shortfall must remain even if forward travel gets extra
        # flight time. A real backward release can be addressed sooner.
        optimistic_along = projected_along + short['clock_uncertainty_s'] * along_speed.clamp_min(0.)
        shortfall = (-optimistic_along).clamp_min(0.)
        early = _smooth_ramp(elapsed_s, short['early_elapsed_s'], short['early_full_elapsed_s'])
        normal = _smooth_ramp(elapsed_s, config['start_elapsed_s'], config['full_elapsed_s'])
        short_gate = torch.where(along_speed < -.05, early, normal)
        short_gate *= _smooth_ramp(remaining_s, .32, .45) * _smooth_ramp(velocity[:, 2], 0., .20)
        short_gate *= _smooth_ramp(upright, math.cos(math.radians(24.)), math.cos(math.radians(16.)))
        permitted = long_free_ascent & undertravel_allowed
        if undertravel_credit_m is not None:
            raw_gate = torch.where(
                permitted, short_gate * _smooth_ramp(shortfall, short['minimum_shortfall_m'], short['full_shortfall_m']), 0.)
            native_along_excess = torch.where(raw_gate > 0., -(shortfall - config['axis_deadband_m']), excess[:, 0])
            reduced_shortfall = (shortfall - undertravel_credit_m).clamp_min(0.)
            credit_audit = {
                'native_undertravel_gate': raw_gate,
                'applied_undertravel_credit_m': torch.where(raw_gate > 0., shortfall - reduced_shortfall, 0.),
            }
            shortfall = reduced_shortfall
        short_gate *= _smooth_ramp(shortfall, short['minimum_shortfall_m'], short['full_shortfall_m'])
        undertravel_gate = torch.where(permitted, short_gate, 0.)
        use_short = undertravel_gate > 0.
        along_excess = torch.where(use_short, -(shortfall - config['axis_deadband_m']), excess[:, 0])
        excess = torch.stack((along_excess, excess[:, 1]), dim=-1)
        along_gate = torch.where(use_short, undertravel_gate, gate)
    # Hold the original XY budget allocation while credit reduces along demand.
    time_squared = remaining_s.clamp_min(config['minimum_remaining_s']).square()
    reference_accel = -2. * excess / time_squared[:, None]
    if short is not None:
        reference_accel[:, 0].clamp_max_(short['max_forward_accel_m_s2'])
    budget = reference_accel
    if native_along_excess is not None:
        native_along_accel = (-2. * native_along_excess / time_squared).clamp_max(short['max_forward_accel_m_s2'])
        budget = torch.stack((native_along_accel, reference_accel[:, 1]), dim=-1)
    reference_accel *= (config['max_accel_m_s2'] /
                        budget.norm(dim=-1).clamp_min(config['max_accel_m_s2']))[:, None]
    desired_tilt = torch.atan(reference_accel / 9.81)

    world_axes = torch.cat((axes, torch.zeros_like(axes[..., :1])), dim=-1)
    world_to_body = quaternion * quaternion.new_tensor([1., -1., -1., -1.])
    axes_body = rotate(world_to_body[:, None, :].expand(-1, 2, -1), world_axes)
    sine = axes_body[..., 2].clamp(-.99, .99)
    rotation_axes = torch.stack((-axes_body[..., 1], axes_body[..., 0]), dim=-1)
    rotation_axes = rotation_axes / (1. - sine.square()).sqrt()[..., None]
    tilt_error = sine.asin() - desired_tilt
    tilt_rate = (omega_body[:, None, :2] * rotation_axes).sum(-1)
    gains = config['attitude']
    forecast_error = (tilt_error + gains['lookahead_s'] * tilt_rate).abs().rad2deg()
    axis_weight = torch.stack((along_gate, gate), dim=-1) * _smooth_ramp(
        forecast_error, gains['activation_error_deg'], gains['full_error_deg'])
    axis_weight *= excess.ne(0)
    mode_target = (-gains['position_gain_n_rad'] * tilt_error - gains['rate_gain_n_s_rad'] * tilt_rate)
    mode_target = mode_target.clamp(-gains['max_mode_n'], gains['max_mode_n'])

    motor_signs = actions.new_tensor([[1., -1., -1., 1.], [1., 1., -1., -1.]])
    motor_modes = rotation_axes @ motor_signs
    thrust = _thrust_from_duty(.5 * (actions.clamp(-1., 1.) + 1.))
    action_mode = (thrust[:, None, :] * motor_modes).sum(-1) / motor_modes.square().sum(-1).clamp_min(1.e-8)
    return {
        'gate': gate,
        'undertravel_gate': undertravel_gate,
        'active': (axis_weight > 0.).any(dim=-1),
        'axis_weight': axis_weight,
        'projected_along_m': projected_along,
        'projected_cross_m': projected_cross,
        'reference_along_accel_m_s2': reference_accel[:, 0] * along_gate,
        'reference_cross_accel_m_s2': reference_accel[:, 1] * gate,
        'action_along_force_n': action_mode[:, 0],
        'action_cross_force_n': action_mode[:, 1],
        'target_along_force_n': mode_target[:, 0],
        'target_cross_force_n': mode_target[:, 1],
        **credit_audit,
    }
