"""Proven native flight/landing kernels for the OBS73 stationary task.

XY and goal payments remain event-only to avoid the former hover solution.
Robot dynamics, policy inputs and the native reward implementation stay frozen.
"""
import math

import torch


def validate_surplus_height_scale(settings):
    keys = {'start_above_target_m', 'full_above_target_m', 'scale'}
    if not isinstance(settings, dict) or settings.keys() != keys or not all(
            math.isfinite(settings[key]) for key in keys):
        raise ValueError('Invalid surplus height scale settings')
    if not (0. <= settings['start_above_target_m'] < settings['full_above_target_m']
            and 0. < settings['scale'] <= 1.):
        raise ValueError('Surplus height scale needs a rising interval and a positive gain at most one')


def _surplus_gain(excess, mask, settings):
    start = settings['start_above_target_m']
    fraction = ((excess - start) / (settings['full_above_target_m'] - start)).clamp(0., 1.)
    smooth = fraction.square() * (3. - 2. * fraction)
    return torch.where(mask, 1. - (1. - settings['scale']) * smooth, 1.)


def flight_terms(z, vz, target, gravity, energy_width, scales, dt, config, *, surplus_mask=None):
    energy = gravity * z.clamp(min=0.) + .5 * vz.square()
    desired_energy = gravity * (target + config['energy_margin_m']).clamp(min=0.)
    # A reward-only clearance margin; the observation command and strict
    # touchdown height threshold remain the original target. Legacy courses
    # omit this option and retain their original reward centres.
    height_error = (target + config.get('height_reward_margin_m', 0.) - z).abs()
    terms = {
        'energy_tracking': scales['energy_tracking'] * torch.exp(-(energy-desired_energy).abs()/energy_width) * dt,
        'distance_to_z': scales['distance_to_z'] * torch.exp(-height_error/config['height_width_m']) * dt,
        'apex': scales['apex'] * torch.exp(-height_error/config['apex_height_width_m'])
                * torch.exp(-vz.abs()/config['apex_speed_width_m_s']) * dt,
    }
    surplus_scale = config.get('surplus_height_scale')
    if surplus_scale is not None and surplus_mask is not None:
        energy_gain = _surplus_gain(energy / gravity - target, surplus_mask, surplus_scale)
        height_gain = _surplus_gain(z - target, surplus_mask, surplus_scale)
        terms['energy_tracking'] *= energy_gain
        terms['distance_to_z'] *= height_gain
        terms['apex'] *= height_gain
    return terms


def landing_terms(peak, target, foot_error, tilt_deg, valid, landing_config, config, position_plateau_m=0., foot_limit_m=.1):
    from .obs73_jump_cycle import plateau_distance
    strict = valid & (peak >= target) & (foot_error <= foot_limit_m) & (tilt_deg <= 15.)
    foot = torch.where(strict, landing_config['scale']
                       * torch.exp(-.5*(plateau_distance(foot_error, position_plateau_m)
                                        /landing_config['foot_width_m']).square()), 0.)
    deficit = torch.where(valid, -config['apex_deficit_scale']*(target-peak).clamp(min=0.), 0.)
    return foot, deficit
