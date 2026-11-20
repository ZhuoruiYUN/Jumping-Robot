"""Proven native flight/landing kernels for the OBS73 stationary task.

XY and goal payments remain event-only to avoid the former hover solution.
Robot dynamics, policy inputs and the native reward implementation stay frozen.
"""
import torch


def flight_terms(z, vz, target, gravity, energy_width, scales, dt, config):
    energy = gravity * z.clamp(min=0.) + .5 * vz.square()
    desired_energy = gravity * (target + config['energy_margin_m']).clamp(min=0.)
    # A reward-only clearance margin; the observation command and strict
    # touchdown height threshold remain the original target. Legacy courses
    # omit this option and retain their original reward centres.
    height_error = (target + config.get('height_reward_margin_m', 0.) - z).abs()
    return {
        'energy_tracking': scales['energy_tracking'] * torch.exp(-(energy-desired_energy).abs()/energy_width) * dt,
        'distance_to_z': scales['distance_to_z'] * torch.exp(-height_error/config['height_width_m']) * dt,
        'apex': scales['apex'] * torch.exp(-height_error/config['apex_height_width_m'])
                * torch.exp(-vz.abs()/config['apex_speed_width_m_s']) * dt,
    }


def landing_terms(peak, target, foot_error, tilt_deg, valid, landing_config, config, position_plateau_m=0., foot_limit_m=.1):
    from .obs73_jump_cycle import plateau_distance
    strict = valid & (peak >= target) & (foot_error <= foot_limit_m) & (tilt_deg <= 15.)
    foot = torch.where(strict, landing_config['scale']
                       * torch.exp(-.5*(plateau_distance(foot_error, position_plateau_m)
                                        /landing_config['foot_width_m']).square()), 0.)
    deficit = torch.where(valid, -config['apex_deficit_scale']*(target-peak).clamp(min=0.), 0.)
    return foot, deficit
