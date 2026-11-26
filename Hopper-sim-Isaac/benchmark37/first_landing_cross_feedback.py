"""Reward-only lateral guidance for the first landing on a long leg."""

import math

import torch


def validate(config):
    if config['version'] != 'late_ascent_first_landing_cross_v1':
        raise ValueError('unsupported first-landing cross feedback version')
    names = (
        'minimum_step_m', 'start_elapsed_s', 'full_elapsed_s', 'minimum_remaining_s',
        'full_remaining_s', 'ascent_start_vz_m_s', 'ascent_full_vz_m_s',
        'descent_cutoff_m_s', 'maximum_acceleration_m_s2',
        'error_width_m', 'forecast_weight',
    )
    for name in names:
        value = config[name]
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(f'first-landing cross feedback requires positive finite {name}')
    if config['start_elapsed_s'] >= config['full_elapsed_s']:
        raise ValueError('first-landing cross feedback ascent window must have positive width')
    if config['minimum_remaining_s'] >= config['full_remaining_s']:
        raise ValueError('first-landing cross feedback remaining flight window must have positive width')
    if config['ascent_full_vz_m_s'] >= config['ascent_start_vz_m_s']:
        raise ValueError('first-landing cross feedback ascent speed window must have positive width')


def _smooth_ramp(value, start, full):
    fraction = ((value - start) / (full - start)).clamp(0., 1.)
    return fraction.square() * (3. - 2. * fraction)


def first_landing_cross_feedback(foot_reference_xy, velocity, target_xy, edge,
                                 elapsed_s, remaining_s, eligible, actual_acceleration_xy,
                                 radius_m, dt, config):
    """Score a causal touchdown projection and discount limited corrective effort.

    ``foot_reference_xy`` follows the current root with the measured release
    foot offset. ``remaining_s`` is the active hop clock, evaluated before the
    target can advance. The target and measured touchdown scorer are untouched.
    All tensor outputs have one element per lane, in input order.
    """
    direction = torch.nn.functional.normalize(edge, dim=-1)
    normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    remaining_s = remaining_s.clamp_min(0.)
    projected_offset = foot_reference_xy - target_xy + velocity[:, :2] * remaining_s[:, None]
    cross_m = (projected_offset * normal).sum(-1)
    # Longitudinal residuals have their own guidance. Trigger correction only
    # where cross-track error alone would exceed the measured landing radius.
    excess_m = (cross_m.abs() - radius_m).clamp_min(0.)

    ascent = _smooth_ramp(elapsed_s, config['start_elapsed_s'], config['full_elapsed_s'])
    time_left = _smooth_ramp(remaining_s, config['minimum_remaining_s'], config['full_remaining_s'])
    vz = velocity[:, 2]
    late_ascent = _smooth_ramp(-vz, -config['ascent_start_vz_m_s'], -config['ascent_full_vz_m_s'])
    above_descent = _smooth_ramp(vz, -config['descent_cutoff_m_s'], 0.)
    long_leg = edge.norm(dim=-1) + 1.e-6 >= config['minimum_step_m']
    gate = torch.where(eligible & long_leg, ascent * time_left * late_ascent * above_descent, 0.)

    denominator = remaining_s.clamp_min(config['minimum_remaining_s']).square()
    amplitude = (2. * excess_m / denominator).clamp_max(config['maximum_acceleration_m_s2'])
    reference = -cross_m.sign() * amplitude * gate
    actual_cross = (actual_acceleration_xy * normal).sum(-1)
    toward_target = (actual_cross * reference.sign()).clamp_min(0.)
    excused = torch.minimum(toward_target, reference.abs())
    # Exact difference between the original and residual squared cross costs.
    # Along-track work, wrong-way thrust and excess corrective thrust stay charged.
    cruise_discount = excused * (2. * toward_target - excused)

    scaled_error = excess_m / config['error_width_m']
    forecast_cost = dt * config['forecast_weight'] * gate * torch.expm1(-.5 * scaled_error.square())
    return forecast_cost, cruise_discount, {
        'gate': gate,
        'active': (gate > 0.) & (excess_m > 0.),
        'projected_cross_error_m': cross_m,
        'reference_acceleration_m_s2': reference,
    }
