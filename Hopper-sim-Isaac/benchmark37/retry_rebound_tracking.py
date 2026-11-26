"""Anticipate the along-route leg axis at the start of spring support."""

import math

import torch
from torch.nn.functional import normalize


def validate(config):
    keys = (
        'settling_time_constants', 'lead_ramp_s', 'max_preview_s',
        'early_guard_start_s', 'early_guard_full_s', 'early_guard_extra_m',
        'axis_prediction_limit_s', 'spring_centroid_s', 'support_window_s',
        'confirm_window_s',
    )
    if any(not math.isfinite(config[key]) or config[key] <= 0. for key in keys):
        raise ValueError('Retry rebound tracking requires positive, finite bounds')
    if not (1. <= config['settling_time_constants'] <= 3.
            and config['lead_ramp_s'] < config['max_preview_s'] <= .3
            and config['early_guard_start_s'] < config['early_guard_full_s'] <= config['max_preview_s']
            and config['early_guard_extra_m'] <= .05
            and config['spring_centroid_s'] <= config['axis_prediction_limit_s'] <= .15
            and config['spring_centroid_s'] <= config['support_window_s'] <= .1
            and config['confirm_window_s'] <= config['support_window_s']):
        raise ValueError('Invalid retry rebound lead, confidence ramp or support window')


def tracked_rebound_reference(nominal_normal, v76_normal, edge, forecast_error, velocity,
                              clearance, flight_time, eligible, actual_axis, world_omega,
                              step_dt, attitude_tau, attitude_delay_ticks, radius, gravity,
                              tilt_limit_deg, signed_config, config):
    """Keep the V76 reference outside the guarded, actuator-led retry window.

    All direction and velocity projections are in world coordinates. The signed
    overshoot is the forecast foot minus the retained waypoint, on the travelled
    route axis. This reference changes neither the motor commands nor the foot
    acceptance rule. The support-axis prediction is a short constant-rate
    extrapolation for the training objective.
    """
    edge_length = edge.norm(dim=-1)
    axis = edge / edge_length.clamp_min(1.e-6)[:, None]
    overshoot = (forecast_error * axis).sum(-1)
    incoming = (velocity[:, :2] * axis).sum(-1)
    down = -velocity[:, 2]
    height = clearance.clamp_min(0.)
    impact_speed = (down.square() + 2. * gravity * height).sqrt()
    touch_time = 2. * height / (impact_speed + down.abs()).clamp_min(1.e-6)

    ramp = config['lead_ramp_s']
    lead = attitude_delay_ticks * step_dt + config['settling_time_constants'] * attitude_tau
    lead = lead.clamp(0., config['max_preview_s'] - ramp)
    time_gate = ((lead + ramp - touch_time) / ramp).clamp(0., 1.)
    early = ((touch_time - config['early_guard_start_s']) /
             (config['early_guard_full_s'] - config['early_guard_start_s'])).clamp(0., 1.)
    guard = radius + early * config['early_guard_extra_m']
    miss_gate = ((overshoot - guard) / signed_config['full_excess_m']).clamp(0., 1.)
    inward_gate = ((-incoming - signed_config['inward_start_m_s']) /
                   (signed_config['inward_full_m_s'] - signed_config['inward_start_m_s'])).clamp(0., 1.)
    allowed = eligible & (edge_length > .1) & (down > 0.) & (clearance > 0.)
    weight = torch.where(allowed, time_gate * miss_gate * inward_gate, 0.)

    nominal_along = (nominal_normal[:, :2] * axis).sum(-1)
    baseline_along = (v76_normal[:, :2] * axis).sum(-1)
    transverse = v76_normal[:, :2] - baseline_along[:, None] * axis
    transverse_squared = transverse.square().sum(-1)
    desired_velocity = -overshoot / flight_time.clamp_min(.1)
    impulse = signed_config['vertical_impulse_ratio'] * impact_speed
    ratio = (desired_velocity - incoming) / impulse.clamp_min(1.e-6)
    desired_along = ratio * torch.rsqrt(1. + ratio.square()) * (1. - transverse_squared).clamp_min(0.).sqrt()
    tilt_limit = math.sin(math.radians(tilt_limit_deg))
    along_limit = (tilt_limit ** 2 - transverse_squared).clamp_min(0.).sqrt()
    desired_along = desired_along.clamp(min=-along_limit, max=along_limit)
    reduction = (nominal_along - desired_along).clamp_min(0.) * weight
    active = reduction > 0.
    proposed = nominal_along - reduction
    use_tracking = active & (proposed < baseline_along)
    along = torch.where(use_tracking, proposed, baseline_along)
    horizontal = transverse + along[:, None] * axis
    vertical = (1. - horizontal.square().sum(-1)).clamp_min(0.).sqrt()
    reference = torch.where(use_tracking[:, None], torch.cat((horizontal, vertical[:, None]), dim=-1), v76_normal)

    horizon = (touch_time + config['spring_centroid_s']).clamp_max(config['axis_prediction_limit_s'])
    axis_rate = torch.linalg.cross(world_omega, actual_axis)
    predicted_axis = normalize(actual_axis + horizon[:, None] * axis_rate, dim=-1)
    support_along = (predicted_axis[:, :2] * axis).sum(-1)
    return reference, dict(
        active=active, weight=torch.where(active, weight, 0.), axis=axis, reference_along=along,
        predicted_axis_xy=predicted_axis[:, :2], projected_support_along=support_along,
        time_to_touch_s=touch_time, overshoot_m=overshoot,
    )
