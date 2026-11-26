"""Trim unwanted along-route spring impulse on an imminent retained overshoot."""

import math

import torch


def validate_signed_retry_rebound(config):
    keys = (
        'start_before_touch_s', 'full_before_touch_s', 'full_excess_m',
        'inward_start_m_s', 'inward_full_m_s', 'vertical_impulse_ratio',
        'max_reduction_deg',
    )
    if any(not math.isfinite(config[key]) or config[key] <= 0. for key in keys):
        raise ValueError('Signed retry rebound requires positive, finite bounds')
    if not (config['full_before_touch_s'] < config['start_before_touch_s'] <= .3
            and config['inward_start_m_s'] < config['inward_full_m_s']
            and 1. <= config['vertical_impulse_ratio'] <= 2.5
            and config['max_reduction_deg'] <= 10.):
        raise ValueError('Invalid signed retry rebound window, ramp or impulse')


def signed_retry_rebound(normal, travelled_edge, forecast_error, velocity, clearance,
                         flight_time, eligible, radius, gravity, tilt_limit_deg, config):
    """Reduce the descent reference's forward lean when the foot will overshoot.

    The travelled edge remains the last confirmed route edge. The signed forecast
    is predicted foot minus current goal. Contact, target retention and rewards
    are still settled using the measured foot. This gives no motor commands.
    """
    edge_length = travelled_edge.norm(dim=-1)
    axis = travelled_edge / edge_length.clamp_min(1.e-6)[:, None]
    overshoot = (forecast_error * axis).sum(-1)
    incoming = (velocity[:, :2] * axis).sum(-1)
    incoming_down = -velocity[:, 2]
    height = clearance.clamp_min(0.)
    impact_speed = (incoming_down.square() + 2. * gravity * height).sqrt()
    touch_time = 2. * height / (impact_speed + incoming_down.abs()).clamp_min(1.e-6)

    miss_gate = ((overshoot - radius) / config['full_excess_m']).clamp(0., 1.)
    inward_gate = ((-incoming - config['inward_start_m_s']) / (
        config['inward_full_m_s'] - config['inward_start_m_s'])).clamp(0., 1.)
    close_gate = ((config['start_before_touch_s'] - touch_time) / (
        config['start_before_touch_s'] - config['full_before_touch_s'])).clamp(0., 1.)
    allowed = eligible & (edge_length > .1) & (incoming_down > 0.) & (clearance > 0.)
    weight = torch.where(allowed, miss_gate * inward_gate * close_gate, 0.)

    # Predict the launch velocity needed if this landing retains the goal. Use
    # only the retained point, so an advance preview cannot add forward impulse.
    desired = -overshoot / flight_time.clamp_min(.1)
    vertical_impulse = config['vertical_impulse_ratio'] * impact_speed
    ratio = (desired - incoming) / vertical_impulse.clamp_min(1.e-6)

    along = (normal[:, :2] * axis).sum(-1)
    transverse = normal[:, :2] - along[:, None] * axis
    transverse_squared = transverse.square().sum(-1)
    # Solve along / vertical = horizontal impulse / vertical impulse while
    # preserving the inherited transverse reference and unit leg axis.
    desired_along = ratio * torch.rsqrt(1. + ratio.square()) * (1. - transverse_squared).clamp_min(0.).sqrt()
    limit = math.sin(math.radians(tilt_limit_deg))
    along_limit = (limit ** 2 - transverse_squared).clamp_min(0.).sqrt()
    desired_along = desired_along.clamp(min=-along_limit, max=along_limit)
    cap = math.radians(config['max_reduction_deg'])
    minimum_along = along * math.cos(cap) - normal[:, 2] * math.sin(cap)
    reduction = torch.minimum((along - desired_along).clamp_min(0.),
                              (along - minimum_along).clamp_min(0.)) * weight
    adjusted = along - reduction
    active = reduction > 0.
    horizontal = transverse + adjusted[:, None] * axis
    vertical = (1. - horizontal.square().sum(-1)).clamp_min(0.).sqrt()
    corrected = torch.cat((horizontal, vertical[:, None]), dim=-1)
    reference = torch.where(active[:, None], corrected, normal)
    return reference, dict(active=active, weight=weight, overshoot_m=overshoot,
                           incoming_along_m_s=incoming, target_along_m_s=desired,
                           approach_along=along, corrected_along=adjusted)
