"""Pure, reward-only helpers for dangerous moving-landings.

No desired XY flight path or motor command is generated here. Quaternion is
Isaac wxyz; angular rates are body-frame rad/s; world Z is the flat ground up.
"""
import math
import torch


SETTINGS_V13 = {
    'approach_clearance_m': 0.55,
    'approach_descent_speed_m_s': 0.10,
    'prediction_horizon_max_s': 0.30,
    'prediction_rate_limit_rad_s': 4.0,
    'danger_tilt_rad': math.radians(30),
    'danger_width_rad': math.radians(10),
    'approach_risk_scale': -12.0,
    'touchdown_danger_scale': -12.0,
    'touchdown_rate_deadband_rad_s': 1.2,
    'touchdown_rate_scale': -12.0,
    'low_altitude_root_z_max_m': 0.35,
    'low_altitude_vz_max_m_s': 0.20,
    'low_altitude_min_tilt_rad': math.radians(25),
}


def predicted_touchdown_tilt(quat_wxyz, angular_velocity_b, clearance, vz,
                             settings=SETTINGS_V13):
    """First-order total body-Z tilt forecast over bounded time-to-ground.

    Uses d(R33)/dt from the measured body angular velocity, not Euler-angle
    addition (which is singular near the two observed ~90-degree failures).
    The horizon is at most 0.30 s, so this is a risk indicator, not a planner.
    """
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError('expected (N,4) wxyz quaternions')
    if angular_velocity_b.shape != (quat_wxyz.shape[0], 3):
        raise ValueError('expected matching (N,3) body angular rates')
    if clearance.shape != vz.shape or clearance.shape != (quat_wxyz.shape[0],):
        raise ValueError('expected matching (N,) clearance and vertical speed')
    qw, qx, qy, qz = quat_wxyz.unbind(-1)
    cos_tilt = (1.0 - 2.0 * (qx.square() + qy.square())).clamp(-1.0, 1.0)
    tilt = torch.acos(cos_tilt)
    # Third row of body->world rotation: world up expressed in body coordinates.
    up_b_x = 2.0 * (qx * qz - qw * qy)
    up_b_y = 2.0 * (qy * qz + qw * qx)
    cos_tilt_dot = angular_velocity_b[:, 1] * up_b_x - angular_velocity_b[:, 0] * up_b_y
    sin_tilt = torch.sqrt((1.0 - cos_tilt.square()).clamp_min(0.0)).clamp_min(0.10)
    tilt_rate = (-cos_tilt_dot / sin_tilt).clamp(
        -settings['prediction_rate_limit_rad_s'], settings['prediction_rate_limit_rad_s'])
    horizon = (clearance.clamp_min(0.0) / (-vz).clamp_min(0.20)).clamp(
        0.0, settings['prediction_horizon_max_s'])
    forecast = (tilt + horizon * tilt_rate).clamp(0.0, math.pi)
    return tilt, forecast, horizon


def danger_cost(tilt, forecast, settings=SETTINGS_V13):
    """Zero inside a 30-degree cone; bounded risk outside it."""
    return (((torch.maximum(tilt, forecast) - settings['danger_tilt_rad']).clamp_min(0.0)
             / settings['danger_width_rad']).clamp_max(4.0)).square()


def approach_mask(moving, cycle_active, touchdown, initial_drop, recovery_remaining,
                  clearance, vz,
                  settings=SETTINGS_V13):
    return (moving & cycle_active & ~touchdown & ~initial_drop
            & (recovery_remaining == 0)
            & (clearance <= settings['approach_clearance_m'])
            & (vz < -settings['approach_descent_speed_m_s']))


def touchdown_rate_cost(body_rates, settings=SETTINGS_V13):
    if body_rates.ndim != 2 or body_rates.shape[-1] != 3:
        raise ValueError('expected (N,3) body angular rates')
    return ((body_rates[:, :2].abs() - settings['touchdown_rate_deadband_rad_s'])
            .clamp(0, 3).square().sum(-1))


def low_altitude_recovery_mask(remaining, v12_near_ground, initial_drop,
                               root_z, vz, clearance, tilt,
                               settings=SETTINGS_V13):
    # Only samples omitted by the v12 near-foot gate. A tipped foot can rise
    # above 3.5 cm while the root is still almost on the floor and falling.
    return ((remaining > 0) & ~v12_near_ground & ~initial_drop
            & (clearance > 0.035)
            & (root_z < settings['low_altitude_root_z_max_m'])
            & (vz < settings['low_altitude_vz_max_m_s'])
            & (tilt > settings['low_altitude_min_tilt_rad']))
