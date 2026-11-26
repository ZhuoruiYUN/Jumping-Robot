"""Pre-apex reference for V73 policy training with measured actuator timing."""

import math

import torch

from .yaw_thrust_feedback import _thrust_from_duty


VERSION = "imu_link_powered_flight_guidance_v1"
NOMINAL_MASS_KG = 0.1556


def validate(config):
    limits = {
        "minimum_clearance_m": (.0, .10),
        "minimum_vz_m_s": (.1, .4),
        "full_vz_m_s": (.3, 1.),
        "clock_ramp_s": (.02, .25),
        "maximum_clock_change_s": (.02, .25),
        "height_deficit_deadband_m": (.0, .04),
        "collective_cap_n": (.0, .20),
        "collective_cost_weight": (.0, 10.),
        "motor_duty_limit": (.80, 1.),
        "yaw_fade_start_deg": (2., 8.),
        "yaw_fade_end_deg": (5., 10.),
        "roll_pitch_fade_start_deg": (8., 22.),
        "roll_pitch_fade_end_deg": (18., 35.),
    }
    coefficients = config.get("predictor_coefficients", ())
    if (
        config.get("version") != VERSION
        or len(coefficients) != 5
        or not all(math.isfinite(x) for x in coefficients)
        or not 5. < coefficients[-1] < 12.
        or not all(math.isfinite(config[k]) and lo <= config[k] <= hi
                   for k, (lo, hi) in limits.items())
        or config["minimum_vz_m_s"] >= config["full_vz_m_s"]
        or config["yaw_fade_start_deg"] >= config["yaw_fade_end_deg"]
        or config["roll_pitch_fade_start_deg"] >= config["roll_pitch_fade_end_deg"]
    ):
        raise ValueError("Invalid imu_link powered flight guidance")


def predict_powered_flight(height, vz, thrust_z_per_mass, coefficients):
    """Causal fit from other imu_link jumps, in SI units, for rising flight."""
    bias, height_gain, speed_gain, thrust_gain, descent_accel = coefficients
    vz = vz.clamp_min(0.)
    energy_gain = (bias + height_gain * (height - .60) + speed_gain * (vz - 1.25)
                   + thrust_gain * (thrust_z_per_mass.clamp(0., 10.) - 4.)).clamp(.046, .22)
    rise = energy_gain * vz.square()
    apex_time = 2. * energy_gain * vz
    apex = height + rise
    landing_time = apex_time + (2. * (apex - .283).clamp_min(0.) / descent_accel).sqrt()
    return apex, apex_time, landing_time


def ascent_gate(phase, apex_seen, clearance, vz, elapsed, config):
    """Gracefully enter free ascent and leave before the velocity sign change."""
    active = ((phase == 2) | (phase == 3)) & ~apex_seen & (clearance >= config["minimum_clearance_m"])
    rise = ((vz - config["minimum_vz_m_s"])
            / (config["full_vz_m_s"] - config["minimum_vz_m_s"])).clamp(0., 1.)
    takeoff_ramp = (elapsed / config["clock_ramp_s"]).clamp(0., 1.)
    return active * rise * takeoff_ramp


def bounded_clock(previous, estimate, gate, minimum, maximum, bound):
    delta = (estimate.clamp(minimum, maximum) - previous).clamp(-bound, bound)
    return previous + gate * delta


def collective_reference(apex, target_apex, height, gate, up, yaw_now, yaw_forecast,
                         duty_previous, duty_applied, thrust, thrust_scale, voltage_gain, config):
    """Teaching reference in force space. Equal force per motor spends available reserve only."""
    yaw = torch.maximum(yaw_now.abs(), yaw_forecast.abs()).rad2deg()
    yaw_margin = ((config["yaw_fade_end_deg"] - yaw)
                  / (config["yaw_fade_end_deg"] - config["yaw_fade_start_deg"])).clamp(0., 1.)
    tilt = up.clamp(-1., 1.).acos().rad2deg()
    tilt_margin = ((config["roll_pitch_fade_end_deg"] - tilt)
                   / (config["roll_pitch_fade_end_deg"] - config["roll_pitch_fade_start_deg"])).clamp(0., 1.)
    duty_limit = config["motor_duty_limit"] * voltage_gain
    commanded = (duty_previous * voltage_gain[:, None]).clamp(0., 1.)
    occupied_duty = torch.maximum(duty_applied, commanded).amax(dim=-1)
    spare = 4. * thrust_scale * (_thrust_from_duty(duty_limit) - _thrust_from_duty(occupied_duty)).clamp_min(0.)
    deficit = (target_apex - apex - config["height_deficit_deadband_m"]).clamp_min(0.)
    distance = (apex - height).clamp_min(.07)
    requested = (NOMINAL_MASS_KG * 9.81 * deficit / (distance * up.clamp_min(.70))).clamp_max(
        config["collective_cap_n"])
    assist = gate * yaw_margin * tilt_margin * torch.minimum(spare, requested)
    return thrust + assist, assist, spare, deficit


def collective_cost(command, target, assist, moving, dt, config):
    norm = max(config["collective_cap_n"], 1.e-6)
    short = ((target - command) / norm).clamp(0., 1.)
    # The force is not injected. PPO learns only from its own allocated action.
    return -config["collective_cost_weight"] * dt * torch.where(moving & (assist > 0.), short.square(), 0.)


def sample_ascent(env, config, clearance):
    data = env._robot.data
    event = env._foot_event
    height = data.root_link_pos_w[:, 2] - env._terrain.env_origins[:, 2]
    vz = data.root_link_lin_vel_w[:, 2]
    up = 1. - 2. * data.root_quat_w[:, 1:3].square().sum(-1)
    thrust = env._thrust[:, 0, 2]
    mass = NOMINAL_MASS_KG * env._imu_link_plant.mass_scale
    apex, apex_time, landing_time = predict_powered_flight(
        height, vz, thrust * up / mass, config["predictor_coefficients"])
    gate = ascent_gate(event.phase, event.apex_seen, clearance, vz, event.air_steps * env.step_dt, config)
    return height, up, thrust, apex, apex_time, landing_time, gate
