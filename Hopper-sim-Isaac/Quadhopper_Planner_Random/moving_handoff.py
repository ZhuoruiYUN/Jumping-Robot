"""Opt-in v11 task shaping. No actuator or trajectory controller."""
import math


SETTINGS_V11 = {
    'moving_contact_recovery_all_hops': True,
    'moving_long_recovery_window_s': 0.12,
    'moving_long_recovery_angle_deadband_rad': math.radians(10.0),
    'moving_long_recovery_rate_deadband_rad_s': 0.8,
    'moving_long_recovery_angle_penalty_scale': -20.0,
    'moving_long_recovery_rate_penalty_scale': -8.0,
    'moving_launch_stability_window_s': 0.20,
    'moving_launch_rate_deadband_rad_s': 0.8,
    'moving_launch_rate_penalty_scale': -8.0,
}


def apply_settings(cfg):
    for name, value in SETTINGS_V11.items():
        setattr(cfg, name, value)


def moving_contact_event(touchdown, final_touchdown, stationary, initial_drop,
                         *, all_hops=False):
    event = touchdown if all_hops else final_touchdown
    return event & ~stationary & ~initial_drop


def moving_launch_mask(spring_position, spring_velocity, cycle_active,
                       launch_remaining, initial_drop, stationary, touchdown):
    contact = spring_position > 0.002
    late_stance = contact & ~cycle_active & (spring_velocity < 0.0)
    early_flight = ~contact & cycle_active & (launch_remaining > 0)
    return ~stationary & ~initial_drop & ~touchdown & (late_stance | early_flight)


def moving_rate_cost(rates, deadband):
    """Bound the squared excess body roll/pitch rate; angles are not targets."""
    if rates.shape[-1] != 2 or deadband < 0:
        raise ValueError('expected roll/pitch rates and a non-negative deadband')
    return (rates.abs() - deadband).clamp(0.0, 3.0).square().sum(-1)
