"""Opt-in training profiles; no change to the nominal physical plant.

v9 uses the same direct four-motor interface, but a larger *shaped* yaw
envelope. It is contract v15, NOT compatible with the pinned v8/v12 launcher.
"""

V9_OVERRIDES = {
    'launch_stability_window_s': 0.15,
    'launch_rate_deadband_rad_s': 0.18,
    'launch_roll_rate_penalty_scale': -12.0,
    'launch_pitch_rate_penalty_scale': -24.0,
    'preapex_yaw_braking_vz_mps': 1.0,
    'preapex_yaw_rate_deadband_rad_s': 0.20,
    'preapex_yaw_rate_penalty_scale': -30.0,
    'deployment_max_airborne_yaw_diagonal_pwm_diff': 220.0,
    # Training uncertainty, not a measured identification result. Nominal
    # play_motor_time_constant remains 0.0674 s; rise/fall still share one tc.
    'train_motor_time_constant_max': 0.100,
}


# Start again from the real-flight-better v8/v12 policy.  v9's larger yaw
# command envelope and blanket pitch rewards did not survive real flight.
# Keep the deployment actuator contract unchanged and expose a training-only
# early-flight pitch disturbance plus a soft cost on the command that actually
# reaches the actuator model.  These are hypotheses to validate, not a
# calibrated replacement for the real motor model.
V10_OVERRIDES = {
    'launch_stability_window_s': 0.16,
    'launch_pitch_command_deadband_pwm': 150.0,
    'launch_pitch_command_penalty_scale': -40.0,
    'train_launch_pitch_impulse_max_nm': 0.006,
}


V11_OVERRIDES = {
    **V10_OVERRIDES,
    # Same v12 motor envelope as v10.  These terms score the accumulated yaw
    # state at physical events rather than rewarding a large diagonal command.
    'stationary_yaw_event_penalty_scale': -260.0,
    'stationary_yaw_hop_change_penalty_scale': -90.0,
    'preapex_yaw_braking_vz_mps': 1.0,
    'preapex_yaw_rate_deadband_rad_s': 0.12,
    'preapex_yaw_rate_penalty_scale': -45.0,
}


# Separate candidate initialized from v8, not a continuation of v11's high
# value-loss run.  Score the state handed to the *next* jump and expose short,
# independent per-hop body torques in randomized training only.  These bounds
# are a robustness hypothesis, not identified hardware parameters.
V12_OVERRIDES = {
    'launch_stability_window_s': 0.10,
    'train_launch_wrench_probability': 0.8,
    'train_launch_roll_impulse_max_nm': 0.012,
    'train_launch_pitch_wrench_max_nm': 0.012,
    'train_launch_yaw_impulse_max_nm': 0.003,
    'stationary_touchdown_angle_penalty_scale': -8.0,
    'stationary_touchdown_rate_penalty_scale': -3.0,
    'stationary_yaw_event_penalty_scale': -20.0,
    'preapex_yaw_braking_vz_mps': 1.0,
    'preapex_yaw_rate_deadband_rad_s': 0.18,
    'preapex_yaw_rate_penalty_scale': -12.0,
}


# Height-robust refinement of v12.  The v12 qualification had no simulated
# deaths and passed all nominal attitude gates, but only 84.7% of hard samples
# reached 0.95 m.  Center the existing measured ballistic-apex shaping more
# strongly at 1 m instead of adding a one-sided collective reward: the latter
# would fix low-thrust samples by making high-thrust samples overshoot further.
V13_OVERRIDES = {
    **V12_OVERRIDES,
    'stationary_apex_progress_scale': 22.0,
    'stationary_predicted_apex_width_m': 0.10,
    'stationary_predicted_apex_reward_scale': 36.0,
    'stationary_apex_error_penalty_scale': -220.0,
    'stationary_apex_shortfall_penalty_scale': -360.0,
}


# v14 responds to the v13 real-flight diagnosis.  It keeps the v13 height and
# actuator contract, but changes *training-only task uncertainty*: the launch
# wrench evolves across hops rather than being an independent pulse.  The roll
# axis gets the stronger early-flight cost because the measured 203839 log
# links early roll rate most strongly to both current and next-hop tilt.
V14_OVERRIDES = {
    **V13_OVERRIDES,
    'launch_stability_window_s': 0.16,
    'launch_rate_deadband_rad_s': 0.18,
    'launch_roll_rate_penalty_scale': -24.0,
    'launch_pitch_rate_penalty_scale': -12.0,
    'train_launch_wrench_probability': 1.0,
    'train_launch_roll_impulse_max_nm': 0.012,
    'train_launch_pitch_wrench_max_nm': 0.010,
    'train_launch_yaw_impulse_max_nm': 0.003,
    'train_launch_wrench_persistence': 0.70,
    'stationary_touchdown_angle_penalty_scale': -12.0,
    'stationary_touchdown_rate_penalty_scale': -6.0,
}


# v15 targets the 20260919 hardware result: the instability begins during the
# touchdown-to-liftoff handoff, where the raw actor asks for near-saturated
# differential commands.  The envelope is an explicit deployment contract,
# not a deployment-only safety hack: the same collective-preserving 450-PWM
# spread cap is inside the training motor path and its executed history.
# Yaw uncertainty is increased because hardware yaw P90 exceeded v14's hard
# simulation envelope.  The stable USD, motor order, rate and four-action
# interface remain unchanged.
V15_OVERRIDES = {
    **V14_OVERRIDES,
    'deployment_launch_recovery_window_s': 0.16,
    'deployment_launch_recovery_spread_limit_pwm': 450.0,
    'launch_recovery_raw_spread_penalty_scale': -12.0,
    'train_launch_wrench_persistence': 0.80,
    'train_launch_yaw_impulse_max_nm': 0.006,
    'stationary_yaw_event_penalty_scale': -80.0,
    'stationary_yaw_hop_change_penalty_scale': -25.0,
    'preapex_yaw_rate_penalty_scale': -18.0,
}


# v15.1 is a controlled ablation from v14.  v15 simultaneously tightened the
# actuator cap, enlarged yaw uncertainty, and amplified event costs, so its
# robust failure could not identify the cause.  Keep v14's learned yaw/task
# scales, suppress only the first 80 ms of the contact handoff, and restore
# differential authority during the second 80 ms.
V15_1_OVERRIDES = {
    **V14_OVERRIDES,
    'deployment_launch_recovery_window_s': 0.16,
    'deployment_launch_recovery_hold_s': 0.08,
    'deployment_launch_recovery_spread_limit_pwm': 450.0,
    'deployment_launch_recovery_spread_limit_end_pwm': 600.0,
    'launch_recovery_raw_spread_penalty_scale': -4.0,
}


# v16 is a weights-only refinement of the accepted v15.1 controller.  The
# 20260919 real run lasted 915 hops, so this is not a wholesale plant rewrite:
# preserve v17's 80 ms / 450 PWM then 80 ms / 600 PWM actuator contract and
# add only the missing *persistent* cross-hop uncertainty.  In particular,
# rotor reaction-torque mismatch is separate from a common yaw bias, and a
# short contact-only wrench exposes the roll/pitch state handed to liftoff.
# All costs remain smooth per-step/event costs at v14/v15.1 magnitude; do not
# reintroduce v15's oversized yaw-event return spike.
V16_OVERRIDES = {
    **V15_1_OVERRIDES,
    'train_yaw_reaction_imbalance_max': 0.14,
    'train_launch_wrench_persistence': 0.80,
    'train_launch_roll_impulse_max_nm': 0.014,
    'train_launch_pitch_wrench_max_nm': 0.012,
    'train_launch_yaw_impulse_max_nm': 0.004,
    'train_touchdown_wrench_probability': 1.0,
    'train_touchdown_roll_wrench_max_nm': 0.008,
    'train_touchdown_pitch_wrench_max_nm': 0.006,
    'train_touchdown_yaw_wrench_max_nm': 0.002,
    'train_touchdown_wrench_window_s': 0.06,
    'train_touchdown_wrench_persistence': 0.75,
    'stationary_contact_recovery_window_s': 0.08,
    'stationary_contact_recovery_spread_deadband': 0.38,
    'stationary_contact_recovery_spread_penalty_scale': -2.0,
    'stationary_contact_recovery_delta_deadband': 0.20,
    'stationary_contact_recovery_delta_penalty_scale': -1.0,
    'launch_roll_rate_penalty_scale': -30.0,
    'launch_pitch_rate_penalty_scale': -15.0,
    'stationary_touchdown_angle_penalty_scale': -14.0,
    'stationary_touchdown_rate_penalty_scale': -8.0,
}


# v17 preserves v15.1's actuator contract but targets stationary corridor
# behavior: 15 cm success radius, 5 cm zero-hop zone, attitude-first shaping.
# Stage A is nominal-plant behavior shaping; Stages B/C add progressive hardware
# randomization only after Stage A passes qualification.
V17_OVERRIDES = {
    **V15_1_OVERRIDES,
    # Stationary corridor parameters will be handled by spatial_tracking_env
    # via --landing-precision-width and target generation logic
    # Re-enable height shaping from v13/v14 (needed because spatial_tracking_env
    # defaults disable these)
    'stationary_apex_progress_scale': 22.0,
    'stationary_predicted_apex_width_m': 0.10,
    'stationary_predicted_apex_reward_scale': 36.0,
    'stationary_apex_error_penalty_scale': -220.0,
    'stationary_apex_shortfall_penalty_scale': -360.0,
}


# v18 is deliberately not another dynamics or actuator experiment. It reuses
# the accepted v15.1 stability/recovery envelope while a separate launcher
# changes only the alternating short/long waypoint ranges. Keeping the route
# profile distinct makes later checkpoint qualification auditable.
TWO_CIRCLE_V18_OVERRIDES = {
    **V15_1_OVERRIDES,
}


def apply_stability_profile(cfg, name):
    if name == 'baseline':
        return {}
    if name == 'takeoff_v9':
        overrides = V9_OVERRIDES
    elif name == 'launch_robust_v10':
        overrides = V10_OVERRIDES
    elif name == 'yaw_event_v11':
        overrides = V11_OVERRIDES
    elif name == 'touchdown_robust_v12':
        overrides = V12_OVERRIDES
    elif name == 'height_robust_v13':
        overrides = V13_OVERRIDES
    elif name == 'cross_hop_v14':
        overrides = V14_OVERRIDES
    elif name == 'cross_hop_v15':
        overrides = V15_OVERRIDES
    elif name == 'cross_hop_v15_1':
        overrides = V15_1_OVERRIDES
    elif name == 'cross_hop_v16':
        overrides = V16_OVERRIDES
    elif name == 'stationary_corridor_v17':
        overrides = V17_OVERRIDES
    elif name == 'two_circle_v18':
        overrides = TWO_CIRCLE_V18_OVERRIDES
    else:
        raise ValueError(f'Unknown stability profile: {name}')
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return dict(overrides)
