"""Training-only strength curriculum for already sampled hardware variation."""
import torch


def severity_at_step(step, steps_per_iteration, stage_iterations):
    """Use 0.50, 0.75, then 1.00 for successive PPO update blocks."""
    iteration = float(step) / float(steps_per_iteration)
    return min(1.0, 0.50 + 0.25 * int(iteration // stage_iterations))


def scale_hard_reset(env, env_ids, severity):
    """Contract one hard episode's sampled offsets toward calibrated values."""
    if severity >= 1.0 or len(env_ids) == 0:
        return
    hard_ids = env_ids[env._hardware_hard_mask[env_ids]]
    if len(hard_ids) == 0:
        return

    for name, nominal in (
        ('dr_mass_multi', 1.0),
        ('dr_inertia_multi', 1.0),
        ('dr_tm', env.cfg.play_motor_time_constant),
        ('dr_thrust_scale', env.cfg.play_thrust_scale),
        ('dr_thrust_curve_shape', env.cfg.play_thrust_curve_shape),
        ('_hardware_global_thrust_scale', 1.0),
        ('_hardware_yaw_effectiveness', 1.0),
        ('_hardware_yaw_bias_torque_nm', 0.0),
        ('_hardware_yaw_reaction_scale', 1.0),
        ('_hardware_spring_stiffness_scale', 1.0),
        ('_hardware_spring_damping_scale', 1.0),
        ('_hardware_body_com_offset_b', 0.0),
    ):
        value = getattr(env, name)
        value[hard_ids] = nominal + severity * (value[hard_ids] - nominal)

    rp = env._hardware_rp_effectiveness
    identity = torch.eye(2, device=rp.device, dtype=rp.dtype)
    rp[hard_ids] = identity + severity * (rp[hard_ids] - identity)

    if env.cfg.comprehensive_hardware_randomization:
        env._write_body_com_offsets(hard_ids, env._hardware_body_com_offset_b[hard_ids])
        env._robot.write_joint_stiffness_to_sim(
            env._default_spring_stiffness[hard_ids]
            * env._hardware_spring_stiffness_scale[hard_ids],
            joint_ids=[env._spring_joint_id], env_ids=hard_ids,
        )
        env._robot.write_joint_damping_to_sim(
            env._default_spring_damping[hard_ids]
            * env._hardware_spring_damping_scale[hard_ids],
            joint_ids=[env._spring_joint_id], env_ids=hard_ids,
        )
