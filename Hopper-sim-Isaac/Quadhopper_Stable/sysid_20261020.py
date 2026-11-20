"""Opt-in 2026-10-20 effective airborne plant; local fit, not a full-jump calibration.

Numbers are transcribed from /home/terry/Downloads/参数辨识交接文档.md.
The referenced parameters.json was not present alongside the supplied document.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class IdentifiedPlant20261020:
    total_mass_kg: float = 0.155
    inertia_kg_m2: tuple[float, float, float] = (
        0.001016642785, 0.0009759686404, 0.00055,
    )
    motor_arm_component_m: float = 0.0813
    motor_time_constant_s: float = 0.080
    voltage_reference_v: float = 3.9812498092651367
    thrust_quadratic_n: float = -0.3647607454907179
    thrust_linear_n: float = 0.9884448461365533
    thrust_constant_n: float = 0.0028762422544365922
    yaw_torque_per_thrust_m: float = 0.011069143101433988
    yaw_acceleration_nm_s: float = 0.001083365948332391
    calibrated_state_min: float = 0.2437326064
    calibrated_state_max: float = 0.7331279315


NOMINAL = IdentifiedPlant20261020()
YAW_SIGNS = (-1.0, 1.0, -1.0, 1.0)


def actuator_step_and_wrench(state, pwm_compare, voltage_v, dt_s, plant=NOMINAL):
    """Step w and return interval-mean per-motor force and body torque.

    PWM is the actual 0..999 compare divided by 1000. The 20 ms fitted
    command/Vicon offset is deliberately not added as motor delay. Values
    outside the identified thrust interval are returned with an explicit mask;
    their forces are extrapolations and cannot validate jump dynamics.
    """
    if state.shape[-1] != 4 or pwm_compare.shape != state.shape or dt_s <= 0:
        raise ValueError('four motor states/PWM channels and positive dt required')
    if not torch.isfinite(state).all() or not torch.isfinite(pwm_compare).all():
        raise ValueError('motor state and PWM must be finite')
    if torch.any((pwm_compare < 0) | (pwm_compare > 999)):
        raise ValueError('PWM compare must be in [0,999]')
    voltage = torch.as_tensor(voltage_v, device=state.device, dtype=state.dtype)
    while voltage.ndim < state.ndim:
        voltage = voltage.unsqueeze(-1)
    if not torch.isfinite(voltage).all() or torch.any(voltage <= 0):
        raise ValueError('voltage must be finite and positive')
    target = (pwm_compare / 1000.0) * voltage / plant.voltage_reference_v
    h = dt_s / plant.motor_time_constant_s
    decay = math.exp(-h)
    next_state = target + (state - target) * decay
    d = state - target
    e1 = -math.expm1(-h) / h
    e2 = -math.expm1(-2.0 * h) / (2.0 * h)
    mean_state = target + d * e1
    mean_square = target.square() + 2.0 * target * d * e1 + d.square() * e2
    forces = (plant.thrust_quadratic_n * mean_square
              + plant.thrust_linear_n * mean_state + plant.thrust_constant_n)
    signs = state.new_tensor(YAW_SIGNS)
    f0, f1, f2, f3 = forces.unbind(dim=-1)
    arm = plant.motor_arm_component_m
    yaw_aero = plant.yaw_torque_per_thrust_m * (forces * signs).sum(dim=-1)
    yaw_acc = plant.yaw_acceleration_nm_s * (
        ((next_state - state) / dt_s) * signs
    ).sum(dim=-1)
    torque = torch.stack((
        arm * (f0 - f1 - f2 + f3),
        arm * (f0 + f1 - f2 - f3),
        yaw_aero + yaw_acc,
    ), dim=-1)
    outside = ((torch.minimum(state, next_state) < plant.calibrated_state_min)
               | (torch.maximum(state, next_state) > plant.calibrated_state_max))
    return next_state, forces, torque, outside
