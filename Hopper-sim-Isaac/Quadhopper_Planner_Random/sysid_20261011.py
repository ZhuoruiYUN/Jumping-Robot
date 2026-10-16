"""2026-10-11 mounted SysID plant parameters and offline equations.

These are identified *plant* parameters, not flight-controller constants.
Stage2 v37 uses these parameters in a spring-leg proxy training plant. It
does not implement the unprovided FC CTBR/LQR controller or rigid leg.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MountedRigidPlant20261011:
    total_mass_kg: float = 0.194
    inertia_kg_m2: tuple[float, float, float] = (0.0011, 0.0012, 0.00065)
    motor_arm_component_m: float = 0.230 / (2.0 * math.sqrt(2.0))
    motor_time_constant_s: float = 0.07069769079261506
    voltage_reference_v: float = 3.7
    voltage_exponent: float = 1.44630397739629
    thrust_quadratic_n: float = -0.34246762310623274
    thrust_linear_n: float = 0.8917171678850943
    yaw_torque_per_thrust_m: float = 0.010
    yaw_ratio_validated: bool = False


NOMINAL = MountedRigidPlant20261011()


def motor_state_step(state, pwm_compare, loaded_voltage_v, dt_s,
                     plant=NOMINAL):
    """Exact zero-order-hold step of the identified motor-state ODE.

    pwm_compare is the actual FC integer compare (0..999), not policy action.
    The state is an effective thrust state, not measured RPM. This lag must
    not be counted again as a communication delay.
    """
    duty = torch.as_tensor(pwm_compare, device=state.device, dtype=state.dtype) / 999.0
    voltage = torch.as_tensor(loaded_voltage_v, device=state.device, dtype=state.dtype)
    if duty.shape != state.shape or dt_s <= 0.0:
        raise ValueError('motor state and PWM shapes must match and dt must be positive')
    if torch.any((duty < 0.0) | (duty > 1.0)) or torch.any(voltage <= 0.0):
        raise ValueError('FC compare must be 0..999 and loaded voltage must be positive')
    while voltage.ndim < duty.ndim:
        voltage = voltage.unsqueeze(-1)
    target = duty * (voltage / plant.voltage_reference_v).pow(plant.voltage_exponent)
    alpha = 1.0 - math.exp(-dt_s / plant.motor_time_constant_s)
    return state + alpha * (target - state)


def motor_forces_n(state, plant=NOMINAL):
    """Nominal mean-channel thrust; low-PWM dead zone remains unidentified."""
    return (plant.thrust_quadratic_n * state.square()
            + plant.thrust_linear_n * state).clamp_min(0.0)


def body_wrench_from_forces(forces_n, plant=NOMINAL):
    """Return collective N and body torque N*m for F1,F2,F3,F4 order."""
    if forces_n.shape[-1] != 4:
        raise ValueError('last dimension must be four FC motor channels')
    f1, f2, f3, f4 = forces_n.unbind(dim=-1)
    arm = plant.motor_arm_component_m
    collective = f1 + f2 + f3 + f4
    torque = torch.stack((
        arm * (f1 + f4 - f2 - f3),
        arm * (f1 + f2 - f3 - f4),
        plant.yaw_torque_per_thrust_m * (-f1 + f2 - f3 + f4),
    ), dim=-1)
    return collective, torque
