"""Pure helpers for airborne direct-motor stability shaping.

The policy still controls four motors directly.  The action-rate helper removes
the common mode before measuring a change, so a collective take-off command is
never penalized as a side effect of suppressing attitude chatter.
"""
import math

import torch


def stability_phase_masks(
    spring_position, spring_velocity, cycle_active, launch_remaining,
    initial_drop_pending, stationary_command, root_vz, preapex_vz,
):
    """Reward gates only: late stance/first flight samples, and late flight.

    A compressed spring moving toward zero is extending. Exclude the initial
    hand-release drop, compression, and stale launch timers at touchdown.
    """
    contact = spring_position > 0.002
    enabled = stationary_command & ~initial_drop_pending
    late_stance = contact & ~cycle_active & (spring_velocity < 0.0)
    airborne = cycle_active & ~contact
    launch = enabled & (late_stance | (airborne & (launch_remaining > 0)))
    yaw_braking = enabled & airborne & (root_vz <= preapex_vz)
    return launch, yaw_braking


def angular_rate_excess_cost(angular_rate, deadband):
    """Symmetric squared excess rate in rad/s; no motor/collective penalty."""
    if deadband < 0.0:
        raise ValueError('angular-rate deadband must be non-negative')
    return (angular_rate.abs() - deadband).clamp_min(0.0).square()


def touchdown_roll_pitch_cost(angles, rates, angle_deadband, rate_deadband):
    """Bounded event costs for the roll/pitch state passed to the next hop."""
    if angles.shape != rates.shape or angles.shape[-1] != 2:
        raise ValueError('expected matching (..., 2) roll/pitch tensors')
    if angle_deadband < 0 or rate_deadband < 0:
        raise ValueError('touchdown deadbands must be non-negative')
    angle = ((angles.abs() - angle_deadband).clamp_min(0.0)
             / 0.17453292519943295).clamp_max(3.0).square().sum(-1)
    rate = ((rates.abs() - rate_deadband).clamp_min(0.0)
            / 1.0).clamp_max(3.0).square().sum(-1)
    return angle, rate


def launch_pitch_command_cost(target_u: torch.Tensor, deadband_pwm: float):
    """Cost the *executed target* pitch pair difference, not raw actor output.

    The signed quantity matches the physical pitch mixer F1+F2-F3-F4.  It
    removes collective and yaw diagonal commands by construction, and is a
    soft training cost rather than a deployment-time command clamp.
    """
    if target_u.shape[-1] != 4 or not 0.0 <= deadband_pwm <= 1000.0:
        raise ValueError("expected four motors and deadband PWM in [0, 1000]")
    pitch_pwm = 500.0 * (target_u[..., 0] + target_u[..., 1]
                         - target_u[..., 2] - target_u[..., 3])
    cost = ((pitch_pwm.abs() - deadband_pwm).clamp_min(0.0) / 100.0).square()
    return cost, pitch_pwm


def differential_action_rate_cost(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    deadband: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a normalized differential-action delta cost and its RMS size."""
    if actions.shape != previous_actions.shape or actions.shape[-1] != 4:
        raise ValueError("expected matching (..., 4) motor action tensors")
    if not 0.0 <= deadband <= 2.0:
        raise ValueError("action-rate deadband must lie in [0, 2]")
    residual = actions - actions.mean(dim=-1, keepdim=True)
    previous_residual = previous_actions - previous_actions.mean(dim=-1, keepdim=True)
    delta_rms = torch.sqrt(torch.mean(torch.square(residual - previous_residual), dim=-1))
    cost = torch.square((delta_rms - deadband).clamp_min(0.0) / 0.10)
    return cost, delta_rms


def correlated_launch_wrench(
    previous: torch.Tensor,
    innovation: torch.Tensor,
    initialized: torch.Tensor,
    correlation: float,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """Bounded AR(1) launch disturbance, with a full first-hop sample.

    The latent wrench is training-only.  Keeping it across touchdowns lets a
    policy encounter the measured failure mode where a nonzero attitude/rate
    at one landing makes the following launch more likely to be difficult.
    """
    if previous.shape != innovation.shape or previous.shape[-1] != 3:
        raise ValueError('expected matching (..., 3) launch wrench tensors')
    if initialized.shape != previous.shape[:-1]:
        raise ValueError('initialized mask must match wrench batch shape')
    if bounds.shape != (3,) or torch.any(bounds < 0):
        raise ValueError('bounds must be a non-negative three-axis tensor')
    if not 0.0 <= correlation < 1.0:
        raise ValueError('correlation must lie in [0, 1)')
    blended = (correlation * previous
               + math.sqrt(1.0 - correlation ** 2) * innovation)
    sample = torch.where(initialized.unsqueeze(-1), blended, innovation)
    return torch.maximum(torch.minimum(sample, bounds), -bounds)


def rotor_wrench_torque_from_com_offset(
    thrust_z: torch.Tensor,
    com_offset_b: torch.Tensor,
) -> torch.Tensor:
    """Torque needed when rotor thrust acts at the link origin, not its CoM.

    ``com_offset_b`` is the CoM position relative to the Body link origin.  The
    rotor resultant is modelled at that origin, hence the lever arm from CoM to
    rotor center is ``-com_offset_b``.  For body-Z thrust this gives
    ``[-dy*Fz, dx*Fz, 0]``.  PhysX still receives the randomized CoM itself so
    contact and articulated dynamics see the same mass distribution.
    """
    if com_offset_b.shape[-1] != 3:
        raise ValueError("expected (..., 3) body-frame CoM offsets")
    if thrust_z.shape != com_offset_b.shape[:-1]:
        raise ValueError("thrust shape must match CoM offset batch dimensions")
    zeros = torch.zeros_like(thrust_z)
    return torch.stack(
        (-com_offset_b[..., 1] * thrust_z,
         com_offset_b[..., 0] * thrust_z,
         zeros),
        dim=-1,
    )
