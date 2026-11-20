"""Reward-only extraction from the user supplied stable_vertical_20261020 package.

Source: reference_packages/stable_vertical_20261020/training_environment/
quadhopper-jump/quadhopper_env.py. No controller, physics or observations imported.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


CONTACT = 0


ASCENT = 1


DESCENT = 2


LANDING = 3


GRAVITY_M_S2 = 9.81


POWER_FALLBACK_SCALE = 16.0


SPRING_CONTACT_THRESHOLD_M = 0.002


VERTICAL_ENERGY_SCALE_M = 0.10


APEX_SCALE_M = 0.06


ANGULAR_VELOCITY_HUBER_BETA_RAD_S = 0.8


XY_PROGRESS_CLIP_M = 0.05


GOAL_RADIUS_M = 0.10


HOVER_MIN_HEIGHT_M = 0.50


HOVER_MAX_VERTICAL_SPEED_M_S = 0.15


HOVER_DWELL_THRESHOLD_S = 0.20


@dataclass(frozen=True)
class RewardInputs:
    xy_error: torch.Tensor
    height: torch.Tensor
    target_height: torch.Tensor
    yaw_error: torch.Tensor
    quaternion_xy: torch.Tensor
    body_horizontal_velocity: torch.Tensor
    motor_commands: torch.Tensor
    spring_position: torch.Tensor
    spring_velocity: torch.Tensor
    vertical_velocity: torch.Tensor
    body_rates: torch.Tensor
    phase: torch.Tensor
    previous_xy_distance: torch.Tensor
    previous_executed_action: torch.Tensor
    executed_action: torch.Tensor
    action_rate_ready: torch.Tensor
    apex_height: torch.Tensor
    apex_event: torch.Tensor
    qualified_landing_event: torch.Tensor
    failure_event: torch.Tensor
    contact: torch.Tensor
    hover_dwell_time: torch.Tensor


@dataclass(frozen=True)
class RewardWeights:
    survival: float = 5.0
    xy: float = 13.0
    height: float = 20.0
    yaw: float = 4.0
    attitude: float = 18.0
    horizontal_velocity: float = 4.0
    power: float = 0.0
    synergy: float = 0.02
    anti_synergy: float = 0.02
    lazy_ground: float = 10.0
    vertical_energy: float = 4.0
    apex: float = 8.0
    angular_velocity: float = 1.5
    xy_progress: float = 8.0
    goal_bonus: float = 3.0
    flight_power: float = 0.55
    hover: float = 40.0
    action_rate_channels: tuple[float, float, float, float] = (0.12, 0.55, 0.55, 0.22)
    termination: float = 50.0


@dataclass(frozen=True)
class RewardOutput:
    reward: torch.Tensor
    terms: dict[str, torch.Tensor]
    next_xy_distance: torch.Tensor
    next_hover_dwell_time: torch.Tensor
    next_action_rate_ready: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def predicted_apex_height(height: torch.Tensor, vertical_velocity: torch.Tensor) -> torch.Tensor:
    if height.shape != vertical_velocity.shape:
        raise ValueError("height and vertical_velocity must have identical shapes")
    upward_velocity = vertical_velocity.clamp_min(0.0)
    return height + upward_velocity.square() / (2.0 * GRAVITY_M_S2)


def huber_loss(value: torch.Tensor, *, beta: float) -> torch.Tensor:
    threshold = float(beta)
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError("beta must be finite and positive")
    absolute = value.abs()
    return torch.where(absolute <= threshold, 0.5 * value.square() / threshold, absolute - 0.5 * threshold)


def _validate_reward(inputs: RewardInputs, weights: RewardWeights) -> int:
    if inputs.xy_error.ndim != 2 or inputs.xy_error.shape[1] != 2:
        raise ValueError("xy_error must be [N,2]")
    batch = inputs.xy_error.shape[0]
    expected = {
        "height": (batch,), "target_height": (batch,), "yaw_error": (batch,),
        "quaternion_xy": (batch, 2), "body_horizontal_velocity": (batch, 2),
        "motor_commands": (batch, 4), "spring_position": (batch,), "spring_velocity": (batch,),
        "vertical_velocity": (batch,), "body_rates": (batch, 3), "phase": (batch,),
        "previous_xy_distance": (batch,), "previous_executed_action": (batch, 4),
        "executed_action": (batch, 4), "action_rate_ready": (batch,), "apex_height": (batch,),
        "apex_event": (batch,), "qualified_landing_event": (batch,), "failure_event": (batch,),
        "contact": (batch,), "hover_dwell_time": (batch,),
    }
    for name, shape in expected.items():
        if getattr(inputs, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if inputs.phase.dtype not in (torch.int32, torch.int64):
        raise ValueError("phase must have an integer dtype")
    for name in ("action_rate_ready", "apex_event", "qualified_landing_event", "failure_event", "contact"):
        if getattr(inputs, name).dtype != torch.bool:
            raise ValueError(f"{name} must have boolean dtype")
    if len(weights.action_rate_channels) != 4:
        raise ValueError("action_rate_channels must contain four weights")
    if any(value < 0.0 or not math.isfinite(value) for value in weights.action_rate_channels):
        raise ValueError("action_rate_channels must be finite and non-negative")
    return batch


def compute_reward(inputs: RewardInputs, weights: RewardWeights, dt_policy: float) -> RewardOutput:
    batch = _validate_reward(inputs, weights)
    dt = float(dt_policy)
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt_policy must be finite and positive")
    xy_distance = torch.linalg.vector_norm(inputs.xy_error, dim=1)
    power_proxy = POWER_FALLBACK_SCALE * torch.sum(inputs.motor_commands.square(), dim=1)
    spring_contact = inputs.spring_position > SPRING_CONTACT_THRESHOLD_M
    zeros = torch.zeros_like(inputs.height)
    terms = {
        "survival": torch.full_like(inputs.height, weights.survival * dt),
        "xy": weights.xy * torch.exp(-xy_distance / 0.80) * dt,
        "height": weights.height * torch.exp(-(inputs.target_height - inputs.height).abs() / 0.20) * dt,
        "yaw": -weights.yaw * inputs.yaw_error.square() * dt,
        "attitude": -weights.attitude * torch.sum(inputs.quaternion_xy.square(), dim=1)
        * (1.0 + 5.0 * torch.clamp(1.0 - inputs.height, 0.0, 1.0)) * dt,
        "horizontal_velocity": -weights.horizontal_velocity
        * torch.sum(inputs.body_horizontal_velocity.square(), dim=1) * dt,
        "power": -weights.power * power_proxy * dt,
        "synergy": torch.where(spring_contact & (inputs.spring_velocity < -0.5), weights.synergy * power_proxy * dt, zeros),
        "anti_synergy": torch.where(spring_contact & (inputs.spring_velocity > 0.5), -weights.anti_synergy * power_proxy * dt, zeros),
        "lazy_ground": torch.where(inputs.height < 0.3, torch.full_like(inputs.height, -weights.lazy_ground * dt), zeros),
    }
    predicted_apex = predicted_apex_height(inputs.height, inputs.vertical_velocity)
    predicted_apex_error = (predicted_apex - inputs.target_height).abs()
    actual_apex_error = (inputs.apex_height - inputs.target_height).abs()
    ascending = inputs.phase == ASCENT
    airborne = (inputs.phase == ASCENT) | (inputs.phase == DESCENT)
    xy_progress = (inputs.previous_xy_distance - xy_distance).clamp(-XY_PROGRESS_CLIP_M, XY_PROGRESS_CLIP_M)
    goal_hit = inputs.qualified_landing_event & (xy_distance <= GOAL_RADIUS_M)
    hover_condition = airborne & ~inputs.contact & (inputs.height > HOVER_MIN_HEIGHT_M) & (
        inputs.vertical_velocity.abs() < HOVER_MAX_VERTICAL_SPEED_M_S
    )
    next_hover_dwell = torch.where(
        hover_condition, inputs.hover_dwell_time + dt, torch.zeros_like(inputs.hover_dwell_time)
    )
    hover_trigger = hover_condition & (next_hover_dwell > HOVER_DWELL_THRESHOLD_S)
    action_delta = inputs.executed_action - inputs.previous_executed_action
    action_rate_weights = inputs.executed_action.new_tensor(weights.action_rate_channels)
    weighted_action_delta_sq = (action_delta.square() * action_rate_weights).sum(dim=1)
    action_delta_sq = torch.where(inputs.action_rate_ready, action_delta.square().sum(dim=1), zeros)
    additional_terms = {
        "vertical_energy": torch.where(
            ascending, weights.vertical_energy * torch.exp(-predicted_apex_error / VERTICAL_ENERGY_SCALE_M) * dt, zeros
        ),
        "apex": torch.where(inputs.apex_event, weights.apex * torch.exp(-actual_apex_error / APEX_SCALE_M), zeros),
        "angular_velocity": -weights.angular_velocity
        * huber_loss(inputs.body_rates, beta=ANGULAR_VELOCITY_HUBER_BETA_RAD_S).sum(dim=1) * dt,
        "xy_progress": weights.xy_progress * xy_progress,
        "goal_bonus": torch.where(goal_hit, torch.full_like(inputs.height, weights.goal_bonus), zeros),
        "flight_power": torch.where(airborne, -weights.flight_power * power_proxy * dt, zeros),
        "hover": torch.where(hover_trigger, torch.full_like(inputs.height, -weights.hover * dt), zeros),
        "action_rate": torch.where(inputs.action_rate_ready, -weighted_action_delta_sq * dt, zeros),
        "termination": torch.where(inputs.failure_event, torch.full_like(inputs.height, -weights.termination), zeros),
    }
    terms.update(additional_terms)
    reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0)
    diagnostics = {
        "predicted_apex_error": predicted_apex_error,
        "actual_apex_error": torch.where(inputs.apex_event, actual_apex_error, zeros),
        "hover_trigger": hover_trigger.to(dtype=inputs.height.dtype),
        "action_delta_sq": action_delta_sq,
        "goal_hit": goal_hit.to(dtype=inputs.height.dtype),
        "failure_termination": inputs.failure_event.to(dtype=inputs.height.dtype),
        "power_proxy": power_proxy,
    }
    return RewardOutput(
        reward=reward, terms=terms, next_xy_distance=xy_distance,
        next_hover_dwell_time=next_hover_dwell,
        next_action_rate_ready=torch.ones(batch, dtype=torch.bool, device=inputs.height.device),
        diagnostics=diagnostics,
    )
