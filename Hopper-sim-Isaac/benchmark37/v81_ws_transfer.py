"""State-derived ascent, rebound and landing references for full AB training."""

import math

import torch

from .ring_geometry import rotate
from .v81_ascent_budget import (
    limit_queued_ascent, retained_ascent_pulse_cost, validate as validate_ascent, validate_retained_pulse,
)
from .v81_rebound_pitch import rebound_pitch_reference, validate as validate_rebound
from .v81_first_landing import first_landing_reference, validate as validate_landing

TERMS = ("ws_queued_ascent", "ws_forward_rebound", "ws_first_landing")
CROSS_REBOUND_TERM = "ws_cross_rebound"
CONTINUATION_FORECAST_VERSION = "bounded_continuation_acceleration_v1"


def terms(config):
    return TERMS + (CROSS_REBOUND_TERM,) if config.get("cross_rebound") is not None else TERMS


def validate(config):
    optional = {"opening_release_speed_m_s", "cross_rebound", "retained_ascent_pulse",
                "continuation_forecast", "continuation_cruise"}
    if set(config) - optional != {"ascent", "rebound", "landing", "cost_per_s"}:
        raise ValueError("Invalid WS teaching configuration")
    opening_speed = config.get("opening_release_speed_m_s")
    if opening_speed is not None and not .15 <= opening_speed <= .35:
        raise ValueError("Invalid first-step rebound speed")
    weights = config["cost_per_s"]
    if set(weights) != set(TERMS) or any(not 0. < weights[key] <= 20. for key in TERMS):
        raise ValueError("V81 teaching costs must be finite and positive")
    validate_ascent(config["ascent"])
    if config.get("retained_ascent_pulse") is not None:
        validate_retained_pulse(config["retained_ascent_pulse"])
    validate_rebound(config["rebound"])
    validate_landing(config["landing"])
    forecast = config.get("continuation_forecast")
    if forecast is not None and (
        set(forecast) != {"version", "minimum_step_m", "early_credit_m", "late_credit_m"}
        or forecast["version"] != CONTINUATION_FORECAST_VERSION
        or not .45 <= forecast["minimum_step_m"] <= .50
        or not 0. < forecast["late_credit_m"] <= .02 < forecast["early_credit_m"] <= .06
        or config["landing"].get("undertravel") is None
    ):
        raise ValueError("Invalid continuation undertravel forecast")
    if config.get("continuation_cruise") is not None:
        from .continuation_cruise import validate as validate_continuation_cruise
        validate_continuation_cruise(config["continuation_cruise"])
        if forecast is None:
            raise ValueError("Continuation cruise credit requires the confirmed-handoff history")
    cross = config.get("cross_rebound")
    if cross is not None:
        limits = {
            "cost_per_s": (0., 10.),
            "minimum_step_m": (.30, .50),
            "axis_half_width_deg": (1., 22.5),
            "residual_start_m_s": (.02, .15),
            "residual_width_m_s": (.05, .35),
            "tilt_prediction_s": (.01, .15),
        }
        if (cross.keys() != limits.keys() | {"version"}
                or cross["version"] != "signed_transverse_preimpact_v1"
                or any(not lo <= cross[key] <= hi for key, (lo, hi) in limits.items())
                or cross["cost_per_s"] <= 0.):
            raise ValueError("Invalid transverse rebound teaching envelope")


class ContinuationForecast:
    """Track the last confirmed first hit and a same-flight velocity window."""

    HISTORY_TICKS = 8

    def __init__(self, num_envs, device, config):
        self.config = config
        slots = self.HISTORY_TICKS + 1
        self.target = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.velocity = torch.zeros(slots, num_envs, 2, device=device)
        self.air_age = self.target.new_full((slots, num_envs), -1)
        self.sample_target = self.air_age.clone()
        self.push_pending = torch.zeros_like(self.air_age, dtype=torch.bool)
        self.steps = [-1] * slots

    def reset(self, ids):
        self.target[ids] = -1
        self.air_age[:, ids] = -1
        self.sample_target[:, ids] = -1

    def touchdown(self, settled, qualified, new_target):
        self.target.copy_(torch.where(settled, torch.where(qualified, new_target, -1), self.target))

    def credit(self, step, dt, target, air_age, velocity, edge, prior_edge, remaining, eligible, pending_push):
        period = len(self.steps)
        old_slot = (step - self.HISTORY_TICKS) % period
        slot = step % period
        world_velocity = velocity[:, :2]
        speed_delta = world_velocity - self.velocity[old_slot]
        clean_window = ((self.steps[old_slot] == step - self.HISTORY_TICKS)
                        & (self.air_age[old_slot] + self.HISTORY_TICKS == air_age)
                        & (self.air_age[old_slot] > 0)
                        & (self.sample_target[old_slot] == target))
        if pending_push is not None:
            # An injected velocity step is not sustained acceleration.
            clean_window &= ~(self.push_pending[old_slot] & ~pending_push)
            self.push_pending[slot].copy_(pending_push)
        self.velocity[slot].copy_(world_velocity)
        self.air_age[slot].copy_(air_age)
        self.sample_target[slot].copy_(target)
        self.steps[slot] = step

        axis = torch.nn.functional.normalize(edge, dim=-1)
        prior_length = prior_edge.norm(dim=-1)
        minimum = self.config["minimum_step_m"] - 1.e-6
        straight = ((edge.norm(dim=-1) >= minimum) & (prior_length >= minimum)
                    & ((prior_edge * axis).sum(-1) >= .996 * prior_length))
        forward = (world_velocity * axis).sum(-1) > 0.
        allowed = eligible & clean_window & (target == self.target) & straight & forward
        acceleration = ((speed_delta * axis).sum(-1) / (self.HISTORY_TICKS * dt)).clamp(0., 1.10)
        elapsed = air_age * dt
        progress = ((elapsed - .27) / .10).clamp(0., 1.)
        fade = ((elapsed - .24) / .03).clamp(0., 1.) * ((.44 - elapsed) / .07).clamp(0., 1.)
        early, late = self.config["early_credit_m"], self.config["late_credit_m"]
        limit = fade * (early + (late - early) * progress)
        decay = .30
        time_left = remaining.clamp_min(0.)
        integral = decay * (time_left + decay * torch.expm1(-time_left / decay))
        return torch.where(allowed, torch.minimum(acceleration * integral, limit), 0.), allowed


def _cross_release_forecast(axis, quaternion, omega_body, velocity, projected, landing_speed,
                            time_to_touch, flight_time, release_vz, prediction_horizon):
    """Estimate signed cross speed after rebound from the predicted impact lean."""
    side = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    side_world = torch.cat((side, torch.zeros_like(side[:, :1])), dim=-1)
    world_to_body = quaternion * quaternion.new_tensor([1., -1., -1., -1.])
    side_body = rotate(world_to_body, side_world)
    sine = side_body[:, 2].clamp(-.99, .99)
    rotation_axis = torch.stack((-side_body[:, 1], side_body[:, 0]), dim=-1)
    tilt_rate = (omega_body[:, :2] * rotation_axis).sum(-1) / (1. - sine.square()).sqrt()
    impact_lean = (sine.asin() + time_to_touch.clamp_max(prediction_horizon) * tilt_rate).clamp(-.35, .35)
    # A tilted leg redirects part of the vertical spring impulse into
    # the signed transverse axis during the next support phase.
    predicted = (velocity[:, :2] * side).sum(-1) + (landing_speed + release_vz) * impact_lean.tan()
    target = (-(projected * side).sum(-1) / flight_time.clamp_min(.5)).clamp(-.25, .25)
    return predicted, target


def references(env, actions, clearance, height, up, remaining):
    from .calibration_20261028 import PARAMS
    from .imu_link_flight import NOMINAL_MASS_KG
    from .moving_phase_reward import heading_motion

    config = env.ws_transfer_teaching
    route, plant = env.omni_route, env._imu_link_plant
    event, data = env._foot_event, env._robot.data
    velocity = data.root_link_lin_vel_w
    vz = velocity[:, 2]
    foot = data.body_link_pos_w[:, env._foot_body_id, :2]
    goal = env._two_ground_pt_xy_w
    edge = env._current_route_edge()
    following = env._two_ground_pt1_xy_w - goal
    edge_norm, next_norm = edge.norm(dim=-1), following.norm(dim=-1)
    axis = torch.nn.functional.normalize(edge, dim=-1)
    route_open = ~route.stationary_mask
    guard = getattr(env, "key_guard", None)
    if guard is not None:
        route_open &= ~guard.mask
    moving = route_open & (route.index > 0) & (edge_norm >= .30)

    first = moving & (env.last_attempted_target != route.index)
    airborne = ((event.phase == event.CANDIDATE) | (event.phase == event.FLIGHT)) & (clearance > .015)
    rising = airborne & ~event.apex_seen & (vz > .20)
    elapsed = event.air_steps * env.step_dt
    heading, heading_rate = heading_motion(data.root_quat_w, data.root_ang_vel_b)
    yaw_config = env.yaw_prediction_config
    error = heading - yaw_config["target_heading_rad"]
    forecast = error + yaw_config["response_horizon_s"] * heading_rate
    yaw_peak = torch.maximum(torch.atan2(error.sin(), error.cos()).abs(),
                             torch.atan2(forecast.sin(), forecast.cos()).abs()).rad2deg()
    yaw_margin = ((yaw_config["peak_heading_goal_deg"] - yaw_peak) / 4.).clamp(0., 1.)
    upright = ((up - math.cos(math.radians(25.))) / .06).clamp(0., 1.)

    timing = plant.parameters[plant.profile]
    ascent_gate = (moving & rising) * yaw_margin * upright * ((height - .55) / .09).clamp(0., 1.)
    ascent_state = (
        actions, ascent_gate, height, vz, up, NOMINAL_MASS_KG * plant.mass_scale,
        PARAMS["air"]["ascending_quadratic_Ns2_m2_whole"] * timing[:, 3],
        plant.collective_scale, plant.voltage_gain, plant.collective_u, torch.stack(tuple(plant.history), dim=1),
        timing[:, 0], timing[:, 1], env.step_dt,
    )
    _, coast, requested, trim = limit_queued_ascent(*ascent_state, config["ascent"])

    gravity = abs(env.sim.cfg.gravity[2])
    landing_speed = torch.sqrt(vz.square() + 2. * gravity * clearance.clamp_min(0.))
    time_to_touch = 2. * clearance.clamp_min(0.) / (landing_speed + vz.abs()).clamp_min(1.e-6)
    projected = foot + velocity[:, :2] * time_to_touch[:, None] - goal
    same_direction = (next_norm >= .30) & ((axis * following).sum(-1) >= .97 * next_norm)
    planned_continuation = route.index <= route.ab_pairs.a_index
    lead = timing[:, 7] * env.step_dt + 2. * timing[:, 6]
    near_touch = ((lead + .12 - time_to_touch) / .12).clamp(0., 1.)
    projected_distance = projected.norm(dim=-1)
    plausible = ((.12 - projected_distance) / .06).clamp(0., 1.)
    rebound_gate = (first & same_direction & planned_continuation) * plausible
    rebound_axis = axis
    opening_target = None
    opening_speed = config.get("opening_release_speed_m_s")
    if opening_speed is not None:
        # Only prepare the move before a plausible hold touchdown. The
        # committed preview, rather than the zero-length hold edge, owns
        # the direction of the first moving rebound.
        opening = route_open & (route.index < route.ab_pairs.a_index) & (edge_norm <= .05) & (next_norm >= .40)
        open_weight = ((.08 - projected_distance) / .04).clamp(0., 1.)
        rebound_gate = torch.where(opening, open_weight, rebound_gate)
        rebound_axis = torch.where(opening[:, None], torch.nn.functional.normalize(following, dim=-1), axis)
        opening_target = opening * opening_speed
    rebound_gate *= airborne & (vz < 0.)
    rebound_gate = rebound_gate * near_touch * yaw_margin * upright
    pitch_action, pitch_target, release_gate, release_speed, _ = rebound_pitch_reference(
        actions, data.root_quat_w, data.root_ang_vel_b, rebound_axis, (velocity[:, :2] * rebound_axis).sum(-1),
        landing_speed, time_to_touch, rebound_gate, config["rebound"], opening_target=opening_target,
    )
    pitch_shortfall = (pitch_target - pitch_action).clamp_min(0.)

    # A near-upright touchdown removes most of the in-flight lean offset. Use
    # link position with the measured release offset, as in the existing first-cross ledger.
    offsets = env.first_cross_foot_offset
    use_release_offset = env.first_cross_release_step >= 0
    foot_for_landing = torch.where(use_release_offset[:, None], data.root_link_pos_w[:, :2] + offsets, foot)
    first_landing = first & rising
    height_ready = (requested >= 1.025) | (height >= 1.02)
    undertravel_allowed = height_ready & (yaw_margin > .35)
    forecast = getattr(env, "ws_continuation_forecast", None)
    credit = None
    if forecast is not None:
        rows = env._robot._ALL_INDICES
        prior_edge = (route.points[rows, (route.index - 1).clamp_min(0)]
                      - route.points[rows, (route.index - 2).clamp_min(0)])
        robustness = getattr(env, "_physical_robustness", None)
        credit, continuation = forecast.credit(
            env.common_step_counter, env.step_dt, route.index, event.air_steps, velocity, edge, prior_edge,
            remaining, first_landing & undertravel_allowed, None if robustness is None else robustness.push_pending,
        )
    landing = first_landing_reference(
        actions, data.root_quat_w, data.root_ang_vel_b, foot_for_landing, velocity, goal,
        edge, elapsed, remaining, first_landing, config["landing"],
        undertravel_allowed=undertravel_allowed, undertravel_credit_m=credit,
    )
    modes = torch.stack((landing["target_along_force_n"] - landing["action_along_force_n"],
                         landing["target_cross_force_n"] - landing["action_cross_force_n"]), dim=-1)
    landing_error = ((modes / .065).square().clamp_max(1.) * landing["axis_weight"]).mean(-1)
    ascent = config["ascent"]
    reserve = ((coast - ascent["protected_height_m"]) / .025).clamp(0., 1.)
    projected_excess = requested - coast.clamp_min(ascent["target_apex_m"])
    height_error = reserve * (projected_excess / .05).clamp(0., 1.).square()
    weight = config["cost_per_s"]
    costs = {
        TERMS[0]: -env.step_dt * weight[TERMS[0]] * ascent_gate * height_error,
        TERMS[1]: -env.step_dt * weight[TERMS[1]] * release_gate * (pitch_shortfall / .06).clamp_max(1.).square(),
        TERMS[2]: -env.step_dt * weight[TERMS[2]] * yaw_margin * upright * landing_error,
    }
    audit = dict(
        trim_n=trim, coast_apex_m=coast, requested_apex_m=requested,
        release_gate=release_gate, predicted_release_m_s=release_speed, pitch_shortfall_n=pitch_shortfall,
        landing_axis_weight=landing["axis_weight"],
        undertravel_gate=landing["undertravel_gate"],
    )
    if credit is not None:
        audit.update(continuation_forecast_m=landing["applied_undertravel_credit_m"],
                     continuation_native_short_gate=landing["native_undertravel_gate"])
    if config.get("continuation_cruise") is not None:
        from .continuation_cruise import forward_allowance
        audit.update(
            continuation_cruise_allowance_m_s2=forward_allowance(
                landing, velocity, edge, elapsed, remaining, coast, requested, continuation, env.landing_radius_m),
            continuation_cruise_target=route.index.clone(),
            continuation_cruise_step=env.common_step_counter + 1,
        )
    retained = config.get("retained_ascent_pulse")
    if retained is not None:
        ascent_cost, predicted, protected = retained_ascent_pulse_cost(
            *ascent_state, env.overheight_band['upper_m'], weight[TERMS[0]], retained,
        )
        costs[TERMS[0]] = ascent_cost
        audit.update(retained_pulse_gate=ascent_gate, retained_pulse_requested_apex_m=predicted,
                     retained_pulse_reduced_apex_m=protected)
    if opening_target is not None:
        audit["opening_release_gate"] = release_gate * (opening_target > 0.)
    cross = config.get("cross_rebound")
    if cross is not None:
        long_step = (edge_norm + 1.e-6 >= cross["minimum_step_m"]) & (next_norm + 1.e-6 >= cross["minimum_step_m"])
        y_route = axis[:, 1].abs() >= math.cos(math.radians(cross["axis_half_width_deg"]))
        straight = (axis * following).sum(-1) >= .99 * next_norm
        # Teach the transverse handoff from an incoming first-attempt A to
        # B. The inherited gate supplies the plausible touchdown, calibrated
        # motor lead, heading margin and upright descent conditions.
        cross_gate = rebound_gate * (first & long_step & y_route & straight)
        predicted, target = _cross_release_forecast(
            axis, data.root_quat_w, data.root_ang_vel_b, velocity, projected, landing_speed,
            time_to_touch, env.rebound_plan.flight, config["rebound"]["assumed_release_vz_m_s"],
            cross["tilt_prediction_s"],
        )
        excess = ((predicted - target).abs() - cross["residual_start_m_s"]) / cross["residual_width_m_s"]
        cross_cost = -env.step_dt * cross["cost_per_s"] * cross_gate * excess.clamp(0., 1.).square()
        costs[CROSS_REBOUND_TERM] = cross_cost
        audit.update(cross_rebound_gate=cross_gate, cross_rebound_prediction_m_s=predicted,
                     cross_rebound_target_m_s=target, cross_rebound_cost=cross_cost)
    return costs, audit
