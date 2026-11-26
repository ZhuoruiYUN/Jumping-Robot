"""Pre-impact estimate of the horizontal velocity produced by spring support."""
import torch

from .ring_geometry import rotate


def predict_support_velocity(leg, q, body_rate, velocity, clearance, gravity, impulse_ratio, support_time):
    """First-order leg sweep to contact, followed by rotation about the planted foot.

    Only used for scoring. The short pivot approximation assumes a planted foot,
    nearly upright body and no intervening control. Release measurements decide
    whether a predicted error actually materialized.
    """
    length = leg.norm(dim=-1, keepdim=True).clamp_min(.05)
    axis = leg / length
    down = -velocity[:, 2]
    height = clearance.clamp_min(0.)
    impact_speed = (down.square() + 2 * gravity * height).sqrt()
    # The clock is used only on free descent. Bound extrapolation on other lanes.
    time = 2 * height / (impact_speed + down.abs()).clamp_min(1.e-6)
    axis_rate = torch.linalg.cross(rotate(q, body_rate), axis)
    impact_axis = axis + axis_rate * time.clamp(max=.2)[:, None]
    impact_axis = torch.nn.functional.normalize(impact_axis, dim=-1)
    impact_velocity = torch.cat((velocity[:, :2], -impact_speed[:, None]), dim=-1)
    tangent = impact_velocity - (impact_velocity * impact_axis).sum(-1, keepdim=True) * impact_axis
    support_axis = torch.nn.functional.normalize(impact_axis + support_time * tangent / length, dim=-1)
    # Vertical impulse times horizontal/vertical axis ratio. Limit the model to
    # its near-upright domain; it is not an impact solver at extreme inclinations.
    delta_xy = impulse_ratio * impact_speed[:, None] * support_axis[:, :2] / support_axis[:, 2:3].clamp_min(.8)
    return velocity[:, :2] + delta_xy, time


class SupportPreview:
    def __init__(self, num_envs, device):
        self.velocity_sum = torch.zeros(num_envs, 2, device=device)
        self.count = torch.zeros(num_envs, device=device)
        self.predicted = torch.zeros_like(self.velocity_sum)
        self.incoming = torch.zeros_like(self.velocity_sum)
        self.straight = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.retry = torch.zeros_like(self.straight)
        self.pending_cost = torch.zeros_like(self.count)

    def reset(self, ids):
        for value in (self.velocity_sum, self.count, self.predicted, self.incoming, self.straight, self.retry, self.pending_cost):
            value[ids] = 0

    def sample(self, valid, predicted):
        self.velocity_sum += torch.where(valid[:, None], predicted, 0.)
        self.count += valid

    def contact(self, touch, straight, retry, velocity):
        observed = self.count > 0
        self.straight[touch] = (straight & observed)[touch]
        self.retry[touch] = (retry & observed)[touch]
        self.incoming[touch] = velocity[touch]
        self.predicted[touch] = (self.velocity_sum / self.count.clamp_min(1.)[:, None])[touch]
        self.velocity_sum[touch] = 0.
        self.count[touch] = 0.

    def release(self, launch, velocity, target, flight_time, threshold, width, weight):
        error = velocity - target
        error_norm = error.norm(dim=-1)
        direction = error / error_norm.clamp_min(1.e-6)[:, None]
        predicted_along_error = ((self.predicted - target) * direction).sum(-1)
        axis = torch.nn.functional.normalize(target, dim=-1)
        cross_speed = (velocity[:, 0] * axis[:, 1] - velocity[:, 1] * axis[:, 0]).abs()
        wrong_direction = ((velocity * axis).sum(-1) < 0) | (cross_speed > threshold)
        eligible = launch & (self.retry | (self.straight & wrong_direction))
        entry_error = (self.incoming - target).norm(dim=-1)
        floor = torch.where(self.retry, entry_error.clamp_min(threshold), threshold)
        # For a retry test whether the predicted impulse grows the error along
        # the direction later observed. Require the actual result to breach both gates.
        incoming_along_error = ((self.incoming - target) * direction).sum(-1)
        preview_floor = torch.where(self.retry, incoming_along_error, threshold)
        excess = torch.minimum(error_norm - floor, predicted_along_error - preview_floor).clamp_min(0.)
        cost = weight * torch.expm1(-.5 * (excess * flight_time / width).square())
        self.straight &= ~launch
        self.retry &= ~launch
        budget = torch.where(eligible, cost, 0.)
        self.pending_cost = torch.where(launch, budget, self.pending_cost)
        return budget

    def settle(self, touchdown, physical, foot_error, radius):
        failed = physical | (touchdown & (foot_error > radius))
        cost = torch.where(failed, self.pending_cost, 0.)
        self.pending_cost = torch.where(touchdown | physical, 0., self.pending_cost)
        return cost
