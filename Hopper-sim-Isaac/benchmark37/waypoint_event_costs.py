"""Reward-only measurements at liftoff and the first physical apex of a jump."""
import torch

from .precision_release import FirstFlightApex


class StraightFirstApex:
    def __init__(self, count, device):
        self.apex = FirstFlightApex(count, device)
        self.release_step = torch.full((count,), -1, device=device, dtype=torch.long)
        self.foot_offset = torch.zeros(count, 2, device=device)

    def reset(self, ids):
        self.apex.reset(ids)
        self.release_step[ids] = -1
        self.foot_offset[ids] = 0.

    def step(self, launch, opening_ascent, eligible, free, foot, root, velocity, goal,
             edge, flight, step, dt, radius):
        # Direct A has a physical free ascent after the initial drop contact but
        # no previous scored waypoint to arm the normal rebound-plan release.
        start = eligible & (launch | (opening_ascent & (self.release_step < 0)))
        self.release_step[start] = step
        self.foot_offset[start] = (foot - root)[start]
        reached = self.apex.step(start, eligible, free, velocity[:, 2])
        remaining = flight - (step - self.release_step) * dt
        active = reached & (remaining > 0)
        error = root + self.foot_offset + velocity[:, :2] * remaining[:, None] - goal
        excess = signed_cross_excess(error, edge, radius)
        return torch.where(active, excess, 0.), active


def signed_cross_excess(error, edge, radius):
    direction = torch.nn.functional.normalize(edge, dim=-1)
    along = (error * direction).sum(-1)
    cross = direction[:, 0] * error[:, 1] - direction[:, 1] * error[:, 0]
    allowance = (radius**2 - along.square()).clamp_min(0.).sqrt()
    return cross.sign() * (cross.abs() - allowance).clamp_min(0.)


class WaypointOutcome:
    """Settle pending measurements on the old target before contact advances the queue."""
    def __init__(self, count, device):
        self.cross = torch.zeros(count, device=device)
        self.growth = torch.zeros_like(self.cross)
        self.velocity_error = torch.zeros(count, 2, device=device)

    def reset(self, ids):
        self.cross[ids] = 0.
        self.growth[ids] = 0.
        self.velocity_error[ids] = 0.

    def step(self, apex, cross, release, growth, velocity_error, touchdown, physical,
             foot_error, edge, radius, weight):
        self.cross[apex] = cross[apex]
        self.growth[release] = growth[release]
        self.velocity_error[release] = velocity_error[release]
        landing_cross = signed_cross_excess(foot_error, edge, radius)
        persistent = torch.minimum(self.cross.abs(), (self.cross.sign() * landing_cross).clamp_min(0.))
        missed = touchdown & (foot_error.norm(dim=-1) > radius)
        lateral = torch.where(physical, self.cross.abs(), persistent)
        cross_cost = torch.where(missed | physical, weight * torch.expm1(-.5 * (lateral / radius).square()), 0.)
        carried_past = (self.velocity_error * foot_error).sum(-1) > 0.
        growth_cost = torch.where((missed & carried_past) | physical, self.growth, 0.)
        self.reset(touchdown | physical)
        return cross_cost, growth_cost



def retry_rebound_growth(release_error, contact_error, flight, active, limit, radius, weight):
    """Charge only new above-limit velocity error still present after spring release."""
    excess = (release_error - contact_error.clamp_min(limit)).clamp_min(0.)
    cost = weight * torch.expm1(-.5 * (excess * flight / radius).square())
    return torch.where(active, cost, 0.)
