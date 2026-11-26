"""Compare equal, consecutive jumps at matching ascent and descent heights."""

import math

import torch

from .ring_geometry import rotate


def straight_pair(first_edge, second_edge):
    first_length = first_edge.norm(dim=-1)
    second_length = second_edge.norm(dim=-1)
    alignment = (first_edge * second_edge).sum(-1)
    return ((first_length >= .25) & (second_length >= .25)
            & ((first_length - second_length).abs() <= .03)
            & (alignment >= .99 * first_length * second_length))


class HopRepeatability:
    """Record a clean A in travel coordinates and compare the matching B."""

    def __init__(self, num_envs, device, config):
        levels = config['sample_heights_m']
        if (config['version'] != 'height_phase_matched_ab_v1'
                or len(levels) != 4 or sorted(set(levels)) != levels
                or not .36 <= levels[0] < levels[-1] <= .97
                or not 0. < config['max_cost_per_sample'] <= .6):
            raise ValueError('Invalid straight-hop repeatability configuration')
        self.levels = torch.tensor(levels, device=device).repeat(2)
        self.ascending = torch.arange(len(self.levels), device=device) < len(levels)
        self.max_cost = config['max_cost_per_sample']
        self.scale = torch.tensor([math.radians(3.), math.radians(2.), .45, .35, .045], device=device)
        self.reference = torch.zeros(num_envs, len(self.levels), len(self.scale), device=device)
        self.seen = torch.zeros(num_envs, len(self.levels), dtype=torch.bool, device=device)
        self.compared = torch.zeros_like(self.seen)
        self.capturing = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.comparing = torch.zeros_like(self.capturing)
        self.ready = torch.zeros_like(self.capturing)
        self.previous_height = torch.zeros(num_envs, device=device)
        self.previous_state = torch.zeros(num_envs, len(self.scale), device=device)
        self.release_velocity = torch.zeros(num_envs, 2, device=device)
        self.release_attitude = torch.zeros(num_envs, 4, device=device)

    def reset(self, ids):
        self.seen[ids] = False
        self.compared[ids] = False
        self.capturing[ids] = False
        self.comparing[ids] = False
        self.ready[ids] = False
        self.previous_height[ids] = 0.
        self.previous_state[ids] = 0.

    def step(self, height, age, quaternion, omega_body, velocity_xy, edge,
             launch, free, contact, clean_a, record_a, eligible_b):
        axis = torch.nn.functional.normalize(edge, dim=-1)
        transverse = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
        body_up = torch.zeros_like(omega_body)
        body_up[:, 2] = 1.
        up = rotate(quaternion, body_up)
        up_dot = torch.cross(rotate(quaternion, omega_body), up, dim=-1)
        sine = torch.stack(((up[:, :2] * axis).sum(-1), (up[:, :2] * transverse).sum(-1)), dim=-1)
        derivative = torch.stack(((up_dot[:, :2] * axis).sum(-1), (up_dot[:, :2] * transverse).sum(-1)), dim=-1)
        sine = sine.clamp(-.99, .99)
        state = torch.cat((sine.asin(), derivative / (1. - sine.square()).sqrt(), age[:, None]), dim=-1)

        start_a = launch & record_a
        self.seen &= ~start_a[:, None]
        self.ready &= ~start_a
        self.release_velocity.copy_(torch.where(start_a[:, None], velocity_xy, self.release_velocity))
        release_attitude = state[:, :4]
        self.release_attitude.copy_(torch.where(start_a[:, None], release_attitude, self.release_attitude))
        release_delta = ((velocity_xy - self.release_velocity) * axis).sum(-1).abs()
        side_delta = ((velocity_xy - self.release_velocity) * transverse).sum(-1).abs()
        first_b = launch & eligible_b & self.ready
        self.ready &= ~first_b
        start_b = first_b & (release_delta <= .35) & (side_delta <= .22)
        self.compared &= ~start_b[:, None]
        self.capturing |= start_a
        self.capturing &= ~(free & (up[:, 2] < math.cos(math.radians(23.))))
        self.comparing |= start_b
        self.comparing &= eligible_b

        old_h, new_h = self.previous_height[:, None], height[:, None]
        rising = (old_h < self.levels) & (new_h >= self.levels)
        falling = (old_h > self.levels) & (new_h <= self.levels)
        crossed = torch.where(self.ascending, rising, falling) & free[:, None] & ~launch[:, None]
        interval = new_h - old_h
        blend = torch.where(interval.abs() > 1.e-6, self.levels - old_h, 0.)
        blend = (blend / torch.where(interval.abs() > 1.e-6, interval, 1.)).clamp(0., 1.)
        sampled = torch.lerp(self.previous_state[:, None], state[:, None], blend[:, :, None])

        compare = crossed & self.comparing[:, None] & self.seen & ~self.compared
        self.compared |= compare
        difference = ((sampled - self.reference).abs() / self.scale - 1.).clamp_min(0.)
        cost = torch.where(compare, self.max_cost * torch.expm1(-.5 * difference.square().sum(-1)), 0.).sum(-1)
        launch_difference = ((release_attitude - self.release_attitude).abs() / self.scale[:4] - 1.).clamp_min(0.)
        cost += torch.where(start_b, self.max_cost * torch.expm1(-.5 * launch_difference.square().sum(-1)), 0.)
        capture = crossed & self.capturing[:, None] & ~self.seen
        # Let a good A provide the main lean pattern. Large sideways
        # corrections on A are not targets for the following jump.
        reference = sampled.clone()
        reference[:, :, 1].clamp_(-math.radians(4.), math.radians(4.))
        reference[:, :, 3].clamp_(-.45, .45)
        self.reference.copy_(torch.where(capture[:, :, None], reference, self.reference))
        self.seen |= capture
        self.ready |= clean_a & self.capturing & (self.seen.sum(-1) >= 6)
        self.capturing &= ~contact
        self.comparing &= ~contact
        self.previous_height.copy_(height)
        self.previous_state.copy_(state)
        return cost, compare.sum(-1) + start_b
