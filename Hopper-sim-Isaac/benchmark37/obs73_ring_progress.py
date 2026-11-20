"""Bounded new-frontier reward for the designated Stage B ring flight only."""
import math

import torch


class ForwardProgress:
    def __init__(self, n, device, course):
        c = course['stage_b']['forward_progress']
        if c['version'] not in ('confirmed_flight_new_frontier_v1','ascending_approach_new_frontier_v2'):
            raise ValueError('Unknown Stage B progress reward')
        self.config = c
        goal = torch.tensor(course['landing_xy_m'], device=device)
        self.length = float(torch.linalg.vector_norm(goal))
        if self.length <= 0 or not math.isfinite(c['budget']) or c['budget'] <= 0:
            raise ValueError('Positive finite route length and progress budget required')
        self.direction = goal / self.length
        self.side = torch.stack((-self.direction[1], self.direction[0]))
        self.low, self.high = c['root_height_range_m']
        if not (0 < self.low < self.high and c['lateral_limit_m'] > 0):
            raise ValueError('Invalid progress corridor')
        self.early = None
        if c['version'] == 'ascending_approach_new_frontier_v2':
            self.early = c['early_approach']
            self.early_end = float((goal.new_tensor(course['center_local_m'][:2])*self.direction).sum())
            if not (0 < self.early['root_height_min_m'] < self.low
                    and 0 < self.early_end < self.length):
                raise ValueError('Early ascent must stay above ground and before the ring')
        self.best = torch.zeros(n, device=device)
        self.paid = torch.zeros(n, device=device)
        self.total = torch.zeros((), device=device)
        self.positive_steps = torch.zeros((), dtype=torch.long, device=device)
        self.early_total = torch.zeros((), device=device)
        self.early_positive_steps = torch.zeros((), dtype=torch.long, device=device)

    def coordinate(self, local_position):
        return (local_position[:, :2] * self.direction).sum(-1).clamp(0., self.length)

    def reset(self, ids):
        self.best[ids] = 0.
        self.paid[ids] = 0.

    def prime(self, ids, local_position):
        # Existing displacement on the first landing is never paid retroactively.
        self.best[ids] = self.coordinate(local_position)

    def step(self, local_position, active, confirmed_free_flight, unsafe, vertical_velocity=None):
        position = self.coordinate(local_position)
        finite = torch.isfinite(local_position).all(-1)
        frontier = torch.maximum(self.best, torch.where(finite, position, self.best))
        lateral = (local_position[:, :2] * self.side).sum(-1).abs()
        height = local_position[:, 2]
        early = torch.zeros_like(active)
        if self.early is not None:
            if vertical_velocity is None:
                raise ValueError('Early ascent reward requires actual world vertical velocity')
            early = (torch.isfinite(vertical_velocity) & (vertical_velocity > 0.)
                     & (height >= self.early['root_height_min_m']) & (height < self.low)
                     & (position < self.early_end))
        height_window = ((height >= self.low) & (height <= self.high)) | early
        eligible = (active & confirmed_free_flight & ~unsafe & finite
                    & (lateral <= self.config['lateral_limit_m']) & height_window)
        increment = (frontier - self.best).clamp(min=0.)
        reward = torch.where(eligible, increment * (self.config['budget'] / self.length), 0.)
        reward = torch.minimum(reward, (self.config['budget'] - self.paid).clamp(min=0.))
        # Track even ineligible advances: leaving the corridor and returning, or
        # moving on the ground then taking off, cannot reclaim the same distance.
        self.best.copy_(torch.where(active, frontier, self.best))
        self.paid += reward
        self.total += reward.sum()
        self.positive_steps += (reward > 0).sum()
        self.early_total += torch.where(early,reward,0.).sum()
        self.early_positive_steps += ((reward > 0) & early).sum()
        return reward
