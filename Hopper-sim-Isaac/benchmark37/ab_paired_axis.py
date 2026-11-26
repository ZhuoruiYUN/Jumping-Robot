"""Repeat a release in the same simulator lane with the opposite B command."""

import torch


class PairedAxisStarts:
    def __init__(self, count, device):
        ids = torch.arange(count, device=device)
        # Half the lanes continue the original AB sampler. The others cover both axes
        # and counterbalance the order of signs within each same-lane release pair.
        self.lanes = ids % 4 < 2
        self.axis = 2 * (ids % 2)
        self.first_sign = (ids // 4) % 2
        self.episodes = torch.zeros_like(ids)
        self.direction = torch.zeros_like(ids)
        self.replaying = torch.zeros_like(ids, dtype=torch.bool)
        self.starts = ids.new_zeros(4)
        self.issued = ids.new_zeros(4)
        self.landings = ids.new_zeros(4)
        self.hits = ids.new_zeros(4)
        self.cached = {}

    def reset(self, route, ids):
        paired = ids[self.lanes[ids]]
        ordinal = self.episodes[paired]
        replay = ordinal % 2
        self.replaying[paired] = replay.bool()
        direction = self.axis[paired] + (self.first_sign[paired] ^ ((ordinal // 2) % 2) ^ replay)
        self.direction[paired] = direction
        self.episodes[paired] += 1
        self.starts.scatter_add_(0, direction, torch.ones_like(direction))
        route.points[paired, :5] = 0.

    def match(self, ids, key, values):
        """Store each fresh paired draw and reuse it on that lane's next reset."""
        values = values.clone()
        if key not in self.cached:
            self.cached[key] = values.new_empty((len(self.lanes), *values.shape[1:]))
        cache = self.cached[key]
        reuse = self.replaying[ids]
        fresh = self.lanes[ids] & ~reuse
        cache[ids[fresh]] = values[fresh]
        values[reuse] = cache[ids[reuse]]
        return values

    def anchor(self, route, ids, local_xy):
        paired = self.lanes[ids]
        route.points[ids[paired], :5] = local_xy[paired, None, :]

    def publish_departure(self, route, ids, terminal):
        ids = ids[self.lanes[ids] & (route.index[ids] == 2)]
        directions = self.direction[ids]
        steps = route.points.new_tensor([[.5, 0.], [-.5, 0.], [0., .5], [0., -.5]])
        target = route.points[ids, 1] + steps[directions]
        route.points[ids, 2:5] = target[:, None]
        # On a terminal A retry the queue can advance for final reward
        # bookkeeping. That B never reaches the policy.
        self.issued.scatter_add_(0, directions, (~terminal[ids]).long())

    def record_arrival(self, route, valid, hit):
        moving_b = self.lanes & (route.index == 2) & valid
        self.landings.scatter_add_(0, self.direction, moving_b.long())
        self.hits.scatter_add_(0, self.direction, (moving_b & hit).long())

    def metrics(self):
        values = {}
        for i, label in enumerate(('positive_x', 'negative_x', 'positive_y', 'negative_y')):
            values[f'AB/paired_{label}_starts'] = self.starts[i]
            values[f'AB/paired_{label}_issued'] = self.issued[i]
            values[f'AB/paired_{label}_landings'] = self.landings[i]
            values[f'AB/paired_{label}_hits_5cm'] = self.hits[i]
        return values
