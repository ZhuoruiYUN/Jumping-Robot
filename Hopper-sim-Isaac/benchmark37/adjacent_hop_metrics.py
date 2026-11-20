"""First-attempt landing accuracy for ordinary moving-route training lanes."""
import math

import torch


class AdjacentHopMetrics:
    def __init__(self, n, device, radius=.10):
        self.radius = radius
        self.ids = torch.arange(n, device=device)
        self.last_attempt = torch.full_like(self.ids,-1)
        self.has_moved = torch.zeros(n,dtype=torch.bool,device=device)
        # Eight world-heading sectors, then reversals and holds after motion.
        self.count = torch.zeros(10,2,dtype=torch.long,device=device)

    def reset(self, ids):
        self.last_attempt[ids] = -1
        self.has_moved[ids] = False

    def observe(self, route, valid, error):
        i = route.index
        pt = route.points[self.ids,i]
        previous = route.points[self.ids,(i-1).clamp_min(0)]
        edge = pt-previous
        incoming = previous-route.points[self.ids,(i-2).clamp_min(0)]
        first = valid & (i>0) & (self.last_attempt!=i)
        moving = edge.ne(0).any(-1)
        sector = (torch.atan2(edge[:,1],edge[:,0])*(4/math.pi)+.5).floor().long()%8
        reached = error <= self.radius
        attempts = first & moving
        self.count[:8].index_add_(0,sector,torch.stack((attempts,attempts & reached),dim=1).long())
        # Allow half a degree for float32 coordinates and exact reversals near the boundary.
        reverse = (attempts & incoming.ne(0).any(-1) &
                   ((edge*incoming).sum(-1) <= -math.cos(math.radians(.5))*edge.norm(dim=-1)*incoming.norm(dim=-1)))
        hold = first & ~moving & self.has_moved
        special = torch.stack((reverse,hold))
        self.count[8:,0] += special.sum(1)
        self.count[8:,1] += (special & reached).sum(1)
        self.last_attempt[valid] = i[valid]
        self.has_moved |= attempts

    def log_rollout(self, writer, iteration):
        values = self.count.tolist()
        self.count.zero_()
        writer.add_scalar('Route/moving_first_attempts',sum(n for n,_ in values[:8]),iteration)
        if all(n for n,_ in values[:8]):
            writer.add_scalar(f'Route/worst_heading_{self.radius*100:g}cm_rate', min(h/n for n,h in values[:8]),iteration)
        for name,(n,hits) in zip(('reverse','hold_after_move'),values[8:]):
            writer.add_scalar(f'Route/{name}_attempts',n,iteration)
            if n:
                writer.add_scalar(f'Route/{name}_{self.radius*100:g}cm_rate',hits/n,iteration)
