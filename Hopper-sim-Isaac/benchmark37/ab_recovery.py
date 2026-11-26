"""Use a real missed A landing as the start of the second training jump."""
import torch


class ABRecovery:
    CONTINUE, RETRY, LOCAL_HOLD = range(3)
    names = ('retry', 'local_hold')

    def __init__(self, count, device, seed, config):
        retry, hold = config['retry_probability'], config['local_hold_probability']
        if not (retry > 0 and hold > 0 and retry + hold < 1):
            raise ValueError('AB recovery needs positive retry, hold and ordinary AB shares')
        self.retry_probability = retry
        self.recovery_probability = retry + hold
        self.rng = torch.Generator(device=device).manual_seed(seed + 490787)
        self.assignment = torch.zeros(count, device=device, dtype=torch.long)
        self.active = torch.zeros_like(self.assignment)
        self.modes = self.active.new_tensor([self.RETRY, self.LOCAL_HOLD])[:, None]
        self.entered = self.active.new_zeros(2)
        self.landings = self.active.new_zeros(2)
        self.hits = self.active.new_zeros(2)
        self.deaths = self.active.new_zeros(2)
        self.releases = self.active.new_zeros(2)
        self.release_error_sum = torch.zeros(2, device=device, dtype=torch.float64)

    def reset(self, ids):
        draw = torch.rand(len(ids), device=ids.device, generator=self.rng)
        self.assignment[ids] = torch.where(draw < self.retry_probability, self.RETRY,
                                           (draw < self.recovery_probability).long() * self.LOCAL_HOLD)
        self.active[ids] = self.CONTINUE

    def start(self, missed_a, prefer_retry=None):
        self.active[missed_a] = self.assignment[missed_a]
        if prefer_retry is not None:
            self.active[missed_a & prefer_retry & (self.assignment == self.CONTINUE)] = self.RETRY
        self.entered += ((self.modes == self.active) & missed_a).sum(-1)
        return missed_a & (self.active == self.LOCAL_HOLD)

    def publish_hold(self, route, ids, foot_local):
        # The previous edge, current waypoint and preview all represent a hold.
        # Only publish after the inherited reward scored the original A.
        idx = route.index[ids]
        for offset in (-1, 0, 1, 2):
            route.points[ids, idx + offset] = foot_local[ids]

    def record_landing(self, second_touch, within_radius, physical):
        groups = self.modes == self.active
        self.landings += (groups & second_touch).sum(-1)
        self.hits += (groups & second_touch & within_radius).sum(-1)
        self.deaths += (groups & physical).sum(-1)

    def record_release(self, launch, error):
        groups = (self.modes == self.active) & launch
        self.releases += groups.sum(-1)
        self.release_error_sum += torch.where(groups, error, 0.).sum(-1)

    def metrics(self):
        result = {}
        for i, name in enumerate(self.names):
            prefix = f'AB/{name}_'
            result.update({
                prefix+'starts': self.entered[i], prefix+'landings': self.landings[i],
                prefix+'hits_within_radius': self.hits[i], prefix+'physical_failures': self.deaths[i],
                prefix+'within_radius_rate': self.hits[i] / self.landings[i].clamp_min(1),
                prefix+'releases': self.releases[i],
                prefix+'mean_release_xy_error_m_s': self.release_error_sum[i] / self.releases[i].clamp_min(1),
            })
        return result
