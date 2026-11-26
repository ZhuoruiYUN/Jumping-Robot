"""Training-only airborne release perturbations for direct A/B episodes."""
import math

import torch


class ABSpawn:
    def __init__(self, count, device, seed, config):
        self.config = config
        if not 0 < config['probability'] < 1:
            raise ValueError('AB spawn probability must preserve both release groups')
        for key in ('xy_radius_m', 'roll_pitch_deg', 'yaw_deg', 'xy_speed_m_s', 'z_speed_m_s',
                    'roll_pitch_rate_rad_s', 'yaw_rate_rad_s'):
            if not math.isfinite(config[key]) or config[key] < 0:
                raise ValueError(f'Invalid AB spawn bound: {key}')
        self.rng = torch.Generator(device=device).manual_seed(seed + 255803)
        self.perturbed = torch.zeros(count, dtype=torch.bool, device=device)
        self.resets = torch.zeros(2, dtype=torch.long, device=device)
        self.touchdowns = torch.zeros((2, 2, 2), dtype=torch.long, device=device)

    def sample(self, ids, paired=None):
        c = self.config
        draw = torch.rand((len(ids), 12), generator=self.rng, device=ids.device)
        if paired is not None:
            draw = paired.match(ids, "spawn", draw)
        selected = draw[:, 0] < c['probability']
        self.perturbed[ids] = selected
        self.resets += torch.stack(((~selected).sum(), selected.sum()))
        draw = draw[selected]

        def disk(radius, angle, limit):
            heading = angle * (2 * math.pi)
            return (limit * radius.sqrt()[:, None]) * torch.stack((heading.cos(), heading.sin()), dim=-1)

        offset = disk(draw[:, 1], draw[:, 2], c['xy_radius_m'])
        angles = (2 * draw[:, 3:6] - 1) * draw.new_tensor(
            [c['roll_pitch_deg'], c['roll_pitch_deg'], c['yaw_deg']]).deg2rad()
        velocity = torch.cat((disk(draw[:, 6], draw[:, 7], c['xy_speed_m_s']),
                              (2 * draw[:, 8:9] - 1) * c['z_speed_m_s'],
                              disk(draw[:, 9], draw[:, 10], c['roll_pitch_rate_rad_s']),
                              (2 * draw[:, 11:12] - 1) * c['yaw_rate_rad_s']), dim=-1)
        return selected, offset, angles, velocity

    def record_touchdowns(self, a_touch, b_touch, strict):
        # Axes: original/perturbed release, A/B, attempts/strict hits.
        phase = torch.stack((a_touch, b_touch))
        groups = torch.stack((~self.perturbed, self.perturbed))[:, None, :] & phase
        self.touchdowns += torch.stack((groups.sum(-1), (groups & strict).sum(-1)), dim=-1)

    def metrics(self):
        return {f'AB/{group}_{hop}_strict_rate': hits / count.clamp_min(1)
                for g, group in enumerate(('original_spawn', 'perturbed_spawn'))
                for hop, (count, hits) in zip(('A', 'B'), self.touchdowns[g])}
