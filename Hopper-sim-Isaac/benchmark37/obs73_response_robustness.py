"""Episode-persistent uncertainty in roll/pitch response for OBS73 fine-tuning.

Uses the existing dimensionless inertia randomization channel (torque / DR),
without editing the nominal robot, thrust, motor timing, actions or rewards.
This is an engineering response envelope, not a newly identified motor model.
"""
import math

import torch


CASE_VERSION = 'obs73_vertical_response64_nominal64_weaker_v1'


class ResponseEnvelope:
    def __init__(self, n, device, config, seed, evaluation=False):
        self.config = config
        self.evaluation = evaluation
        low, high = config['roll_pitch_gain_range']
        self.training_gain_range = config.get('training_roll_pitch_gain_range', (low, high))
        self.tail_fraction = config.get('training_tail_fraction_of_weaker', 0.)
        training_low, training_high = self.training_gain_range
        if (not all(math.isfinite(x) for x in (low, high)) or not 0 < low < high <= 1
                or config['nominal_fraction'] != .5):
            raise ValueError('Expected bounded weaker response and50% nominal episodes')
        if not low <= training_low < training_high <= high:
            raise ValueError('Training response range must lie within the evaluation envelope')
        if (not math.isfinite(self.tail_fraction) or not 0. <= self.tail_fraction < 1.
                or (self.tail_fraction and training_low == low)):
            raise ValueError('Training response tail needs a fraction below one and an unused gain interval')
        self.gain = torch.ones(n, device=device)
        self.base_inertia = torch.ones(n, 3, device=device)
        self.rng = torch.Generator(device=device).manual_seed(int(seed)+731118)

    def evaluation_gains(self, ids):
        """One128-case gate, split by case ID; single-environment play is nominal."""
        result = torch.ones(len(ids), device=self.gain.device)
        if len(self.gain) != 128:
            return result
        low, high = self.config['roll_pitch_gain_range']
        weak = ids >= 64
        # Midpoints avoid putting a nominal endpoint in the weaker stratum.
        result[weak] = low+(high-low)*(ids[weak]-64+.5)/64
        return result

    def apply(self, ids, inertia, base, gains):
        ids = ids.to(device=self.gain.device, dtype=torch.long)
        base = base.to(device=self.gain.device, dtype=inertia.dtype).clone()
        gains = gains.to(device=self.gain.device, dtype=inertia.dtype)
        low, high = self.config['roll_pitch_gain_range']
        if (base.shape != (len(ids), 3) or gains.shape != (len(ids),)
                or not torch.isfinite(base).all() or (base <= 0).any()
                or not torch.isfinite(gains).all()
                or ((gains != 1) & ((gains < low) | (gains > high))).any()):
            raise ValueError('Invalid saved response case')
        self.base_inertia[ids] = base
        self.gain[ids] = gains
        effective = base.clone()
        effective[:, :2] /= gains[:, None]
        inertia[ids] = effective

    def reset(self, ids, inertia, full_range_lanes=None):
        if self.evaluation:
            gains = self.evaluation_gains(ids)
        else:
            low, high = self.training_gain_range
            draw = torch.rand(len(ids), generator=self.rng, device=self.gain.device)
            gains = low + (high - low) * draw
            group_draw = torch.rand(len(ids), generator=self.rng, device=self.gain.device)
            if self.tail_fraction:
                evaluation_low = self.config['roll_pitch_gain_range'][0]
                # Reserve part of the old weaker group for the uncovered gains.
                # The conditional draw and all other gains retain their old values.
                tail = (group_draw >= .5) & (group_draw < .5 + .5 * self.tail_fraction)
                gains = torch.where(tail, evaluation_low + (low - evaluation_low) * draw, gains)
            if full_range_lanes is not None:
                full_low, full_high = self.config['roll_pitch_gain_range']
                gains = torch.where(full_range_lanes, full_low + (full_high - full_low) * draw, gains)
            nominal = group_draw < .5
            gains[nominal] = 1.
        # The base environment already resampled its original DR for these IDs.
        self.apply(ids, inertia, inertia[ids], gains)

    def restore(self, cases, inertia):
        ids = torch.arange(len(cases), device=self.gain.device)
        saved = [c.get('response_gain') is not None for c in cases]
        if any(saved) and not all(saved):
            raise ValueError('Mixed response case formats')
        if all(saved):
            base = torch.tensor([c['base_inertia_multiplier'] for c in cases], device=self.gain.device)
            gains = torch.tensor([c['response_gain'] for c in cases], device=self.gain.device)
            if not torch.allclose(gains, self.evaluation_gains(ids)):
                raise ValueError('Saved cases differ from the fixed64/64 response gate')
        else:
            base = torch.tensor([c['inertia_multiplier'] for c in cases], device=self.gain.device)
            gains = self.evaluation_gains(ids)
        # Apply to saved BASE values, never to already weakened effective DR.
        self.apply(ids, inertia, base, gains)

    def annotate(self, cases):
        for case in cases:
            i = case['case_id']
            case.update(response_gain=float(self.gain[i]),
                        base_inertia_multiplier=self.base_inertia[i].cpu().tolist(),
                        response_group='nominal' if i < 64 else 'weaker')

    def log(self):
        result = {'Response/roll_pitch_gain_mean': self.gain.mean(),
                  'Response/roll_pitch_gain_min': self.gain.min(),
                  'Response/weaker_fraction': (self.gain < 1).float().mean()}
        if self.tail_fraction and not self.evaluation:
            result['Response/below_core_gain_fraction'] = (self.gain < self.training_gain_range[0]).float().mean()
        return result
