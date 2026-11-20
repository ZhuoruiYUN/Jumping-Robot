"""Whole-flight quality for stationary direct-motor touchdown payments only."""
import math

import torch


class FlightQuality:
    def __init__(self, n, device, config):
        self.config = config
        for key in ('span_free_deg', 'span_width_deg', 'rate_free_rad_s', 'rate_width_rad_s',
                    'free_spring_qdot_max_m_s'):
            if not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'Invalid flight-quality setting: {key}')
        self.tracking = torch.zeros(n, dtype=torch.bool, device=device)
        self.low = torch.zeros(n, 2, device=device)
        self.high = self.low.clone()
        self.rate_sq = torch.zeros(n, device=device)
        self.pitch_rate_sq = self.rate_sq.clone()
        self.samples = self.rate_sq.clone()
        refinement = config.get('settled_pitch_refinement')
        if refinement is not None:
            mode = refinement.get('mode', 'multiply')
            if mode not in ('multiply', 'bounded_cost'):
                raise ValueError('Invalid settled pitch refinement mode')
            if mode == 'bounded_cost' and (not math.isfinite(refinement['cost_scale']) or refinement['cost_scale'] <= 0):
                raise ValueError('Invalid settled pitch cost scale')
            tail = refinement.get('span_tail_scale', 0.)
            if (not math.isfinite(tail) or tail < 0 or (tail and mode != 'bounded_cost')):
                raise ValueError('Pitch span tail requires a nonnegative scale and bounded_cost mode')
            if not isinstance(refinement['warmup_landings'], int) or refinement['warmup_landings'] < 0:
                raise ValueError('Invalid settled pitch warmup count')
            for key in ('span_free_deg', 'span_width_deg', 'rate_free_rad_s', 'rate_width_rad_s'):
                if not math.isfinite(refinement[key]) or refinement[key] <= 0:
                    raise ValueError(f'Invalid settled pitch setting: {key}')

    def reset(self, ids):
        self.tracking[ids] = False
        for value in (self.low, self.high, self.rate_sq, self.pitch_rate_sq, self.samples):
            value[ids] = 0.

    def step(self, quaternion, omega_body, phase, spring, spring_rate, physical, valid, precision_payment,
             completed_landings=None):
        # Only pre-contact flight contributes: touchdown impulses and the
        # initial release cannot inflate the flight angular-rate statistic.
        contact = spring > .002
        active = (((phase == 2) | (phase == 3)) & ~contact & ~physical
                  & (spring_rate.abs() <= self.config['free_spring_qdot_max_m_s']))
        w, x, y, z = quaternion.unbind(-1)
        angles = torch.stack((torch.atan2(2*(w*x+y*z), 1-2*(x.square()+y.square())),
                              torch.asin((2*(w*y-z*x)).clamp(-1., 1.))), -1)
        start = active & ~self.tracking
        self.low = torch.where(start[:, None], angles, self.low)
        self.high = torch.where(start[:, None], angles, self.high)
        self.rate_sq = torch.where(start, 0., self.rate_sq)
        self.pitch_rate_sq = torch.where(start, 0., self.pitch_rate_sq)
        self.samples = torch.where(start, 0., self.samples)
        self.tracking |= start
        self.low = torch.where(active[:, None], torch.minimum(self.low, angles), self.low)
        self.high = torch.where(active[:, None], torch.maximum(self.high, angles), self.high)
        self.rate_sq += torch.where(active, omega_body[:, :2].square().sum(-1), 0.)
        self.pitch_rate_sq += torch.where(active, omega_body[:, 1].square(), 0.)
        self.samples += active.float()
        span = torch.rad2deg((self.high-self.low).amax(-1))
        rms = torch.sqrt(self.rate_sq/self.samples.clamp(min=1.))
        c = self.config
        span_excess = (span-c['span_free_deg']).clamp(min=0.)/c['span_width_deg']
        rate_excess = (rms-c['rate_free_rad_s']).clamp(min=0.)/c['rate_width_rad_s']
        complete = valid & self.tracking & ~physical
        quality = torch.where(complete, torch.exp(-.5*(span_excess.square()+rate_excess.square())), 0.)
        pitch_span = torch.rad2deg(self.high[:, 1]-self.low[:, 1])
        pitch_rms = torch.sqrt(self.pitch_rate_sq/self.samples.clamp(min=1.))
        settled = torch.zeros_like(complete)
        settled_cost = torch.zeros_like(quality)
        refinement = c.get('settled_pitch_refinement')
        if refinement is not None:
            if completed_landings is None:
                raise ValueError('Settled pitch refinement requires the existing completed landing count')
            settled = complete & (completed_landings >= refinement['warmup_landings'])
            excess_span = (pitch_span-refinement['span_free_deg']).clamp(min=0.)/refinement['span_width_deg']
            excess_rate = (pitch_rms-refinement['rate_free_rad_s']).clamp(min=0.)/refinement['rate_width_rad_s']
            pitch_quality = torch.exp(-.5*(excess_span.square()+excess_rate.square()))
            if refinement.get('mode', 'multiply') == 'bounded_cost':
                # Keep the complete v12 precision quality. This base event
                # cost is capped independently of height/foot payment; the
                # opt-in span tail below continues to charge larger swings.
                settled_cost = torch.where(settled, -refinement['cost_scale']*(1.-pitch_quality), 0.)
                if refinement.get('span_tail_scale', 0.):
                    # Preserve the old cost up to one span-width beyond the
                    # deadband (6deg); larger swings must not all cost ~4.
                    tail = refinement['span_tail_scale']*(excess_span.square()-1.).clamp(min=0.)
                    settled_cost -= torch.where(settled, tail, 0.)
            else:
                quality = quality*torch.where(settled, pitch_quality, 1.)
        # An event payment, not a cost per control step. Height/energy and the
        # strict evaluator are independent and retain their existing meanings.
        discount = torch.where(valid, -(1.-quality)*precision_payment.clamp(min=0.), 0.)
        self.tracking &= ~(contact | physical | (phase == 0))
        return dict(discount=discount, quality=quality, span_deg=span,
                    rate_rms_rad_s=rms, complete=complete, settled=settled,
                    pitch_span_deg=pitch_span, pitch_rate_rms_rad_s=pitch_rms,
                    settled_pitch_cost=settled_cost)
