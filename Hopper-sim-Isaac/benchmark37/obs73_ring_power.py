"""Reduce one existing effort cost during a bounded second-hop powered ascent.

This changes rewards only. Motor commands, thrust, physics and observation
history are untouched; the net flight-power term remains nonpositive.
"""
import math

import torch


class AscentPowerRelief:
    def __init__(self,n,device,config):
        self.config=config
        if config['version']!='bounded_second_hop_ascent_power_v1':
            raise ValueError('Unknown ascent power reward')
        numeric=[config[k] for k in ('retained_fraction','duration_s','fade_out_s',
                                    'zero_relief_vz_m_s','full_relief_vz_m_s')]
        if (not all(math.isfinite(x) for x in numeric)
                or not 0<config['retained_fraction']<1
                or not 0<config['fade_out_s']<config['duration_s']
                or not 0<=config['zero_relief_vz_m_s']<config['full_relief_vz_m_s']):
            raise ValueError('Invalid bounded ascent power reward parameters')
        self.started=torch.zeros(n,dtype=torch.bool,device=device)
        self.elapsed=torch.zeros(n,device=device)
        self.total=torch.zeros((),device=device)
        self.active_steps=torch.zeros((),dtype=torch.long,device=device)

    def reset(self,ids):
        self.started[ids]=False
        self.elapsed[ids]=0.

    def step(self,native_cost,active,airborne,spring_free,corridor,vz,unsafe,passed,dt):
        start=active & airborne & spring_free & ~self.started
        self.started |= start
        self.elapsed[start]=0.
        c=self.config
        fade=((c['duration_s']-self.elapsed)/c['fade_out_s']).clamp(0.,1.)
        upward=((vz-c['zero_relief_vz_m_s'])
                /(c['full_relief_vz_m_s']-c['zero_relief_vz_m_s'])).clamp(0.,1.)
        allowed=(active & self.started & airborne & spring_free & corridor
                 & ~unsafe & ~passed & torch.isfinite(vz))
        weight=torch.where(allowed,fade*upward,0.)
        # native_cost already includes its original scale and control dt.
        # Adding this partial cancellation leaves between25% and100% of the
        # original cost; there is no positive thrust/power payment.
        delta=(c['retained_fraction']-1.)*native_cost.clamp(max=0.)*weight
        self.total+=delta.sum()
        self.active_steps+=(delta>0).sum()
        # Time always advances after takeoff, even outside the corridor or
        # during a contact pause. It cannot restart in the same episode.
        self.elapsed+=torch.where(active & self.started,dt,0.)
        return delta
