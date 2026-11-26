"""Local release shaping for a long commanded step immediately before reversal."""
import torch


class ReversalShortfall:
    """At the end of ascent, score only the predicted distance short of the landing disk."""
    def __init__(self,n,device,bias):
        self.bias=bias
        self.crossing=FirstFlightApex(n,device)
        self.foot_offset=torch.zeros(n,2,device=device)
        self.last_prediction=torch.full((n,),float('nan'),device=device)

    def reset(self,ids):
        self.crossing.reset(ids)
        self.last_prediction[ids]=float('nan')

    def step(self,launch,first,free,in_flight,velocity,root_xy,foot_xy,target,edge,preview,
             remaining,radius,weight):
        self.foot_offset.copy_(torch.where(launch[:,None],foot_xy-root_xy,self.foot_offset))
        self.last_prediction.masked_fill_(launch,float('nan'))
        reverse=torch.nn.functional.cosine_similarity(edge,preview,dim=-1)<-.866
        eligible=launch & first & reverse & (edge.norm(dim=-1)>=.40) & (preview.norm(dim=-1)>=.20)
        vz=velocity[:,2]
        sample=self.crossing.step(eligible,in_flight,free,vz-.2) & (vz>0.)
        projected=root_xy+self.foot_offset+velocity[:,:2]*remaining.clamp_min(0.)[:,None]
        along=((projected-target)*torch.nn.functional.normalize(edge,dim=-1)).sum(-1)+self.bias
        self.last_prediction.copy_(torch.where(sample,along,self.last_prediction))
        short=(-along-radius).clamp_min(0.)/radius
        return torch.where(sample,weight*torch.expm1(-.5*short.square()),0.)


class FirstFlightApex:
    """One apex per eligible release, using two consecutive free-flight samples."""
    def __init__(self,count,device):
        self.armed=torch.zeros(count,dtype=torch.bool,device=device)
        self.rising=torch.zeros_like(self.armed)

    def reset(self,ids):
        self.armed[ids]=False
        self.rising[ids]=False

    def step(self,release,in_flight,free,vz):
        self.armed &= in_flight
        self.armed |= release
        reached=self.armed & self.rising & free & (vz<=0)
        self.armed &= ~reached
        self.rising.copy_(self.armed & free & (vz>0))
        return reached


def launch_discount_scale(reference, velocity, edge, fade):
    axis=torch.nn.functional.normalize(edge,dim=-1)
    deficit=((reference-velocity)*axis).sum(-1)
    low,high=fade
    return ((high-deficit)/(high-low)).clamp(0.,1.)


def long_step_window_reference(reference, edge, eligible, discount, scale=None):
    axis=torch.nn.functional.normalize(edge,dim=-1)
    forward=(reference*axis).sum(-1).clamp(min=0.,max=discount)
    if scale is not None:
        forward=forward*scale
    active=eligible & (edge.norm(dim=-1)>=.40)
    return reference-torch.where(active,forward,0.)[:,None]*axis


def pre_reversal_geometry(edge, preview):
    reversal=torch.nn.functional.cosine_similarity(edge,preview,dim=-1)<-.866
    return reversal & (edge.norm(dim=-1)>=.40) & (preview.norm(dim=-1)>=.20)


def pre_reversal_loss(edge, preview, velocity, reference, flight, release, config):
    axis=torch.nn.functional.normalize(edge,dim=-1)
    eligible=release & pre_reversal_geometry(edge, preview)
    deficit=((reference-velocity)*axis).sum(-1)-config['allowance_m_s']
    excess=(deficit.clamp_min(0.)*flight/.05).square()
    return torch.where(eligible,config['weight']*torch.expm1(-.5*excess),0.)


class OpeningReversalRelease:
    """Latch only the A-release reference at drop contact, independently of rebound history."""

    def __init__(self, count, device):
        self.pending = torch.zeros(count, dtype=torch.bool, device=device)
        self.reference = torch.zeros(count, 2, device=device)
        self.counts = torch.zeros(3, dtype=torch.long, device=device)

    def reset(self, ids):
        self.pending[ids] = False

    def step(self, drop_contact, free_ascent, foot, target, edge, preview, velocity, flight, speed_limit, config):
        desired = (target - foot) / flight[:, None]
        desired *= (speed_limit / desired.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.)
        self.reference.copy_(torch.where(drop_contact[:, None], desired, self.reference))
        self.pending |= drop_contact
        launch = self.pending & free_ascent
        self.pending &= ~launch
        loss = pre_reversal_loss(edge, preview, velocity, self.reference, flight, launch, config)
        eligible = launch & pre_reversal_geometry(edge, preview)
        full_reverse = (eligible & (edge.norm(dim=-1) >= .5 - 1e-6)
                        & (preview.norm(dim=-1) >= .5 - 1e-6) & ((edge + preview).norm(dim=-1) < 1e-5))
        self.counts += torch.stack((eligible.sum(), full_reverse.sum(), (full_reverse & (loss < 0)).sum()))
        return loss

    def metrics(self):
        return dict(zip(('eligible', 'full_reverse', 'full_reverse_penalized'), self.counts.unbind()))
