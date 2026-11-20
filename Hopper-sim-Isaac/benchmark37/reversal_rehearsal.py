"""Training-only variable-heading hold, approach, long step and reversal sequences."""
import math

import torch


def rehearse(route, ids, fraction, reversal_kinds):
    if not 0. < fraction < 1. or route.evaluation:
        raise ValueError('Rehearsal requires an opt-in fraction on training routes')
    device=route.points.device
    kinds=route.sequence_kind[ids]
    eligible=(kinds==reversal_kinds[0]) | (kinds==reversal_kinds[1])
    chosen=ids[eligible & (torch.rand(len(ids),device=device,generator=route.rng)<fraction)]
    count=route.points.shape[1]
    blocks=(count+7)//8
    def draw():
        return torch.rand((len(chosen),blocks),device=device,generator=route.rng)
    def vector(angle):
        return torch.stack((angle.cos(),angle.sin()),-1)
    angle=draw()*2*math.pi
    approach=vector(angle+(.25+draw()*1.5)*math.pi)*(.04+.26*draw())[:,:,None]
    length=.4+.1*draw()
    length=torch.where(draw()<.4,.5,length)
    outward=approach+length[:,:,None]*vector(angle)
    returning=outward-length[:,:,None]*vector(angle+(draw()-.5)*math.pi/6)
    slots=torch.arange(8,device=device)[None,None,:]-(1+(draw()*3).long())[:,:,None]
    points=torch.zeros((len(chosen),blocks,8,2),device=device,dtype=route.points.dtype)
    for slot,point in enumerate((approach,outward,returning)):
        points=torch.where((slots==slot)[:,:,:,None],point[:,:,None,:],points)
    route.points[chosen]=points.flatten(1,2)[:,:count]
    route.sequence_kind[chosen]=8
