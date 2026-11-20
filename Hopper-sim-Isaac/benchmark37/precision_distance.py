"""Training-only per-lane progression; new difficulty applies only at route reset."""
import torch


class DistancePractice:
    def __init__(self, count, device):
        self.ids=torch.arange(count,device=device)
        self.mask=self.ids%8==3
        self.level=torch.zeros_like(self.ids)
        self.caps=torch.tensor([.35,.425,.5],device=device)
        self.last=torch.full_like(self.ids,-1)
        self.attempts=torch.zeros_like(self.ids)
        self.hits=torch.zeros_like(self.ids)
        self.died=torch.zeros_like(self.mask)

    def observe(self, route, valid, strict, physical):
        index=route.index
        edge=route.points[self.ids,index]-route.points[self.ids,(index-1).clamp_min(0)]
        scored=valid & self.mask & (index>0) & (index!=self.last) & (edge.norm(dim=-1)>=.8*self.caps[self.level]-1e-6)
        self.attempts+=scored
        self.hits+=scored & strict
        self.last.copy_(torch.where(valid,index,self.last))
        self.died|=physical & self.mask

    def reset(self, route, ids):
        chosen=ids[self.mask[ids]]
        attempts=self.attempts[chosen]
        hits=self.hits[chosen]
        enough=attempts>=6
        up=enough & (hits*5>=attempts*4) & ~self.died[chosen]
        down=(enough & (hits*2<attempts)) | self.died[chosen]
        self.level[chosen]=(self.level[chosen]+up.long()-down.long()).clamp(0,len(self.caps)-1)
        scale=self.caps[self.level[chosen]]/.5
        route.points[chosen]=route.points[chosen]*scale[:,None,None]
        self.last[chosen]=-1
        self.attempts[chosen]=0
        self.hits[chosen]=0
        self.died[chosen]=False
