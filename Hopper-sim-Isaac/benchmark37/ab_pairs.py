"""Training-only A/B commands and first-touch metrics for ordinary moving lanes."""
import math

import torch


class ABPairs:
    def __init__(self,n,device,seed,focus=False):
        self.mask=torch.arange(n,device=device)%8==7
        self.a_index=torch.ones(n,device=device,dtype=torch.long)
        self.seen_a=torch.zeros_like(self.mask)
        self.hit_a=torch.zeros_like(self.mask)
        self.rng=torch.Generator(device=device).manual_seed(seed+80731)
        self.totals=torch.zeros(7,device=device,dtype=torch.long)
        self.focus=focus
        if focus:
            self.b_by_start=torch.zeros(4,device=device,dtype=torch.long)

    def reset(self,route,ids):
        ids=ids[self.mask[ids]]
        draw=torch.rand((len(ids),9),device=ids.device,generator=self.rng)
        angle=2*math.pi*draw[:,0]
        mode=(draw[:,1]*10).long()
        if self.focus:
            # 25% each full straight and reverse; keep both holds and quarter turns.
            buckets=(draw[:,1]*20).long()
            lookup=draw.new_tensor([0,0,1,1,2,2,2,2,2,3,3,3,3,3,4,4,5,5,5,5],dtype=torch.long)
            mode=lookup[buckets]
            exact=((mode==2)|(mode==3)) & (draw[:,5]<.5)
            angle=torch.where(exact,(angle/(math.pi/4)).round()*(math.pi/4),angle)
        # 10% hold at A, 10% hold at B, 10% full straight, 10% full
        # reversal, 10% quarter turn. Other pairs use independent headings.
        length_a=torch.where(mode==0,0.,.5*draw[:,2])
        length_b=torch.where(mode==1,0.,.5*draw[:,3])
        full=(mode==2)|(mode==3)
        length_a=torch.where(full,.5,length_a)
        length_b=torch.where(full,.5,length_b)
        relative=2*math.pi*draw[:,4]
        relative=torch.where(mode==2,0.,torch.where(mode==3,math.pi,relative))
        relative=torch.where(mode==4,torch.sign(draw[:,5]-.5)*math.pi/2,relative)
        def offset(direction,length):
            return torch.stack((direction.cos(),direction.sin()),dim=-1)*length[:,None]
        warm=draw[:,6]<.5
        start=offset(2*math.pi*draw[:,7],.1+.4*draw[:,8])*warm[:,None]
        index=1+warm.long()
        a=start+offset(angle,length_a)
        b=a+offset(angle+relative,length_b)
        preview=torch.rand((len(ids),2),device=ids.device,generator=self.rng)
        c=b+offset(2*math.pi*preview[:,0],.5*preview[:,1])
        route.points[ids,1]=start
        route.points[ids,index]=a
        route.points[ids,index+1]=b
        route.points[ids,index+2]=c
        self.a_index[ids]=index
        self.seen_a[ids]=False
        self.hit_a[ids]=False
        self.totals[0]+=len(ids)

    def touchdown(self,index,valid,strict):
        at_a=self.mask & valid & (index==self.a_index)
        first=at_a & ~self.seen_a
        self.hit_a |= first & strict
        self.seen_a |= at_a
        at_b=self.mask & valid & (index==self.a_index+1)
        paired=at_b & self.hit_a
        if self.focus:
            warm=self.a_index==2
            self.b_by_start+=torch.stack([m.sum() for m in (at_b&~warm,at_b&~warm&strict,at_b&warm,at_b&warm&strict)])
        self.totals[1:]+=torch.stack([m.sum() for m in (first,first&strict,at_b,at_b&strict,paired,paired&strict)])
        return at_b

    def metrics(self):
        starts,a,hit_a,b,hit_b,paired,both=self.totals.unbind()
        metrics={'AB/episodes_started':starts,'AB/A_first_strict_rate':hit_a/a.clamp_min(1),
                'AB/B_touches':b,'AB/B_first_strict_rate':hit_b/b.clamp_min(1),
                'AB/B_strict_given_A_strict_and_B_reached':both/paired.clamp_min(1),
                'AB/both_first_strict':both}
        if self.focus:
            cold,hit_cold,warm,hit_warm=self.b_by_start.unbind()
            metrics.update({'AB/B_strict_direct_start':hit_cold/cold.clamp_min(1),
                            'AB/B_strict_after_moving_warmup':hit_warm/warm.clamp_min(1)})
        return metrics
