"""Training-only A/B commands and first-touch metrics for dedicated or mixed batches."""
import math

import torch


class ABPairs:
    def __init__(self,n,device,seed,focus=False,all_lanes=False,recovery_config=None,paired_axes=False,
                 opening_hold_probability=0., repeated_hold_fraction=0.):
        self.all_lanes = all_lanes
        if not 0. <= opening_hold_probability <= .5 or (opening_hold_probability and (not all_lanes or paired_axes)):
            raise ValueError('Opening hold rehearsal requires independent AB-only routes')
        self.opening_hold_probability = opening_hold_probability
        if not 0. <= repeated_hold_fraction <= 1. or (repeated_hold_fraction and (not all_lanes or paired_axes)):
            raise ValueError('Repeated holds require independent AB-only routes')
        self.repeated_hold_fraction = repeated_hold_fraction
        self.repeated_hold = torch.zeros(n, dtype=torch.bool, device=device)
        self.repeated_hold_starts = torch.zeros((), dtype=torch.long, device=device)
        self.recovery = None
        if recovery_config is not None:
            from .ab_recovery import ABRecovery
            if not all_lanes:
                raise ValueError('Two-touch recovery requires the AB-only course')
            self.recovery = ABRecovery(n, device, seed, recovery_config)
        self.paired_axes = None
        if paired_axes:
            if not all_lanes or self.recovery is None:
                raise ValueError('Paired-axis releases require full AB with missed-A recovery')
            from .ab_paired_axis import PairedAxisStarts
            self.paired_axes = PairedAxisStarts(n, device)
        self.mask = (torch.ones(n, dtype=torch.bool, device=device) if all_lanes
                     else torch.arange(n, device=device) % 8 == 7)
        self.a_index=torch.ones(n,device=device,dtype=torch.long)
        self.seen_a=torch.zeros_like(self.mask)
        self.hit_a=torch.zeros_like(self.mask)
        self.rng=torch.Generator(device=device).manual_seed(seed+80731)
        self.totals=torch.zeros(7,device=device,dtype=torch.long)
        self.focus=focus
        if focus and not all_lanes:
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
        if self.repeated_hold_fraction:
            # Full-AB lanes do not use draw[:, 8] for their warmup distance.
            # Convert a subset of existing zero-A starts to consecutive holds without moving other lanes or the RNG.
            hold_pair = (mode == 0) & (draw[:, 8] < self.repeated_hold_fraction)
            length_b = torch.where(hold_pair, 0., length_b)
            self.repeated_hold[ids] = hold_pair
            self.repeated_hold_starts += hold_pair.sum()
        relative=2*math.pi*draw[:,4]
        relative=torch.where(mode==2,0.,torch.where(mode==3,math.pi,relative))
        relative=torch.where(mode==4,torch.sign(draw[:,5]-.5)*math.pi/2,relative)
        def offset(direction,length):
            return torch.stack((direction.cos(),direction.sin()),dim=-1)*length[:,None]
        warm=torch.zeros_like(mode,dtype=torch.bool) if self.all_lanes else draw[:,6]<.5
        start=offset(2*math.pi*draw[:,7],.1+.4*draw[:,8])*warm[:,None]
        index=1+warm.long()
        a=start+offset(angle,length_a)
        b=a+offset(angle+relative,length_b)
        preview=torch.rand((len(ids),2),device=ids.device,generator=self.rng)
        c=b+offset(2*math.pi*preview[:,0],.5*preview[:,1])
        if self.repeated_hold_fraction:
            c = torch.where(hold_pair[:, None], b, c)
        route.points[ids,1]=start
        route.points[ids,index]=a
        route.points[ids,index+1]=b
        route.points[ids,index+2]=c
        if self.all_lanes:
            route.points[ids, 0] = 0.  # preceding point for the first moving edge
            # Give the policy a real stationary touchdown with a visible A
            # preview. A and B keep their original command and event indices.
            opening_hold = (draw[:, 6] < self.opening_hold_probability) & (length_a >= .40)
            route.index[ids] = torch.where(opening_hold, 0, index)
            route.points[ids, index+3] = c  # hold preview if B is advanced on the terminal tick
        self.a_index[ids]=index
        self.seen_a[ids]=False
        self.hit_a[ids]=False
        if self.recovery is not None:
            self.recovery.reset(ids)
        if self.paired_axes is not None:
            self.paired_axes.reset(route, ids)
            paired = ids[self.paired_axes.lanes[ids]]
            self.recovery.assignment[paired] = self.recovery.RETRY
        self.totals[0]+=len(ids)

    def touchdown(self,index,valid,strict):
        at_a=self.mask & valid & (index==self.a_index)
        first=at_a & ~self.seen_a
        self.hit_a |= first & strict
        self.seen_a |= at_a
        at_b=self.mask & valid & (index==self.a_index+1)
        if self.recovery is not None:
            at_b |= valid & (self.recovery.active != self.recovery.CONTINUE)
        paired=at_b & self.hit_a
        if self.focus and not self.all_lanes:
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
        if self.focus and not self.all_lanes:
            cold,hit_cold,warm,hit_warm=self.b_by_start.unbind()
            metrics.update({'AB/B_strict_direct_start':hit_cold/cold.clamp_min(1),
                            'AB/B_strict_after_moving_warmup':hit_warm/warm.clamp_min(1)})
        if self.recovery is not None:
            metrics.update(self.recovery.metrics())
        if self.repeated_hold_fraction:
            metrics['AB/repeated_hold_starts'] = self.repeated_hold_starts
        return metrics
