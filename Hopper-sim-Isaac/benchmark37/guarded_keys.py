"""Batched keyboard route for simulation. Distances in release/world axes.

An index here counts scored contacts, not commands. The FIFO has its own
accounting: a paused/recovery hop never earns a commanded arrival.
"""
import math
import torch

HOLD, PREVIEW, STEP, PAUSED, RECOVER = range(5)
FIELDS = ('requested','ignored_recover','launched','arrived','strict_arrived','cancelled',
          'pauses','resumes','recoveries','paused_ticks','landings','outside_10cm',
          'paused_landings','recovery_landings','completed_recoveries','pause_yaw_outside','pause_predicted_yaw_outside','pause_rp_excess',
          'pause_missing_approach','pending_at_end')


class GuardedKeys:
    def __init__(self, route, dt, evaluation, training_moving_fraction=.25, training_pair_gap_s=0.):
        self.route, self.dt = route, dt
        self.radius = getattr(route,'config',{}).get('strict_foot_m',.10)
        self.fields = tuple(f'outside_{self.radius*100:g}cm' if key=='outside_10cm' else key for key in FIELDS)
        n = len(route.index); device = route.index.device
        self.ids = torch.arange(n, device=device)
        if training_moving_fraction not in (.25, .375, .5):
            raise ValueError('Guarded-key training share of moving lanes must be 1/4, 3/8 or 1/2')
        self.mask = torch.ones_like(self.ids,dtype=torch.bool) if evaluation else (self.ids % 8 == 1) & ~route.stationary_mask
        self.direction = (self.ids if evaluation else self.ids // 8) % 4
        if not evaluation and training_moving_fraction > .25:
            # Original keyboard lanes keep their commands and start times.
            # Add one or two moving lanes per block of sixteen; balance their
            # first command across all four headings over the training batch.
            stride = 16 if training_moving_fraction == .375 else 8
            extra = (self.ids % stride == 5) & ~route.stationary_mask
            self.mask |= extra
            self.direction = torch.where(extra, (self.ids // stride) % 4, self.direction)
        self.state, self.head, self.tail, self.since, self.ticks = [torch.zeros_like(self.ids) for _ in range(5)]
        self.first = torch.zeros_like(self.mask)
        self.terminal_hold = torch.zeros_like(self.mask)
        self.xy = route.points[:,0].clone(); self.next = self.xy.clone()
        self.fifo = route.points.new_zeros(n,32,2)  # at most 7 bursts of 3 per 60 s
        self.rp, self.yaw_rate = [self.xy[:,0].clone() for _ in range(2)]
        self.step_dir=torch.zeros_like(self.ids)
        self.by_key=torch.zeros(n,4,4,device=device,dtype=torch.long)  # launch, arrival, strict arrival, miss
        self.count = torch.zeros(n,len(FIELDS),device=device,dtype=torch.long)
        self.peak_rate = self.xy[:,0].clone(); self.peak_error = self.peak_rate.clone()
        self.peak_yaw = self.peak_rate.clone()
        self.dirs = self.xy.new_tensor([[1,0],[-1,0],[0,1],[0,-1]])
        self.start = round(2.60/dt) + (0 if evaluation else (self.ids//32 % 11)*round(.07/dt))
        self.period = round(9./dt); self.min_preview = math.ceil(.45/dt-1e-8)
        self.pair_gap = 0 if evaluation else round(training_pair_gap_s/dt)
        if training_pair_gap_s and not 0 < round(training_pair_gap_s/dt) < self.min_preview:
            raise ValueError('Paired presses must arrive on distinct ticks within one preview window')
        if self.pair_gap:
            self.pair_open = torch.zeros_like(self.mask)
            self.tickets = torch.full((n,2),-1,dtype=torch.long,device=device)
            self.pair_hits = torch.zeros((n,3),dtype=torch.bool,device=device)
            self.pair_totals = torch.zeros(3,dtype=torch.long,device=device)
            self.second_command = torch.zeros_like(self.mask)
            self.second_in_flight = torch.zeros_like(self.mask)
            self.second_ticket = torch.zeros_like(self.fifo[:,:,0],dtype=torch.bool)
            self.second_results = torch.zeros(4,2,dtype=torch.long,device=device)

    def observe_free_liftoff(self, launch):
        self.second_in_flight |= self.second_command & launch & (self.state==STEP)

    def _finish_second(self, ids, inside):
        # A flight can cross a rollout or a keyboard window. Count it once,
        # on measured touchdown, or as a miss if the episode ends in flight.
        flew = self.second_in_flight[ids]
        outcomes = torch.stack((flew, flew & inside),dim=1).long()
        self.second_results.index_add_(0,self.step_dir[ids],outcomes)
        self.second_in_flight[ids] = False

    def _close_pairs(self, ids):
        opened = self.pair_open[ids]
        flags = self.pair_hits[ids]
        self.pair_totals += torch.stack((opened.sum(), (opened & flags[:,0]).sum(),
                                         (opened & flags[:,1] & flags[:,2]).sum()))
        self.pair_open[ids] = False
        self.tickets[ids] = -1
        self.pair_hits[ids] = False

    def log_pairs(self, writer, iteration):
        windows, launches, arrivals = self.pair_totals.tolist()
        self.pair_totals.zero_()
        writer.add_scalar('Keyboard/completed_pair_windows',windows,iteration)
        if windows:
            writer.add_scalar('Keyboard/first_launch_rate',launches/windows,iteration)
            writer.add_scalar('Keyboard/both_arrived_rate',arrivals/windows,iteration)
        for prefix,counts in (('Keyboard/',self.second_results.sum(0)),('Keyboard/neg_y_',self.second_results[3])):
            started, inside = counts.tolist()
            writer.add_scalar(prefix+'second_liftoffs_resolved',started,iteration)
            if started:
                writer.add_scalar(prefix+f'second_liftoff_{100*self.radius:g}cm_rate',inside/started,iteration)
        self.second_results.zero_()

    def reset(self, ids):
        ids = ids[self.mask[ids]]
        if self.pair_gap:
            self._close_pairs(ids)
            self._finish_second(ids,False)
            self.second_command[ids] = False
        self.terminal_hold[ids]=False
        for x in (self.state,self.head,self.tail,self.ticks,self.first,self.count,self.by_key,self.step_dir,
                  self.xy,self.next,self.peak_rate,self.peak_error,self.peak_yaw): x[ids]=0
        self.since[ids]=-100000; self.rp[ids]=torch.inf; self.yaw_rate[ids]=torch.inf
        self.route.points[ids,:2]=0.

    def publish(self, ids):
        # The FIFO stores absolute positions, like the deployed queue. No
        # rotation of W/A/S/D when the body yaws. Only headings are body-relative.
        self.route.points[ids,self.route.index[ids]]=self.xy[ids]
        self.route.points[ids,self.route.index[ids]+1]=self.next[ids]

    def update(self, valid, strict, reached, foot, error, yaw, heading_rate, rp, descending, free, physical):
        active=self.mask & ~physical; hit=valid & active; self.ticks+=1
        capture=active & descending
        self.rp=torch.where(capture,rp,self.rp); self.yaw_rate=torch.where(capture,heading_rate,self.yaw_rate)
        ready=(yaw.abs()<=10)&((yaw+.14*self.yaw_rate).abs()<=10)&(self.rp<=1.2)
        self.peak_rate=torch.maximum(self.peak_rate,torch.where(active&free,rp,0.))
        self.peak_yaw=torch.maximum(self.peak_yaw,torch.where(active&free,yaw.abs(),0.))
        self.peak_error=torch.maximum(self.peak_error,torch.where(hit,error,0.))
        # Keyboard input is processed before the landing handler.
        age=self.ticks-self.start; burst=active & (age>=0) & (age.remainder(self.period)==0)
        presses=3
        if self.pair_gap:
            first_press = burst
            burst = burst | (active & (age>=0) & (age.remainder(self.period)==self.pair_gap))
            presses = 1
        if hasattr(self,'two_press_ticks'):
            first_tick,second_tick=self.two_press_ticks
            burst=active & ((self.ticks==first_tick)|(self.ticks==second_tick))
            presses=1
        if not (hit|burst).any():
            self.count[:,9]+=active & (self.state==PAUSED)
            return self.ids[:0],torch.zeros_like(hit)
        request=burst & (self.state!=RECOVER); rid=self.ids[request]
        if self.pair_gap:
            # An ignored first press is still a requested pair. Absolute FIFO
            # tickets distinguish late arrivals from the previous command window.
            opened = self.ids[first_press]
            self._close_pairs(opened)
            self.pair_open[opened] = True
            self.tickets[rid,(~first_press[rid]).long()] = self.tail[rid]
            self.second_ticket[rid,self.tail[rid]%32] = ~first_press[rid] & (self.tickets[rid,0]>=0)
        if len(rid):
            empty=self.tail[rid]==self.head[rid]; show=empty & (self.state[rid]!=PAUSED)
            base=torch.where(empty[:,None],self.xy[rid],self.fifo[rid,(self.tail[rid]-1)%32])
            sign=torch.ones_like(rid) if hasattr(self,'two_press_ticks') else 1-2*((age[rid]//self.period)%2)
            edge=.5*sign[:,None]*self.dirs[self.direction[rid]]
            for j in range(presses): self.fifo[rid,(self.tail[rid]+j)%32]=base+(j+1)*edge
            self.tail[rid]+=presses
            sid=rid[show]; self.next[sid]=self.fifo[sid,self.head[sid]%32]; self.since[sid]=self.ticks[sid]
            self.state[rid]=torch.where(self.state[rid]==HOLD,PREVIEW,self.state[rid])
        old=self.state.clone(); self.first|=hit&strict
        pending=self.tail-self.head
        arrived=hit & (old==STEP) & reached
        miss=hit & ((old==STEP)|(old==PREVIEW)|(old==PAUSED)) & ~reached
        paused=hit & reached & ~ready & ((old==PREVIEW)|((old==STEP)&(pending>0)))
        resume=hit & reached & ready & self.first & (old==PAUSED)
        recovered=hit & strict & ready & (old==RECOVER)
        elapsed=self.ticks-self.since>=self.min_preview
        launch=hit & ready & reached & elapsed & (((old==PREVIEW)&self.first)|((old==STEP)&(pending>0)))
        if self.pair_gap:
            self.pair_hits[:,0] |= launch & (self.head==self.tickets[:,0])
            arrived_ticket = self.head-1
            self.pair_hits[:,1:] |= arrived[:,None] & (arrived_ticket[:,None]==self.tickets)
            landed_ids = self.ids[hit]
            self._finish_second(landed_ids,error[landed_ids]<=self.radius)
            self.second_command[landed_ids] = (launch[landed_ids]
                & self.second_ticket[landed_ids,self.head[landed_ids]%32])
        self.state[recovered | (arrived & (pending==0))]=HOLD
        self.state[arrived & ready & (pending>0) & ~elapsed]=PREVIEW
        self.state[paused]=PAUSED; self.next[paused]=self.xy[paused]
        self.state[resume]=PREVIEW; self.next[resume]=self.fifo[self.ids[resume],self.head[resume]%32]; self.since[resume]=self.ticks[resume]
        for col,mask in ((1,arrived),(2,arrived&strict),(3,miss&(old==STEP))):
            sel=self.ids[mask];self.by_key[sel,self.step_dir[sel],col]+=1
        lid=self.ids[launch]; dest=self.fifo[lid,self.head[lid]%32]
        if len(lid):
            self.step_dir[lid]=((dest-self.xy[lid])@self.dirs.T).argmax(-1)
            self.by_key[lid,self.step_dir[lid],0]+=1
        self.xy[lid]=dest; self.head[lid]+=1
        self.next[lid]=torch.where((self.tail[lid]>self.head[lid])[:,None],self.fifo[lid,self.head[lid]%32],self.xy[lid]); self.since[lid]=self.ticks[lid]; self.state[lid]=STEP
        self.state[miss]=RECOVER; self.xy[miss]=foot[miss]; self.next[miss]=foot[miss]; self.head[miss]=self.tail[miss]; self.first[miss]=False
        self.terminal_hold=(self.terminal_hold | (arrived & (pending==0))) & (self.state==HOLD) & (self.head==self.tail)
        for j,v in enumerate((presses*burst,presses*(burst&~request),launch,arrived,arrived&strict,
                              torch.where(miss,pending+(old==STEP).long(),0),paused,resume,miss,
                              active&(old==PAUSED),hit,hit&(error>self.radius),hit&(old==PAUSED),hit&(old==RECOVER),recovered,
                              paused&(yaw.abs()>10),paused&torch.isfinite(self.yaw_rate)&((yaw+.14*self.yaw_rate).abs()>10),
                              paused&torch.isfinite(self.rp)&(self.rp>1.2),paused&~torch.isfinite(self.rp+self.yaw_rate))):
            self.count[:,j]+=v
        self.rp[hit]=torch.inf; self.yaw_rate[hit]=torch.inf
        self.count[:,-1]=self.tail-self.head+(self.state==STEP).long()
        return self.ids[hit|request],launch
