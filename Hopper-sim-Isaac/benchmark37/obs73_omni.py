"""OBS73 short omnidirectional bridge from the stationary v23 foundation.

Reuse the historical pre-visible two-ground-point queue, with physical misses
retaining both targets. The ring is nuisance observation context only.
"""
import math

import torch

TASK='obs73_omni_ground_targets_v1'
CASE_VERSION='obs73_omni_short_response64_64_first128_v1'
SERIES='benchmark73_omni_v1'
DIRECT_REVERSAL_KIND = 6
PAUSED_REVERSAL_KIND = 7


def navigation_advance(valid,strict,error,tilt,index,stationary,config=None,foot_limit_m=.1):
    """Keep height scoring strict without repeating a reached moving point."""
    if not config:return valid&strict
    if config.get('version')!='position_and_tilt_after_first_hop_v1':
        raise ValueError('Unknown moving navigation policy')
    reached=valid&(error<=foot_limit_m)&(tilt<=15.)&(index>0)&~stationary
    return (valid&strict)|reached


def case_version(course):
    if course.get('omni',{}).get('case_version'):
        return course['omni']['case_version']
    if course.get('observation_version')=='direct_motor_crossing_quaternion73_history40_20_v1':
        return 'obs73_quaternion_omni_response64_64_first128_v1'
    if course.get('omni',{}).get('distance_curriculum'):
        return 'obs72_omni_range08_response64_64_first128_v1'
    return CASE_VERSION


def distance_bins(course):
    """Half-open reporting intervals, with the final sampled ceiling included."""
    c=course['omni']
    edges=c.get('evaluation_distance_bin_edges_m',[.1,.2,.4,.6,.8])
    low,high=c['moving_distance_range_m']
    if (len(edges)<2 or not all(math.isfinite(v) for v in edges)
            or any(a>=b for a,b in zip(edges,edges[1:]))
            or edges[0]>=low or edges[-1]<high):
        raise ValueError('Distance reporting bins must cover the configured moving range')
    return list(zip(edges,edges[1:]))


def straight_eval_spec(course,spacing,jumps):
    """Explicit evaluation route; never mutate the saved training course."""
    low,high=course['omni']['moving_distance_range_m']
    count=course['omni']['route_points']
    if (not math.isfinite(spacing) or not low<=spacing<=high
            or not isinstance(jumps,int) or not 1<=jumps<count-1):
        raise ValueError('Straight evaluation needs in-range spacing and1..route_points-2 jumps')
    return dict(version='obs73_straight_x_response64_64_first128_v1',
        evaluation_scope=f'straight_x_step{spacing:g}_jumps{jumps}',
        spacing_m=spacing,requested_moving_targets=jumps,route_points=count,
        route_scope='First hop stationary; constant world+X spacing, no holds. Continue along+X for the entire15s; score first requested targets, all15s landing errors/yaw/deaths.')


class GroundRoute:
    def __init__(self,n,device,config,seed,evaluation=False,training_mixture=None):
        self.config=config;self.evaluation=evaluation
        self.waypoint_source = None
        low,high=config['moving_distance_range_m']
        self.curriculum=config.get('distance_curriculum')
        ceiling=.8 if self.curriculum else .2
        if not (.1<low<=high<=ceiling and config['zero_probability']==.1
                and config['workspace_radius_m']>=2*high and config['route_points']>=24):
            raise ValueError('Invalid omni distance/workspace contract; ranges above20cm need an explicit curriculum')
        if self.curriculum:
            c=self.curriculum
            if not (c.get('version')=='linear_max_distance_v1'
                    and low<=c['initial_max_m']<=high and c['final_max_m']==high
                    and c['ramp_updates']>0 and c['steps_per_update']==256):
                raise ValueError('Invalid explicit omni distance curriculum')
        self.rng=torch.Generator(device=device).manual_seed(int(seed)+1800427)
        self.points=torch.zeros(n,config['route_points'],2,device=device)
        self.index=torch.zeros(n,dtype=torch.long,device=device)
        self.advances=torch.zeros((),dtype=torch.long,device=device)
        self.strict_advances=torch.zeros_like(self.advances)
        self.advances_without_strict=torch.zeros_like(self.advances)
        self.misses=torch.zeros((),dtype=torch.long,device=device)
        self.sampled_max=torch.zeros(n,device=device)
        self.training_mixture=training_mixture
        self.sequence_kind=torch.zeros(n,dtype=torch.long,device=device)
        long_steps = config.get('training_long_steps')
        if long_steps and (training_mixture
                or not low < long_steps['min_distance_m'] <= high
                or not 0. < long_steps['fraction'] < 1.):
            raise ValueError('Long-step sampling requires a bounded, free-heading training route')
        self.training_long_steps = None if evaluation else long_steps
        # Saved qualification routes retain their original distance strata.
        self.radius_floor = low if evaluation else config.get('training_min_radius_m', low)
        if not 0 <= self.radius_floor <= low:
            raise ValueError('Invalid free-route training radius floor')
        self.adjacent_hops = None if evaluation else config.get('training_adjacent_hops')
        self.adjacent_metrics = None
        if self.adjacent_hops is not None:
            mix = self.adjacent_hops
            turns = [mix[k] for k in ('straight_probability','right_angle_probability','reverse_probability')]
            if (not all(0 < p < 1 for p in turns) or sum(turns) >= 1
                    or not 0 < mix['full_step_probability'] < 1-config['zero_probability']
                    or self.training_long_steps or training_mixture
                    or 'training_small_turn_fraction' in config or 'training_w_sw_s_small_turn_fraction' in config):
                raise ValueError('Invalid adjacent-hop training mixture')
            from .adjacent_hop_metrics import AdjacentHopMetrics
            self.adjacent_metrics = AdjacentHopMetrics(n,device,config.get("strict_foot_m",.1))
        # Uniform full-circle headings put 1/12 of proposed turns within
        # +/-15 degrees. Only the training free-route sampler can reweight it.
        small_turn_fraction=config.get('training_small_turn_fraction',1./12.)
        hard_heading_fraction=config.get('training_w_sw_s_small_turn_fraction',small_turn_fraction)
        if not (0.<small_turn_fraction<1. and 0.<hard_heading_fraction<1.):
            raise ValueError('Small-turn sampling must preserve both angular ranges')
        self.training_small_turn_fraction=(
            small_turn_fraction if not evaluation and small_turn_fraction!=1./12. else None)
        self.training_w_sw_s_small_turn_fraction=(
            hard_heading_fraction if not evaluation and hard_heading_fraction!=small_turn_fraction else None)
        reversals = config.get('training_reversals')
        if reversals:
            direct, paused = (reversals[key] for key in ('direct_fraction', 'paused_fraction'))
            y_share=reversals.get('fixed_axis_y_fraction',.25)
            if not 0<y_share<1:
                raise ValueError('Fixed reversal headings must retain both Y and non-Y coverage')
            spacing = reversals['spacing_m']
            ranges = [reversals[key] for key in ('warmup_hops', 'travel_hops', 'pause_hops')]
            paused_travel = reversals.get('paused_travel_hops')
            if paused_travel is not None:
                ranges.append(paused_travel)
            pair = reversals.get('paired_long_steps')
            if pair and (not low <= pair['min_spacing_m'] < spacing
                         or not 0 < pair['full_spacing_probability'] < 1
                         or reversals['travel_hops'] != [1, 1] or paused_travel != [2, 2]):
                raise ValueError('Long pairs require one outbound leg for reversals and two for straight pairs')
            if any(len(r) != 2 or any(type(v) is not int for v in r)
                   or not 1 <= r[0] <= r[1] for r in ranges):
                raise ValueError('Invalid hop counts for reversal routes')
            max_travel = max(reversals['travel_hops'][1], (paused_travel or reversals['travel_hops'])[1])
            if (training_mixture or reversals['version'] != 'omni_repeated_reversal_v1'
                    or not (direct > 0 and paused > 0 and direct + paused < 1)
                    or not 0 < reversals['fixed_heading_fraction'] < 1
                    or not low <= spacing <= self.current_max()
                    or max_travel * spacing > config['workspace_radius_m']
                    or ranges[0][1] + 2 * (max_travel + ranges[2][1]) >= config['route_points']):
                raise ValueError('Invalid omnidirectional reversal training routes')
        self.training_reversals = None if evaluation else reversals
        if training_mixture:
            m=training_mixture
            queued=m.get('version')=='cardinal_queued_segments_v2'
            minimum=0 if queued else 1e-12
            if (not minimum<=m['positive_x_fraction']<1 or not minimum<=m['other_heading_fraction']<1
                    or not math.isfinite(m.get('step_hold_fraction',0.))
                    or m.get('step_hold_fraction',0.)<0
                    or m['positive_x_fraction']+m['other_heading_fraction']+m.get('step_hold_fraction',0.)>=1
                    or (m.get('step_hold_fraction',0.)>0 and
                        (not isinstance(m.get('step_hold_warmup_hops'),int)
                         or not 1<=m['step_hold_warmup_hops']<config['route_points']-1))
                    or not low<=m['spacing_m']<=high):
                raise ValueError('Invalid training-only continuous route mixture')
            if not low<=m.get('step_hold_spacing_m',m['spacing_m'])<=high:
                raise ValueError('Invalid independent step/hold training spacing')
            if queued:
                fractions=[m[k] for k in ('cardinal_fraction','queued_fraction')]
                if (any(not math.isfinite(v) or v<=0 for v in fractions)
                        or sum(fractions)+m['positive_x_fraction']+m['other_heading_fraction']+m.get('step_hold_fraction',0.)>=1):
                    raise ValueError('Invalid balanced queued mixture')

    def current_max(self,common_step_counter=0):
        if self.evaluation or not self.curriculum:
            return self.config['moving_distance_range_m'][1]
        c=self.curriculum
        progress=min(max(float(common_step_counter)/(c['ramp_updates']*c['steps_per_update']),0.),1.)
        return c['initial_max_m']+(c['final_max_m']-c['initial_max_m'])*progress

    def reset(self,ids,common_step_counter=0):
        c=self.config;low=self.radius_floor;high=self.current_max(common_step_counter);n=len(ids)
        self.sampled_max[ids]=high
        if self.adjacent_metrics is not None:
            self.adjacent_metrics.reset(ids)
        self.index[ids]=0;self.points[ids,0]=0.
        if self.waypoint_source is not None:
            self.points[ids] = 0.
            self._fill_preview(ids, self.index[ids])
            return
        if self.evaluation and 'precision_eval_seed' in c:
            from .precision_routes import routes
            if not hasattr(self,'precision_bank'):
                self.precision_bank=self.points.new_tensor(routes(c['precision_eval_seed'],self.points.shape[1],
                                                                 c.get('opposite_straight_audit',False),
                                                                 c.get('continuous_route_eval',False),c.get('ab_preview_eval')))
            self.points[ids]=self.precision_bank[ids]
            return
        previous=self.points[ids,0]
        for step in range(1,c['route_points']):
            distance_draw = torch.rand(n, device=self.points.device, generator=self.rng)
            radius = low + (high - low) * distance_draw
            if self.adjacent_hops is not None:
                # The independent hold draw takes its share first. The full-step
                # probability in the course refers to all free-route edges.
                full = self.adjacent_hops['full_step_probability'] / (1-c['zero_probability'])
                continuous = low + (high-low)*(distance_draw-full)/(1-full)
                radius = torch.where(distance_draw < full, high, continuous)
            elif self.training_long_steps:
                # Inverse transform for a mixture of two uniform distances.
                # Holds and directions still use their independent draws.
                fraction = self.training_long_steps['fraction']
                long_min = min(high, self.training_long_steps['min_distance_m'])
                long_distance = long_min + (high - long_min) * (distance_draw / fraction)
                full_distance = low + (high - low) * ((distance_draw - fraction) / (1 - fraction))
                radius = torch.where(distance_draw < fraction, long_distance, full_distance)
            hold=torch.rand(n,device=self.points.device,generator=self.rng)<c['zero_probability']
            angle=2*math.pi*torch.rand(n,device=self.points.device,generator=self.rng)
            if self.adjacent_hops is not None and step>1:
                incoming = previous-self.points[ids,step-2]
                heading = torch.atan2(incoming[:,1],incoming[:,0])
                turn = torch.rand(n,device=self.points.device,generator=self.rng)
                straight = self.adjacent_hops['straight_probability']
                right_angle = self.adjacent_hops['right_angle_probability']
                right_end = straight+right_angle
                reverse_end = right_end+self.adjacent_hops['reverse_probability']
                quarter = torch.where(turn < straight+right_angle/2, math.pi/2, -math.pi/2)
                relative_heading = torch.where(turn < straight, heading,
                    torch.where(turn < right_end, heading+quarter, heading+math.pi))
                # After a zero edge there is no incoming direction. The next
                # nonzero edge uses the independent uniform world heading.
                angle = torch.where(incoming.ne(0).any(dim=1) & (turn<reverse_end),relative_heading,angle)
            if step>1 and (self.training_small_turn_fraction is not None
                           or self.training_w_sw_s_small_turn_fraction is not None):
                # Both intervals retain a uniform conditional density. After
                # a hold there is no incoming direction, so keep the free draw.
                incoming=previous-self.points[ids,step-2]
                signed_draw=angle-math.pi
                quantile=signed_draw.abs()/math.pi
                heading=torch.atan2(incoming[:,1],incoming[:,0])
                reweight=incoming.norm(dim=-1)>.1
                fraction=self.training_small_turn_fraction or 1./12.
                if self.training_w_sw_s_small_turn_fraction is not None:
                    # Same 45-degree bins as evaluation: W=4, SW=5, S=6.
                    sector=(heading/(math.pi/4)+.5).floor().remainder(8)
                    hard_heading=(sector>=4)&(sector<=6)
                    fraction=torch.where(hard_heading,self.training_w_sw_s_small_turn_fraction,fraction)
                    if self.training_small_turn_fraction is None:
                        reweight=reweight & hard_heading
                cone=math.pi/12.
                magnitude=torch.where(quantile<fraction,cone*quantile/fraction,
                    cone+(math.pi-cone)*(quantile-fraction)/(1.-fraction))
                angle=torch.where(reweight,heading+torch.copysign(magnitude,signed_draw),angle)
            if self.evaluation and step==1:
                # Balanced first moving edge,16 cases per heading sector.
                hold.zero_()
                angle=(ids%8)*math.pi/4+(angle/(2*math.pi)-.5)*math.pi/4
            radius=torch.where(hold,0.,radius)
            candidate=previous+radius[:,None]*torch.stack((angle.cos(),angle.sin()),-1)
            # Bounded rejection keeps edge length intact. It never clips the
            # endpoint into the10cm hit tolerance. Boundaries bias direction.
            for _ in range(8):
                bad=candidate.norm(dim=-1)>c['workspace_radius_m']
                if not bad.any():break
                a=2*math.pi*torch.rand(int(bad.sum()),device=self.points.device,generator=self.rng)
                candidate[bad]=previous[bad]+radius[bad,None]*torch.stack((a.cos(),a.sin()),-1)
            bad=candidate.norm(dim=-1)>c['workspace_radius_m']
            if bad.any():
                toward=-previous[bad]/previous[bad].norm(dim=-1,keepdim=True).clamp(min=1e-6)
                candidate[bad]=previous[bad]+radius[bad,None]*toward
            self.points[ids,step]=candidate
            previous=candidate
        self.sequence_kind[ids]=0
        if self.training_reversals:
            self._sample_training_reversals(ids)
            if c.get('reversal_rehearsal_fraction'):
                from .reversal_rehearsal import rehearse
                rehearse(self,ids,c['reversal_rehearsal_fraction'],(DIRECT_REVERSAL_KIND,PAUSED_REVERSAL_KIND))
        if not self.evaluation and c.get('ten_step_fraction'):
            chosen=ids[torch.rand(n,device=ids.device,generator=self.rng)<c['ten_step_fraction']]
            heading=torch.rand(len(chosen),device=ids.device,generator=self.rng)*2*math.pi
            if c.get('straight_heading_weights'):
                from .straight_heading import weighted_heading
                heading=weighted_heading(heading,c['straight_heading_weights'])
            heading=torch.where(torch.rand(len(chosen),device=ids.device,generator=self.rng)<.5,
                                (heading/(math.pi/4)).round()*(math.pi/4),heading)
            slots=torch.arange(self.points.shape[1],device=ids.device)%24
            levels=torch.minimum(slots,(22-slots).clamp_min(0)).clamp_max(10)*.5
            self.points[chosen]=levels[None,:,None]*torch.stack((heading.cos(),heading.sin()),-1)[:,None,:]
            self.sequence_kind[chosen]=4
        if self.training_mixture and not self.evaluation:
            m=self.training_mixture;draw=torch.rand(n,device=self.points.device,generator=self.rng)
            x=draw<m['positive_x_fraction']
            other=(draw>=m['positive_x_fraction'])&(draw<m['positive_x_fraction']+m['other_heading_fraction'])
            angle=2*math.pi*torch.rand(n,device=self.points.device,generator=self.rng)
            angle=torch.where(x,0.,angle)
            direction=torch.stack((angle.cos(),angle.sin()),-1)
            distance=torch.arange(self.points.shape[1],device=self.points.device)*m['spacing_m']
            selected=x|other
            # Infinite flat plane, isolated environment collision groups:
            # long straight training commands deliberately extend beyond the
            # random-route sampling disk. Formal evaluation remains unchanged.
            self.points[ids[selected]]=distance[None,:,None]*direction[selected,None,:]
            self.sequence_kind[ids[x]]=1;self.sequence_kind[ids[other]]=2
            if m.get('step_hold_fraction',0.)>0:
                boundary=m['positive_x_fraction']+m['other_heading_fraction']
                step_hold=(draw>=boundary)&(draw<boundary+m['step_hold_fraction'])
                chosen=ids[step_hold]
                # Explicit command exposure: no instantaneous goal teleport;
                # the next destination appears one complete warmup hop early.
                from .step_hold import DIRECTIONS
                cardinal=torch.randint(4,(len(chosen),),device=self.points.device,generator=self.rng)
                vector=self.points.new_tensor(DIRECTIONS)[cardinal]*m.get('step_hold_spacing_m',m['spacing_m'])
                self.points[chosen]=0.
                self.points[chosen,m['step_hold_warmup_hops']:]=vector[:,None,:]
                self.sequence_kind[chosen]=3
            if m.get('version')=='cardinal_queued_segments_v2':
                from .step_hold import DIRECTIONS
                from .cardinal_queue import queued_points
                start=m['positive_x_fraction']+m['other_heading_fraction']+m.get('step_hold_fraction',0.)
                for fraction,kind in ((m['cardinal_fraction'],4),(m['queued_fraction'],5)):
                    chosen=ids[(draw>=start)&(draw<start+fraction)];start+=fraction
                    cardinal=torch.randint(4,(len(chosen),),device=self.points.device,generator=self.rng)
                    direction=self.points.new_tensor(DIRECTIONS)[cardinal]
                    if kind==4:
                        levels=torch.arange(self.points.shape[1],device=self.points.device)*m['spacing_m']
                    else:
                        levels=self.points.new_tensor(queued_points(self.points.shape[1],m['spacing_m'],0,
                            m['queued_warmup_hops'],m['queued_moves'],m['queued_hold_hops']))[:,0]
                    self.points[chosen]=levels[None,:,None]*direction[:,None,:]
                    self.sequence_kind[chosen]=kind

    def _sample_training_reversals(self, ids):
        config = self.training_reversals
        device = self.points.device
        draw = torch.rand(len(ids), device=device, generator=self.rng)
        selected = draw < config['direct_fraction'] + config['paused_fraction']
        chosen = ids[selected]
        paused = draw[selected] >= config['direct_fraction']
        count = chosen.numel()

        def integers(key):
            low, high = config[key]
            return torch.randint(low, high + 1, (count,), device=device, generator=self.rng)

        warmup = integers('warmup_hops')
        travel = integers('travel_hops')
        if 'paused_travel_hops' in config:
            travel = torch.where(paused, integers('paused_travel_hops'), travel)
        hold_hops = integers('pause_hops')
        pause = hold_hops * paused
        continuous = 2 * math.pi * torch.rand(count, device=device, generator=self.rng)
        # Reweight only the exact-heading subset of training reversal routes.
        # Mirror signs within each axis; the unrestricted heading draw is unchanged.
        y_share=config.get('fixed_axis_y_fraction',.25)
        if count and y_share!=.25:
            weights=self.points.new_full((8,), (1.-y_share)/6.)
            weights[2::4]=y_share/2.
            fixed=(math.pi/4)*torch.multinomial(weights,count,replacement=True,generator=self.rng)
        else:
            fixed=(math.pi/4)*torch.randint(8,(count,),device=device,generator=self.rng)
        use_fixed = torch.rand(count, device=device, generator=self.rng) < config['fixed_heading_fraction']
        angle = torch.where(use_fixed, fixed, continuous)
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        distance = config['spacing_m']
        pair = config.get('paired_long_steps')
        if pair:
            draw = torch.rand(count, device=device, generator=self.rng)
            fraction = pair['full_spacing_probability']
            # Atom at full length, otherwise uniform over the long-step interval.
            distance = config['spacing_m'] - (config['spacing_m'] - pair['min_spacing_m']) * (
                (draw - fraction).clamp_min(0.) / (1. - fraction))
            distance = distance[:, None]
        step = direction * distance

        # Begin at the origin, travel out, pause, travel back, pause, then repeat.
        # Zero pauses yield adjacent 180-degree turns with no intermediate hold.
        slots = torch.arange(self.points.shape[1], device=device) - warmup[:, None]
        half_cycle = (travel + pause)[:, None]
        # A direct long pair returns immediately, then rehearses holding the origin.
        cycle_length = 2 * half_cycle
        if pair:
            cycle_length = cycle_length + (hold_hops * ~paused)[:, None]
        cycle = slots % cycle_length
        outward = (cycle + 1).minimum(travel[:, None])
        returning = (travel[:, None] - (cycle - half_cycle + 1)).clamp_min(0)
        level = torch.where(cycle < half_cycle, outward, returning)
        level = torch.where(slots >= 0, level, 0)
        self.points[chosen] = level[:, :, None] * step[:, None, :]
        self.sequence_kind[chosen] = torch.where(paused, PAUSED_REVERSAL_KIND, DIRECT_REVERSAL_KIND)

    def pair(self,ids):
        i=self.index[ids]
        return self.points[ids,i],self.points[ids,i+1]

    def _fill_preview(self, ids, index):
        from .waypoint_stream import next_preview
        current = self.points[ids, index].tolist()
        previews = [next_preview(self.waypoint_source, lane, i+1, xy)
                    for lane, i, xy in zip(ids.tolist(), index.tolist(), current)]
        if previews:
            self.points[ids, index+1] = self.points.new_tensor(previews)

    def advance(self,valid,strict,navigation=None):
        advance=valid&strict if navigation is None else valid&(strict|navigation)
        ids=advance.nonzero(as_tuple=False).flatten()
        if len(ids):
            new_index = self.index[ids]+1
            if (new_index+1>=self.points.shape[1]).any():
                if self.waypoint_source is None:
                    raise RuntimeError('Omni route exhausted; do not silently reuse targets')
                self.points = torch.cat((self.points, torch.zeros_like(self.points)), dim=1)
            if self.waypoint_source is not None:
                self._fill_preview(ids, new_index)
            self.index[ids] = new_index
            self.advances += len(ids)
        self.misses+=(valid & ~strict).sum()
        self.strict_advances+=(valid&strict).sum()
        self.advances_without_strict+=(advance&~strict).sum()
        return ids


def prepare_source(state,previous,course):
    """Fold only formerly duplicate target columns; keep all ring8 weights."""
    if previous['task']==TASK:return []
    if previous['task']!='obs73_stationary_foundation_v1':
        raise ValueError('Initialize omni from its stationary73D foundation or own model')
    changed=[]
    for side,letter in (('actor','a'),('critic','c')):
        for suffix in ('_mean','_var','_std'):
            value=state[f'{side}_obs_normalizer.{suffix}']
            if not (torch.equal(value[:,:2],value[:,2:4]) and torch.equal(value[:,:2],value[:,4:6])):
                raise ValueError('Stationary source must have identical duplicate-target statistics')
        key=f'memory_{letter}.rnn.weight_ih_l0';w=state[key]
        if w.shape!=(1024,73):raise ValueError('Expected73D single-layer LSTM256')
        w[:,:2]+=w[:,2:4]+w[:,4:6];w[:,2:6]=0.;changed.append(key)
    return changed


def env_class(evaluation=False):
    from .obs73_vertical import env_class as vertical_class
    from .two_ground_targets import _set_active_goal

    class OmniEnv(vertical_class(evaluation=evaluation)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ab_failure_settlement = None
            config = self.course.get('ab_failure_settlement')
            if config is not None and not evaluation:
                if not self.course['omni'].get('ab_pair_only'):
                    raise ValueError('AB failure settlement requires the all-AB training course')
                from .ab_failure_settlement import ABFailureSettlement
                self.ab_failure_settlement = ABFailureSettlement(config, self.step_dt, self.device)
                self._episode_sums['ab_failure_settlement'] = torch.zeros(self.num_envs, device=self.device)

        def _reset_task_targets(self,ids):
            if not hasattr(self,'omni_route'):
                self.omni_route=GroundRoute(self.num_envs,self.device,self.course['omni'],self.cfg.seed,evaluation,
                    training_mixture=self.course.get('moving_training_sequences'))
            # New episodes sample at the current range; already queued points
            # remain fixed while the global PPO-update curriculum advances.
            self.omni_route.waypoint_source = getattr(self, 'waypoint_source', None)
            self.omni_route.reset(ids,getattr(self,'common_step_counter',0))
            train_keys = not evaluation and not self.course['omni'].get('ab_pair_only')
            if self.course.get('guarded_keys') and (train_keys or getattr(self,'guarded_diagnostic',False)):
                from .guarded_keys import GuardedKeys
                if not hasattr(self,'key_guard'):
                    opt=self.course['guarded_keys']
                    if (opt['version']!='batched_keyboard_guard_v1' or not 1<=opt['launch_multiplier']<=2
                            or abs(self.step_dt-.01)>1e-8 or self.course['long_moving_horizon']['moving_s']!=60.
                            or not self.course.get('omni_rebound_handoff')
                            or self.course['launch_rotation_refinement'].get('window_s')!=.18):
                        raise ValueError('Unsupported guarded-key course or control clock')
                    self.key_guard=GuardedKeys(self.omni_route,self.step_dt,evaluation,
                        training_moving_fraction=opt.get('training_moving_fraction',.25),
                        training_pair_gap_s=opt.get('training_pair_gap_s',0.))
                self.key_guard.reset(ids)
            self._bind_targets(ids)
            self._desired_pos_w[ids,2]=self._terrain.env_origins[ids,2]+1.

        def _bind_targets(self,ids):
            pt,nxt=self.omni_route.pair(ids);origin=self._terrain.env_origins[ids,:2]
            self._two_ground_pt_xy_w[ids]=pt+origin
            self._two_ground_pt1_xy_w[ids]=nxt+origin
            self._two_ground_route_index[ids]=self.omni_route.index[ids]
            _set_active_goal(self,ids)

        def set_next_waypoints(self, ids, points_xy):
            """Replace previews on online evaluation lanes before building the next observation."""
            route = self.omni_route
            if getattr(route, 'waypoint_source', None) is None:
                raise ValueError('Online evaluation is required to replace a preview')
            points = torch.as_tensor(points_xy, device=route.points.device, dtype=route.points.dtype)
            index = route.index[ids]
            current = route.points[ids, index]
            if (points.shape != current.shape or not torch.isfinite(points).all()
                    or ((points-current).norm(dim=-1) > .500001).any()):
                raise ValueError('Expected one finite XY per lane within 0.5 m of its current waypoint')
            route.points[ids, index+1] = points
            self._two_ground_pt1_xy_w[ids] = self._terrain.env_origins[ids, :2] + points

        def _get_rewards(self):
            d=self._robot.data;f=self._foot_event;q=d.root_quat_w
            physical=(d.root_link_pos_w[:,2]<.05)|(q[:,1:3].square().sum(-1)>.5)
            valid=((f.phase==f.FLIGHT)&(f.air_steps>=5)
                &(d.joint_pos[:,self._spring_joint_id]>.002)
                &(f.apex_seen|(d.root_com_lin_vel_w[:,2]<=0))&~physical)
            peak=torch.maximum(f.peak,d.root_link_pos_w[:,2])-self._terrain.env_origins[:,2]
            error=(d.body_link_pos_w[:,self._foot_body_id,:2]-self._two_ground_pt_xy_w).norm(dim=-1)
            tilt=torch.rad2deg(torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1,1)))
            strict=valid&(peak>=1.)&(error<=self.landing_radius_m)&(tilt<=15.)
            ab=getattr(self.omni_route,'ab_pairs',None)
            recovery = ab.recovery if ab is not None else None
            if ab is not None:
                opening = self.omni_route.index < ab.a_index
                ab_done=ab.touchdown(self.omni_route.index,valid,strict)
                if ab.paired_axes is not None:
                    ab.paired_axes.record_arrival(self.omni_route, valid, error <= self.landing_radius_m)
                if hasattr(self, 'ab_spawn'):
                    self.ab_spawn.record_touchdowns(valid & ~ab_done & ~opening, ab_done, strict)
                    self.extras.setdefault('log', {}).update(self.ab_spawn.metrics())
            stationary=getattr(self.omni_route,'stationary_mask',torch.zeros_like(valid))
            advance=navigation_advance(valid,strict,error,tilt,self.omni_route.index,
                stationary,self.course.get('moving_navigation'),self.landing_radius_m)
            if recovery is not None:
                recovery.record_landing(ab_done, error <= self.landing_radius_m, physical)
                missed_a = valid & ~ab_done & ~advance & ~opening
                prefer_retry = None
                if self.course.get('retry_compression') is not None:
                    from .retry_compression import projected_release_at_contact
                    cfg = self.course['retry_compression']
                    foot = d.body_link_pos_w[:, self._foot_body_id]
                    velocity = d.root_link_lin_vel_w
                    # Use the same post-touch flight EMA as the reference latched
                    # after the queue update, but preview the retain branch only.
                    ema = self.rebound_config['duration_ema']
                    flight = (1.-ema)*self.rebound_plan.flight + ema*(f.air_steps*self.step_dt).clamp(.6,1.2)
                    target = (self._two_ground_pt_xy_w-foot[:, :2])/flight[:, None]
                    limit = self.landing_guidance_config['speed_limit_m_s']
                    target *= (limit/target.norm(dim=-1,keepdim=True).clamp_min(1.e-6)).clamp(max=1.)
                    predicted = projected_release_at_contact(
                        d.root_link_pos_w-foot, q, d.root_ang_vel_b, velocity,
                        cfg['sweep_horizon_s'], cfg['vertical_impulse_ratio'])
                    threshold = cfg['threshold_m_s']
                    # Only moving A misses inside the small-angle impact model.
                    ids = torch.arange(self.num_envs, device=self.device)
                    edge = self.omni_route.points[ids,self.omni_route.index] - self.omni_route.points[
                        ids,(self.omni_route.index-1).clamp_min(0)]
                    prefer_retry = ((edge.norm(dim=-1) > 1.e-6) & (tilt <= 15.) & (velocity[:, 2] < 0)
                                    & ((velocity[:, :2]-target).norm(dim=-1) <= threshold)
                                    & ((predicted-target).norm(dim=-1) > threshold))
                    self._compression_promoted = missed_a & prefer_retry & (recovery.assignment == recovery.CONTINUE)
                start_hold = recovery.start(missed_a, prefer_retry)
                # A rehearsal hold advances on a strict landing. After A,
                # ordinary AB advances and the recovery branches retain A.
                advance = valid & (((recovery.active == recovery.CONTINUE) & ~opening) | advance)
            elif ab is not None and ab.all_lanes:
                advance = valid & (~opening | advance)
            # Parent reward and outer OBS73 event histories use the old goal.
            guard=getattr(self,'key_guard',None)
            metrics = self.omni_route.adjacent_metrics
            practice=getattr(self.omni_route,'distance_practice',None)
            if practice is not None:
                practice.observe(self.omni_route,valid,strict,physical)
            if metrics is not None:
                ordinary = ~stationary if guard is None else ~(stationary | guard.mask)
                metrics.observe(self.omni_route, valid & ordinary, error)
            if guard is not None:
                from .moving_phase_reward import heading_motion
                angle,rate=heading_motion(q,d.root_ang_vel_b)
                foot=d.body_link_pos_w[:,self._foot_body_id]-self._terrain.env_origins
                airborne=(f.phase==2)|(f.phase==3)
                free=airborne & (d.joint_pos[:,self._spring_joint_id]<=.002)
                changed,launched=guard.update(valid, strict, (error<=self.landing_radius_m)&(tilt<=15), foot[:,:2],error,
                    angle.rad2deg(),rate.rad2deg(),d.root_ang_vel_b[:,:2].norm(dim=-1),
                    free&(foot[:,2]>.032)&(d.root_link_lin_vel_w[:,2]<0),free,physical)
            reward=super()._get_rewards()
            if guard is None:
                ids=self.omni_route.advance(valid,strict,advance)
                if ab is not None and ab.paired_axes is not None:
                    ab.paired_axes.publish_departure(self.omni_route, ids, ab_done | self.reset_buf)
            else:
                ids=self.omni_route.advance(valid & ~guard.mask,strict,advance)
                # Keep the old current as the preceding edge at touchdown;
                # a recovery anchor and its preview can both change on a miss.
                hit=guard.mask & valid
                self.omni_route.index[hit]+=1
                if (self.omni_route.index[hit]+1>=self.omni_route.points.shape[1]).any():
                    raise RuntimeError('Guard contact history exhausted')
                guard.publish(changed)
                ids=torch.cat((ids,changed))
                advance=torch.where(guard.mask,launched,advance)
            self._omni_last_strict=strict
            if ab is not None:
                # DirectRLEnv resets after _get_rewards. Keep the full B
                # touchdown reward and mark the curriculum cut as a timeout.
                if self.ab_failure_settlement is not None:
                    failed = self.reset_terminated & physical
                    charge = self.ab_failure_settlement.charge(self.episode_length_buf, failed)
                    reward = reward + charge
                    self._episode_sums['ab_failure_settlement'] += charge
                    # A fall coinciding with the time limit is a true terminal for PPO.
                    self.reset_time_outs &= ~failed
                self.reset_time_outs |= ab_done & ~self.reset_terminated
                self.reset_buf |= ab_done
                self.extras.setdefault('log',{}).update(ab.metrics())
                if ab.paired_axes is not None:
                    self.extras['log'].update(ab.paired_axes.metrics())
            self._omni_last_navigation_advance=advance
            advanced_count = len(ids) if guard is None else advance.sum()
            if recovery is not None:
                hold_ids = start_hold.nonzero().flatten()
                foot_local = d.body_link_pos_w[:,self._foot_body_id,:2] - self._terrain.env_origins[:,:2]
                recovery.publish_hold(self.omni_route, hold_ids, foot_local)
                ids = torch.cat((ids, hold_ids))
            if len(ids):
                self._bind_targets(ids)
                self._two_ground_total_advances+=advanced_count
            if guard is not None:
                self.extras.setdefault('log',{}).update({
                    'Keyboard/paused_lane_fraction':(guard.state[guard.mask]==3).float().mean(),
                    'Keyboard/recover_lane_fraction':(guard.state[guard.mask]==4).float().mean()})
            self.extras.setdefault('log',{}).update({
                'Omni/strict_target_advances_total':self.omni_route.strict_advances,
                'Omni/navigation_advances_total':self.omni_route.advances,
                'Omni/navigation_advances_without_strict_total':self.omni_route.advances_without_strict,
                'Omni/missed_landings_total':self.omni_route.misses,
                'Omni/current_edge_max_m':(self._two_ground_pt1_xy_w-self._two_ground_pt_xy_w).norm(dim=-1).max(),
                'Omni/sampling_distance_max_m':self.omni_route.current_max(self.common_step_counter)})
            return reward

    return OmniEnv
