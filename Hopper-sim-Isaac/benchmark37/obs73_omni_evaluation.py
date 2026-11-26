"""First128 episodes for the short OBS73 omnidirectional bridge; no updates."""
import json
import math

import torch

from .obs73_omni import TASK, CASE_VERSION, case_version, distance_bins
from .ring_evaluation import RingEvaluation
from .metrics import distribution, summarize, continuity_metrics, landing_precision_metrics,rebound_motion_metrics,straight_batch_metrics

DIRECTIONS=('E','NE','N','NW','W','SW','S','SE')


class OmniEvaluation(RingEvaluation):
    def __init__(self,env):
        super().__init__(env)
        self.attempts=[{} for _ in self.episodes]
        self.launch_snapshot=None
        self.apex_snapshot=None
        self.release_window=None
        self.window_motion=None
        self.release_vertical_state=None
        self.flight_yaw=None
        self.retry_unloading=None
        self.approach_history=None
        self.landing_forecasts=None
        self.forecast_clock=None
        self.plane_peak_target=None
        if 'precision_eval_seed' in env.course['omni'] and hasattr(env,'plane_totals'):
            initial=env.plane_totals[:,:,4]
            self.plane_peak_seen=torch.zeros_like(initial)
            self.plane_peak_target=torch.full_like(initial,-1,dtype=torch.long)
            self.plane_peak_step=torch.full_like(self.plane_peak_target,-1)

    def record_plane_peaks(self, target):
        # Only first episodes enter this evaluation. Use the target from before
        # touchdown advanced the route and the same peak as the acceptance gate.
        peaks=self.env.plane_totals[:,:,4]
        raised=peaks>self.plane_peak_seen
        self.plane_peak_target.copy_(torch.where(raised,target[:,None],self.plane_peak_target))
        self.plane_peak_step.masked_fill_(raised,self.env.common_step_counter)
        self.plane_peak_seen.copy_(peaks)

    def record_release(self,env):
        """Latch the existing reward audit at physical release; no policy inputs or RNG."""
        audit=getattr(env,'_moving_phase_last_audit',None)
        if not audit or audit.get('rebound') is None:return
        rebound=audit['rebound']
        active=rebound['launch']
        if self.launch_snapshot is None:
            self.release_body_rate=torch.zeros_like(env._robot.data.root_ang_vel_b)
            if hasattr(env,'continuation_cosine'):
                self.apex_snapshot=(torch.full_like(env.omni_route.index,-1),
                                    torch.full_like(env.omni_route.index,-1),
                                    torch.zeros_like(rebound['actual_launch_velocity']),
                                    torch.zeros_like(rebound['actual_launch_velocity']))
            self.launch_snapshot=(torch.full_like(env.omni_route.index,-1),
                                  torch.full_like(env.omni_route.index,-1),
                                  torch.full_like(env.omni_route.index,-1),
                                  torch.zeros_like(rebound['actual_launch_velocity']),
                                  torch.zeros_like(rebound['target_launch_velocity']),
                                  torch.zeros_like(rebound['actual_launch_velocity']),
                                  torch.zeros_like(rebound['actual_launch_velocity']),
                                  torch.zeros_like(env.rebound_plan.flight))
        index,contact,tick,actual,desired,release_root,release_foot,duration=self.launch_snapshot
        index.copy_(torch.where(active,env.omni_route.index,index))
        contact.copy_(torch.where(active,env.rebound_plan.contact_step,contact))
        tick.masked_fill_(active,env.common_step_counter)
        self.release_body_rate.copy_(torch.where(active[:,None],env._robot.data.root_ang_vel_b,self.release_body_rate))
        actual.copy_(torch.where(active[:,None],rebound['actual_launch_velocity'],actual))
        desired.copy_(torch.where(active[:,None],rebound['target_launch_velocity'],desired))
        data=env._robot.data
        release_root.copy_(torch.where(active[:,None],data.root_link_pos_w[:,:2],release_root))
        release_foot.copy_(torch.where(active[:,None],data.body_link_pos_w[:,env._foot_body_id,:2],release_foot))
        duration.copy_(torch.where(active,env.rebound_plan.flight,duration))
        if 'clearance' in audit and hasattr(env,'continuation_cosine'):
            # Pair the available launch height and vertical speed with the
            # actual flight duration. The flight-clock comparison is offline.
            if self.release_vertical_state is None:
                self.release_vertical_state=(torch.zeros_like(duration),torch.zeros_like(duration))
            vz,clearance=self.release_vertical_state
            vz.copy_(torch.where(active,data.root_link_lin_vel_w[:,2],vz))
            clearance.copy_(torch.where(active,audit['clearance'],clearance))
        if 'launch_velocity_active' in rebound:
            if self.release_window is None:
                self.release_window=(torch.zeros_like(tick),torch.zeros_like(duration))
            samples,cost=self.release_window
            samples.masked_fill_(active,0)
            cost.masked_fill_(active,0.)
            scored=(rebound['launch_velocity_active'] & (tick>=0)
                    & (index==env.omni_route.index) & (contact==env.rebound_plan.contact_step))
            samples.add_(scored)
            # Accumulate the configured reward, including its actual clearance
            # mask. A window need not contain all eighteen control samples.
            cost.add_(torch.where(scored,audit['deltas']['moving_launch_velocity'],0.))
            if self.plane_peak_target is not None:
                if self.window_motion is None:
                    self.window_motion=duration.new_zeros((len(tick),2,3))
                self.window_motion.masked_fill_(active[:,None,None],0.)
                late=(env.common_step_counter-tick)*env.step_dt>=.5*env.launch_velocity_window_config['window_s']
                mask=torch.stack((scored & ~late,scored & late),dim=1)
                delta=data.root_link_lin_vel_w[:,:2]-desired
                self.window_motion[:,:,0]+=mask
                self.window_motion[:,:,1:]+=torch.where(mask[:,:,None],delta[:,None,:],0.)
        if 'handoff_confidence' in rebound:
            # sample() reads this slot before the current tick overwrites it.
            guide=audit.get('guidance')
            has_forecasts=guide is not None and 'projected_foot_xy' in guide
            if self.approach_history is None:
                history_s=.35 if self.plane_peak_target is not None and has_forecasts else (.25 if has_forecasts else .10)
                delay=round(history_s/env.step_dt)
                shape=(delay,len(index))
                columns=9 if 'actual_approach_normal_xy' in rebound else 2
                self.approach_history=(index.new_full(shape,-1),index.new_full(shape,-1),
                                       duration.new_zeros((*shape,columns)),[-1]*delay)
                if has_forecasts:
                    self.landing_forecasts=duration.new_zeros((*shape,2,2))
                    if self.plane_peak_target is not None:
                        self.forecast_clock=duration.new_zeros(shape)
            goal_ids,contact_ids,preview,steps=self.approach_history
            slot=env.common_step_counter%len(steps)
            desc=rebound['air'] & (data.root_link_lin_vel_w[:,2]<0)
            goal_ids[slot].copy_(torch.where(desc,env.omni_route.index,-1))
            contact_ids[slot].copy_(env.rebound_plan.contact_step)
            predicted_error=(rebound['handoff_predicted_landing_xy']-env._two_ground_pt_xy_w).norm(dim=-1)
            preview[slot,:,0].copy_(predicted_error)
            preview[slot,:,1].copy_(rebound['handoff_confidence'])
            if self.landing_forecasts is not None:
                self.landing_forecasts[slot,:,0].copy_(guide['projected_foot_xy'])
                self.landing_forecasts[slot,:,1].copy_(rebound['handoff_predicted_landing_xy'])
                if self.forecast_clock is not None:
                    self.forecast_clock[slot].copy_(guide['remaining_time_s'])
            if preview.shape[-1]==9:
                preview[slot,:,2:4].copy_(rebound['actual_approach_normal_xy'])
                preview[slot,:,4:6].copy_(rebound['desired_approach_normal_xy'])
                preview[slot,:,6:9].copy_(data.root_ang_vel_b)
            steps[slot]=env.common_step_counter
        if 'retry_unloading_active' in rebound:
            if self.retry_unloading is None:
                self.retry_unloading=(torch.full_like(contact,-1),torch.zeros_like(contact),torch.zeros_like(contact))
            counted_contact,ticks,at_launch=self.retry_unloading
            ticks.masked_fill_(counted_contact!=env.rebound_plan.contact_step,0)
            counted_contact.copy_(env.rebound_plan.contact_step)
            ticks.add_(rebound['retry_unloading_active'])
            at_launch.copy_(torch.where(active,ticks,at_launch))
        air=(rebound['air'] & (tick>=0) & (index==env.omni_route.index)
             & (contact==env.rebound_plan.contact_step))
        if self.apex_snapshot is not None:
            # First positive-to-nonpositive vz transition at consecutive free
            # control samples of this release. Skip incomplete flights.
            last_rising,apex_tick,apex_xy,apex_vxy=self.apex_snapshot
            now=env.common_step_counter
            up=data.root_link_lin_vel_w[:,2]>0
            apex_tick.masked_fill_(active,-1)
            reached=air & ~up & (tick<now) & (apex_tick<0) & (last_rising==now-1)
            apex_tick.masked_fill_(reached,now)
            apex_xy.copy_(torch.where(reached[:,None],data.root_link_pos_w[:,:2],apex_xy))
            apex_vxy.copy_(torch.where(reached[:,None],data.root_link_lin_vel_w[:,:2],apex_vxy))
            last_rising.masked_fill_(air & up,now)
        yaw=audit.get('yaw_prediction')
        if yaw is not None:
            magnitude=torch.rad2deg(yaw['error'].abs())
            if self.flight_yaw is None:
                self.flight_yaw=magnitude.new_full((len(magnitude),3),-1.)
            at_release,rising,falling=self.flight_yaw.unbind(1)
            at_release.copy_(torch.where(active,magnitude,at_release))
            rising.masked_fill_(active,-1.)
            falling.masked_fill_(active,-1.)
            up=data.root_link_lin_vel_w[:,2]>=0
            rising.copy_(torch.where(air & up,torch.maximum(rising,magnitude),rising))
            falling.copy_(torch.where(air & ~up,torch.maximum(falling,magnitude),falling))

    def restore_cases(self,path):
        e=self.env;payload=json.loads(path.read_text());cases=payload['cases'];ids=e._robot._ALL_INDICES
        straight=getattr(e,'straight_evaluation_spec',None)
        step_hold=getattr(e,'step_hold_evaluation_spec',None)
        from .step_hold import restore_step_hold_route
        restore_step_hold_route(e,payload)
        from .cardinal_queue import restore_route
        queued=restore_route(e,payload)
        saved_straight=payload.get('straight_evaluation')
        if saved_straight is not None and saved_straight!=straight:
            raise ValueError('Straight cases require their explicit matching straight evaluation scope')
        saved=['omni_route_local_m' in case for case in cases]
        if any(saved) and not all(saved):raise ValueError('Mixed saved omni route formats')
        if (not straight and not step_hold and not queued and all(saved) and e.course['omni'].get('distance_curriculum')
                and payload.get('omni_distance_range_m')!=e.course['omni']['moving_distance_range_m']):
            raise ValueError('Distance curriculum evaluation requires saved cases at its full final range; do not reuse short v1/v2 routes')
        super().restore_cases(path)
        if queued:
            restore_route(e,payload,apply=True)
        elif straight:
            from .ring_play import straight_x_points
            points=e.omni_route.points.new_tensor(straight_x_points(
                straight['route_points'],straight['spacing_m']))
            e.omni_route.points.copy_(points.unsqueeze(0).expand_as(e.omni_route.points))
            e.omni_route.index.zero_()
        elif step_hold:
            restore_step_hold_route(e,payload,apply=True)
        elif all(saved):
            points=torch.tensor([case['omni_route_local_m'] for case in cases],device=e.device)
            if points.shape!=e.omni_route.points.shape or not torch.isfinite(points).all():
                raise ValueError('Saved omni route shape/data differ')
            lengths=points.diff(dim=1).norm(dim=-1)
            low,high=e.course['omni']['moving_distance_range_m']
            if (points[:,0].abs().max()>1e-6
                    or points.norm(dim=-1).max()>e.course['omni']['workspace_radius_m']+1e-5
                    or ((lengths>1e-6)&((lengths<low-1e-5)|(lengths>high+1e-5))).any()):
                raise ValueError('Saved route violates the configured omni contract')
            e.omni_route.points.copy_(points);e.omni_route.index.zero_()
        # Without saved omni routes, use this task's already sampled seed7001
        # routes; only releases/base DR are restored from the older case file.
        e._bind_targets(ids)
        e.ring.center[:]=torch.tensor([case['ring_center_world'] for case in cases],device=e.device)
        if e.response_envelope is not None:e.response_envelope.restore(cases,e.dr_inertia_multi)

    def freeze_cases(self,folder):
        super().freeze_cases(folder)
        path=folder/'cases.json';payload=json.loads(path.read_text())
        payload.update(version=case_version(self.env.course),task=TASK,
            omni_distance_range_m=self.env.course['omni']['moving_distance_range_m'],
            workspace_radius_m=self.env.course['omni']['workspace_radius_m'],
            route_scope='first hop stationary; first moving edge balanced16 per8 directions; later10% holds',
            response_scope='cases0..63 nominal,64..127 weaker roll/pitch; same source response envelope')
        straight=getattr(self.env,'straight_evaluation_spec',None)
        if straight:
            payload.update(version=straight['version'],straight_evaluation=straight,
                omni_distance_range_m=None,workspace_radius_m=None,route_scope=straight['route_scope'])
        step_hold=getattr(self.env,'step_hold_evaluation_spec',None)
        if step_hold:
            payload.update(version=step_hold['version'],step_hold_evaluation=step_hold,
                route_scope=step_hold['route_scope'])
        for case,points in zip(payload['cases'],self.env.omni_route.points.cpu().tolist()):
            if getattr(self.env,'waypoint_source',None) is None:
                case['omni_route_local_m']=points
            else:
                case['initial_target_pair_local_m']=points[:2]
        if getattr(self.env,'waypoint_source',None) is not None:
            payload.update(version='online_waypoints',omni_distance_range_m=[0.,.5],workspace_radius_m=None,
                           route_scope='Online preview after each navigation advance. Touchdown targets are in episodes.json.')
        queued=getattr(self.env,'cardinal_queue_evaluation_spec',None)
        precision=self.env.course['omni'].get('precision_eval_seed')
        if precision is not None:
            payload.update(version='obs73_precision10_v1',seed=precision,workspace_radius_m=None,
                omni_distance_range_m=[0.,.5],route_scope='64 omni and 64 straight; origin plus ten first-attempt commands')
            if self.env.course['omni'].get('opposite_straight_audit'):
                payload.update(version='obs73_opposite_straight_audit_v1',
                               route_scope='Paired diagnostic with the 64 straight headings reversed')
            if self.env.course['omni'].get('continuous_route_eval'):
                payload.update(version='continuous60_v1',route_scope='60s; 64 random omni and 64 straight routes, 96 waypoints')
            if self.env.course['omni'].get('ab_preview_eval'):
                payload.update(version=self.env.course['omni']['ab_preview_eval'],route_scope='Paired full-step preview; 64 direct and 64 after moving warmup')
        if queued:
            payload.update(version=queued['version'],cardinal_queue_evaluation=queued,
                omni_distance_range_m=None,workspace_radius_m=None,route_scope=queued['route_scope'])
        if self.env.response_envelope is not None:self.env.response_envelope.annotate(payload['cases'])
        path.write_text(json.dumps(payload,indent=2)+'\n')

    def sample(self):
        lengths=[len(ep.events) for ep in self.episodes]
        super().sample()  # Actual landing geometry, before the target advances.
        route=self.env.omni_route
        for i,ep in enumerate(self.episodes):
            if len(ep.events)==lengths[i]:continue
            index=int(route.index[i]);target=route.points[i,index]
            if self.launch_snapshot is not None:
                ep.events[-1]['touchdown_body_rate_rad_s']=self.env._robot.data.root_ang_vel_b[i].tolist()
            edge=target-route.points[i,max(0,index-1)]
            length=float(edge.norm());moving=length>.1
            attempt=self.attempts[i].get(index,0)+1;self.attempts[i][index]=attempt
            direction=int(math.floor(math.atan2(float(edge[1]),float(edge[0]))/(math.pi/4)+.5))%8 if moving else None
            if self.approach_history is not None:
                goal_ids,contact_ids,preview,steps=self.approach_history
                now=self.env.common_step_counter
                delay=round(.10/self.env.step_dt)
                slot=(now-delay)%len(steps)
                if (steps[slot]==now-delay and int(goal_ids[slot,i])==index
                        and int(contact_ids[slot,i])==int(self.env.rebound_plan.contact_step[i])):
                    sample=preview[slot,i].tolist()
                    ep.events[-1].update(predicted_error_100ms_before_touchdown_m=sample[0],
                                         advance_confidence_100ms_before_touchdown=sample[1])
                    if len(sample)==9:
                        ep.events[-1].update(approach_normal_xy_100ms_before_touchdown=sample[2:4],
                                             desired_normal_xy_100ms_before_touchdown=sample[4:6],
                                             preimpact_body_rate_100ms_rad_s=sample[6:9])
                if self.landing_forecasts is not None:
                    foot=self.env._robot.data.body_link_pos_w[i,self.env._foot_body_id,:2]
                    for ms in ((100,150,250,350) if self.forecast_clock is not None else (100,250)):
                        tick=now-round(ms*.001/self.env.step_dt)
                        slot=tick%len(steps)
                        if (steps[slot]==tick and int(goal_ids[slot,i])==index
                                and int(contact_ids[slot,i])==int(self.env.rebound_plan.contact_step[i])):
                            dense,preview_error=(self.landing_forecasts[slot,i]-foot).tolist()
                            event=ep.events[-1]
                            event[f'dense_forecast_minus_landing_xy_{ms}ms_m']=dense
                            if self.forecast_clock is not None:
                                event[f'dense_remaining_s_{ms}ms']=float(self.forecast_clock[slot,i])
                            event[f'preview_forecast_minus_landing_xy_{ms}ms_m']=preview_error
                # The last reward audit belongs to the preceding control
                # sample. Require its timestamp, goal and flight to match the impact.
                prior=(now-1)%len(steps)
                if (steps[prior]==now-1 and int(goal_ids[prior,i])==index
                        and int(contact_ids[prior,i])==int(self.env.rebound_plan.contact_step[i])):
                    if preview.shape[-1]==9:
                        ep.events[-1]['preimpact_body_rate_10ms_rad_s']=preview[prior,i,6:9].tolist()
                    rebound=self.env._moving_phase_last_audit['rebound']
                    if 'actual_approach_normal_xy' in rebound and bool(rebound['preparation_active'][i]):
                        if 'continuation_reference_velocity_xy' in rebound and bool(rebound['lead_changed'][i]):
                            ep.events[-1]['continuation_reference_xy_m_s_before_touchdown']=rebound['continuation_reference_velocity_xy'][i].tolist()
                        ep.events[-1].update(
                            approach_normal_xy_10ms_before_touchdown=rebound['actual_approach_normal_xy'][i].tolist(),
                            desired_normal_xy_10ms_before_touchdown=rebound['desired_approach_normal_xy'][i].tolist(),
                            root_velocity_10ms_before_touchdown_m_s=[
                                *rebound['actual_launch_velocity'][i].tolist(),float(rebound['approach_velocity_z'][i])])
            matched_release=False
            if self.launch_snapshot is not None:
                idx,contact,tick=self.launch_snapshot[:3]
                # The contact timestamp distinguishes retries on the same target.
                matched_release=(int(idx[i])==index and int(contact[i])==int(self.env.rebound_plan.contact_step[i])
                                 and 0<=int(contact[i])<int(tick[i])<self.env.common_step_counter)
            if matched_release and self.flight_yaw is not None:
                for key,angle in zip(('yaw_at_release_deg','rising_air_yaw_peak_deg','falling_air_yaw_peak_deg'),
                                     self.flight_yaw[i].tolist()):
                    if angle>=0:
                        ep.events[-1][key]=angle
            if matched_release:
                ep.events[-1]['release_body_rate_rad_s']=self.release_body_rate[i].tolist()
                if hasattr(self.env,'reversal_shortfall'):
                    prediction=float(self.env.reversal_shortfall.last_prediction[i])
                    if math.isfinite(prediction):
                        ep.events[-1]['late_rising_forecast_along_m']=prediction
            if matched_release and attempt>1 and self.retry_unloading is not None:
                actual,desired=self.launch_snapshot[3:5]
                ep.events[-1].update(retry_unloading_ticks=int(self.retry_unloading[2][i]),
                    retry_target_speed_m_s=float(desired[i].norm()),
                    retry_release_error_m_s=float((actual[i]-desired[i]).norm()))
            if moving and hasattr(self.env,'continuation_cosine'):
                # Signed components of the same actual foot error scored at touchdown.
                # Local goals share world XY axes; positive cross is left of travel.
                foot=self.env._robot.data.body_link_pos_w[i,self.env._foot_body_id,:2]
                residual=foot-self.env._two_ground_pt_xy_w[i]
                axis=edge/length
                ep.events[-1].update(foot_error_along_m=float(residual@axis),
                    foot_error_cross_m=float(axis[0]*residual[1]-axis[1]*residual[0]))
                if matched_release:
                    _,_,tick,actual,desired,release_root,release_foot,duration=self.launch_snapshot
                    err=actual[i]-desired[i]
                    T=(self.env.common_step_counter-int(tick[i]))*self.env.step_dt
                    root=self.env._robot.data.root_link_pos_w[i,:2]
                    # Signed XY accounting identity, not an estimate of force or causality.
                    # In order: plan at release, flight duration, release speed, subsequent
                    # root motion, and changing foot-to-root offset (attitude/spring).
                    parts=torch.stack((
                        release_foot[i]-self.env._two_ground_pt_xy_w[i]+desired[i]*duration[i],
                        desired[i]*(T-duration[i]), err*T,
                        (root-release_root[i])-actual[i]*T,
                        (foot-root)-(release_foot[i]-release_root[i])))
                    ep.events[-1].update(
                        release_velocity_xy_m_s=actual[i].tolist(),
                        desired_release_velocity_xy_m_s=desired[i].tolist(),
                        launch_error_along_m_s=float(err@axis),
                        launch_error_cross_m_s=float(axis[0]*err[1]-axis[1]*err[0]),
                        release_to_touch_s=T,
                        foot_error_components_along_m=(parts@axis).tolist(),
                        foot_error_components_cross_m=(axis[0]*parts[:,1]-axis[1]*parts[:,0]).tolist())
                    if self.release_vertical_state is not None:
                        vz,clearance=self.release_vertical_state
                        ep.events[-1].update(release_vertical_speed_m_s=float(vz[i]),
                                             release_foot_clearance_m=float(clearance[i]))
                    if self.release_window is not None:
                        samples,cost=self.release_window
                        ep.events[-1].update(release_window_samples=int(samples[i]),
                                             release_window_reward=float(cost[i]),
                                             planned_flight_time_at_release_s=float(duration[i]))
                        if self.window_motion is not None:
                            halves=self.window_motion[i]
                            totals=(halves[:,1:]@axis).tolist()
                            counts=halves[:,0].tolist()
                            ep.events[-1]['window_along_error_halves_m_s']=[
                                value/count if count else None for value,count in zip(totals,counts)]
                    if self.apex_snapshot is not None:
                        _,apex_tick,apex_xy,apex_vxy=self.apex_snapshot
                        if int(tick[i])<int(apex_tick[i])<self.env.common_step_counter:
                            ascent=(int(apex_tick[i])-int(tick[i]))*self.env.step_dt
                            descent=T-ascent
                            # These signed displacement terms sum to parts[3]:
                            # motion accumulated before apex, carry of the speed
                            # change at apex, then motion relative to that speed.
                            motion=torch.stack((apex_xy[i]-release_root[i]-actual[i]*ascent,
                                (apex_vxy[i]-actual[i])*descent,root-apex_xy[i]-apex_vxy[i]*descent))
                            ep.events[-1].update(release_to_apex_sample_s=ascent,
                                post_release_motion_along_m=(motion@axis).tolist(),
                                post_release_motion_cross_m=(axis[0]*motion[:,1]-axis[1]*motion[:,0]).tolist())
            ep.events[-1].update(target_index=index,target_local_xy_m=target.cpu().tolist(),
                commanded_edge_m=length,moving=moving,direction=DIRECTIONS[direction] if moving else None,
                attempt_for_target=attempt,first_attempt_for_target=attempt==1)
            if getattr(route, 'waypoint_source', None) is not None:
                ep.events[-1]['preview_local_xy_m'] = route.points[i,index+1].tolist()

    def finish(self):
        start=len(self.rows);super().finish()
        for row in self.rows[start:]:
            i=row['case_id'];e=self.env
            if hasattr(e,'plane_totals'):
                row['flight_plane_totals'] = e.plane_totals[i].tolist()
                if self.plane_peak_target is not None:
                    row['flight_plane_peak_at']={
                        mode:dict(target_index=index,control_step=tick) if index>=0 else None
                        for mode,index,tick in zip(('hold','travel'),self.plane_peak_target[i].tolist(),
                                                   self.plane_peak_step[i].tolist())}
            if row['physical_death'] and self.launch_snapshot is not None:
                data=e._robot.data
                tilt=torch.acos((1-2*data.root_quat_w[i,1:3].square().sum()).clamp(-1,1))
                index=int(e.omni_route.index[i])
                terminal=dict(target_index=index,root_height_m=float(data.root_link_pos_w[i,2]-e._terrain.env_origins[i,2]),
                              tilt_deg=float(torch.rad2deg(tilt)),body_rate_rad_s=data.root_ang_vel_b[i].tolist(),
                              confirmed_release=False)
                idx,contact,tick,actual,desired=self.launch_snapshot[:5]
                # Do not attribute the previous hop's liftoff to a terminal
                # before release, including a retry on the same waypoint.
                if (int(idx[i])==index and int(contact[i])==int(e.rebound_plan.contact_step[i])
                        and 0<=int(contact[i])<int(tick[i])<=e.common_step_counter):
                    terminal.update(confirmed_release=True,seconds_since_release=(e.common_step_counter-int(tick[i]))*e.step_dt,
                                    release_velocity_xy_m_s=actual[i].tolist(),desired_release_velocity_xy_m_s=desired[i].tolist(),
                                    release_body_rate_rad_s=self.release_body_rate[i].tolist())
                row['terminal_flight']=terminal
            first_index=getattr(e,'step_hold_evaluation_spec',{}).get('warmup_hops',1)
            row.update(response_group='nominal' if i<64 else 'weaker',
                moving_hits=sum(ev['strict'] and ev['moving'] for ev in row['events']),
                first_moving_strict=any(ev['target_index']==first_index and ev['first_attempt_for_target']
                    and ev['strict'] for ev in row['events']),
                first_stationary_strict=bool(row['events'] and row['events'][0]['strict']),
                stationary_pitch_totals=e.stationary_episode_pitch[i].cpu().tolist(),
                vertical_yaw_totals=e.vertical_episode_yaw[i].cpu().tolist())
            if getattr(e,'rebound_plan',None) is not None:
                row['rebound_totals']=e.rebound_plan.totals[i].cpu().tolist()
                row['body_rate_totals']=e.body_rate_totals[i].cpu().tolist()
            edge=e.omni_route.points[i,first_index]-e.omni_route.points[i,first_index-1]
            direction=int(math.floor(math.atan2(float(edge[1]),float(edge[0]))/(math.pi/4)+.5))%8
            row['first_moving_direction']=DIRECTIONS[direction]
            row['three_moving_hits']=row['moving_hits']>=3
            for key in ('attempted_takeoff','confirmed_takeoff','root_crossed','full_crossed','ring_collision',
                    'reverse','bypass','partial_root_only','minimum_signed_clearance_m','crossing_window',
                    'first_ring_contact','joint_success','joint_and_survive_episode','joint_and_three_later_strict'):
                row.pop(key,None)

    def result(self):
        assert len(self.rows)==128 and len({row['case_id'] for row in self.rows})==128
        rows=sorted(self.rows,key=lambda row:row['case_id'])
        events=[ev for row in rows for ev in row['events']]
        moving=[ev for ev in events if ev['moving']]
        first=[ev for ev in moving if ev['first_attempt_for_target']]
        def stats(values):
            return dict(attempts=len(values),strict=sum(ev['strict'] for ev in values),
                strict_rate=sum(ev['strict'] for ev in values)/len(values) if values else 0.,
                foot_error_m=distribution([ev['foot_error_m'] for ev in values]),
                apex_m=distribution([ev['apex_root_link_world_m'] for ev in values]),
                touchdown_tilt_deg=distribution([ev['touchdown_tilt_deg'] for ev in values]))
        # Include missing takeoffs/touchdowns and timeouts in the fixed first
        # moving-edge denominator; later easy targets/retries cannot replace it.
        directions={key:dict(attempts=sum(row['first_moving_direction']==key for row in rows),
            strict=sum(row['first_moving_direction']==key and row['first_moving_strict'] for row in rows))
                for key in DIRECTIONS}
        for group in directions.values():group['strict_rate']=group['strict']/max(1,group['attempts'])
        counts={key:sum(bool(row[key]) for row in rows) for key in
            ('first_stationary_strict','first_moving_strict','three_moving_hits','physical_death')}
        first_stats=stats(first);gate=self.env.course['omni']['promotion']
        passed=(first_stats['strict_rate']>=gate['moving_first_attempt_strict_min']
            and counts['first_moving_strict']/128>=gate['moving_first_attempt_strict_min']
            and counts['first_stationary_strict']>=116
            and counts['three_moving_hits']>=gate['episodes_with_three_moving_hits_min']
            and counts['physical_death']<=gate['physical_deaths_max']
            and all(v['attempts']>=gate['per_direction_samples_min']
                and v['strict_rate']>=gate['per_direction_strict_min'] for v in directions.values()))
        precision_limit=gate.get('foot_error_p95_m_max')
        if precision_limit is not None:
            p95=first_stats['foot_error_m']['p95']
            passed=passed and p95 is not None and p95<=precision_limit
        maximum_goal=self.env.course['omni'].get('landing_precision_goal_m')
        precision=landing_precision_metrics(rows,maximum_goal if maximum_goal is not None else .1)
        if maximum_goal is not None:passed=passed and precision['passed']
        totals=torch.tensor([row['stationary_pitch_totals'] for row in rows]).sum(0).tolist()
        flights=max(1,totals[0])
        summary=dict(episodes=128,seed=7001,case_version=case_version(self.env.course),task=TASK,
            case_source=self.case_source,denominator='all128 first15s episodes; moving statistics count nonzero target attempts; retries never replace first attempts',
            counts=counts,rates={key:value/128 for key,value in counts.items()},
            moving_first_attempt=first_stats,moving_all_attempts=stats(moving),
            moving_retries=stats([ev for ev in moving if not ev['first_attempt_for_target']]),
            stationary_hold_attempts=stats([ev for ev in events if not ev['moving'] and ev['target_index']>0]),
            per_direction_first_attempt=directions,
            response_groups={group:dict(episodes=64,
                moving_first_attempt=stats([ev for row in rows if row['response_group']==group
                    for ev in row['events'] if ev['moving'] and ev['first_attempt_for_target']]),
                physical_deaths=sum(row['physical_death'] for row in rows if row['response_group']==group))
                    for group in ('nominal','weaker')},
            pitch=dict(completed_flights=int(totals[0]),mean_span_deg=totals[1]/flights,
                span_gt10_rate=totals[2]/flights,mean_rate_rms_rad_s=totals[3]/flights),
            failure_counts_overlapping=dict(no_completed_jump=sum(not row['events'] for row in rows),
                first_moving_not_strict=128-counts['first_moving_strict'],
                fewer_than_three_moving_hits=128-counts['three_moving_hits'],
                moving_underheight=sum(ev['apex_root_link_world_m']<1 for ev in moving),
                **{f'moving_foot_gt{self.env.landing_radius_m*100:g}cm':sum(ev['foot_error_m']>self.env.landing_radius_m for ev in moving)},
                moving_tilt_gt15deg=sum(ev['touchdown_tilt_deg']>15 for ev in moving)),
            promotion_gate=dict(**gate,first_stationary_strict_min=116,passed=passed),
            continuous_jump_metrics=summarize(rows),moving_continuity=continuity_metrics(rows),
            landing_precision=precision)
        if self.env.course.get('rebound_transition'):
            summary['rebound_diagnostics']=rebound_motion_metrics(rows,self.env.course['moving_yaw_prediction'])
            yaw_pass=summary['rebound_diagnostics']['yaw_gate']['passed']
            summary['promotion_gate']['moving_yaw_passed']=yaw_pass
            summary['promotion_gate']['passed']=summary['promotion_gate']['passed'] and yaw_pass
            from .step_hold import body_rate_metrics
            summary['body_rate_diagnostics']=body_rate_metrics(rows)
        if maximum_goal is not None:
            summary['promotion_gate']['all_target_landing_error_max_m']=maximum_goal
        if self.env.course['omni'].get('distance_curriculum'):
            summary['distance_range_m']=self.env.course['omni']['moving_distance_range_m']
            bins=distance_bins(self.env.course)
            summary['per_distance_first_attempt']={
                f'{low:.2f}..{high:.2f}m':stats([ev for ev in first
                    if low<ev['commanded_edge_m']<=(high+1e-5 if high==bins[-1][1] else high)])
                for low,high in bins}
            summary['moving_first_position_accuracy']={
                f'within_{int(limit*100)}cm':sum(ev['foot_error_m']<=limit for ev in first)/max(1,len(first))
                for limit in (.03,.05,.10)}
        if self.env.course.get('moving_navigation'):
            summary['navigation']=dict(
                policy=self.env.course['moving_navigation']['version'],
                advances=sum(bool(ev.get('navigation_advanced')) for ev in events),
                advances_without_strict=sum(bool(ev.get('navigation_advanced_without_strict')) for ev in events),
                strict_jump_scoring_unchanged=True,
                scope='Navigation advancement does not credit a strict jump; height failures remain failed first attempts/prefixes.')
        straight=getattr(self.env,'straight_evaluation_spec',None)
        if straight:
            report=straight_batch_metrics(rows,straight,self.env.course['moving_yaw_prediction'],self.env.landing_radius_m)
            summary.update(case_version=straight['version'],evaluation_scope=straight['evaluation_scope'])
            # Keep full-omni diagnostics explicitly separate: an unobserved
            # direction is never accepted by a fixed-direction evaluation.
            summary['omni_promotion_diagnostic']=dict(**summary['promotion_gate'],
                applicable=False,scope='Full omnidirectional gate; not evaluated by this straight route')
            summary['straight_evaluation']={k:v for k,v in report.items() if k!='cases'}
            for row,case in zip(rows,report['cases']):row['straight_evaluation']=case
            summary['promotion_gate']=dict(scope=straight['evaluation_scope'],passed=report['passed'],
                checks=report['checks'],full_omni_evaluated=False,hardware_deployment_accepted=False)
        step_hold=getattr(self.env,'step_hold_evaluation_spec',None)
        if step_hold:
            from .step_hold import step_hold_batch_metrics
            report=step_hold_batch_metrics(rows,step_hold,self.env.course['moving_yaw_prediction'],self.env.landing_radius_m)
            summary.update(case_version=step_hold['version'],evaluation_scope=step_hold['evaluation_scope'])
            summary['omni_promotion_diagnostic']=dict(**summary['promotion_gate'],
                applicable=False,scope='Full omnidirectional gate; not evaluated by this step/hold route')
            summary['step_hold_evaluation']={k:v for k,v in report.items() if k!='cases'}
            summary['counts']=report['counts']
            summary['rates']={k:report['counts'][k]/128 for k in ('moving_first_strict','strict_sequence_and_survived','physical_deaths')}
            summary['failure_counts_overlapping']=dict(
                moving_first_not_strict=128-report['counts']['moving_first_strict'],
                not_enough_later_holds=128-report['counts']['enough_later_holds'],
                sequence_not_strict_and_survived=128-report['counts']['strict_sequence_and_survived'])
            for row,case in zip(rows,report['cases']):row['step_hold_evaluation']=case
            summary['promotion_gate']=dict(scope=step_hold['evaluation_scope'],passed=report['passed'],
                checks=report['checks'],full_omni_evaluated=False,hardware_deployment_accepted=False)
        queued=getattr(self.env,'cardinal_queue_evaluation_spec',None)
        if queued:
            from .cardinal_queue import batch_metrics
            report=batch_metrics(rows,queued,self.env.course['moving_yaw_prediction'],self.env.landing_radius_m)
            summary.update(case_version=queued['version'],evaluation_scope=queued['evaluation_scope'],denominator=queued['route_scope'])
            summary['omni_promotion_diagnostic']=dict(**summary['promotion_gate'],applicable=False,
                scope='Original15s omni gate is not evaluated by the60s cardinal route')
            summary['cardinal_queue_evaluation']={k:v for k,v in report.items() if k!='cases'}
            summary['landing_precision']=report['precision']
            summary['rebound_diagnostics']=report['motion']
            summary['body_rate_diagnostics']=report['body_rates']
            summary['counts']=report['counts']
            summary['rates']={k:report['counts'][k]/128 for k in ('enough_landings','strict_sequence_and_survived','physical_deaths')}
            summary['failure_counts_overlapping']=dict(not_enough_landings=128-report['counts']['enough_landings'],
                sequence_not_strict_and_survived=128-report['counts']['strict_sequence_and_survived'])
            for row,case in zip(rows,report['cases']):row['cardinal_queue_evaluation']=case
            summary['promotion_gate']=dict(scope=queued['evaluation_scope'],passed=report['passed'],
                checks=report['checks'],full_omni_evaluated=False,hardware_deployment_accepted=False)
        if self.env.course.get('flight_plane_limits'):
            from .flight_plane import flight_plane_report
            limits = self.env.course['flight_plane_limits']
            report = flight_plane_report(rows,limits,('hold','travel'))
            summary['flight_plane'] = dict(limits=limits, **report)
            gate = summary['promotion_gate']
            gate['flight_plane_passed'] = all(mode['passed'] for mode in report.values())
            gate['passed'] = gate['passed'] and gate['flight_plane_passed']
        return summary,rows


def eval_env_class(base):
    class OmniEvalEnv(base):
        def __init__(self,*args,**kwargs):
            self.evaluation=None
            super().__init__(*args,**kwargs)
            self.evaluation=OmniEvaluation(self)

        def _get_rewards(self):
            lengths=([len(ep.events) for ep in self.evaluation.episodes]
                if self.evaluation is not None and self.course.get('moving_navigation') else None)
            if self.evaluation is not None:self.evaluation.sample()
            target=(self.omni_route.index.clone() if self.evaluation is not None
                    and self.evaluation.plane_peak_target is not None else None)
            reward=super()._get_rewards()
            if target is not None:
                self.evaluation.record_plane_peaks(target)
            if self.evaluation is not None and (getattr(self,'cardinal_queue_evaluation_spec',None)
                                                or self.course.get('omni_rebound_handoff')):
                self.evaluation.record_release(self)
            if lengths is not None:
                for i,ep in enumerate(self.evaluation.episodes):
                    if len(ep.events)>lengths[i]:
                        event=ep.events[-1]
                        advanced=int(self.omni_route.index[i])>event['target_index']
                        event.update(navigation_advanced=advanced,
                            navigation_advanced_without_strict=advanced and not event['strict'])
            if self.evaluation is not None:self.evaluation.finish()
            return reward
    return OmniEvalEnv
