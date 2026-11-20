"""First-episode, contact-to-contact diagnostics; separate from promotion gates."""
import statistics
import math

import torch

from .guarded_keys import FIELDS, HOLD, STEP


class HoldProbe:
    def __init__(self, env, direction):
        guard=env.key_guard
        self.counter_fields=guard.fields
        self.landing_radius_m=env.landing_radius_m
        if direction == 'four-axis-rebound':
            guard.direction.copy_(guard.ids % 4)
            self.pending_kick = torch.zeros_like(guard.mask)
            self.kick_count = torch.zeros_like(guard.head)
            self.final_guard_counts = torch.zeros_like(guard.count)
            self.command_touch = torch.full((env.num_envs,2,3),float('nan'),device=env.device)
            self.command_release = torch.full((env.num_envs,2,4),float('nan'),device=env.device)
        else:
            guard.direction.fill_(2 if direction=='+Y' else 3)
        guard.two_press_ticks=(round(14./env.step_dt),round(14.25/env.step_dt))
        self.live=torch.ones_like(guard.mask)
        self.dead=torch.zeros_like(self.live)
        self.arrivals=torch.zeros_like(guard.head)
        self.strict_arrivals=torch.zeros_like(guard.head)
        self.sent=torch.zeros_like(guard.head)
        self.over_excursion=torch.zeros_like(self.live)
        self.count=torch.zeros((env.num_envs,2),dtype=torch.long,device=env.device)
        self.samples=torch.full((env.num_envs,2,10,5),float('nan'),device=env.device)
        self.peak=torch.zeros((env.num_envs,4),device=env.device)

    def before(self,env):
        guard=env.key_guard
        landings=guard.count[:,FIELDS.index('landings')].clone()
        before=(guard.count[:,FIELDS.index('requested')]==0) & (landings>0)
        after=guard.terminal_hold & (guard.count[:,FIELDS.index('arrived')]==2)
        stage=after.long()
        active=self.live & (before|after) & (guard.state==HOLD) & (self.count.gather(1,stage[:,None])[:,0]<10)
        data=env._robot.data
        goal=env._two_ground_pt_xy_w
        quat=data.root_quat_w
        tilt=torch.rad2deg(torch.acos((1-2*quat[:,1:3].square().sum(-1)).clamp(-1.,1.)))
        if hasattr(self,'command_touch'):
            self.in_step = self.live & (guard.state==STEP)
            self.command_index = guard.count[:,FIELDS.index('launched')]-1
            axis = guard.dirs[guard.direction]
            self.cross_axis = torch.stack((-axis[:,1],axis[:,0]),dim=1)
            delta = data.body_link_pos_w[:,env._foot_body_id,:2]-goal
            self.touch_values = torch.stack(((delta*axis).sum(1),(delta*self.cross_axis).sum(1),tilt),dim=1)
        values=torch.stack((tilt,(data.root_link_pos_w[:,:2]-goal).norm(dim=-1),
                            data.root_link_lin_vel_w[:,:2].norm(dim=-1),
                            data.root_link_pos_w[:,2]-env._terrain.env_origins[:,2]),dim=-1)
        self.over_excursion |= active & after & (values[:,1]>.30)
        foot_error=(data.body_link_pos_w[:,env._foot_body_id,:2]-goal).norm(dim=-1)
        self.peak=torch.where(active[:,None],torch.maximum(self.peak,values),0.)
        return active,stage,landings,foot_error

    def after(self,env,snapshot):
        guard=env.key_guard
        active,stage,old_landings,foot_error=snapshot
        hit=guard.count[:,FIELDS.index('landings')]>old_landings
        if hasattr(self,'command_touch'):
            self.final_guard_counts[self.live] = guard.count[self.live]
            landed = hit & self.in_step
            self.command_touch[landed,self.command_index[landed]] = self.touch_values[landed]
            released = self.in_step & env._moving_phase_last_audit['rebound']['launch'] & ~hit
            ids = released.nonzero().flatten()
            axis = guard.dirs[guard.direction[ids]]
            cross = self.cross_axis[ids]
            v = env._robot.data.root_link_lin_vel_w[ids,:2]
            ref = env.rebound_plan.velocity[ids]
            self.command_release[ids,self.command_index[ids]] = torch.stack(
                ((v*axis).sum(1),(ref*axis).sum(1),(v*cross).sum(1),(ref*cross).sum(1)),dim=1)
        ids=(active & hit).nonzero().flatten()
        phase=stage[ids]
        self.samples[ids,phase,self.count[ids,phase]]=torch.cat((self.peak[ids],foot_error[ids,None]),dim=-1)
        self.count[ids,phase]+=1
        self.peak[hit]=0.
        self.dead |= self.live & env.reset_terminated
        for target,name in ((self.arrivals,'arrived'),(self.strict_arrivals,'strict_arrived'),(self.sent,'requested')):
            target.copy_(torch.where(self.live,guard.count[:,FIELDS.index(name)],target))
        self.live &= ~(env.reset_terminated | env.reset_time_outs | (self.count[:,1]>=10))
        if hasattr(self,'pending_kick'):
            launch = env._moving_phase_last_audit['rebound']['launch']
            self.pending_kick.copy_(self.live & guard.terminal_hold & (self.arrivals==2)
                                    & (self.kick_count<6) & launch)

    def apply_rebound(self,env):
        ids = self.pending_kick.nonzero().flatten()
        if len(ids):
            guard = env.key_guard
            velocity = env._robot.data.root_link_vel_w[ids].clone()
            velocity[:,:2] -= .10*guard.dirs[guard.direction[ids]]
            env._robot.write_root_link_velocity_to_sim(velocity,env_ids=ids)
            self.kick_count[ids] += 1
            self.pending_kick[ids] = False

    def report(self):
        names=('peak_tilt_deg','peak_root_offset_m','peak_xy_speed_m_s','apex_m','landing_foot_error_m')
        rows=[]
        for index,phases in enumerate(self.samples.cpu().tolist()):
            row=dict(case_id=index,death=bool(self.dead[index]),arrivals=int(self.arrivals[index]),
                     strict_arrivals=int(self.strict_arrivals[index]),requested=int(self.sent[index]),
                     endpoint_over_30cm=bool(self.over_excursion[index]))
            if hasattr(self,'kick_count'):
                row.update(direction=('+X','-X','+Y','-Y')[index%4],rebound_count=int(self.kick_count[index]))
                row['guard_counts'] = dict(zip(self.counter_fields,self.final_guard_counts[index].tolist()))
                row['steps'] = []
                for touch,release in zip(self.command_touch[index].tolist(),self.command_release[index].tolist()):
                    info = {}
                    if math.isfinite(touch[0]):
                        info.update(zip(('foot_along_m','foot_cross_m','tilt_deg'),touch))
                    if math.isfinite(release[0]):
                        info.update(zip(('launch_along_m_s','desired_along_m_s','launch_cross_m_s','desired_cross_m_s'),release))
                    row['steps'].append(info)
            for label,values in zip(('warm','endpoint'),phases):
                expected=int(self.count[index,0 if label=='warm' else 1])
                if not all(math.isfinite(value) for hop in values[:expected] for value in hop):
                    raise ValueError('Nonfinite completed-hop diagnostic')
                row[label]=[dict(zip(names,hop)) for hop in values[:expected]]
            rows.append(row)
        paired=[row for row in rows if len(row['warm'])>=8 and len(row['endpoint'])>=8]
        def typical(label,key):
            return (statistics.median(statistics.median(hop[key] for hop in row[label][2:8])
                                      for row in paired) if paired else None)
        return dict(cases=rows,landing_radius_m=self.landing_radius_m,episodes=len(rows),paired_through_hop8=len(paired),
                    all_ten=sum(len(row['endpoint'])==10 for row in rows),
                    endpoint_over_30cm=sum(row['endpoint_over_30cm'] for row in rows),
                    deaths=sum(row['death'] for row in rows),both_arrived=sum(row['arrivals']==2 for row in rows),
                    requested=sum(row['requested'] for row in rows),strict_arrivals=sum(row['strict_arrivals'] for row in rows),
                    paired_hops_3_to_8={label:{key:typical(label,key) for key in names} for label in ('warm','endpoint')})
