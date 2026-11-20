"""One 128-case development evaluation, sampled before auto-reset.

Reuses the baseline Episode flight/landing scorer; never updates the policy.
Ring contact is the frozen conservative virtual-envelope test, not PhysX truth.
"""
import json
import math
import torch
from .metrics import Episode, distribution, summarize
from .ring_env import RingEnv
from .ring_geometry import rotate

PROTOCOL = dict(contact_q_on_m=.002, takeoff_q_max_m=.001,
                minimum_flight_samples=5, takeoff_confirm_samples=3,
                apex_min_world_m=1., foot_error_max_m=.10, touchdown_tilt_max_deg=15.)


class RingEvaluation:
    def __init__(self, env):
        self.env=env
        protocol=dict(PROTOCOL, foot_error_max_m=getattr(env,'landing_radius_m',PROTOCOL['foot_error_max_m']))
        self.episodes=[Episode(protocol) for _ in range(env.num_envs)]
        self.finished=[False]*env.num_envs
        self.rows=[]
        self.window=[dict(before_entry=None, entry=None, root_crossing=None, complete_exit=None,
                          maximum_tilt_deg=0., maximum_angular_speed_rad_s=0.) for _ in self.episodes]
        self.previous=[None]*env.num_envs
        self.launched=[False]*env.num_envs
        self.confirmed=[False]*env.num_envs
        self.duty_integral=[0.]*env.num_envs
        self.action_change=[0.]*env.num_envs
        self.contacts=[None]*env.num_envs
        self.case_source=None

    def restore_cases(self,path):
        """Restore the original128 releases and DR directly, without resampling."""
        payload=json.loads(path.read_text());cases=payload['cases'];e=self.env
        if payload['seed']!=7001 or len(cases)!=128 or [c['case_id'] for c in cases]!=list(range(128)):
            raise ValueError('Expected the original ordered128 seed7001 cases')
        device=e.device;ids=e._robot._ALL_INDICES
        pose=torch.tensor([c['root_link_pose_world'] for c in cases],device=device)
        # Both evaluations use the same128 environment grid and ground origin.
        if not torch.allclose(pose[:,:2],e._terrain.env_origins[:,:2],atol=1e-5):
            raise ValueError('Saved release tiles differ from the current128-environment grid')
        if any(any(v!=0 for v in c['root_com_velocity_world']) for c in cases):
            raise ValueError('Saved release must use the inherited zero-velocity contract')
        e._robot.write_root_pose_to_sim(pose,ids)
        e._robot.write_root_velocity_to_sim(torch.zeros(e.num_envs,6,device=device),ids)
        e._robot.write_joint_state_to_sim(torch.tensor([c['joint_pos'] for c in cases],device=device),
                                        torch.tensor([c['joint_vel'] for c in cases],device=device),env_ids=ids)
        from .flight_plane import restore_rigid_scales
        restore_rigid_scales(e,ids,cases)
        for attr,key in (('dr_mass_multi','mass_multiplier'),('dr_inertia_multi','inertia_multiplier'),('dr_tm','motor_time_constant_s')):
            getattr(e,attr)[:]=torch.tensor([c[key] for c in cases],device=device)
        e._two_ground_pt_xy_w[:]=torch.tensor([c['ground_target_xy'] for c in cases],device=device)
        e._two_ground_pt1_xy_w[:]=torch.tensor([c['next_ground_target_xy'] for c in cases],device=device)
        from .two_ground_targets import _set_active_goal
        _set_active_goal(e,ids)
        d=e._robot.data
        e.ring.reset(ids,d.root_link_pos_w,d.root_quat_w,d.joint_pos[:,e._spring_joint_id],e._terrain.env_origins)
        self.case_source=str(path.resolve())

    def freeze_cases(self, folder):
        e=self.env;d=e._robot.data
        fields=dict(root_link_pose_world=torch.cat((d.root_link_pos_w,d.root_quat_w),-1),
                    root_com_velocity_world=d.root_com_lin_vel_w,
                    joint_pos=d.joint_pos,joint_vel=d.joint_vel,
                    mass_multiplier=e.dr_mass_multi,inertia_multiplier=e.dr_inertia_multi,
                    motor_time_constant_s=e.dr_tm,ground_target_xy=e._two_ground_pt_xy_w,
                    next_ground_target_xy=e._two_ground_pt1_xy_w,ring_center_world=e.ring.center)
        if hasattr(e,'rigid_scales'):
            fields['physical_mass_inertia_scale'] = e.rigid_scales
        values={k:v.detach().cpu().tolist() for k,v in fields.items()}
        cases=[dict(case_id=i,**{k:v[i] for k,v in values.items()}) for i in range(e.num_envs)]
        for i,episode in enumerate(self.episodes):
            episode.start=dict(root_z_m=cases[i]['root_link_pose_world'][2],
                               joint_q_m=cases[i]['joint_pos'][e._spring_joint_id],
                               joint_qdot_m_s=cases[i]['joint_vel'][e._spring_joint_id],target_z=1.)
        (folder/'cases.json').write_text(json.dumps(dict(
            version='ring_r1_seed7001_initial128_v1',seed=e.cfg.seed,
            release='same0.8..1.2m/fixed5deg mixture and DR as training; noise retained',
            quota='exactly first episode per environment; all128 included',cases=cases),indent=2)+'\n')

    def physics(self):
        e=self.env;d=e._robot.data;f=e._foot_event;r=e.ring
        if r.previous is None:return
        quat=d.root_quat_w
        tilt=torch.rad2deg(torch.acos((1-2*quat[:,1:3].square().sum(-1)).clamp(-1,1)))
        vertices=r.shape.vertices(d.root_link_pos_w,quat,d.joint_pos[:,e._spring_joint_id])
        signed=((vertices-r.center[:,None,None,:])*r.normal[:,None,None,:]).sum(-1)
        data=torch.cat((d.root_link_pos_w-e._terrain.env_origins,quat,d.root_com_lin_vel_w,
                        d.root_ang_vel_b,d.joint_pos[:,e._spring_joint_id,None],tilt[:,None],
                        signed.amax((1,2))[:,None],r.passed[:,None].float(),
                        r.root_crossed[:,None].float(),f.phase[:,None].float(),f.attempt_index[:,None].float(),
                        r.collision[:,None].float()),-1).detach().cpu().tolist()
        for i,a in enumerate(data):
            if self.finished[i]:continue
            snap=dict(position_local_m=a[:3],quaternion_wxyz=a[3:7],velocity_world_m_s=a[7:10],
                      angular_velocity_body_rad_s=a[10:13],spring_q_m=a[13],tilt_deg=a[14])
            w=self.window[i]
            if a[18] in (f.CANDIDATE,f.FLIGHT) and a[19]==0:
                self.launched[i]=True
                self.confirmed[i]|=a[18]==f.FLIGHT
                if a[15]>=-e.course['axial_thickness_m']/2 and w['entry'] is None:
                    w['before_entry']=self.previous[i];w['entry']=snap
                if w['entry'] is not None and w['complete_exit'] is None:
                    w['maximum_tilt_deg']=max(w['maximum_tilt_deg'],a[14])
                    w['maximum_angular_speed_rad_s']=max(w['maximum_angular_speed_rad_s'],math.sqrt(sum(x*x for x in a[10:13])))
            if a[17] and w['root_crossing'] is None:w['root_crossing']=snap
            if a[16] and w['complete_exit'] is None:w['complete_exit']=snap
            if a[20] and self.contacts[i] is None:
                contact=dict(before=self.previous[i],current=snap,control_step=int(e.episode_length_buf[i]),
                             flight_phase=int(a[18]),attempt_index=int(a[19]))
                context=getattr(r.shape,'sweep_context',None)
                if context is not None:
                    old=context[0]
                    contact['sweep_previous']=dict(position_local_m=(old[0][i]-e._terrain.env_origins[i]).cpu().tolist(),
                        quaternion_wxyz=old[1][i].cpu().tolist(),spring_q_m=float(old[2][i]))
                self.contacts[i]=contact
            self.previous[i]=snap

    def sample(self):
        # Called BEFORE RingEnv advances the ground queue or clears flight peak.
        e=self.env;d=e._robot.data;q=d.root_quat_w
        foot=d.body_link_pos_w[:,e._foot_body_id]
        local=torch.zeros_like(foot);local[:,2]=-.273+d.joint_pos[:,e._spring_joint_id]
        estimate=d.root_link_pos_w+rotate(q,local)
        tilt=torch.rad2deg(torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1,1)))
        physical=(d.root_link_pos_w[:,2]<.05)|(q[:,1:3].square().sum(-1)>.5)
        data=torch.cat((d.root_link_pos_w[:,2:3],d.joint_pos[:,e._spring_joint_id,None],
            d.root_com_lin_vel_w,tilt[:,None],
            torch.linalg.vector_norm(foot[:,:2]-e._desired_pos_w[:,:2],dim=-1)[:,None],
            torch.linalg.vector_norm(d.root_link_pos_w[:,:2]-e._desired_pos_w[:,:2],dim=-1)[:,None],
            e._desired_pos_w[:,2:3],((e._actions*.5+.5).clamp(0,1)).amax(-1)[:,None],
            e._motor_u.abs().amax(-1)[:,None],e._thrust[:,0,2:3],
            torch.linalg.vector_norm(estimate-foot,dim=-1)[:,None],physical[:,None].float(),
            e._motor_u.sum(-1)[:,None],(e._actions-e._previous_actions).square().sum(-1)[:,None]),-1).detach().cpu().tolist()
        collision=e.ring.collision.cpu().tolist()
        for i,a in enumerate(data):
            if self.finished[i]:continue
            self.duty_integral[i]+=a[14]*e.step_dt
            self.action_change[i]+=a[15]*e.step_dt
            self.episodes[i].sample(dict(z=a[0],q=a[1],vz=a[4],velocity=a[2:5],tilt=a[5],
                foot_error=a[6],root_error=a[7],target_z=a[8],duty_max=a[9],motor_max=a[10],
                force=a[11],foot_formula_error=a[12],died=bool(a[13] or collision[i])))

    def finish(self):
        e=self.env;r=e.ring;d=e._robot.data
        ids=torch.nonzero(e.reset_buf).flatten().cpu().tolist()
        for i in ids:
            if self.finished[i]:continue
            quat=d.root_quat_w[i]
            physical=bool((d.root_link_pos_w[i,2]<.05)|(quat[1:3].square().sum()>.5))
            collision=bool(r.collision[i])
            row=self.episodes[i].finish(physical or collision,bool(e.reset_time_outs[i]),e.step_dt)
            clearance=float(r.min_clearance[i])
            row.update(case_id=i,attempted_takeoff=self.launched[i],confirmed_takeoff=self.confirmed[i],
                       root_crossed=bool(r.root_crossed[i]),full_crossed=bool(r.passed[i]),
                       ring_collision=collision,physical_death=physical,reverse=bool(r.reverse[i]),
                       bypass=bool(r.bypass[i]),partial_root_only=bool(r.root_crossed[i] and not r.passed[i]),
                       minimum_signed_clearance_m=clearance if math.isfinite(clearance) else None,
                       crossing_window=self.window[i],first_ring_contact=self.contacts[i],motor_duty_sum_integral_s=self.duty_integral[i],
                       action_change_squared_integral_s=self.action_change[i])
            row['joint_success']=row['full_crossed'] and row['first_strict']
            row['joint_and_survive_episode']=row['joint_success'] and not row['died']
            row['joint_and_three_later_strict']=(row['joint_and_survive_episode'] and len(row['events'])>=4
                                                and all(x['strict'] for x in row['events'][1:4]))
            self.rows.append(row);self.finished[i]=True

    def result(self):
        assert len(self.rows)==128 and len({r['case_id'] for r in self.rows})==128
        rows=sorted(self.rows,key=lambda r:r['case_id']);n=len(rows)
        keys=('attempted_takeoff','confirmed_takeoff','root_crossed','full_crossed','first_strict',
              'joint_success','joint_and_survive_episode','joint_and_three_later_strict',
              'ring_collision','physical_death','reverse','bypass','partial_root_only')
        counts={k:sum(bool(r[k]) for r in rows) for k in keys}
        first=[r['events'][0] for r in rows if r['events']]
        failure=dict(no_confirmed_takeoff=n-counts['confirmed_takeoff'],no_first_touchdown=n-len(first),
                     no_full_crossing=n-counts['full_crossed'],
                     first_underheight=sum(e['apex_root_link_world_m']<1. for e in first),
                     first_foot_gt10cm=sum(e['foot_error_m']>.10 for e in first),
                     first_tilt_gt15deg=sum(e['touchdown_tilt_deg']>15. for e in first))
        clearance=[r['minimum_signed_clearance_m'] for r in rows if r['minimum_signed_clearance_m'] is not None]
        good_clearance=[r['minimum_signed_clearance_m'] for r in rows if r['full_crossed'] and r['minimum_signed_clearance_m'] is not None]
        out=dict(episodes=n,seed=7001,case_version='ring_r1_seed7001_initial128_v1',
                 case_source=self.case_source,geometry_version=self.env.course['envelope']['version'],
                 denominator='all128 first episodes; initial drop never a jump; no retries replace first attempt',
                 counts=counts,rates={k:v/n for k,v in counts.items()},failure_counts_overlapping=failure,
                 minimum_clearance_m=distribution(clearance),full_crossing_clearance_m=distribution(good_clearance),
                 promotion_gate=dict(joint_rate_min=.90,physical_deaths_max=0,
                     passed=counts['joint_success']/n>=.90 and counts['physical_death']==0),
                 continuous_jump_metrics=summarize(rows),
                 collision_scope='conservative virtual swept boxes/annulus, not physical ring contacts',
                 energy_scope='motor-duty/action-change proxies per case; not calibrated joules')
        return out,rows


class RingEvalEnv(RingEnv):
    def __init__(self,*args,**kwargs):
        self.evaluation=None
        super().__init__(*args,**kwargs)
        self.evaluation=RingEvaluation(self)

    def _ring_sample(self):
        super()._ring_sample()
        if self.evaluation is not None:self.evaluation.physics()

    def _get_rewards(self):
        if self.evaluation is not None:self.evaluation.sample()
        reward=super()._get_rewards()
        if self.evaluation is not None:self.evaluation.finish()
        return reward
