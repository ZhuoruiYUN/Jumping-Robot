"""Capture a small set of first-contact segments; replay geometry without Isaac.

Does not change the collision detector, ring course, policy, rewards or physics.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'outputs/benchmark37_ring_r1/eval_seed7001_20261105_024407_947007Z'
CHECKPOINT=ROOT/'outputs/benchmark37_ring_r1/train_seed42_20261104_132720Z/model_49.pt'
CASE_IDS=(1,4,0,6,104,9,11,2)
PART_NAMES=('body_arms_motors_box','fixed_base_box','sliding_leg_rod_foot_box',
            'rotor_xminus_yminus_box','rotor_xminus_yplus_box',
            'rotor_xplus_yminus_box','rotor_xplus_yplus_box')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def capture(args):
    saved=json.loads((args.cases/'cases.json').read_text())['cases']
    case_ids=tuple(args.case_ids or CASE_IDS)
    if not 2<=len(case_ids)<=16 or len(set(case_ids))!=len(case_ids) or any(i<0 or i>=len(saved) for i in case_ids):
        raise ValueError('Select2..16 distinct saved case IDs for the small multi-env diagnostic')
    course=json.loads((args.checkpoint.parent/'ring_config.json').read_text())
    if course.get('task')=='obs73_stationary_foundation_v1':
        raise ValueError('Stationary foundation has no ring contacts; use eval or play')
    from .ring_obs73 import observation_dim
    obs_dim=observation_dim(course)
    min_loop=course.get('task')=='drop_stationary_ring_stationary2_v1'
    from isaaclab.app import AppLauncher
    app=AppLauncher(args).app
    raw=None
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    folder=args.checkpoint.resolve().parent.parent/f'contact_diagnostic_seed7001_{stamp}'
    folder.mkdir()
    manifest=dict(status='running',seed=7001,case_ids=list(case_ids),checkpoint=str(args.checkpoint),
                  scope='initial-state replay for contact diagnosis only; not128-case evaluation or exact RNG replay')
    write(folder/'manifest.json',manifest)
    print('RING_CONTACT_RUN='+str(folder),flush=True)
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure
        from .ring_env import RingEnv
        if min_loop:
            from .ring_min_loop import env_class
            RingEnv=env_class()
            if course.get('height_randomization'):
                from .ring_height_loop import env_class as height_env_class
                RingEnv=height_env_class(evaluation=True)
            if course.get('targeted_adjustments'):
                from .ring_targeted_adjustments import env_class as targeted_env_class
                RingEnv=targeted_env_class(evaluation=True)
            if obs_dim==73:
                from .ring_obs73 import wrap_env
                RingEnv=wrap_env(RingEnv)

        class ContactEnv(RingEnv):
            def __init__(self,*a,**kw):
                self.contact_probe=False
                super().__init__(*a,**kw)
                self.finished=[False]*self.num_envs
                self.contacts=[];self.terminals=[]
                self.paths=[[] for _ in range(self.num_envs)]

            def _ring_sample(self):
                if not self.contact_probe:return super()._ring_sample()
                previous=self.ring.previous
                collision=self.ring.collision.clone()
                super()._ring_sample()
                fresh=torch.nonzero(self.ring.collision&~collision).flatten().cpu().tolist()
                for i in fresh:
                    if self.finished[i]:continue
                    d=self._robot.data;origin=self._terrain.env_origins[i]
                    def pose(p,q,s):return dict(position_local_m=(p[i]-origin).cpu().tolist(),quaternion_wxyz=q[i].cpu().tolist(),spring_q_m=float(s[i]))
                    f=self._foot_event
                    self.contacts.append(dict(case_id=case_ids[i],control_step=int(self.episode_length_buf[i]),
                        physics_step=int(self._sim_step_counter),previous=pose(*previous),current=pose(*self.ring.previous),
                        root_link_velocity_world_m_s=d.root_link_lin_vel_w[i].cpu().tolist(),
                        angular_velocity_body_rad_s=d.root_ang_vel_b[i].cpu().tolist(),
                        spring_velocity_m_s=float(d.joint_vel[i,self._spring_joint_id]),
                        flight_phase=int(f.phase[i]),attempt_index=int(f.attempt_index[i]),
                        root_crossed=bool(self.ring.root_crossed[i]),full_crossed=bool(self.ring.passed[i]),
                        group='first_entry' if not bool(self.ring.root_crossed[i]) else 'first_exit',
                        ring_center_local_m=(self.ring.center[i]-origin).cpu().tolist(),
                        ground_target_local_m=(self._desired_pos_w[i]-origin).cpu().tolist()))

            def _get_rewards(self):
                if self.contact_probe:
                    d=self._robot.data;f=self._foot_event
                    stage=self.loop.stage if min_loop else f.attempt_index
                    # Before the landing tracker clears its peak and before
                    # auto-reset: one10ms sample, only the first episode.
                    samples=torch.cat((d.root_link_pos_w-self._terrain.env_origins,
                        d.root_link_lin_vel_w,d.root_quat_w,
                        d.body_link_pos_w[:,self._foot_body_id]-self._terrain.env_origins,
                        d.joint_pos[:,self._spring_joint_id,None],f.phase[:,None],
                        f.attempt_index[:,None],stage[:,None],f.peak[:,None],
                        self._desired_pos_w[:,2:3],self.ring.center[:,2:3]-self._terrain.env_origins[:,2:3],
                        d.root_ang_vel_b[:,1:2],d.joint_vel[:,self._spring_joint_id,None],
                        self._moment[:,0,1:2],self._actions,self._motor_u),-1).cpu().tolist()
                result=super()._get_rewards()
                if self.contact_probe:
                    flags=torch.stack((self.ring.passed,self.ring.collision,self.ring.bypass,
                        self.ring.root_crossed,self._foot_event.last_qualified,self.reset_buf),-1).cpu().tolist()
                    for i,sample in enumerate(samples):
                        if not self.finished[i]:self.paths[i].append(sample[:18]+flags[i]+sample[18:])
                    for i in torch.nonzero(self.reset_buf).flatten().cpu().tolist():
                        if self.finished[i]:continue
                        self.terminals.append(dict(case_id=case_ids[i],seconds=float(self.episode_length_buf[i])*self.step_dt,
                            ring_collision=bool(self.ring.collision[i]),full_crossed=bool(self.ring.passed[i]),
                            first_joint_success=bool(self.ring_joint_success[i]),timed_out=bool(self.reset_time_outs[i]),
                            final_stage=int(self.loop.stage[i]) if min_loop else None,
                            ring_height_local_m=float(self.ring.center[i,2]-self._terrain.env_origins[i,2]),
                            bypass=bool(self.ring.bypass[i]),root_crossed=bool(self.ring.root_crossed[i])))
                        self.finished[i]=True
                return result

        torch.manual_seed(7001)
        cfg=QuadhopperEnvCfg();configure(cfg)
        cfg.seed=7001;cfg.scene.num_envs=len(case_ids);cfg.sim.device=args.device;cfg.observation_space=obs_dim
        cfg.power_model_path=str(ROOT/'Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt')
        cfg.csv_log_path=str(folder/'unused.csv')
        raw=ContactEnv(cfg,course)
        wrapped=RslRlVecEnvWrapper(raw)
        d=raw._robot.data;device=raw.device;ids=raw._robot._ALL_INDICES
        selected=[saved[i] for i in case_ids]
        poses=torch.tensor([r['root_link_pose_world'] for r in selected],device=device)
        # Source release XY is its tile origin. Translate tiles, preserving the
        # saved local pose, quaternion, spring and all DR values.
        poses[:,:2]=raw._terrain.env_origins[:,:2]
        raw._robot.write_root_pose_to_sim(poses,ids)
        assert all(max(map(abs,r['root_com_velocity_world']))==0 for r in selected)
        raw._robot.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=device),ids)
        raw._robot.write_joint_state_to_sim(torch.tensor([r['joint_pos'] for r in selected],device=device),
                                           torch.tensor([r['joint_vel'] for r in selected],device=device),env_ids=ids)
        for attr,key in (('dr_mass_multi','mass_multiplier'),('dr_inertia_multi','inertia_multiplier'),('dr_tm','motor_time_constant_s')):
            getattr(raw,attr)[:]=torch.tensor([r[key] for r in selected],device=device)
        if min_loop:
            from .two_ground_targets import _set_active_goal
            raw._reset_task_targets(ids);_set_active_goal(raw,ids)
        raw.ring.reset(ids,d.root_link_pos_w,d.root_quat_w,d.joint_pos[:,raw._spring_joint_id],raw._terrain.env_origins)
        if course.get('height_randomization'):raw.restore_height_cases(ids,selected)
        rc=json.loads((args.checkpoint.parent/'runner_config.json').read_text());rc['seed']=7001
        runner=OnPolicyRunner(wrapped,rc,log_dir=None,device=args.device)
        runner.alg.policy.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=False)['model_state_dict'],strict=True)
        policy=runner.get_inference_policy(device=args.device)
        obs=wrapped.get_observations();raw.contact_probe=True
        write(folder/'selected_cases.json',selected)
        with torch.inference_mode():
            for step in range(raw.max_episode_length+1):
                obs,_,done,_=wrapped.step(policy(obs));runner.alg.policy.reset(done)
                if step%250==0:print(f'RING_CONTACT_STEP={step} COMPLETE={sum(raw.finished)}/{len(case_ids)} CONTACTS={len(raw.contacts)}',flush=True)
                if all(raw.finished):break
        assert all(raw.finished),'Incomplete diagnostic quota'
        write(folder/'contacts.json',dict(course=course,case_ids=list(case_ids),scope=manifest['scope'],contacts=raw.contacts,terminals=raw.terminals))
        columns=['root_x','root_y','root_z','vx','vy','vz','qw','qx','qy','qz',
                 'foot_x','foot_y','foot_z','spring_q','flight_phase','attempt_index','loop_stage','peak_z',
                 'passed','collision','bypass','root_crossed','strict_touchdown','terminal']
        columns+=['goal_z','ring_center_z','body_wy','spring_qdot','moment_y',
                  'action0','action1','action2','action3','motor0','motor1','motor2','motor3']
        write(folder/'trajectories.json',dict(scope=manifest['scope'],step_dt_s=raw.step_dt,columns=columns,
            course=course,trajectories=[dict(case_id=c,samples=s) for c,s in zip(case_ids,raw.paths)],terminals=raw.terminals))
        manifest.update(status='complete',contact_segments=len(raw.contacts),trace=str(folder/'contacts.json'))
        print('RING_CONTACT_TRACE='+str(folder/'contacts.json'),flush=True)
    except BaseException as exc:
        import traceback
        manifest.update(status='failed',error=str(exc));traceback.print_exc();raise
    finally:
        write(folder/'manifest.json',manifest)
        if raw is not None:raw.close()
        app.close()


def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','replay','analyze'))
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--cases',type=Path,default=EVAL)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--case-id',type=int)
    parser.add_argument('--case-ids',type=int,nargs='+',help='Capture2..16 selected saved releases; defaults to the legacy diagnostic set')
    parser.add_argument('--geometry',choices=('v1','v2'),default='v1')
    parser.add_argument('--assert-clear',action='store_true')
    if len(sys.argv)>1 and sys.argv[1] in ('replay','analyze'):
        args=parser.parse_args()
        if args.mode=='replay':replay(args)
        else:analyze(args)
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args();capture(args)


def replay(args):
    import torch
    from .ring_geometry import RingEvents
    from .ring_geometry_v2 import RingEventsV2
    trace=json.loads(args.trace.read_text());reproduced=0;processed=0
    for row in trace['contacts']:
        if args.case_id is not None and row['case_id']!=args.case_id:continue
        processed+=1
        def tensors(s):return (torch.tensor([s['position_local_m']]),torch.tensor([s['quaternion_wxyz']]),torch.tensor([s['spring_q_m']]))
        old=tensors(row['previous']);new=tensors(row['current'])
        event_type=RingEventsV2 if args.geometry=='v2' else RingEvents
        row_course=dict(trace['course'],center_local_m=row.get('ring_center_local_m',trace['course']['center_local_m']))
        events=event_type(1,'cpu',row_course);events.reset(torch.tensor([0]),*old,torch.zeros(1,3))
        events.sample(*new,torch.ones(1,dtype=torch.bool))
        hit=bool(events.collision[0]);reproduced+=int(hit)
        print(f"case={row['case_id']} geometry={args.geometry} contact={hit} control_step={row['control_step']}")
    print(f'CONTACT_REPLAY={reproduced}/{processed}')
    if args.assert_clear:assert reproduced==0,f'{reproduced} captured contact segments still trigger termination'


def hull(points):
    points=sorted(set(points))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def chain(seq):
        out=[]
        for p in seq:
            while len(out)>=2 and cross(out[-2],out[-1],p)<=0:out.pop()
            out.append(p)
        return out
    return chain(points)[:-1]+chain(reversed(points))[:-1] if len(points)>1 else points


def radial_range(points):
    """Exact radial extrema of a convex plane projection, including interior."""
    polygon=hull(points)
    maximum=max(math.hypot(*p) for p in points)
    if len(polygon)==1:return maximum,maximum
    inside=len(polygon)>=3
    minimum=math.inf
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        cross=a[0]*b[1]-a[1]*b[0]
        if cross < -1e-12:inside=False
        v=(b[0]-a[0],b[1]-a[1]);den=v[0]**2+v[1]**2
        t=max(0.,min(1.,-(a[0]*v[0]+a[1]*v[1])/den)) if den else 0.
        minimum=min(minimum,math.hypot(a[0]+t*v[0],a[1]+t*v[1]))
    return (0. if inside else minimum),maximum


def clipped_points(vertices,edges,center,half):
    points=[]
    for v in vertices:
        if abs(v[0]-center[0])<=half:points.append((v[1]-center[1],v[2]-center[2]))
    for i,j in edges:
        a,b=vertices[i],vertices[j];den=b[0]-a[0]
        if abs(den)<1e-12:continue
        for sign in (-1,1):
            alpha=(center[0]+sign*half-a[0])/den
            if 0<=alpha<=1:points.append((a[1]+alpha*(b[1]-a[1])-center[1],a[2]+alpha*(b[2]-a[2])-center[2]))
    return points


def analyze(args):
    import torch
    from .ring_geometry import FullEnvelope, rotate
    from .ring_geometry_v2 import RingEventsV2
    trace=json.loads(args.trace.read_text());course=trace['course']
    assert course['normal_world']==[1.,0.,0.],'This diagnostic targets the frozen x-normal course'
    shape=FullEnvelope('cpu');edges=shape.edges.tolist();center=course['center_local_m']
    radius=course['inner_radius_m'];half=course['axial_thickness_m']/2;delta=course['delta_m']
    outer=radius+course['tube_radial_width_m']

    def tensor(s):return (torch.tensor([s['position_local_m']]),torch.tensor([s['quaternion_wxyz']]),torch.tensor([s['spring_q_m']]))
    def interpolate(a,b,t):
        p,q,s=a;bp,bq,bs=b
        bq=torch.where(((q*bq).sum(-1)<0)[:,None],-bq,bq)
        return p+(bp-p)*t,torch.nn.functional.normalize(q+(bq-q)*t,dim=-1),s+(bs-s)*t
    def inflation(a,b):
        p,q,s=a;bp,bq,bs=b
        distance=float(torch.linalg.vector_norm(bp-p))/2
        angle=2*math.acos(min(1.,abs(float((q*bq).sum()))))
        rotation=2*shape.radius*math.sin(angle/4)
        spring=abs(float(bs-s))/2
        return dict(translation_m=distance,rotation_m=rotation,spring_m=spring,delta_m=delta,total_m=distance+rotation+spring+delta)
    def disk_vertices(p,q,x,y):
        count=32;r=.1/math.cos(math.pi/count)
        local=torch.tensor([[x+r*math.cos(2*math.pi*i/count),y+r*math.sin(2*math.pi*i/count),z]
                             for z in (-.008,.008) for i in range(count)])
        v=rotate(q[:,None,:].expand(-1,len(local),-1),local[None])+p[:,None,:]
        e=[(i,(i+1)%count) for i in range(count)]+[(count+i,count+(i+1)%count) for i in range(count)]+[(i,count+i) for i in range(count)]
        return v[0].tolist(),e
    def detail(state,margin,disks=False):
        p,q,s=state;vertices=shape.vertices(p,q,s)[0].tolist();out=[]
        for i,v in enumerate(vertices):
            local_edges=edges
            if disks and i>=3:
                x=(-.0813 if i<5 else .0813);y=(-.0813 if i in (3,5) else .0813)
                v,local_edges=disk_vertices(p,q,x,y)
            points=clipped_points(v,local_edges,center,half+margin)
            if not points:continue
            low,high=radial_range(points)
            upper=high+margin;lower=max(0.,low-margin)
            part=dict(part=PART_NAMES[i],minimum_radial_m=low,maximum_radial_m=high,
                      signed_inner_clearance_m=radius-upper,
                      annulus_hit=upper>=radius and lower<=outer,
                      slab_half_with_margin_m=half+margin)
            out.append(part)
        return out

    results=[]
    for row in trace['contacts']:
        if args.case_id is not None and row['case_id']!=args.case_id:continue
        center=row.get('ring_center_local_m',course['center_local_m'])
        a,b=tensor(row['previous']),tensor(row['current']);mid=interpolate(a,b,.5)
        bounds=inflation(a,b)
        broad=detail(mid,bounds['total_m'])
        endpoints=[detail(s,delta) for s in (a,b)]
        refined=[];disks=[]
        for i in range(16):
            left,right=interpolate(a,b,i/16),interpolate(a,b,(i+1)/16)
            submid=interpolate(left,right,.5);margin=inflation(left,right)['total_m']
            refined.extend(d for d in detail(submid,margin) if d['annulus_hit'])
            disks.extend(d for d in detail(submid,margin,disks=True) if d['annulus_hit'])
        quat=b[1][0]
        tilt=math.degrees(math.acos(max(-1.,min(1.,1-2*float(quat[1:3].square().sum())))))
        result=dict(case_id=row['case_id'],control_step=row['control_step'],position_local_m=row['current']['position_local_m'],
                    tilt_deg=tilt,root_velocity_world_m_s=row['root_link_velocity_world_m_s'],
                    attempt_index=row['attempt_index'],flight_phase=row['flight_phase'],full_crossed=row['full_crossed'],
                    ground_target_local_m=row['ground_target_local_m'],sweep_inflation=bounds,
                    exact_projection_at_original_sweep=broad,
                    endpoint_contact_parts=sorted({d['part'] for end in endpoints for d in end if d['annulus_hit']}),
                    refined16_box_contact_parts=sorted({d['part'] for d in refined}),
                    refined16_disk_contact_parts=sorted({d['part'] for d in disks}),
                    clearance_by_part_at_endpoint=endpoints[-1],
                    diagnosis=('loose_radial_lower_bound' if not any(d['annulus_hit'] for d in broad)
                               else 'coarse_sweep_only' if not refined else 'square_rotor_only' if not disks
                               else 'contact_persists_in_refined_envelope'))
        corrected=RingEventsV2(1,'cpu',dict(course,center_local_m=center))
        corrected.reset(torch.tensor([0]),*a,torch.zeros(1,3))
        corrected.sample(*b,torch.ones(1,dtype=torch.bool))
        result['v2_contact']=bool(corrected.collision[0])
        results.append(result)
        print(json.dumps({k:result[k] for k in ('case_id','position_local_m','tilt_deg','sweep_inflation','endpoint_contact_parts','refined16_box_contact_parts','refined16_disk_contact_parts','diagnosis')},ensure_ascii=False))
    output=args.trace.parent/'contact_analysis.json'
    diagnoses={name:sum(r['diagnosis']==name for r in results) for name in sorted({r['diagnosis'] for r in results})}
    write(output,dict(scope='captured5ms segments only; same frozen course; no policy rollout or success-rate claim',
                      refinement='16 swept subintervals; exact convex radial minimum;32-sided circumscribed100mm rotor disks',
                      diagnosis_counts=diagnoses,v1_contacts=len(results),v2_contacts=sum(r['v2_contact'] for r in results),
                      contacts=results))
    print('CONTACT_ANALYSIS='+str(output))


if __name__=='__main__':main()
