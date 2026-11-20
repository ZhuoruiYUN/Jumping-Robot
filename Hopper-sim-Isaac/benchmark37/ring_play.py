"""Isaac GUI playback of a ring checkpoint; no training or scored evaluation."""
import argparse
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT=ROOT/'outputs/benchmark37_ring_r1_v2/train_seed42_20261105_063018Z/model_49.pt'
DEFAULT_CASES=ROOT/'outputs/benchmark37_ring_r1/eval_seed7001_20261105_024407_947007Z/cases.json'


def straight_x_points(count,spacing):
    """Preview-only, world+X route; no change to trained workspace/config."""
    if count<4 or not math.isfinite(spacing) or spacing<=0:
        raise ValueError('Straight preview requires positive finite spacing and at least4 points')
    return [[i*spacing,0.] for i in range(count)]


def straight_episode_stats(events,requested):
    moving=[e for e in events if 1<=e['target_index']<=requested]
    first={}
    for event in moving:first.setdefault(event['target_index'],event)
    prefix=0
    for i in range(1,requested+1):
        if i not in first or not first[i]['strict']:break
        prefix+=1
    errors=[e['foot_error_m'] for e in moving]
    return dict(completed_moving_targets=sum(e['strict'] for e in moving),moving_attempts=len(moving),
        navigation_targets_reached=len({e['target_index'] for e in moving if e.get('navigation_advanced',e['strict'])}),
        navigation_advances_without_strict=sum(e.get('navigation_advanced',e['strict']) and not e['strict'] for e in moving),
        retries=sum(e['attempt_for_target']>1 for e in moving),
        first_attempt_strict=sum(e['strict'] for e in first.values()),
        initial_consecutive_moving_hits=prefix,consecutive_first_attempt_all=prefix==requested,
        foot_error_mean_m=sum(errors)/len(errors) if errors else None,
        foot_error_max_m=max(errors) if errors else None)


class RingPreviewVisuals:
    def __init__(self,env,show_rotors=True):
        from pxr import Gf,Sdf,UsdGeom,UsdShade
        import omni.usd
        self.env=env;self.Gf=Gf;self.UsdGeom=UsdGeom
        stage=omni.usd.get_context().get_stage();self.stage=stage
        # Render only env0 while retaining the multi-env actuator/DR code path.
        for i in range(1,env.num_envs):
            prim=stage.GetPrimAtPath(f'/World/envs/env_{i}')
            if prim.IsValid():UsdGeom.Imageable(prim).MakeInvisible()
        base='/Visuals/RingPlayback'
        course=env.course;r=course['inner_radius_m'];outer=r+course['tube_radial_width_m']
        half=course['axial_thickness_m']/2;segments=128
        points=[Gf.Vec3f(x,rad*math.cos(2*math.pi*i/segments),rad*math.sin(2*math.pi*i/segments))
                for x in (-half,half) for rad in (r,outer) for i in range(segments)]
        faces=[]
        for i in range(segments):
            j=(i+1)%segments
            for a,b in ((0,1),(2,3),(0,2),(1,3)):
                faces.extend((a*segments+i,a*segments+j,b*segments+j,b*segments+i))
        self.ring=UsdGeom.Mesh.Define(stage,base+'/Ring')
        self.ring.CreatePointsAttr(points);self.ring.CreateFaceVertexCountsAttr([4]*(4*segments))
        self.ring.CreateFaceVertexIndicesAttr(faces);self.ring.CreateSubdivisionSchemeAttr('none')
        self.ring.CreateDoubleSidedAttr(True)
        if course.get('task') in ('obs73_stationary_foundation_v1','obs73_omni_ground_targets_v1'):
            UsdGeom.Imageable(self.ring.GetPrim()).MakeInvisible()
        self.ring_transform=UsdGeom.Xformable(self.ring).AddTranslateOp()
        self.last_color=None
        self.targets=[]
        for name,color,size in (('P_t',(1.,.35,0.),.035),('P_t_plus_1',(.6,.1,.9),.03)):
            sphere=UsdGeom.Sphere.Define(stage,base+'/'+name);sphere.CreateRadiusAttr(size)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            self.targets.append(UsdGeom.Xformable(sphere).AddTranslateOp())
        if getattr(env,'preview_straight_spacing',None) is not None:
            origin=env._terrain.env_origins[0].detach().cpu().tolist()
            spacing=env.preview_straight_spacing
            for i in range(env.preview_straight_jumps+1):
                point=UsdGeom.Sphere.Define(stage,base+f'/StraightPoint_{i}')
                point.CreateRadiusAttr(.018);point.CreateDisplayColorAttr([Gf.Vec3f(.45,.55,.6)])
                UsdGeom.Xformable(point).AddTranslateOp().Set(Gf.Vec3d(origin[0]+i*spacing,origin[1],.018))
            line=UsdGeom.BasisCurves.Define(stage,base+'/StraightPath')
            line.CreateTypeAttr('linear');line.CreateCurveVertexCountsAttr([2])
            line.CreatePointsAttr([Gf.Vec3f(origin[0],origin[1],.006),
                Gf.Vec3f(origin[0]+env.preview_straight_jumps*spacing,origin[1],.006)])
            line.CreateWidthsAttr([.008]);line.CreateDisplayColorAttr([Gf.Vec3f(.45,.55,.6)])
        self.rotors=[]
        if show_rotors:
            material=UsdShade.Material.Define(stage,base+'/RotorEnvelopeMaterial')
            shader=UsdShade.Shader.Define(stage,base+'/RotorEnvelopeMaterial/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.05,.7,.85))
            shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.22)
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.6)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            for i in range(4):
                cylinder=UsdGeom.Cylinder.Define(stage,base+f'/RotorEnvelope_{i}')
                cylinder.CreateRadiusAttr(course['envelope']['rotor_radius_m'])
                cylinder.CreateHeightAttr(2*course['envelope']['rotor_half_height_m']);cylinder.CreateAxisAttr('Z')
                UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(material)
                transform=UsdGeom.Xformable(cylinder)
                self.rotors.append((transform.AddTranslateOp(),transform.AddOrientOp()))
        # No CollisionAPI or rigid-body API is attached to any preview marker.
        self.update()

    def update(self):
        import torch
        from .ring_geometry import rotate
        env=self.env;Gf=self.Gf
        center=env.ring.center[0].detach().cpu().tolist()
        self.ring_transform.Set(Gf.Vec3d(*center))
        color=(1.,.08,.04) if bool(env.ring.collision[0]) else (.1,.85,.25) if bool(env.ring.passed[0]) else (1.,.7,.1)
        if color!=self.last_color:
            self.ring.CreateDisplayColorAttr([Gf.Vec3f(*color)]);self.last_color=color
        for op,point in zip(self.targets,(env._two_ground_pt_xy_w[0],env._two_ground_pt1_xy_w[0])):
            xy=point.detach().cpu().tolist();op.Set(Gf.Vec3d(xy[0],xy[1],.02))
        if self.rotors:
            d=env._robot.data;q=d.root_quat_w[0]
            offsets=d.root_link_pos_w.new_tensor([[-.0813,-.0813,0.],[-.0813,.0813,0.],[.0813,-.0813,0.],[.0813,.0813,0.]])
            positions=(rotate(q[None,:].expand(4,-1),offsets)+d.root_link_pos_w[0]).detach().cpu().tolist()
            a=q.detach().cpu().tolist();orientation=Gf.Quatf(a[0],Gf.Vec3f(*a[1:]))
            for (translate,orient),position in zip(self.rotors,positions):
                translate.Set(Gf.Vec3d(*position));orient.Set(orientation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--ring-config',type=Path,help='Optional explicit task configuration for a bounded inference startup check')
    parser.add_argument('--cases-file',type=Path,default=DEFAULT_CASES)
    parser.add_argument('--case-id',type=int,default=0)
    parser.add_argument('--seed',type=int,default=7001)
    parser.add_argument('--training-difficulty',action='store_true',help='Vertical playback: use the saved training level and its release distribution instead of the full1m evaluation setup')
    parser.add_argument('--stationary-retention',action='store_true',help='Show the retained stationary behavior for a rehearsal checkpoint')
    parser.add_argument('--slow-motion',type=float,default=2.,help='Wall seconds per simulation second;0 disables pacing')
    parser.add_argument('--pause-on-done',type=float,default=.75,help='Hold the terminal pose before resetting, GUI only')
    parser.add_argument('--steps',type=int,default=0,help='Control steps;0 plays until the window closes')
    parser.add_argument('--hide-rotor-envelope',action='store_true')
    parser.add_argument('--trace-output',type=Path,help='Optional env0 ring-loop or vertical pitch/motor CSV; no control changes')
    parser.add_argument('--straight-x-step-m',type=float,
        help='Preview an omni policy with fixed world+X target spacing, e.g.0.4m. First hop stays stationary; no training/config changes.')
    parser.add_argument('--straight-jumps',type=int,default=10,
        help='With --straight-x-step-m, stop after this many moving targets or the first episode end; retries are printed separately.')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.slow_motion<0 or args.pause_on_done<0 or args.steps<0:parser.error('Durations and steps must be nonnegative')
    checkpoint=args.checkpoint.resolve(strict=True)
    selected=None
    if not args.training_difficulty:
        case_data=json.loads(args.cases_file.read_text())['cases']
        cases={c['case_id']:c for c in case_data}
        if args.case_id not in cases:parser.error('--case-id is not in the saved cases')
        selected=cases[args.case_id]
    course=json.loads((args.ring_config or checkpoint.parent/'ring_config.json').read_text())
    straight=args.straight_x_step_m is not None
    if straight:
        if (course.get('task')!='obs73_omni_ground_targets_v1'
                or args.stationary_retention or args.training_difficulty):
            parser.error('Straight preview requires an omni checkpoint and a saved release case')
        low,high=course['omni']['moving_distance_range_m']
        if (not math.isfinite(args.straight_x_step_m) or not low<=args.straight_x_step_m<=high
                or not 1<=args.straight_jumps<=course['omni']['route_points']-3):
            parser.error('Straight spacing must fit the trained edge range, with jumps1..route_points-3')
    if args.training_difficulty and not course.get('vertical_pretraining'):
        parser.error('--training-difficulty requires a vertical checkpoint')
    level=None
    if args.training_difficulty:
        level=json.loads((checkpoint.parent/'manifest.json').read_text())['vertical_curriculum']['level']
    from .ring_obs73 import observation_dim,validate_checkpoint_course
    validate_checkpoint_course(checkpoint,course)
    obs_dim=observation_dim(course)
    if args.trace_output:
        if course.get('task')!='drop_stationary_ring_stationary2_v1' and not course.get('vertical_pretraining'):
            parser.error('--trace-output requires the minimal-loop or vertical task')
        if args.trace_output.exists():parser.error('--trace-output already exists; choose a new path')
    trace_rows=[]
    app=AppLauncher(args).app;raw=None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure
        from .ring_env import RingEnv
        from .two_ground_targets import _set_active_goal
        min_loop=course.get('task')=='drop_stationary_ring_stationary2_v1'
        stage_b=course.get('task')=='obs73_fixed_wide_ring_stage_b_v1'
        stationary=course.get('task')=='obs73_stationary_foundation_v1'
        omni=course.get('task')=='obs73_omni_ground_targets_v1'
        if stationary:
            from .obs73_stationary import env_class
            RingEnv=env_class()
            if course.get('vertical_pretraining'):
                from .obs73_vertical import env_class as vertical_env_class
                RingEnv=vertical_env_class(evaluation=True,evaluation_level=level)
        if min_loop:
            from .ring_min_loop import env_class
            RingEnv=env_class()
            if course.get('height_randomization'):
                from .ring_height_loop import env_class as height_env_class
                RingEnv=height_env_class(evaluation=True)
            if course.get('targeted_adjustments'):
                from .ring_targeted_adjustments import env_class as targeted_env_class
                RingEnv=targeted_env_class(evaluation=True)
        if stage_b:
            from .obs73_ring_stage_b import env_class as stage_b_env_class
            RingEnv=stage_b_env_class(evaluation=True)
        if omni:
            if course.get('stationary_retention'):
                from .stationary_retention import env_class as retention_env_class
                RingEnv=retention_env_class(evaluation=True,stationary_only=args.stationary_retention)
            else:
                from .obs73_omni import env_class as omni_env_class
                RingEnv=omni_env_class(evaluation=True)
        if omni and course.get('moving_phase_reward'):
            from .moving_phase_reward import wrap_env as motion_reward_wrapper
            RingEnv=motion_reward_wrapper(RingEnv)
        if course.get('observation_version')=='direct_motor_crossing_quaternion73_history40_20_v1':
            from .crossing_obs73_quat import wrap_env
            RingEnv=wrap_env(RingEnv)
        elif obs_dim==73:
            from .ring_obs73 import wrap_env
            RingEnv=wrap_env(RingEnv)
        elif obs_dim==72:
            from .crossing_obs72 import wrap_env
            RingEnv=wrap_env(RingEnv)

        class PreviewEnv(RingEnv):
            loop_continue=min_loop
            def __init__(self,*a,**kw):
                self.preview_visuals=None;self.preview_episode=0
                self.preview_straight_spacing=args.straight_x_step_m
                self.preview_straight_jumps=args.straight_jumps
                self.preview_straight_events=[];self.preview_straight_died=False
                self.preview_straight_complete=False;self.preview_straight_timeout=False
                self.preview_straight_finished=False
                super().__init__(*a,**kw)

            def _reset_idx(self,ids):
                if ids is None:ids=self._robot._ALL_INDICES
                visible_reset=bool((ids==0).any())
                if straight and visible_reset and self.preview_episode>0:
                    d=self._robot.data
                    self.preview_straight_died=bool((d.root_link_pos_w[0,2]<.05)|(d.root_quat_w[0,1:3].square().sum()>.5))
                    self.preview_straight_timeout=bool(self.reset_time_outs[0])
                if visible_reset and self.preview_visuals is not None:
                    self.preview_visuals.update()
                    print(f'RING_PLAY episode={self.preview_episode} full_pass={bool(self.ring.passed[0])} first_joint={bool(self.ring_joint_success[0])} ring_contact={bool(self.ring.collision[0])}',flush=True)
                    if self.sim.has_gui():
                        self.sim.render()
                        if args.pause_on_done:time.sleep(args.pause_on_done)
                super()._reset_idx(ids)
                if not visible_reset or not hasattr(self,'ring'):return
                if selected is None:
                    self.preview_episode+=1
                    return  # Native training release/height/DR were set by super.
                d=self._robot.data;one=ids.new_tensor([0]);origin=self._terrain.env_origins[0]
                pose=d.root_link_pos_w.new_tensor([selected['root_link_pose_world']])
                pose[0,:2]=origin[:2]
                self._robot.write_root_pose_to_sim(pose,one)
                self._robot.write_root_velocity_to_sim(torch.zeros(1,6,device=self.device),one)
                self._robot.write_joint_state_to_sim(pose.new_tensor([selected['joint_pos']]),pose.new_tensor([selected['joint_vel']]),env_ids=one)
                from .flight_plane import restore_rigid_scales
                restore_rigid_scales(self,one,[selected])
                for attr,key in (('dr_mass_multi','mass_multiplier'),('dr_inertia_multi','inertia_multiplier'),('dr_tm','motor_time_constant_s')):
                    getattr(self,attr)[0]=pose.new_tensor(selected[key])
                if min_loop or stationary or stage_b or omni:
                    self._reset_task_targets(one)
                    if straight:
                        # Only env0's command queue is replaced. The ground
                        # plane is infinite; this diagnostic may go beyond the
                        # training route's2m sampling radius. Relative commands,
                        # physics, history and the saved navigation rule remain.
                        self.omni_route.points[0]=pose.new_tensor(straight_x_points(
                            self.omni_route.points.shape[1],args.straight_x_step_m))
                        self.omni_route.index[0]=0;self._bind_targets(one)
                    elif omni and not args.stationary_retention and 'omni_route_local_m' in selected:
                        saved=pose.new_tensor(selected['omni_route_local_m'])
                        route=self.omni_route.points[0]
                        if saved.shape!=route.shape:
                            raise ValueError(f'Saved preview route shape {tuple(saved.shape)} differs from '
                                             f'checkpoint route shape {tuple(route.shape)}; use a matching case '
                                             'or specify --straight-x-step-m for a straight preview')
                        route.copy_(saved)
                        self.omni_route.index[0]=0
                        self._bind_targets(one)
                else:
                    old_xy=pose.new_tensor(selected['root_link_pose_world'][:2])
                    self._two_ground_pt_xy_w[0]=origin[:2]+pose.new_tensor(selected['ground_target_xy'])-old_xy
                    self._two_ground_pt1_xy_w[0]=origin[:2]+pose.new_tensor(selected['next_ground_target_xy'])-old_xy
                _set_active_goal(self,one)
                self.ring.reset(one,d.root_link_pos_w,d.root_quat_w,d.joint_pos[:,self._spring_joint_id],self._terrain.env_origins)
                if course.get('height_randomization'):self.restore_height_cases(one,[selected])
                self.preview_episode+=1

            def _get_rewards(self):
                sample=None
                vertical_strict=False
                landing=None
                if straight:
                    d=self._robot.data;f=self._foot_event
                    physical=bool((d.root_link_pos_w[0,2]<.05)|(d.root_quat_w[0,1:3].square().sum()>.5))
                    valid=bool((f.phase[0]==f.FLIGHT)&(f.air_steps[0]>=5)
                        &(d.joint_pos[0,self._spring_joint_id]>.002)
                        &(f.apex_seen[0]|(d.root_com_lin_vel_w[0,2]<=0))) and not physical
                    if valid:
                        index=int(self.omni_route.index[0]);origin=self._terrain.env_origins[0,:2]
                        target=self._two_ground_pt_xy_w[0]-origin
                        foot=d.body_link_pos_w[0,self._foot_body_id,:2]-origin
                        q=d.root_quat_w[0]
                        landing=dict(target_index=index,target_local_xy_m=target.tolist(),
                            foot_local_xy_m=foot.tolist(),foot_error_m=float((foot-target).norm()),
                            apex_world_m=float(torch.maximum(f.peak[0],d.root_link_pos_w[0,2])-self._terrain.env_origins[0,2]),
                            touchdown_tilt_deg=float(torch.rad2deg(torch.acos((1-2*q[1:3].square().sum()).clamp(-1,1)))),
                            step=int(self.common_step_counter))
                if args.trace_output:
                    from .ring_pitch_diagnostic import snapshot
                    sample=snapshot(self)
                    if not hasattr(self,'loop'):
                        d=self._robot.data;f=self._foot_event;q=d.root_quat_w[0]
                        tilt=torch.rad2deg(torch.acos((1-2*q[1:3].square().sum()).clamp(-1,1)))
                        vertical_strict=bool((f.phase[0]==f.FLIGHT)&(f.air_steps[0]>=5)
                            &(d.joint_pos[0,self._spring_joint_id]>.002)
                            &(f.apex_seen[0]|(d.root_com_lin_vel_w[0,2]<=0))
                            &(torch.maximum(f.peak[0],d.root_link_pos_w[0,2])>=1.)
                            &(torch.linalg.vector_norm(d.body_link_pos_w[0,self._foot_body_id,:2]-self._two_ground_pt_xy_w[0])<=self.landing_radius_m)
                            &(tilt<=15)&~self.reset_terminated[0])
                reward=super()._get_rewards()
                if landing is not None:
                    index=landing['target_index']
                    # Navigation advancement alone is not strict jump success.
                    landing['strict']=bool(self._omni_last_strict[0])
                    landing['navigation_advanced']=int(self.omni_route.index[0])==index+1
                    landing['attempt_for_target']=1+sum(e['target_index']==index for e in self.preview_straight_events)
                    self.preview_straight_events.append(landing)
                    self.preview_straight_finished=(index==args.straight_jumps and landing['navigation_advanced'])
                    self.preview_straight_complete=(self.preview_straight_finished and
                        straight_episode_stats(self.preview_straight_events,args.straight_jumps)['completed_moving_targets']==args.straight_jumps)
                    print('STRAIGHT_X_LANDING='+json.dumps(landing),flush=True)
                if sample is not None:
                    sample.extend((int(self.loop.stage[0]) if hasattr(self,'loop') else -1,
                                   int(self._foot_event.last_qualified[0]) if hasattr(self,'loop') else int(vertical_strict),
                                   int(self.reset_buf[0]),int(self.ring.passed[0])))
                    trace_rows.append(sample)
                return reward

        torch.manual_seed(args.seed)
        cfg=QuadhopperEnvCfg();configure(cfg)
        cfg.seed=args.seed;cfg.scene.num_envs=16;cfg.sim.device=args.device;cfg.observation_space=obs_dim
        cfg.debug_vis=False
        cfg.viewer.origin_type='env';cfg.viewer.env_index=0
        cfg.viewer.eye=(1.8,-2.,1.45);cfg.viewer.lookat=(.35,0.,.65)
        if straight:
            span=args.straight_jumps*args.straight_x_step_m
            cfg.viewer.eye=(span/2+1.8,-max(3.,span*.9),max(2.,span*.7))
            cfg.viewer.lookat=(span/2,0.,.55)
        cfg.power_model_path=str(ROOT/'Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt')
        raw=PreviewEnv(cfg,course);wrapped=RslRlVecEnvWrapper(raw)
        visuals=RingPreviewVisuals(raw,show_rotors=not args.hide_rotor_envelope);raw.preview_visuals=visuals
        rc=json.loads((checkpoint.parent/'runner_config.json').read_text());rc['seed']=args.seed
        runner=OnPolicyRunner(wrapped,rc,log_dir=None,device=args.device)
        runner.alg.policy.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=False)['model_state_dict'],strict=True)
        def forbidden(*a,**kw):raise AssertionError('Optimizer updates forbidden during playback')
        runner.alg.optimizer.step=forbidden
        policy=runner.get_inference_policy(device=args.device);obs=wrapped.get_observations()
        print(f'RING_PLAY checkpoint={checkpoint} case={args.case_id} geometry={course["envelope"]["version"]}',flush=True)
        if course.get('vertical_pretraining'):
            print('VERTICAL_PLAY '+json.dumps(dict(target_apex_m=raw.vertical_curriculum.current['target_apex_m'],
                release=raw.vertical_curriculum.current['release_height_m'],
                frozen_level=raw.vertical_curriculum.level,training_difficulty=args.training_difficulty)),flush=True)
        print('RING_PLAY colors: amber=pending green=full-pass red=contact; orange=P_t purple=P_t+1 cyan=engineering rotor envelopes. Playback is not a scored evaluation.',flush=True)
        if straight:
            print('STRAIGHT_X_PLAY '+json.dumps(dict(spacing_m=args.straight_x_step_m,
                moving_targets=args.straight_jumps,direction='world+X',initial_stationary_hop=True,
                targets_continue_beyond_preview=True,release_case=args.case_id,
                response_group=selected.get('response_group'),strict_foot_m=raw.landing_radius_m,strict_apex_m=1.,strict_tilt_deg=15.,
                scope='single16-env-context preview; not the formal128-case gate or hardware validation')),
                flush=True)
        step=0
        with torch.inference_mode():
            while app.is_running() and (args.steps==0 or step<args.steps):
                started=time.perf_counter()
                obs,_,done,_=wrapped.step(policy(obs));runner.alg.policy.reset(done)
                visuals.update();step+=1
                if straight and (raw.preview_straight_finished or bool(done[0])):
                    if raw.sim.has_gui() and args.pause_on_done:
                        raw.sim.render();time.sleep(args.pause_on_done)
                    break
                remaining=raw.step_dt*args.slow_motion-(time.perf_counter()-started)
                if remaining>0:time.sleep(remaining)
        print(f'RING_PLAY stopped steps={step}',flush=True)
        if straight:
            reason=('target_count' if raw.preview_straight_complete else 'physical_death' if raw.preview_straight_died
                else 'target_count_with_failed_jumps' if raw.preview_straight_finished
                else 'time_out' if raw.preview_straight_timeout else 'step_limit' if args.steps and step>=args.steps
                else 'window_closed')
            print('STRAIGHT_X_RESULT='+json.dumps(dict(checkpoint=str(checkpoint),spacing_m=args.straight_x_step_m,
                requested_moving_targets=args.straight_jumps,
                course_version=course.get('version'),
                source_policy_trained_for_course=course.get('version')==json.loads((checkpoint.parent/'ring_config.json').read_text()).get('version'),
                **straight_episode_stats(raw.preview_straight_events,args.straight_jumps),
                complete=raw.preview_straight_complete,physical_death=raw.preview_straight_died,
                timed_out=raw.preview_straight_timeout,steps=step,stop_reason=reason,optimizer_updates=0,
                scope='one unscored playback episode, including misses/retries; not a deployment acceptance')),
                flush=True)
        assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
        if course.get('post_ring_stability'):
            print('RING_POST_STABILITY env0_episode_costs='+json.dumps({
                name:float(raw._episode_sums[name][0]) for name in ('post_ring_pitch','post_ring_pitch_rate')}),flush=True)
        if course.get('height_randomization'):
            print('RING_HEIGHT_PLAY '+json.dumps(dict(env0_height_m=float(raw._ring_height[0]),
                sampled_min_m=float(raw._ring_height.min()),sampled_max_m=float(raw._ring_height.max()))),flush=True)
        if course.get('targeted_adjustments'):
            print('RING_TARGETED_CHECK '+json.dumps(dict(cost_totals={k:float(v) for k,v in raw.targeted_totals.items()},
                active_samples={k:float(v) for k,v in raw.targeted_active_samples.items()},
                flight_metrics={k:float(v) for k,v in raw.flight_pitch_stats.log().items()})),flush=True)
    except KeyboardInterrupt:
        pass
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:
        if args.trace_output and trace_rows:
            import csv
            from .ring_pitch_diagnostic import TRACE_COLUMNS
            args.trace_output.parent.mkdir(parents=True,exist_ok=True)
            with args.trace_output.open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(TRACE_COLUMNS);writer.writerows(trace_rows)
            print(f'RING_PITCH_TRACE={args.trace_output}',flush=True)
        if raw is not None:raw.close()
        app.close()


if __name__=='__main__':main()
