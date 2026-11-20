"""Bounded16x256 inference wiring check. No learning or checkpoint saving."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COURSE=ROOT/'benchmark37/configs/obs73_omni_course_v1.json'


def offline_check(course):
    import torch
    from .obs73_omni import GroundRoute,prepare_source
    source=ROOT/course['omni']['initial_checkpoint']
    previous=json.loads((source.parent/'ring_config.json').read_text())
    assert course['vertical_pretraining']==previous['vertical_pretraining']
    assert course['stationary_pretraining']==previous['stationary_pretraining']
    state=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    original={key:value.clone() for key,value in state.items()}
    changed=prepare_source(state,previous,course)
    for key,value in state.items():
        if key not in changed:assert torch.equal(value,original[key]),key
        else:
            assert not value[:,2:6].any()
            assert torch.equal(value[:,6:],original[key][:,6:])
            # Equal target vectors preserve the old recurrent preactivation.
            x=torch.linspace(-1,1,73).repeat(4,1)
            x[:,2:4]=x[:,0:2];x[:,4:6]=x[:,0:2]
            torch.testing.assert_close(x@value.T,x@original[key].T,rtol=1e-5,atol=2e-6)
    route=GroundRoute(128,'cpu',course['omni'],7001,evaluation=True)
    ids=torch.arange(128);route.reset(ids)
    lengths=route.points.diff(dim=1).norm(dim=-1)
    assert route.points.norm(dim=-1).max()<=.60001
    assert ((lengths<1e-6)|((lengths>=.12-1e-6)&(lengths<=.2+1e-6))).all()
    edges=route.points[:,1];directions=((torch.atan2(edges[:,1],edges[:,0])/(torch.pi/4)+.5).floor().long()%8)
    assert torch.equal(torch.bincount(directions,minlength=8),torch.full((8,),16))
    valid=torch.zeros(128,dtype=torch.bool);strict=valid.clone()
    valid[:2]=True;strict[0]=True
    before=route.points.clone();route.advance(valid,strict)
    assert route.index[0]==1 and not route.index[1:].any() and route.misses==1
    assert torch.equal(before,route.points)
    route.reset(torch.tensor([1]));assert route.index[0]==1
    assert torch.equal(before[0],route.points[0])
    rc=json.loads((ROOT/course['runner_config_source']).read_text())
    saved_rc=json.loads((source.parent/'runner_config.json').read_text())
    for key in ('algorithm','policy','num_steps_per_env'):
        assert rc[key]==saved_rc[key],key
    return dict(source=str(source),only_changed_tensors=changed,ring8_and_std_preserved=True,
        duplicate_target_function_preserved=True,route_bounds_and_strict_retry_checked=True,
        evaluation_first_edge_direction_counts=[16]*8,rewards_dr_and_ppo_preserved=True)


def main():
    import sys
    obs72='--obs72' in sys.argv
    if obs72:sys.argv.remove('--obs72')
    course_path=ROOT/'benchmark37/configs/obs72_omni_course_v4.json' if obs72 else COURSE
    while '--ring-config' in sys.argv:
        if not obs72:raise ValueError('Explicit startup course is supported for OBS72 only')
        index=sys.argv.index('--ring-config')
        course_path=Path(sys.argv[index+1]).resolve(strict=True)
        del sys.argv[index:index+2]
    course=json.loads(course_path.read_text())
    if '--offline' in sys.argv and course['omni'].get('training_adjacent_hops'):
        from .checks.adjacent_hops_check import check
        print('ADJACENT_HOPS_CHECK='+json.dumps(check(course)),flush=True)
        return
    if '--offline' in sys.argv and course.get('guarded_keys'):
        from .checks.moving_phase_reward_check import guarded_keys_check
        print('GUARDED_KEYS_OFFLINE='+str(guarded_keys_check(course)),flush=True);return
    if '--offline' in sys.argv and course['omni'].get('training_reversals'):
        from .checks.moving_phase_reward_check import omni_reversal_check
        print('OBS73_OMNI_REVERSAL_CHECK='+json.dumps(omni_reversal_check(course_path)),flush=True)
        return
    if '--offline' in sys.argv and course.get('omni_rebound_handoff'):
        from .checks.moving_phase_reward_check import omni_rebound_handoff_check
        print('OBS73_OMNI_HANDOFF_CHECK='+json.dumps(omni_rebound_handoff_check(course_path)),flush=True)
        return
    if '--offline' in sys.argv and course['omni'].get('training_long_steps'):
        from .checks.moving_phase_reward_check import omni_focus_check
        print('OBS73_OMNI_FOCUS_CHECK='+json.dumps(omni_focus_check(course_path)),flush=True)
        return
    if '--offline' in sys.argv and course.get('moving_outward_tilt_work'):
        from .checks.moving_phase_reward_check import outward_tilt_work_check
        print('OBS73_OUTWARD_TILT_WORK_CHECK='+json.dumps(outward_tilt_work_check(course_path)),flush=True)
        return
    if '--offline' in sys.argv and course.get('launch_velocity_window'):
        from .checks.moving_phase_reward_check import continuous_velocity_check
        print('OBS73_CONTINUOUS_VELOCITY_CHECK='+json.dumps(continuous_velocity_check()),flush=True)
        return
    if '--offline' in sys.argv and course.get('version')=='obs73_quaternion_cardinal_queue_repair_v32':
        from .checks.moving_phase_reward_check import queued_repair_check
        print('OBS73_QUEUED_REPAIR_CHECK='+json.dumps(queued_repair_check()),flush=True)
        return
    from .ring_obs73 import observation_dim
    dim=observation_dim(course) if obs72 else 73
    if '--offline' in sys.argv:
        if obs72:
            from .crossing_obs72 import HISTORY_VERSION
            if course.get('moving_phase_reward'):
                from .checks.moving_phase_reward_check import run
            elif course.get('version')=='obs73_quaternion_omni_range05_v8_from_verified_v2':
                from .checks.obs73_range05_check import run
            elif course.get('stationary_retention'):
                from .checks.stationary_retention_check import run
            elif course['observation_version']=='direct_motor_crossing_quaternion73_history40_20_v1':
                from .checks.obs73_quaternion_check import run
            elif course['observation_version']==HISTORY_VERSION:
                from .checks.obs72_history_check import run
            elif course['omni'].get('distance_curriculum'):
                from .checks.obs72_range08_check import run
            else:
                from .checks.crossing_obs72_check import run
            result=run(course_path) if course.get('moving_phase_reward') else run()
            print('OBS72_OFFLINE_RESULT='+json.dumps(result),flush=True);return
        print('OMNI_OFFLINE_RESULT='+json.dumps(offline_check(course)),flush=True);return
    parser=argparse.ArgumentParser(description=__doc__)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if args.headless and '/exts/omni.kit.telemetry/skipDeferredStartup' not in args.kit_args:
        args.kit_args=(args.kit_args+' --/exts/omni.kit.telemetry/skipDeferredStartup=true').strip()
    source=ROOT/course['omni']['initial_checkpoint']
    folder=ROOT/'outputs'/course.get('output_series','benchmark73_omni_v1')/('startup_seed42_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ'))
    folder.mkdir(parents=True,exist_ok=False)
    steps=1280 if course.get('guarded_keys') else 256
    manifest=dict(status='starting',mode='zero_update_startup',seed=42,num_envs=16,steps=steps,
        optimizer_updates=0,checkpoint_saved=False,source=str(source),
        scope='source-policy inference checks task wiring; not an omni promotion evaluation')
    def save():(folder/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    app=None;raw=None;save()
    try:
        app=AppLauncher(args).app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure
        from .obs73_omni import env_class
        if course.get('stationary_retention'):
            from .stationary_retention import env_class
        if course.get('moving_phase_reward'):
            from .moving_phase_reward import wrap_env as motion_reward_wrapper
            parent_env_class=env_class
            env_class=lambda:motion_reward_wrapper(parent_env_class())
        from .obs73_omni_evaluation import eval_env_class
        from .ring_obs73 import wrap_env
        from .ring_transfer import initialize_same_contract
        if obs72:
            from .crossing_obs72 import wrap_env,contract,VERSION,HISTORY_VERSION
            source_course=json.loads((source.parent/'ring_config.json').read_text())
            if course['observation_version']=='direct_motor_crossing_quaternion73_history40_20_v1':
                from .crossing_obs73_quat import wrap_env,contract
                if source_course['observation_version']!=course['observation_version']:
                    from .crossing_transfer import initialize_quaternion as initialize_same_contract
                manifest['observation_contract']=contract()
            elif (source_course['observation_version']==VERSION and course['observation_version']==HISTORY_VERSION):
                from .crossing_transfer import initialize_history as initialize_same_contract
            elif source_course['observation_version']!=course['observation_version']:
                from .crossing_transfer import initialize as initialize_same_contract
            if 'observation_contract' not in manifest:
                manifest['observation_contract']=contract(course['observation_version'])
        from .artifacts import clean
        torch.manual_seed(42)
        cfg=QuadhopperEnvCfg();configure(cfg)
        cfg.seed=42;cfg.scene.num_envs=16;cfg.sim.device=args.device
        cfg.observation_space=dim;cfg.episode_length_s=15.
        cfg.power_model_path=str(ROOT/'Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt')
        cfg.csv_log_path=str(folder/'single_env.csv')
        expected=json.loads((source.parent/'env_config.json').read_text())
        # A long-horizon source already stores60s here. Compare its full
        # calibrated config before applying the explicitly declared task horizon.
        cfg.episode_length_s=expected['episode_length_s']
        actual=clean(cfg.to_dict())
        for key in ('seed','observation_space','csv_log_path'):expected[key]=actual[key]
        expected['scene']['num_envs']=16
        assert actual==expected,'Keep the source calibrated environment configuration'
        if course.get('long_moving_horizon'):
            cfg.episode_length_s=course['long_moving_horizon']['moving_s']
            manifest['task_moving_horizon_s']=cfg.episode_length_s

        class CheckedEnv(eval_env_class(wrap_env(env_class()))):
            goal_switches=0
            def _get_rewards(self):
                index=self.omni_route.index.clone();pt=self._two_ground_pt_xy_w.clone()
                nxt=self._two_ground_pt1_xy_w.clone();before=self._foot_event.qualified.clone()
                phase_before=self._foot_event.phase.clone()
                guard=getattr(self,'key_guard',None)
                if guard is not None:
                    focus=guard.mask & ((guard.state!=2)|self.rebound_plan.handoff_turn)
                if hasattr(self,'last_attempted_target'):
                    first=index!=self.last_attempted_target
                reward=super()._get_rewards()
                if self.course.get('moving_phase_reward'):
                    audit=self._moving_phase_last_audit
                    if hasattr(self,'last_attempted_target'):
                        eligible=audit['moving'] & ~audit['route_hold'] & first & self._omni_last_strict
                    if getattr(self,'center_arrival_bonus',0.):
                        distance=(self._robot.data.body_link_pos_w[:,self._foot_body_id,:2]-pt).norm(dim=-1)
                        expected=self.center_arrival_bonus*(1.-distance/self.landing_radius_m).clamp(0.,1.)
                        torch.testing.assert_close(audit['deltas']['moving_center_arrival'],torch.where(eligible,expected,0.))
                    if getattr(self,'first_arrival_bonus',0.):
                        torch.testing.assert_close(audit['deltas']['moving_first_arrival'],
                                                   self.first_arrival_bonus*eligible.float())
                    for name,value in audit['deltas'].items():
                        if self.plane_cost and name in self.plane_cost:
                            assert torch.isfinite(value).all() and (value<=0).all()
                            continue
                        assert not value[~audit['moving']].any(),'Nonmoving reward changed'
                    self.motion_samples=getattr(self,'motion_samples',0)+int(audit['moving'].sum())
                    self.motion_adjustment_max=max(getattr(self,'motion_adjustment_max',0.),float(audit['adjustment'].abs().max()))
                    if self.course.get('moving_rotation_work_extension'):
                        a=audit['rotation_work_extension'];extra=a['extra_weight'];cost=a['additional_cost']
                        assert torch.isfinite(extra).all() and torch.isfinite(cost).all() and (cost<=0).all()
                        assert ((extra>=0)&(extra<=1-audit['weight']+1e-6)).all()
                        assert not extra[~a['eligible']].any() and not extra[a['gate']==0].any()
                        torch.testing.assert_close(cost,extra*a['base_cost'])
                        mask=audit['moving']
                        torch.testing.assert_close((audit['inherited']['vertical_rotation_work']+
                            audit['deltas']['vertical_rotation_work'])[mask],
                            ((audit['weight']+extra)*a['base_cost'])[mask],atol=1e-5,rtol=1e-5)
                        if not self.course['moving_rotation_work_extension']['enabled']:
                            assert not extra.any() and not cost.any()
                        self.work_extension_eligible=getattr(self,'work_extension_eligible',0)+int(a['eligible'].sum())
                        self.work_extension_negative=getattr(self,'work_extension_negative',0)+int((cost<0).sum())
                    if self.course.get('moving_outward_tilt_work'):
                        cost = audit['deltas']['moving_outward_tilt_work']
                        active = audit['outward_tilt_work_active']
                        assert torch.isfinite(cost).all() and (cost <= 0).all()
                        assert not cost[~active].any()
                        self.outward_work_eligible = getattr(self,'outward_work_eligible',0) + int(active.sum())
                        self.outward_work_negative = getattr(self,'outward_work_negative',0) + int((cost < 0).sum())
                    if self.course.get('moving_body_rate'):
                        r=audit['body_rate'];value=audit['deltas']['moving_transport_body_rate']
                        assert torch.isfinite(value).all() and (value<=0).all()
                        assert not value[~r['active']].any()
                        assert not value[audit['weight']==1].any()
                        assert not value[r['excess_rate']==0].any()
                    if self.course.get('moving_rotation_prediction'):
                        r=audit['rotation_prediction'];value=audit['deltas']['moving_predicted_rotation']
                        assert all(torch.isfinite(v).all() for v in r.values())
                        assert torch.isfinite(value).all() and (value<=0).all()
                        assert not value[~r['active']].any()
                        assert (r['predicted_max_tilt']>=r['current_tilt']-1e-5).all()
                        self.rotation_prediction_samples=getattr(self,'rotation_prediction_samples',0)+int(r['active'].sum())
                        self.rotation_prediction_negative_samples=getattr(self,'rotation_prediction_negative_samples',0)+int((value<0).sum())
                    if self.course.get('rebound_transition'):
                        a=audit['rebound']
                        assert a is not None
                        if 'moving_release_carry' in audit['deltas']:
                            value=audit['deltas']['moving_release_carry']
                            assert torch.isfinite(value).all() and (value<=0).all()
                            assert not value[~a['launch']].any()
                            self.carry_cost_events=getattr(self,'carry_cost_events',0)+int((value<0).sum())
                        if 'lead_changed' in a:
                            active=a['lead_changed']
                            assert not active[~audit['moving']|~a['preparation_active']].any()
                            assert torch.isfinite(a['lead_delta']).all() and not a['lead_delta'][~active].any()
                            self.lead_adjustments=getattr(self,'lead_adjustments',0)+int((a['lead_delta']>1e-6).sum())
                            if 'continuation_velocity_shift_m_s' in a:
                                shift=a['continuation_velocity_shift_m_s']
                                assert torch.isfinite(shift).all() and not shift[~active].any()
                                self.foot_offset_adjustments=getattr(self,'foot_offset_adjustments',0)+int((shift>1e-5).sum())
                        for key in ('preparation_weight','normal','outgoing_velocity','time_floor'):
                            assert torch.isfinite(a[key]).all()
                        assert ((a['preparation_weight']>=0)&(a['preparation_weight']<=1)).all()
                        for key in ('moving_rebound_approach','moving_launch_velocity','moving_cruise_acceleration'):
                            assert torch.isfinite(audit['deltas'][key]).all() and (audit['deltas'][key]<=0).all()
                        velocity_mask=a['launch_velocity_active'] if self.course.get('launch_velocity_window') else a['launch']
                        assert not audit['deltas']['moving_launch_velocity'][~velocity_mask].any()
                        if self.course.get('launch_direction_refinement'):
                            value=audit['deltas']['moving_launch_direction']
                            assert torch.isfinite(value).all() and (value<=0).all()
                            assert not value[~a['launch']].any()
                        if self.course.get('contact_attitude_continuity'):
                            rw=a['contact_reference_weight'];normal=a['contact_reference_normal']
                            assert torch.isfinite(rw).all() and ((rw==0)|(rw==1)).all()
                            assert not rw[~audit['moving']].any() and not rw[a['launch']].any()
                            torch.testing.assert_close(normal.norm(dim=-1),torch.ones_like(rw))
                            for key in ('vertical_tilt','attitude_penalty'):
                                torch.testing.assert_close((audit['inherited'][key]+audit['deltas'][key])[rw>0],
                                    torch.zeros_like(rw[rw>0]),atol=1e-5,rtol=1e-5)
                            previous=getattr(self,'_contact_continuity_previous',None)
                            if previous is not None:
                                stable=(rw>0)&(previous[0]>0)
                                torch.testing.assert_close(normal[stable],previous[1][stable])
                            self._contact_continuity_previous=(rw.clone(),normal.clone())
                            self.contact_continuity_samples=getattr(self,'contact_continuity_samples',0)+int((rw>0).sum())
                        if self.course.get('transition_repair'):
                            repair=self.course['transition_repair']
                            if repair['rebound_consistency']:
                                rw=a['contact_reference_weight']
                                assert torch.isfinite(rw).all() and ((rw>=0)&(rw<=1)).all()
                                assert not rw[~audit['moving']].any()
                                torch.testing.assert_close(a['contact_reference_normal'].norm(dim=-1),torch.ones_like(rw))
                                for key in ('vertical_tilt','attitude_penalty'):
                                    torch.testing.assert_close((audit['inherited'][key]+audit['deltas'][key])[rw>0],
                                        torch.zeros_like(rw[rw>0]),atol=1e-5,rtol=1e-5)
                                self.contact_reference_samples=getattr(self,'contact_reference_samples',0)+int((rw>0).sum())
                        if self.score_yaw_peak:
                            y=audit['yaw_prediction'];value=audit['deltas']['moving_yaw_peak']
                            assert torch.isfinite(value).all() and (value<=0).all()
                            mask=a['air'] if self.air_yaw_peak_weight else audit['moving']
                            assert not value[~mask].any()
                            strength=self.air_yaw_peak_weight or self.course['vertical_pretraining']['yaw_refinement']['heading_weight']
                            expected=-strength*(y['peak_excess_squared']-y['peak_previous_squared'])
                            torch.testing.assert_close(value,expected)
                            self.yaw_peak_guard_samples=getattr(self,'yaw_peak_guard_samples',0)+int((value<0).sum())
                        if self.course.get('launch_endpoint_refinement'):
                            scale=self.course['launch_endpoint_refinement']['width_m']
                            squared=a['launch_error_squared']*(self.rebound_plan.flight/scale).square()
                            expected=torch.where(a['launch'],-self.course['rebound_transition']['launch_weight']*
                                squared/(torch.sqrt(1+squared)+1),0.)
                            torch.testing.assert_close(audit['deltas']['moving_launch_velocity'],expected)
                        if self.course.get('launch_velocity_window'):
                            from .moving_phase_reward import launch_velocity_window_cost
                            option=self.course['launch_velocity_window'];plan=self.rebound_plan
                            expected,check=launch_velocity_window_cost(
                                self._robot.data.root_link_lin_vel_w[:,:2],a['target_launch_velocity'],plan.flight,
                                plan.launch_step,self.common_step_counter,audit['moving'],
                                phase_before,audit['clearance'],self.step_dt,
                                self.course['rebound_transition']['launch_weight'],option)
                            torch.testing.assert_close(audit['deltas']['moving_launch_velocity'],expected)
                            assert torch.equal(a['launch_velocity_active'],check['active'])
                            self.launch_velocity_window_samples=getattr(self,'launch_velocity_window_samples',0)+int(check['active'].sum())
                        if self.course.get('omni_rebound_handoff'):
                            if hasattr(self,'continuation_cosine'):
                                part=audit['deltas']['moving_continuation_cross_velocity']
                                assert torch.isfinite(part).all() and (part<=0).all()
                                assert not part[(phase_before!=1)|~audit['moving']|a['turn_unloading_active']|a['retry_unloading_active']].any()
                                self.continuation_samples=getattr(self,'continuation_samples',0)+int((part<0).sum())
                                if getattr(self,'shortfall_before_hold',False):
                                    short=audit['deltas']['moving_continuation_shortfall']
                                    assert torch.isfinite(short).all() and (short<=0).all()
                                    assert not short[(pt!=nxt).any(-1)|~audit['moving']|(phase_before!=1)].any()
                                    self.before_hold_samples=getattr(self,'before_hold_samples',0)+int((short<0).sum())
                            turn = a['turn_unloading_active']
                            retry = a['retry_unloading_active']
                            loss = audit['deltas']['moving_contact_handoff_velocity']
                            assert torch.isfinite(loss).all() and (loss <= 0).all()
                            assert not (turn & retry).any()
                            assert not ((turn | retry) & (phase_before != 1)).any()
                            assert not loss[~(turn | retry)].any()
                            self.handoff_samples = getattr(self, 'handoff_samples', 0) + int((turn | retry).sum())
                            self.handoff_preview_samples = getattr(self, 'handoff_preview_samples', 0) + int(a['handoff_preparation_active'].sum())
                        if self.course.get('launch_rotation_refinement'):
                            from .moving_phase_reward import launch_rotation_cost
                            value=audit['deltas']['moving_launch_rotation']
                            angular=self.course['launch_rotation_refinement']
                            expected=launch_rotation_cost(self._robot.data.root_ang_vel_b,a['launch_rotation_active'],angular)
                            if angular.get('window_s'):expected*=self.step_dt/angular['window_s']
                            if guard is not None:expected*=torch.where(focus,self.course['guarded_keys']['launch_multiplier'],1.)
                            torch.testing.assert_close(value,expected)
                            assert not value[~a['launch_rotation_active']].any()
                            self.launch_rotation_negative_events=getattr(self,'launch_rotation_negative_events',0)+int((value<0).sum())
                        self.rebound_preparation_samples=getattr(self,'rebound_preparation_samples',0)+int((a['preparation_weight']>0).sum())
                        self.rebound_launch_events=getattr(self,'rebound_launch_events',0)+int(a['launch'].sum())
                        y=audit['yaw_prediction']
                        assert all(torch.isfinite(v).all() for v in y.values())
                    if self.course['moving_phase_reward'].get('impact_alignment'):
                        value=audit['deltas']['moving_impact_alignment']
                        assert torch.isfinite(value).all() and (value<=0).all()
                        assert not value[~audit['impact_alignment']['active']].any()
                        assert torch.isfinite(audit['impact_alignment']['transverse_speed_m_s']).all()
                        self.impact_alignment_samples=getattr(self,'impact_alignment_samples',0)+int((value<0).sum())
                    if self.course['moving_phase_reward'].get('transport_outward_braking'):
                        value=audit['outward_weight']
                        assert torch.isfinite(value).all() and (value>=audit['weight']).all() and (value<=1).all()
                        earlier=audit['moving']&(value>audit['weight']+1e-6)
                        self.outward_early_samples=getattr(self,'outward_early_samples',0)+int(earlier.sum())
                    if self.course['moving_phase_reward'].get('include_route_holds'):
                        hold=audit['route_hold']
                        assert audit['moving'][hold].all(),'Route holds unexpectedly leave phase/guidance scope'
                        self.route_hold_samples=getattr(self,'route_hold_samples',0)+int(hold.sum())
                    if self.course.get('moving_touchdown_stability'):
                        value=audit['deltas']['moving_touchdown_rate']
                        assert (value<=0).all() and torch.isfinite(value).all()
                        self.touchdown_rate_events=getattr(self,'touchdown_rate_events',0)+int((value<0).sum())
                    if self.course['moving_phase_reward'].get('landing_time_s'):
                        assert torch.isfinite(audit['weight']).all()
                        assert ((audit['weight']>=0)&(audit['weight']<=1)).all()
                        earlier=(audit['weight']>audit['legacy_weight']+1e-6)&audit['moving']&(audit['clearance']>.3)
                        self.anticipation_samples=getattr(self,'anticipation_samples',0)+int(earlier.sum())
                    if self.course.get('moving_landing_guidance'):
                        assert audit['guidance'] is not None
                        desired=audit['guidance']['desired_velocity_xy']
                        assert torch.isfinite(desired).all()
                        assert desired.norm(dim=-1).max()<=self.course['moving_landing_guidance']['speed_limit_m_s']+1e-5
                        if self.course['moving_landing_guidance'].get('projection_clock_version'):
                            g=audit['guidance'];mask=audit['moving']&g['projection_clock_active']
                            clearance=audit['clearance'].clamp(min=0.)
                            speed=self._robot.data.root_link_lin_vel_w[:,2].abs()
                            expected=2*clearance/(torch.sqrt(speed.square()+2*9.81*clearance)+speed).clamp(min=1e-9)
                            torch.testing.assert_close(g['remaining_time_s'][mask],expected.clamp(max=1.2)[mask])
                            assert (g['division_time_s']>=.15).all()
                            self.physical_clock_samples=getattr(self,'physical_clock_samples',0)+int(mask.sum())
                        if self.course['moving_landing_guidance'].get('objective')=='projected_touchdown_error_v1':
                            for key in ('projected_foot_xy','projected_error_m'):
                                assert torch.isfinite(audit['guidance'][key]).all()
                            error=audit['guidance']['projected_error_m']
                            assert (error>=0).all()
                            active=audit['moving']&(audit['guidance']['flight_weight']>0)&(error>.1)
                            self.projected_miss_samples=getattr(self,'projected_miss_samples',0)+int(active.sum())
                        if self.course['moving_landing_guidance'].get('projection_origin'):
                            from .moving_phase_reward import touchdown_projection_origin
                            d=self._robot.data
                            reference=touchdown_projection_origin(d.root_link_pos_w,
                                d.body_link_pos_w[:,self._foot_body_id],d.joint_pos[:,self._spring_joint_id],
                                audit['rebound']['normal'])
                            mask=audit['moving']
                            blend_time=self.course['moving_landing_guidance'].get('measured_foot_blend_s',0.)
                            if blend_time:
                                g=audit['guidance']
                                fraction=(1.-(g['remaining_time_s']/blend_time).square()).clamp(0.,1.)
                                active=mask & g['projection_clock_active']
                                measured=d.body_link_pos_w[:,self._foot_body_id,:2]
                                reference=torch.where(active[:,None],
                                    reference+(measured-reference)*fraction[:,None],reference)
                            torch.testing.assert_close(audit['guidance']['projection_origin_xy'][mask],reference[mask])
                            difference=(reference-d.body_link_pos_w[:,self._foot_body_id,:2]).norm(dim=-1)
                            self.reference_projection_samples=getattr(self,'reference_projection_samples',0)+int((mask&(difference>.001)).sum())
                        for name in ('moving_velocity_guidance','moving_landing_error'):
                            assert (audit['deltas'][name]<=0).all()
                        self.guidance_samples=getattr(self,'guidance_samples',0)+int(audit['moving'].sum())
                        self.guidance_landings=getattr(self,'guidance_landings',0)+int((audit['deltas']['moving_landing_error']<0).sum())
                delta=self.omni_route.index-index;changed=delta==1
                assert ((delta==0)|changed).all()
                if guard is not None:
                    g=guard.mask;origin=self._terrain.env_origins[g,:2]
                    torch.testing.assert_close(self._two_ground_pt_xy_w[g],guard.xy[g]+origin)
                    torch.testing.assert_close(self._two_ground_pt1_xy_w[g],guard.next[g]+origin)
                    assert not (guard.mask & self.omni_route.stationary_mask).any()
                normal=~guard.mask if guard is not None else torch.ones_like(changed)
                assert torch.equal(self._two_ground_pt_xy_w[~changed & normal],pt[~changed & normal])
                assert torch.equal(self._two_ground_pt1_xy_w[~changed & normal],nxt[~changed & normal])
                if changed.any():
                    torch.testing.assert_close(self._two_ground_pt_xy_w[changed & normal],nxt[changed & normal])
                    memory=self.crossing_memory if obs72 else self.obs73_memory
                    assert memory.states[changed].abs().sum()>0
                assert int(self._omni_last_strict.sum())==int(self._foot_event.qualified-before)
                assert torch.equal(changed[normal],self._omni_last_navigation_advance[normal])
                if not self.course.get('moving_navigation'):
                    assert int(changed.sum())==int(self._foot_event.qualified-before)
                else:
                    assert not (changed&~self._omni_last_strict&self.omni_route.stationary_mask).any()
                self.goal_switches+=int(changed.sum())
                return reward

        raw=CheckedEnv(cfg,course)
        wrapped=RslRlVecEnvWrapper(raw)
        if course.get('straight_sequence_focus') or course['omni'].get('training_long_steps'):
            # A16-env short check may never draw a hold with the focused
            # training mixture. Exercise one deliberately in a random-route
            # member; leave the actual training/evaluation sampler untouched.
            route=raw.omni_route
            candidates=((route.sequence_kind==0)&~route.stationary_mask).nonzero().flatten()
            if hasattr(raw,'key_guard'):candidates=candidates[~raw.key_guard.mask[candidates]]
            if not candidates.numel() and course.get('moving_training_sequences',{}).get('version')=='cardinal_queued_segments_v2':
                candidates=((route.sequence_kind==4)&~route.stationary_mask).nonzero().flatten()
                if not candidates.numel():candidates=(~route.stationary_mask).nonzero().flatten()
            assert candidates.numel()>0,'Startup needs a moving member for its check-only hold fixture'
            probe=int(candidates[0]);offset=route.points[probe,1].clone()
            route.points[probe,1:]-=offset
            assert not route.points[probe,1].any()
            raw._bind_targets(raw._robot._ALL_INDICES)
            manifest['startup_hold_fixture']=dict(env_id=probe,
                scope='Check-only first route edge changed to a hold; later edge vectors preserved. Formal training/evaluation samplers unchanged.')
        if course.get('rigid_body_randomization'):
            from .flight_plane import audit_rigid_scales
            manifest['rigid_body_randomization'] = audit_rigid_scales(raw)
        obs=wrapped.get_observations()
        rc=json.loads((ROOT/course['runner_config_source']).read_text());rc.update(seed=42,max_iterations=0)
        runner=OnPolicyRunner(wrapped,rc,log_dir=None,device=args.device)
        manifest['transfer']=initialize_same_contract(runner,source,course)
        assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
        def forbidden(*a,**kw):raise AssertionError('Optimizer step forbidden')
        runner.alg.optimizer.step=forbidden
        policy=runner.get_inference_policy(device=args.device)
        assert raw.physics_dt==.005 and raw.step_dt==.01 and wrapped.num_actions==4
        minimum=float('inf');maximum=-float('inf');resets=0
        with torch.inference_mode():
            for step in range(steps):
                action=policy(obs);assert torch.isfinite(action).all()
                obs,reward,done,_=wrapped.step(action);runner.alg.policy.reset(done)
                assert obs['policy'].shape==(16,dim) and torch.isfinite(obs['policy']).all()
                assert torch.isfinite(reward).all()
                assert not raw.ring.collision.any() and not raw.ring.passed.any()
                assert not obs['policy'][done.bool(),19:57].any()
                if obs72:assert not obs['policy'][:,59:].any()  # inactive gate in omni bridge
                else:assert not obs['policy'][done.bool(),58:64].any()
                minimum=min(minimum,float(reward.min()));maximum=max(maximum,float(reward.max()))
                resets+=int(done.sum())
                if step%128==0:print(f'OMNI_STARTUP_STEP={step}',flush=True)
        events=[ev for ep in raw.evaluation.episodes for ev in ep.events]
        assert raw.goal_switches>0 and any(ev['target_index']==0 for ev in events)
        assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
        if hasattr(raw,'continuation_cosine'):
            if raw.adjust_continuation_approach:
                assert raw.lead_adjustments>0,'No descending continuation reference changed'
                manifest['continuation_approach_adjusted_ticks']=raw.lead_adjustments
            if raw.continuation_foot_offset_fraction:
                assert raw.foot_offset_adjustments>0,'No continuation used a displaced forecast foot'
                manifest['continuation_foot_offset_adjusted_ticks']=raw.foot_offset_adjustments
            assert raw.continuation_samples>0,'No continuous same-direction rebound tested'
            manifest['continuation_penalized_ticks']=raw.continuation_samples
            if getattr(raw,'release_carry_weight',0.):
                assert raw.carry_cost_events>0,'No matched continuation liftoff exercised the carry cost'
                manifest['release_carry_cost_events']=raw.carry_cost_events
            if getattr(raw,'shortfall_before_hold',False):
                assert raw.before_hold_samples>0,'No final continuation into a hold was exercised'
                manifest['before_hold_penalized_ticks']=raw.before_hold_samples
        if course.get('guarded_keys'):
            g=raw.key_guard
            expected_lanes=int((~raw.omni_route.stationary_mask).sum()*course['guarded_keys'].get('training_moving_fraction',.25))
            assert int(g.mask.sum())==expected_lanes and not (g.mask & raw.omni_route.stationary_mask).any()
            assert g.count[:,2].sum()>0
            manifest['guarded_queue_smoke']=dict(lanes=expected_lanes,launched=int(g.count[:,2].sum()),arrived=int(g.count[:,3].sum()))
        if course.get('stationary_retention'):
            assert int(raw.omni_route.stationary_mask.sum())==8
            assert not raw.omni_route.points[raw.omni_route.stationary_mask].any()
            assert raw.omni_route.points[~raw.omni_route.stationary_mask].abs().sum()>0
            manifest['rehearsal_allocation']=dict(stationary_envs=8,moving_envs=8)
        if course.get('moving_phase_reward'):
            assert raw.motion_samples>0 and raw.motion_adjustment_max>0
            manifest['moving_phase_reward_check']=dict(active_samples=raw.motion_samples,
                max_absolute_reward_delta=raw.motion_adjustment_max,nonmoving_rewards_unchanged=True)
            if course.get('moving_outward_tilt_work'):
                assert raw.outward_work_eligible > 0 and raw.outward_work_negative > 0
                manifest['outward_tilt_work_check'] = dict(eligible_samples=raw.outward_work_eligible,
                    penalized_samples=raw.outward_work_negative, optimizer_updates=0)
            if course.get('moving_rotation_work_extension'):
                enabled=course['moving_rotation_work_extension']['enabled']
                assert raw.work_extension_eligible>0 and raw.physical_clock_samples>0
                if enabled:assert raw.work_extension_negative>0
                manifest['rotation_work_extension_check']=dict(enabled=enabled,
                    eligible_free_flight_samples=raw.work_extension_eligible,
                    negative_extra_cost_samples=raw.work_extension_negative,
                    actual_motor_work_and_complementary_landing_factor_checked=True,
                    stationary_first_hop_masks_checked=True,optimizer_updates=0,
                    scope='Wiring only; learned precision/rate/retention require manual paired training')
            elif course.get('contact_attitude_continuity'):
                assert raw.contact_continuity_samples>0 and raw.physical_clock_samples>0
                manifest['contact_attitude_continuity_check']=dict(
                    stance_reference_samples=raw.contact_continuity_samples,
                    physical_clock_samples=raw.physical_clock_samples,
                    no_free_liftoff_reference_extension=True,nonmoving_masks_checked=True,
                    optimizer_updates=0,scope='Wiring only; no learned precision or retention claim')
            elif course.get('clock_ablation'):
                enabled=course['clock_ablation']['enabled']
                if enabled:assert raw.physical_clock_samples>0
                manifest['clock_ablation_check']=dict(enabled=enabled,
                    physical_clock_samples=getattr(raw,'physical_clock_samples',0),
                    unchanged_v26_other_objectives=True,optimizer_updates=0,
                    scope='Equal-budget source/config checked on CPU; GPU wiring only')
            if course['moving_phase_reward'].get('impact_alignment'):
                assert raw.impact_alignment_samples>0,'No precontact alignment cost exercised'
                manifest['impact_alignment_check']=dict(negative_precontact_samples=raw.impact_alignment_samples,
                    finite_nonpositive_and_masked=True,optimizer_updates=0,
                    scope='Wiring only; no learned impact stability claim')
        if course.get('rebound_transition'):
            assert raw.rebound_preparation_samples>0 and raw.rebound_launch_events>0
            assert cfg.scene.replicate_physics and cfg.scene.filter_collisions
            moving=~raw.omni_route.stationary_mask
            if course.get('moving_training_sequences'):
                assert raw.omni_route.sequence_kind[moving].gt(0).any()
            elif course['omni'].get('training_reversals'):
                from .obs73_omni import DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND
                kinds = raw.omni_route.sequence_kind[moving]
                assert ((kinds == 0) | (kinds == DIRECT_REVERSAL_KIND) | (kinds == PAUSED_REVERSAL_KIND)).all()
                manifest['omni_reversal_training_check'] = dict(
                    unrestricted=int((kinds == 0).sum()),
                    direct=int((kinds == DIRECT_REVERSAL_KIND).sum()),
                    paused=int((kinds == PAUSED_REVERSAL_KIND).sum()))
            else:
                assert not raw.omni_route.sequence_kind[moving].any()
            if course.get('omni_rebound_handoff'):
                assert raw.handoff_preview_samples > 0
                if raw.air_yaw_peak_weight:
                    manifest['free_air_yaw_peak_check']=dict(weight=raw.air_yaw_peak_weight,
                        predicted_peak_increments=getattr(raw,'yaw_peak_guard_samples',0))
                manifest['omni_rebound_handoff_check'] = dict(
                    preview_samples=raw.handoff_preview_samples, unloading_samples=raw.handoff_samples)
            manifest['rebound_transition_check']=dict(preparation_samples=raw.rebound_preparation_samples,
                once_liftoff_events=raw.rebound_launch_events,world_heading_prediction_finite=True,
                training_sequence_kinds=raw.omni_route.sequence_kind[moving].tolist(),
                isolated_collision_groups=True,pure_stationary_and_first_hop_rewards_unchanged=True,
                optimizer_updates=0,scope='Wiring only; smoothness/yaw improvement requires manual training')
            if course.get('launch_rotation_refinement'):
                manifest['launch_rotation_check']=dict(
                    launch_events=raw.rebound_launch_events,
                    negative_events=raw.launch_rotation_negative_events,
                    once_only_liftoff_latch_and_reward_masks_checked=True,optimizer_updates=0,
                    response_window_s=course['launch_rotation_refinement'].get('window_s',0.),
                    training_step_hold_envs=int(((raw.omni_route.sequence_kind==3)&moving).sum()))
            if course.get('launch_endpoint_refinement'):
                manifest['launch_endpoint_check']=dict(launch_events=raw.rebound_launch_events,
                    distance_width_m=course['launch_endpoint_refinement']['width_m'],
                    event_cost_and_masks_checked=True,optimizer_updates=0)
            if course.get('launch_velocity_window'):
                assert raw.launch_velocity_window_samples>0
                manifest['launch_velocity_window_check']=dict(
                    response_window_s=course['launch_velocity_window']['window_s'],
                    endpoint_width_m=course['launch_velocity_window']['width_m'],
                    active_samples=raw.launch_velocity_window_samples,
                    full_window_budget_and_masks_checked=True,optimizer_updates=0,
                    scope='Kernel/wiring check; learned precision and stationary retention require evaluation')
            if course.get('transition_repair'):
                assert raw.contact_reference_samples>0 and raw.physical_clock_samples>0
                manifest['transition_repair_check']=dict(
                    physical_clock_samples=raw.physical_clock_samples,
                    contact_reference_samples=raw.contact_reference_samples,
                    predicted_yaw_peak_increments=raw.yaw_peak_guard_samples,
                    restored_v26_launch_loss=True,stationary_masks_checked=True,
                    optimizer_updates=0,scope='Wiring only; learned behavior requires manual training/evaluation')
        if course.get('moving_rotation_prediction'):
            assert raw.rotation_prediction_samples>0
            manifest['rotation_prediction_check']=dict(active_free_flight_samples=raw.rotation_prediction_samples,
                negative_cost_samples=raw.rotation_prediction_negative_samples,
                finite_nonpositive_and_masked=True,optimizer_updates=0,
                scope='Wiring plus CPU real-case replay only; learned stability requires manual training/evaluation')
        if course.get('moving_landing_guidance'):
            assert raw.guidance_samples>0 and raw.guidance_landings>0
            manifest['moving_landing_guidance_check']=dict(active_samples=raw.guidance_samples,
                touchdown_error_events=raw.guidance_landings,
                reference_finite_and_speed_bounded=True,reward_nonpositive=True,nonmoving_rewards_unchanged=True)
            if course['moving_landing_guidance'].get('objective')=='projected_touchdown_error_v1':
                assert raw.projected_miss_samples>0,'No forecast misses exercise the new continuous objective'
                manifest['projected_landing_guidance_check']=dict(
                    active_forecast_miss_gt10cm_samples=raw.projected_miss_samples,
                    forecast_finite=True,optimizer_updates=0,
                    scope='Reward wiring only; forecast and learned improvement not validated by startup')
            if course['moving_landing_guidance'].get('projection_origin'):
                assert raw.reference_projection_samples>0
                manifest['reference_touchdown_projection_check']=dict(
                    changed_origin_samples=raw.reference_projection_samples,
                    root_A_and_reference_leg_checked=True,actual_touchdown_scorer_unchanged=True,
                    optimizer_updates=0)
        if course.get('moving_phase_reward',{}).get('landing_time_s'):
            assert raw.anticipation_samples>0,'No moving descent samples exercise anticipation before30cm'
            manifest['moving_landing_anticipation_check']=dict(
                earlier_than_legacy_30cm_samples=raw.anticipation_samples,
                time_window_s=course['moving_phase_reward']['landing_time_s'],
                phase_weights_finite_and_bounded=True,optimizer_updates=0)
        if course.get('moving_touchdown_stability'):
            assert raw.touchdown_rate_events>0,'No active touchdown-rate event exercised'
            manifest['moving_touchdown_stability_check']=dict(
                negative_touchdown_rate_events=raw.touchdown_rate_events,
                nonmoving_rewards_unchanged=True,reward_nonpositive_and_finite=True)
        if course.get('moving_phase_reward',{}).get('include_route_holds'):
            assert raw.route_hold_samples>0,'No route-hold samples exercised in startup'
            manifest['moving_route_hold_check']=dict(active_hold_samples=raw.route_hold_samples,
                pure_stationary_and_first_hop_unchanged=True,holds_keep_phase_and_guidance=True)
        if course.get('moving_phase_reward',{}).get('transport_outward_braking'):
            manifest['transport_outward_braking_check']=dict(
                earlier_weight_samples=raw.outward_early_samples,weights_finite_bounded_and_not_below_landing=True,
                scope='Runtime weights audited; real case52 native outward/inward kernel checked on CPU; no performance claim')
        if course.get('long_moving_horizon'):
            saved_length=raw.episode_length_buf.clone()
            raw.episode_length_buf[:]=math.ceil(15./raw.step_dt)-1
            _,timeouts=raw._get_dones()
            torch.testing.assert_close(timeouts,raw.omni_route.stationary_mask)
            raw.episode_length_buf.copy_(saved_length)
            manifest['split_timeout_check']='stationary15s,moving60s; actual _get_dones boundary checked'
        manifest.update(status='complete',gpu=torch.cuda.get_device_name(),goal_switches=raw.goal_switches,
            reward_min=minimum,reward_max=maximum,resets=resets,collector_events=len(events),
            target_retry_and_history_checked=True,source_env_config_preserved=not bool(course.get('long_moving_horizon')),
            source_robot_and_actuator_config_preserved=not bool(course.get('rigid_body_randomization')))
        (folder/'ring_config.json').write_text(json.dumps(course,indent=2)+'\n')
        save();print('OMNI_STARTUP_RESULT='+json.dumps(manifest),flush=True)
    except BaseException as exc:
        import traceback
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        if raw is not None:raw.close()
        if app is not None:app.close()


if __name__=='__main__':main()
