"""Deterministic reward/lifecycle checks; not a learned-motion performance gate."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ..moving_phase_reward import REWEIGHTED,PHASE_COSTS,REMOVED_COSTS,late_weight,reward_deltas,wrap_env,landing_guidance,touchdown_rate_cost,outward_weight,impact_alignment_cost,heading_motion,rebound_approach,ReboundPlan
from ..ring_obs73 import validate_checkpoint_course
from ..obs73_vertical import attitude_costs,flight_braking_costs,yaw_costs
from ..obs73_omni import navigation_advance
from ..metrics import continuity_metrics, landing_precision_metrics,rebound_motion_metrics,straight_batch_metrics

ROOT=Path(__file__).resolve().parents[2]


class ContactRewardParent:
    def __init__(self,course):
        self.course=course;self.num_envs=4;self.device='cpu';self.step_dt=.01
        self.extras={};self.common_step_counter=100;self._spring_joint_id=0;self._foot_body_id=0
        self._episode_sums={k:torch.zeros(4) for k in REWEIGHTED+('vertical_yaw_heading','vertical_yaw_rate')}
        self._thrust=torch.zeros(4,1,3)
        q=torch.tensor([[1.,0.,0.,0.]]).repeat(4,1);q[3]=torch.tensor([0.,1.,0.,0.])
        p=torch.zeros(4,3);p[:,2]=.4;foot=p[:,None,:]-torch.tensor([0.,0.,.285])
        v=torch.tensor([[.3,0.,-2.]]).repeat(4,1)
        self._robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=p,root_quat_w=q,
            body_link_pos_w=foot,root_link_lin_vel_w=v,root_com_lin_vel_w=v.clone(),
            root_ang_vel_b=torch.zeros(4,3),joint_pos=torch.zeros(4,1),joint_vel=torch.zeros(4,1)))
        self._terrain=SimpleNamespace(env_origins=torch.zeros(4,3))
        self._foot_event=SimpleNamespace(phase=torch.full((4,),3),DROP=0,FLIGHT=3,
            air_steps=torch.full((4,),80),apex_seen=torch.ones(4,dtype=torch.bool))
        points=torch.zeros(4,5,2);points[:,1:,0]=torch.tensor([.5,1.,1.5,2.])
        self.omni_route=SimpleNamespace(index=torch.tensor([1,0,1,1]),points=points,
            stationary_mask=torch.tensor([True,False,False,False]),evaluation=True)
        self.bind();self._desired_pos_w=torch.zeros(4,3);self._desired_pos_w[:,2]=1.
        self.sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0.,0.,-9.81)))
    def bind(self):
        i=self.omni_route.index
        self._two_ground_pt_xy_w=self.omni_route.points[torch.arange(4),i].clone()
        self._two_ground_pt1_xy_w=self.omni_route.points[torch.arange(4),i+1].clone()
    def _get_rewards(self):
        d=self._robot.data;f=self._foot_event
        physical=d.root_quat_w[:,1:3].square().sum(-1)>.5
        values=attitude_costs(d.root_quat_w,d.root_ang_vel_b,f.phase,physical,.01,
            self.course['vertical_pretraining']['attitude_refinement'])
        values['attitude_penalty']=-.2*d.root_quat_w[:,1:3].square().sum(-1)
        yaw,_,_=yaw_costs(d.root_quat_w,d.root_ang_vel_b[:,2],f.phase,physical,.01,
            self.course['vertical_pretraining']['yaw_refinement']);values.update(yaw)
        for k,v in values.items():self._episode_sums[k]+=v
        valid=(f.phase==3)&(f.air_steps>=5)&(d.joint_pos[:,0]>.002)&~physical
        error=(d.body_link_pos_w[:,0,:2]-self._two_ground_pt_xy_w).norm(dim=-1)
        tilt=torch.rad2deg(torch.acos((1-2*d.root_quat_w[:,1:3].square().sum(-1)).clamp(-1,1)))
        strict=valid&(error<=.1)&(tilt<=15.)
        advance=navigation_advance(valid,strict,error,tilt,self.omni_route.index,
            self.omni_route.stationary_mask,self.course.get('moving_navigation'))
        self.omni_route.index+=advance.long();self.bind()
        f.phase[valid]=1;f.air_steps[valid]=0
        return torch.stack(list(values.values())).sum(0)
    def _reset_idx(self,ids):pass


def rotation_work_extension_check(c):
    """Production wrapper checks for the single-change v31 work ablation."""
    from ..obs73_vertical import rotation_work_cost
    source=ROOT/c['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text());run(source.parent/'ring_config.json')
    option=c['moving_rotation_work_extension'];enabled=option['enabled']
    assert option['version']=='excess_rate_positive_motor_work_v1' and isinstance(enabled,bool)
    assert option['source_checkpoint']==str(source.relative_to(ROOT))
    allowed={'version','status','output_series','runner_config_source','training_iterations',
             'omni','initialization','reward','moving_rotation_work_extension'}
    assert set(c)==set(old)|{'moving_rotation_work_extension'}
    for key in set(c)-allowed:assert c[key]==old[key],key
    assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    assert not any(c.get(key) for key in ('contact_attitude_continuity','transition_repair','launch_endpoint_refinement'))
    assert c['training_iterations']==100 and c['stationary_retention']['training_stationary_fraction']==.5
    assert c['vertical_pretraining']['rotation_work_penalty']['weight']==500.
    assert json.loads((ROOT/c['runner_config_source']).read_text())==json.loads((source.parent/'runner_config.json').read_text())
    validate_checkpoint_course(source,c)
    pair_kind='control' if enabled else 'only'
    pair=json.loads((ROOT/f'benchmark37/configs/obs73_quaternion_work_{pair_kind}_course_v31.json').read_text())
    for key in c:
        if key not in ('version','status','output_series','reward','moving_rotation_work_extension'):
            assert c[key]==pair[key],key
    assert dict(option,enabled=not enabled)==pair['moving_rotation_work_extension']

    class WorkParent(ContactRewardParent):
        def __init__(self,course):
            super().__init__(course);self._moment=torch.zeros(4,1,3)
        def _get_rewards(self):
            d=self._robot.data
            physical=(d.root_link_pos_w[:,2]<.05)|(d.root_quat_w[:,1:3].square().sum(-1)>.5)
            raw=rotation_work_cost(self._moment[:,0],d.root_ang_vel_b,self._foot_event.phase,
                d.joint_pos[:,0],physical,.01,self.course['vertical_pretraining']['rotation_work_penalty'],d.root_quat_w)
            result=super()._get_rewards()
            self._episode_sums['vertical_rotation_work']+=raw
            return result+raw

    def probe(omega=(2.,0.,0.),moment=(.025,0.,0.),phase=2,spring=0.,spring_rate=0.,clearance=.1,vz=2.,hold=False,pitch_deg=0.):
        audits=[]
        for course in (c,old):
            env=wrap_env(WorkParent)(course);d=env._robot.data
            env.common_step_counter=130;env.rebound_plan.launch_step[:]=100
            env._foot_event.phase[:]=phase;d.joint_pos[:]=spring;d.joint_vel[:]=spring_rate
            d.body_link_pos_w[:,:,2]=.012+clearance;d.root_link_pos_w[:,2]=.297+clearance
            d.root_link_lin_vel_w[:,2]=vz;d.root_ang_vel_b[:]=torch.tensor(omega)
            env._moment[:]=torch.tensor(moment)
            half=math.radians(pitch_deg)/2
            d.root_quat_w[2]=torch.tensor([math.cos(half),0.,math.sin(half),0.])
            if hold:env.omni_route.points[:,0]=env.omni_route.points[:,1];env.bind()
            env._get_rewards();audits.append(env._moving_phase_last_audit)
        a,b=audits
        for key in a['deltas']:
            if key!='vertical_rotation_work':torch.testing.assert_close(a['deltas'][key],b['deltas'][key])
            torch.testing.assert_close(a['deltas'][key][[0,1,3]],b['deltas'][key][[0,1,3]])
        return a,b

    a,b=probe();k=2
    raw=a['inherited']['vertical_rotation_work'][k]
    assert raw<0 and not a['deltas']['moving_launch_rotation'][k],'Fixture must be past180ms launch window'
    actual=raw+a['deltas']['vertical_rotation_work'][k]
    if enabled:
        assert actual<0,'High-rate positive motor work is still unpenalized after180ms'
        torch.testing.assert_close(actual,raw)
    else:torch.testing.assert_close(a['deltas']['vertical_rotation_work'],b['deltas']['vertical_rotation_work'])
    # Low-rate useful lean, pure yaw and braking create no added work cost;
    # one braking axis must not cancel acceleration on the other axis.
    for omega,moment in [((.5,0.,0.),(.025,0.,0.)),((2.,0.,0.),(-.025,0.,0.)),((0.,0.,2.),(0.,0.,.025))]:
        a,b=probe(omega,moment)
        torch.testing.assert_close(a['deltas']['vertical_rotation_work'],b['deltas']['vertical_rotation_work'])
    a,b=probe((.9,0.,0.));raw=a['inherited']['vertical_rotation_work'][k]
    torch.testing.assert_close(raw+a['deltas']['vertical_rotation_work'][k],raw*(.5 if enabled else 0.),atol=1e-6,rtol=1e-5)
    a,b=probe((2.,2.,0.),(.025,-.025,0.));raw=a['inherited']['vertical_rotation_work'][k]
    torch.testing.assert_close(raw,torch.tensor(-.25))
    torch.testing.assert_close(raw+a['deltas']['vertical_rotation_work'][k],raw if enabled else torch.tensor(0.))
    a,b=probe((0.,-2.,0.),(0.,-.025,0.),pitch_deg=20.)
    raw=a['inherited']['vertical_rotation_work'][k]
    torch.testing.assert_close(raw,torch.tensor(-.0625),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(raw+a['deltas']['vertical_rotation_work'][k],raw if enabled else torch.tensor(0.),atol=1e-6,rtol=1e-5)
    # Existing late-landing cost is complemented, never charged twice.
    a,b=probe(phase=3,clearance=3*.09+.5*9.81*.09**2,vz=-3.)
    raw=a['inherited']['vertical_rotation_work'][k]
    torch.testing.assert_close(a['weight'][k],torch.tensor(.5),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(raw+a['deltas']['vertical_rotation_work'][k],raw*(1. if enabled else .5),atol=1e-6,rtol=1e-5)
    for kwargs in ({'phase':0},{'phase':1,'spring':.03},{'spring_rate':1.},{'clearance':0.}):
        a,b=probe(**kwargs)
        torch.testing.assert_close(a['deltas']['vertical_rotation_work'],b['deltas']['vertical_rotation_work'])
    a,b=probe(hold=True)
    if enabled:assert a['deltas']['vertical_rotation_work'][k]<b['deltas']['vertical_rotation_work'][k]
    return dict(paired_same_v29_source100_budget_and_only_work_factor_change_checked=True,
        high_rate_positive_work_past180ms_braking_yaw_low_rate_and_axis_cancellation_checked=True,
        continuous_ramp_and_no_landing_double_charge_checked=True,
        stationary_first_hop_drop_contact_spring_motion_terminal_and_holds_checked=True,
        enabled=enabled,scope='Actual reward call-site checks; learned improvement requires manual training/evaluation')


def contact_continuity_check(c):
    """Aligned descent/stance call-site regression, with unchanged v29 costs."""
    source=ROOT/c['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text())
    run(source.parent/'ring_config.json')
    allowed={'version','status','output_series','runner_config_source','training_iterations',
             'omni','initialization','reward','contact_attitude_continuity'}
    assert set(c)==set(old)|{'contact_attitude_continuity'}
    for k in set(c)-allowed:assert c[k]==old[k],k
    assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    assert c['contact_attitude_continuity']['source_checkpoint']==c['omni']['initial_checkpoint']
    assert not c.get('transition_repair') and not c.get('launch_endpoint_refinement')
    assert c['training_iterations']==100 and c['stationary_retention']['training_stationary_fraction']==.5
    assert json.loads((ROOT/c['runner_config_source']).read_text())==json.loads((source.parent/'runner_config.json').read_text())
    validate_checkpoint_course(source,c)
    new=wrap_env(ContactRewardParent)(c);base=wrap_env(ContactRewardParent)(old);k=2
    def step(t):
        new.common_step_counter=t;base.common_step_counter=t
        new._get_rewards();base._get_rewards()
        a=new._moving_phase_last_audit;b=base._moving_phase_last_audit
        for key in a['deltas']:
            if key not in ('vertical_tilt','attitude_penalty','moving_rebound_approach'):
                torch.testing.assert_close(a['deltas'][key],b['deltas'][key])
            torch.testing.assert_close(a['deltas'][key][[0,1,3]],b['deltas'][key][[0,1,3]])
        torch.testing.assert_close(new.rebound_plan.velocity,base.rebound_plan.velocity)
        assert torch.equal(new.omni_route.index,base.omni_route.index)
        return a,b
    a,b=step(100)
    for key in a['deltas']:torch.testing.assert_close(a['deltas'][key],b['deltas'][key])
    normal=a['rebound']['normal'][k].clone()
    assert normal[0].abs()>.01,'Fixture needs a useful leaned approach'
    angle=torch.atan2(normal[0],normal[2])
    for env in (new,base):
        d=env._robot.data
        d.body_link_pos_w[k,0]=torch.tensor([.5,0.,.012])
        d.root_link_pos_w[k]=d.body_link_pos_w[k,0]+.255*normal
        d.root_quat_w[k]=torch.tensor([torch.cos(angle/2),0.,torch.sin(angle/2),0.])
        d.joint_pos[k]=.03
    a,b=step(101)
    # Red on v29: a useful leaned touchdown is charged for not being upright,
    # immediately after the descent normal was its desired attitude.
    for key in ('vertical_tilt','attitude_penalty'):
        assert abs(float(a['inherited'][key][k]+a['deltas'][key][k]))<1e-7, \
            'Aligned rebound reference is still punished as upright at contact'
    assert a['rebound']['contact_reference_weight'][k]==1
    torch.testing.assert_close(new.rebound_plan.contact_normal[k],normal)
    assert abs(float(a['deltas']['moving_rebound_approach'][k]))<1e-6
    # Contact must still brake body rates/leg motion; it is not free attitude.
    for env in (new,base):env._robot.data.root_ang_vel_b[k,:2]=torch.tensor([.8,-.7])
    a,b=step(102)
    assert a['inherited']['vertical_roll_pitch_rate'][k]<0
    torch.testing.assert_close(a['rebound']['contact_reference_normal'][k],normal)
    # The same cycle reference survives compression and extension.
    for env in (new,base):env._robot.data.joint_pos[k]=.01
    a,b=step(103)
    assert a['rebound']['contact_reference_weight'][k]==1
    # Free liftoff immediately restores exact v29 terms; there is no v28
    # post-release attitude extension. Launch velocity/rate costs are intact.
    for env in (new,base):
        d=env._robot.data;env._foot_event.phase[k]=2
        d.joint_pos[k]=0.;d.joint_vel[k]=0.
        d.body_link_pos_w[k,0,2]=.05;d.root_link_pos_w[k,2]=.335
        d.root_link_lin_vel_w[k]=torch.tensor([.4,0.,2.5])
    a,b=step(107)
    assert a['rebound']['launch'][k] and a['rebound']['contact_reference_weight'][k]==0
    for key in a['deltas']:torch.testing.assert_close(a['deltas'][key],b['deltas'][key])
    a,b=step(116)
    for key in a['deltas']:torch.testing.assert_close(a['deltas'][key],b['deltas'][key])
    # Drop and partial reset cannot retain a reference from a prior cycle.
    new._foot_event.phase[k]=0;new._get_rewards()
    assert new._moving_phase_last_audit['rebound']['contact_reference_weight'][k]==0
    new.rebound_plan.contact_reference_valid[0]=True
    new._reset_idx(torch.tensor([k]))
    assert not new.rebound_plan.contact_reference_valid[k] and not new.rebound_plan.incoming_valid[k]
    assert new.rebound_plan.contact_reference_valid[0]
    torch.testing.assert_close(new.rebound_plan.prepared_normal[k],torch.tensor([0.,0.,1.]))
    # A missed target must prepare its retry, not continue the next-point
    # lean. This changes only the stance reference, never precontact preview.
    missed=wrap_env(ContactRewardParent)(c);md=missed._robot.data
    missed._get_rewards();prepared=missed.rebound_plan.prepared_normal[k].clone()
    md.body_link_pos_w[k,0]=torch.tensor([.7,0.,.012])
    md.root_link_pos_w[k]=md.body_link_pos_w[k,0]+torch.tensor([0.,0.,.255])
    md.joint_pos[k]=.03;missed.common_step_counter=101;missed._get_rewards()
    assert missed.omni_route.index[k]==1 and missed.rebound_plan.velocity[k,0]<0
    assert prepared[0]>0 and missed.rebound_plan.contact_normal[k,0]<0
    held=missed.rebound_plan.contact_normal[k].clone()
    missed.common_step_counter=102;missed._get_rewards()
    torch.testing.assert_close(missed.rebound_plan.contact_normal[k],held)
    # A hold uses the same already-existing zero-outgoing approach. A fresh
    # contact with no descent observation must retain the old upright cost.
    hold=wrap_env(ContactRewardParent)(c)
    hold.omni_route.points[k,2]=hold.omni_route.points[k,1];hold.bind();hold._get_rewards()
    hold_normal=hold.rebound_plan.prepared_normal[k].clone();hd=hold._robot.data
    hd.body_link_pos_w[k,0]=torch.tensor([.5,0.,.012])
    hd.root_link_pos_w[k]=hd.body_link_pos_w[k,0]+.255*hold_normal;hd.joint_pos[k]=.03
    hold.common_step_counter=101;hold._get_rewards()
    torch.testing.assert_close(hold.rebound_plan.contact_normal[k],hold_normal)
    assert hold._moving_phase_last_audit['rebound']['contact_reference_weight'][k]==1
    cold=wrap_env(ContactRewardParent)(c);cold._robot.data.joint_pos[k]=.03;cold._get_rewards()
    assert cold._moving_phase_last_audit['rebound']['contact_reference_weight'][k]==0
    return dict(aligned_descent_contact_compression_rebound_reference_checked=True,
        original_rate_leg_work_yaw_launch_and_free_ascent_terms_checked=True,
        stationary_first_hop_drop_terminal_and_partial_reset_checked=True,
        missed_target_retry_hold_and_missing_precontact_fallback_checked=True,
        only_contact_angle_objective_changed_vs_v29=True,
        limitation='Call-site reward checks only; manual training and saved-case evaluation required')


def clock_ablation_check(c):
    """Equal-budget v26 control and single-change clock experiment."""
    ablation=c['clock_ablation'];source=ROOT/c['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text())
    assert ablation['version']=='v26_control_vs_descent_clock_only_v1'
    assert isinstance(ablation['enabled'],bool)
    assert ablation['source_checkpoint']==str(source.relative_to(ROOT))
    assert source.name=='model_499.pt' and json.loads((source.parent/'manifest.json').read_text())['status']=='complete'
    run(source.parent/'ring_config.json')
    allowed={'version','status','output_series','runner_config_source','training_iterations',
             'omni','initialization','reward','moving_landing_guidance','clock_ablation'}
    assert set(c)==set(old)|{'clock_ablation'}
    for k in set(c)-allowed:assert c[k]==old[k],k
    assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    expected=dict(old['moving_landing_guidance'])
    if ablation['enabled']:expected['projection_clock_version']='zero_at_contact_separate_divisor_v2'
    assert c['moving_landing_guidance']==expected
    assert c['training_iterations']==100
    assert not c.get('transition_repair') and not c.get('launch_endpoint_refinement')
    assert c['stationary_retention']['training_stationary_fraction']==.5
    assert json.loads((ROOT/c['runner_config_source']).read_text())==json.loads((source.parent/'runner_config.json').read_text())
    validate_checkpoint_course(source,c)
    # Near impact, including zero clearance and a stale long-flight floor.
    time=torch.tensor([.01,0.,.08]);height=3*time+.5*9.81*time.square()
    v=torch.tensor([[.5,0.,-3.]]).expand(3,-1)
    args=(torch.zeros(3,2),height,v,torch.full((3,),.3),torch.zeros(3,2),torch.ones(3),
          torch.full((3,),3),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool),9.81,.01)
    terms,a=landing_guidance(*args,c['moving_landing_guidance'],torch.ones(3))
    _,baseline=landing_guidance(*args,old['moving_landing_guidance'],torch.ones(3))
    if ablation['enabled']:
        torch.testing.assert_close(a['remaining_time_s'],time,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a['projected_foot_xy'][:,0],.5*time,atol=1e-6,rtol=1e-5)
    else:
        for k in a:torch.testing.assert_close(a[k],baseline[k])
    assert torch.isfinite(a['desired_velocity_xy']).all()
    assert (a['division_time_s']>=.15).all()
    # Stance and ascending floor are deliberately identical to v26. Masked
    # stationary/first-hop terms remain zero even though audit clocks differ.
    for phase in (0,1,2,3):
        up=list(args);up[2]=v.clone();up[2][:,2]=3.;up[6]=torch.full((3,),phase)
        nt,na=landing_guidance(*up,c['moving_landing_guidance'],torch.ones(3))
        ot,oa=landing_guidance(*up,old['moving_landing_guidance'],torch.ones(3))
        for k in nt:torch.testing.assert_close(nt[k],ot[k])
        torch.testing.assert_close(na['remaining_time_s'],oa['remaining_time_s'])
    inactive=list(args);inactive[7]=torch.zeros(3,dtype=torch.bool)
    nt,_=landing_guidance(*inactive,c['moving_landing_guidance'],torch.ones(3))
    assert not any(value.any() for value in nt.values())
    # Counterpart differs only in its flag and the projection-clock setting,
    # apart from labels. This catches accidental bundled reward changes.
    other='control' if ablation['enabled'] else 'only'
    pair=json.loads((ROOT/f'benchmark37/configs/obs73_quaternion_clock_{other}_course_v29.json').read_text())
    for k in c:
        if k not in ('version','status','output_series','reward','moving_landing_guidance','clock_ablation'):
            assert c[k]==pair[k],k
    return dict(equal_budget100_same_source_seed_physics_ppo_and_stationary_mix_checked=True,
        clock_only_enabled=ablation['enabled'],stale_floor_and_zero_clearance_projection_checked=True,
        legacy_ascent_stance_and_inactive_reward_masks_checked=True,
        scope='Numerical/wiring checks only; learned behavior requires manual training and saved-case evaluation')


def transition_repairs_check(c):
    """Actual production-kernel and wrapper regressions for v28 repairs."""
    source=ROOT/c['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text())
    # Keep the entire historical v26 regression suite red-capable as well.
    run(source.parent/'ring_config.json')
    allowed={'version','status','output_series','runner_config_source','training_iterations',
             'omni','initialization','reward','moving_landing_guidance','transition_repair'}
    assert set(c)==set(old)|{'transition_repair'}
    for k in set(c)-allowed:assert c[k]==old[k],k
    assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    assert c['moving_landing_guidance']==dict(old['moving_landing_guidance'],
        projection_clock_version='zero_at_contact_separate_divisor_v2')
    assert not c.get('launch_endpoint_refinement')
    validate_checkpoint_course(source,c)
    g=c['moving_landing_guidance']
    # Same precontact pattern as case122:10ms actual remaining, but the old
    # velocity denominator floor projected another75mm. A stale EMA floor
    # must not override measured descent/contact timing.
    time=torch.tensor([.01,0.,.08]);speed=torch.full((3,),3.)
    height=speed*time+.5*9.81*time.square()
    vel=torch.zeros(3,3);vel[:,0]=.5;vel[:,2]=-speed
    args=(torch.zeros(3,2),height,vel,torch.full((3,),.285),torch.zeros(3,2),
          torch.ones(3),torch.full((3,),3),torch.ones(3,dtype=torch.bool),
          torch.zeros(3,dtype=torch.bool),9.81,.01,g,torch.ones(3))
    _,a=landing_guidance(*args)
    torch.testing.assert_close(a['remaining_time_s'],time,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(a['projected_foot_xy'][:,0],.5*time,atol=1e-6,rtol=1e-5)
    assert torch.isfinite(a['desired_velocity_xy']).all()
    assert a['desired_velocity_xy'].norm(dim=-1).max()<=g['speed_limit_m_s']
    from ..moving_phase_reward import consistent_outgoing_velocity
    # Failed landing precursor: foot already beyond current point; preparing
    # +X next point contradicts the actual retained-target retry to-X.
    foot=torch.tensor([[.7,0.],[.5,0.],[.55,0.]])
    vel=torch.zeros(3,3);vel[:,2]=-3.
    current=torch.tensor([[.5,0.]]).expand(3,-1)
    nxt=torch.tensor([[1.,0.],[1.,0.],[.5,0.]])
    outgoing,confidence,_=consistent_outgoing_velocity(foot,torch.tensor([.03]*3),vel,
        current,nxt,torch.ones(3),torch.zeros(3),9.81,1.2,c['transition_repair'])
    assert outgoing[0,0]<0 and confidence[0]==0,'A missed point still prepares next-point acceleration'
    assert outgoing[1,0]>0 and confidence[1]==1
    assert outgoing[2,0]<0,'A moving-route hold does not correct existing position error'
    assert torch.isfinite(outgoing).all() and outgoing.norm(dim=-1).max()<=1.2
    from ..moving_phase_reward import yaw_peak_increment
    # Actual case127 crosses10deg before contact with sustained negative
    # heading rate. The predictive guard must react while still at7deg.
    error=torch.deg2rad(torch.tensor([-7.,7.,0.,-7.]))
    rate=torch.tensor([-.9,.9,0.,-.9]);active=torch.tensor([True,True,True,False])
    cost,peak=yaw_peak_increment(error,rate,torch.zeros(4),active,.14,10.,40.)
    assert cost[0]<0 and cost[0]==cost[1] and not cost[2:].any()
    repeated,next_peak=yaw_peak_increment(error,rate,peak,active,.14,10.,40.)
    assert not repeated.any();torch.testing.assert_close(next_peak,peak)
    recovery,_=yaw_peak_increment(torch.zeros(4),torch.zeros(4),peak,active,.14,10.,40.)
    assert not recovery.any(),'Recovery pays back peak risk and permits reward cycling'
    # Exercise actual parent navigation,contact latch,stance and free-release
    # call order. No substitute controller/test-only reward implementation.
    from ..obs73_omni import navigation_advance
    from ..ring_geometry import rotate

    yaw_env=wrap_env(ContactRewardParent)(c);yd=yaw_env._robot.data;k=2
    half=math.radians(-7.)/2
    yd.root_quat_w[k]=torch.tensor([math.cos(half),0.,0.,math.sin(half)])
    yd.root_ang_vel_b[k,2]=-.9
    yaw_env._get_rewards();ya=yaw_env._moving_phase_last_audit
    assert ya['deltas'].get('moving_yaw_peak',torch.zeros(4))[k]<0,'Actual wrapper has no predictive yaw signal before10deg'
    assert not ya['deltas']['moving_yaw_peak'][[0,1,3]].any()
    yaw_env._get_rewards()
    assert not yaw_env._moving_phase_last_audit['deltas']['moving_yaw_peak'].any(),'Repeated state charged again'
    # A NEW real free-liftoff starts a new cycle. Goal changes alone do not.
    yaw_env.omni_route.index[k]=2;yaw_env.bind();yaw_env._get_rewards()
    assert yaw_env._moving_phase_last_audit['deltas']['moving_yaw_peak'][k]==0
    yaw_env.rebound_plan.pending[k]=True;yaw_env._foot_event.phase[k]=2
    yd.root_link_lin_vel_w[k,2]=2.;yaw_env.common_step_counter=101
    yaw_env._get_rewards();ya=yaw_env._moving_phase_last_audit
    assert ya['rebound']['launch'][k] and ya['deltas']['moving_yaw_peak'][k]<0
    # An independent switch disables only the new yaw objective.
    yaw_off=dict(c,transition_repair=dict(c['transition_repair'],yaw_peak_guard=False))
    off_env=wrap_env(ContactRewardParent)(yaw_off);od=off_env._robot.data
    od.root_quat_w[k]=yd.root_quat_w[k];od.root_ang_vel_b[k]=yd.root_ang_vel_b[k]
    off_env._get_rewards();assert 'moving_yaw_peak' not in off_env._moving_phase_last_audit['deltas']

    env=wrap_env(ContactRewardParent)(c);d=env._robot.data;k=2
    d.root_link_pos_w[k,0]=.72;d.body_link_pos_w[k,0,0]=.72
    env._get_rewards();a=env._moving_phase_last_audit
    assert a['rebound']['outgoing_velocity'][k,0]<0,'Wrapper still prepares next point before an expected miss'
    assert env.omni_route.index[k]==1
    # A successful contact latches the actual next point, regardless of the
    # earlier predicted miss. Restore a physically tilted8deg contact state.
    angle=math.radians(8);d.root_quat_w[k]=torch.tensor([math.cos(angle/2),0.,math.sin(angle/2),0.])
    d.body_link_pos_w[k,0]=torch.tensor([.5,0.,.012])
    d.root_link_pos_w[k]=d.body_link_pos_w[k,0]+rotate(d.root_quat_w[k],torch.tensor([0.,0.,.255]))
    d.joint_pos[k]=.03;d.root_link_lin_vel_w[k]=torch.tensor([.4,0.,-2.5])
    env.common_step_counter=101;env._get_rewards();a=env._moving_phase_last_audit
    assert env.omni_route.index[k]==2 and env.rebound_plan.pending[k]
    assert env.rebound_plan.contact_reference_valid[k]
    assert env.rebound_plan.velocity[k,0]>0,'Actual advance did not override the earlier predicted retry'
    assert a['rebound']['contact_reference_weight'][k]==1
    assert a['deltas']['moving_rebound_approach'][k]<0
    torch.testing.assert_close(a['inherited']['vertical_tilt'][k]+a['deltas']['vertical_tilt'][k],torch.tensor(0.))
    normal=env.rebound_plan.contact_normal[k].clone()
    env.common_step_counter=102;env._get_rewards()
    torch.testing.assert_close(env.rebound_plan.contact_normal[k],normal)
    assert env._moving_phase_last_audit['rebound']['contact_reference_weight'][k]==1
    # Exactly aligned useful lean gets no angle cost; upright is not required.
    d.root_link_pos_w[k]=d.body_link_pos_w[k,0]+.255*normal
    env.common_step_counter=103;env._get_rewards()
    assert abs(float(env._moving_phase_last_audit['deltas']['moving_rebound_approach'][k]))<1e-6
    # Free liftoff retains the same normal, then fades over existing response
    # horizon. The velocity event remains the original v26 squared error.
    env._foot_event.phase[k]=2;d.joint_pos[k]=0.;d.joint_vel[k]=0.
    d.body_link_pos_w[k,0,2]=.05;d.root_link_pos_w[k,2]=.335
    d.root_link_lin_vel_w[k]=torch.tensor([.4,0.,2.5]);env.common_step_counter=107
    env._get_rewards();a=env._moving_phase_last_audit
    assert a['rebound']['launch'][k] and not env.rebound_plan.pending[k]
    assert a['rebound']['contact_reference_weight'][k]==1
    torch.testing.assert_close(a['deltas']['moving_launch_velocity'][k],
        -4*(d.root_link_lin_vel_w[k,:2]-env.rebound_plan.velocity[k]).square().sum())
    env.common_step_counter=116;env._get_rewards();a=env._moving_phase_last_audit
    torch.testing.assert_close(a['rebound']['contact_reference_weight'][k],torch.tensor(.5))
    assert not a['rebound']['launch'][k] and a['deltas']['moving_launch_velocity'][k]==0
    env.common_step_counter=125;env._get_rewards()
    assert env._moving_phase_last_audit['rebound']['contact_reference_weight'][k]==0
    for j in (0,1,3):
        assert all(v[j]==0 for v in env._moving_phase_last_audit['deltas'].values()),'Repair affected stationary/first-hop/terminal sample'
    # Drop exclusion and partial reset of all newly introduced memories.
    env._foot_event.phase[k]=0;env.common_step_counter=126;env._get_rewards()
    assert all(v[k]==0 for v in env._moving_phase_last_audit['deltas'].values())
    env.rebound_plan.yaw_peak_squared.fill_(1.)
    env._reset_idx(torch.tensor([k]));assert env.rebound_plan.yaw_peak_squared[k]==0
    assert env.rebound_plan.yaw_peak_squared[0]==1
    assert not env.rebound_plan.contact_reference_valid[k] and not env.rebound_plan.incoming_valid[k]
    torch.testing.assert_close(env.rebound_plan.contact_normal[k],torch.tensor([0.,0.,1.]))
    return dict(zero_at_contact_forecast_and_separate_divisor_checked=True,
        predictive_retained_vs_advanced_goal_and_hold_recovery_checked=True,
        predictive_yaw_peak_early_braking_symmetry_masks_and_no_duplicate_charge_checked=True,
        contact_reference_actual_navigation_latch_release_fade_and_partial_reset_checked=True,
        yaw_actual_wrapper_signal_cycle_reset_and_independent_disable_checked=True,
        legacy_launch_cost_and_stationary_first_hop_drop_terminal_masks_checked=True)


def run(course_path=None):
    c=json.loads(Path(course_path or ROOT/'benchmark37/configs/obs73_quaternion_straight_focus_course_v23.json').read_text())
    if c.get('moving_rotation_work_extension'):
        return rotation_work_extension_check(c)
    if c.get('contact_attitude_continuity'):
        return contact_continuity_check(c)
    if c.get('clock_ablation'):
        return clock_ablation_check(c)
    if c.get('transition_repair'):
        return transition_repairs_check(c)
    source=ROOT/c['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text())
    m=json.loads((source.parent/'manifest.json').read_text())
    anticipation=c['moving_phase_reward'].get('landing_time_s')
    precision=(c.get('moving_precision_refinement') if c.get('moving_landing_guidance',{}).get('landing_width_m')
        !=old.get('moving_landing_guidance',{}).get('landing_width_m') else None)
    stability=c.get('moving_touchdown_stability')
    include_holds=c['moving_phase_reward'].get('include_route_holds',False)
    outward=c['moving_phase_reward'].get('transport_outward_braking')
    projection=c.get('projected_landing_refinement')
    impact=c['moving_phase_reward'].get('impact_alignment')
    rebound=c.get('rebound_transition')
    touchdown_projection=c.get('touchdown_projection_reference')
    navigation=c.get('moving_navigation')
    rotation_prediction=c.get('moving_rotation_prediction')
    straight_focus=c.get('straight_sequence_focus')
    body_rate=c.get('moving_body_rate')
    launch_rotation=c.get('launch_rotation_refinement')
    response_window=c.get('launch_response_window_refinement')
    launch_endpoint=c.get('launch_endpoint_refinement')
    if launch_endpoint:
        allowed={'version','status','output_series','runner_config_source','training_iterations','omni','launch_endpoint_refinement','initialization','reward'}
        assert set(c)==set(old)|{'launch_endpoint_refinement'}
        for key in set(c)-allowed:assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert launch_endpoint['source_checkpoint']==str(source.relative_to(ROOT))
        assert launch_endpoint['width_m']==.1
    elif response_window:
        allowed={'version','status','output_series','runner_config_source','training_iterations','omni',
                 'moving_training_sequences','launch_rotation_refinement','launch_response_window_refinement'}
        assert set(c)==set(old)|{'launch_response_window_refinement'}
        for key in set(c)-allowed:assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert response_window['source_checkpoint']==str(source.relative_to(ROOT))
        assert launch_rotation['weight']==old['launch_rotation_refinement']['weight']==4.
        assert launch_rotation['free_rate_rad_s']==old['launch_rotation_refinement']['free_rate_rad_s']==.6
        assert launch_rotation['window_s']==.18
        assert not body_rate
        assert c['moving_training_sequences']==dict(old['moving_training_sequences'],spacing_m=.5,
            step_hold_spacing_m=.4,scope=c['moving_training_sequences']['scope'])
    elif launch_rotation:
        allowed={'version','status','output_series','runner_config_source','omni',
                 'moving_training_sequences','launch_rotation_refinement'}
        assert set(c)==set(old)|{'launch_rotation_refinement'}
        for key in set(c)-allowed:assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert launch_rotation['source_checkpoint']==str(source.relative_to(ROOT))
        assert not body_rate
        expected=dict(old['moving_training_sequences'],positive_x_fraction=.4,step_hold_fraction=.4,
            step_hold_warmup_hops=3,scope=c['moving_training_sequences']['scope'])
        assert c['moving_training_sequences']==expected
    elif body_rate:
        allowed={'version','status','output_series','runner_config_source','omni','moving_body_rate'}
        assert set(c)==set(old)|{'moving_body_rate'}
        for key in set(c)-allowed:assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert body_rate['source_checkpoint']==str(source.relative_to(ROOT))
        assert body_rate['weight']==c['vertical_pretraining']['attitude_refinement']['roll_pitch_rate_weight']==4.
    elif straight_focus:
        # Enforce a single experimental variable. Source/output provenance can
        # change; every physical/task/reward/training parameter must match v22.
        allowed={'version','status','output_series','runner_config_source',
                 'omni','moving_training_sequences','straight_sequence_focus'}
        assert set(c)==set(old)|{'straight_sequence_focus'}
        for key in set(c)-allowed:assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert c['moving_training_sequences']==dict(old['moving_training_sequences'],
            positive_x_fraction=.8,other_heading_fraction=.1,scope=c['moving_training_sequences']['scope'])
        assert straight_focus['source_checkpoint']==str(source.relative_to(ROOT))
        baseline=json.loads((ROOT/straight_focus['source_straight_evaluation']/'summary.json').read_text())
        assert baseline['straight_evaluation']['counts']['strict_sequence_and_survived']==70
        assert baseline['promotion_gate']['passed'] is False
    assert m['status']=='complete' and m['iteration']==int(source.stem.removeprefix('model_'))
    validate_checkpoint_course(source,c)
    for key in ('vertical_pretraining','stationary_pretraining','history_sampling','t_frame_command','stationary_retention'):
        assert c[key]==old[key],key
    assert c['stationary_retention']['training_stationary_fraction']==.5
    assert c['omni']['moving_distance_range_m']==[.12,.5]
    assert c['omni']['promotion']==old['omni']['promotion']
    assert c['omni']['case_version']==old['omni']['case_version']
    assert c['omni']['distance_curriculum']['initial_max_m']==.5
    assert c['vertical_pretraining']['levels'][0]['target_apex_m']==1.
    rc=json.loads((ROOT/c['runner_config_source']).read_text())
    assert rc==json.loads((source.parent/'runner_config.json').read_text())
    if navigation:
        from ..obs73_omni import GroundRoute,navigation_advance
        from ..ring_play import straight_episode_stats
        # Only the physical navigation rule changes; strict rewards/commands
        # retain the source numerical configuration and acceptance thresholds.
        for key in ('moving_phase_reward','moving_landing_guidance','rebound_transition',
                    'moving_yaw_prediction','moving_training_sequences'):
            if key!='moving_training_sequences' or not straight_focus:assert c[key]==old[key],key
        import csv
        recorded=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((ROOT/navigation['diagnosis_source']).open())]
        touch=next(row for row in recorded if row['step']==663)
        peak=max(row['root_z'] for row in recorded if 556<row['step']<=663)
        error=math.hypot(touch['foot_x']-touch['target_x'],touch['foot_y']-touch['target_y'])
        w,x,y,z=[touch[k] for k in ('qw','qx','qy','qz')]
        tilt=math.degrees(math.acos(max(-1.,min(1.,1-2*(x*x+y*y)))))
        assert touch['phase']==3 and touch['spring_q']>.002 and touch['strict']==0
        assert .999<peak<1 and error<.1 and tilt<15
        valid=torch.tensor([True]);strict=torch.tensor([False]);idx=torch.tensor([5]);stationary=torch.tensor([False])
        args=(valid,strict,torch.tensor([error]),torch.tensor([tilt]),idx,stationary)
        assert not navigation_advance(*args).any()
        nav=navigation_advance(*args,navigation);assert nav.item() and not strict.item()
        route=GroundRoute(1,'cpu',c['omni'],7001,evaluation=True);route.reset(torch.tensor([0]));route.index[0]=5
        route.advance(valid,strict,nav)
        assert route.index[0]==6 and route.advances==1 and route.strict_advances==0
        assert route.advances_without_strict==1 and route.misses==1
        for slot,value in ((0,~valid),(2,torch.tensor([.101])),(3,torch.tensor([15.01])),
                           (4,torch.tensor([0])),(5,~stationary)):
            excluded=list(args);excluded[slot]=value
            assert not navigation_advance(*excluded,navigation).any()
        # A reached underheight point must remain a failed jump in playback
        # and in the continuous strict prefix, even when navigation moves on.
        events=[dict(target_index=i,strict=i!=5,navigation_advanced=True,
                     attempt_for_target=1,foot_error_m=.02) for i in range(1,11)]
        result=straight_episode_stats(events,10)
        assert result['navigation_targets_reached']==10 and result['completed_moving_targets']==9
        assert result['first_attempt_strict']==9 and result['initial_consecutive_moving_hits']==4
        assert not result['consecutive_first_attempt_all'] and result['navigation_advances_without_strict']==1
    if anticipation:
        expected=dict(old['moving_phase_reward'])
        for key in ('version','landing_time_s','phase','anticipation_basis'):expected[key]=c['moving_phase_reward'][key]
        if include_holds:
            for key in ('include_route_holds','scope','lateral_velocity'):expected[key]=c['moving_phase_reward'][key]
        if outward:expected['transport_outward_braking']=outward
        if impact:expected['impact_alignment']=impact
        assert c['moving_phase_reward']==expected
        assert c['moving_phase_reward']['version']=='transport_then_landing_reward_v2' and anticipation==.18
        if precision:
            assert precision==dict(version='touchdown_cost_width_only_v1',source_width_m=.1,width_m=.05)
            assert old['moving_landing_guidance']['landing_width_m']==.1
            assert c['moving_landing_guidance']['landing_width_m']==.05
            expected_guide=dict(old['moving_landing_guidance'])
            for key in ('landing_width_m','landing_reward','coefficient_basis'):expected_guide[key]=c['moving_landing_guidance'][key]
            assert c['moving_landing_guidance']==expected_guide
            assert c['moving_phase_reward']==old['moving_phase_reward']
            assert c['omni']['landing_precision_goal_m']==.1
        else:
            expected_guide=dict(old['moving_landing_guidance'])
            if include_holds:expected_guide['scope']=c['moving_landing_guidance']['scope']
            if projection:
                for key in ('objective','projected_error_width_m','velocity_reward'):
                    expected_guide[key]=c['moving_landing_guidance'][key]
            if touchdown_projection:
                expected_guide['projection_origin']=c['moving_landing_guidance']['projection_origin']
            assert c['moving_landing_guidance']==expected_guide
    elif c.get('moving_landing_guidance'):assert c['moving_phase_reward']==old['moving_phase_reward']
    if stability:
        assert c['moving_landing_guidance']==old['moving_landing_guidance']
        assert c['moving_phase_reward']==old['moving_phase_reward']
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert c['moving_precision_refinement']==old['moving_precision_refinement']
    control=c.get('precision_continuation_control')
    if control:
        assert not stability and 'moving_touchdown_stability' not in c
        for key in ('moving_phase_reward','moving_landing_guidance','moving_precision_refinement'):
            assert c[key]==old[key],key
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        comparison=json.loads((ROOT/control['comparison_course']).read_text())
        trained=json.loads((ROOT/control['comparison_training_run']/'manifest.json').read_text())
        assert c['omni']['initial_checkpoint']==comparison['omni']['initial_checkpoint']
        assert c['runner_config_source']==comparison['runner_config_source']
        assert c['training_iterations']==comparison['training_iterations']==control['budget_updates']==100
        assert trained['status']=='complete' and trained['requested_iterations']==100
        assert trained['initial_checkpoint']==str(source.resolve())
        assert control['version']=='v12_same_reward_equal_budget_control_v1'
    if include_holds:
        assert not stability and not control
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    if outward:
        expected=dict(old['moving_phase_reward'],transport_outward_braking=outward)
        if impact:expected['impact_alignment']=impact
        assert c['moving_phase_reward']==expected
        assert outward==dict(start_tilt_deg=10.,full_tilt_deg=20.)
        if not projection:assert c['moving_landing_guidance']==old['moving_landing_guidance']
    if projection:
        expected=dict(old['moving_phase_reward'])
        if impact:expected['impact_alignment']=impact
        assert c['moving_phase_reward']==expected
        assert c['moving_precision_refinement']==old['moving_precision_refinement']
        assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
        assert c['moving_landing_guidance']['objective']=='projected_touchdown_error_v1'
        assert c['moving_landing_guidance']['projected_error_width_m']==c['omni']['landing_precision_goal_m']==.1
        assert old['moving_landing_guidance'].get('objective','velocity_tracking_v1') in ('velocity_tracking_v1','projected_touchdown_error_v1')
        assert c['training_iterations']==100
    if impact:
        assert not old['moving_phase_reward'].get('impact_alignment')
        assert c['moving_landing_guidance']==old['moving_landing_guidance']
        assert impact==dict(weight=4.,free_spring_q_max_m=.001,free_spring_qdot_max_m_s=.05)
        # Retain the evaluated zero-death initialization, not the rejected
        # same-objective continuation. Formal evaluation is read, never rerun.
        report=next(p for p in source.parent.parent.glob('eval*/summary.json')
            if json.loads(p.with_name('manifest.json').read_text()).get('checkpoint')==str(source.resolve())
            and json.loads(p.read_text()).get('task')=='obs73_omni_ground_targets_v1')
        assert json.loads(report.read_text())['counts']['physical_death']==0
    # Eight lifecycle scenarios exercise the actual wrapper around a ledger
    # source, including a queue advance inside its parent reward call.
    class Parent:
        def __init__(self):
            self.course=c;self.num_envs=8;self.device='cpu';self.step_dt=.01;self.extras={}
            self._episode_sums={name:torch.zeros(8) for name in REWEIGHTED}
            for name in ('vertical_yaw_heading','vertical_yaw_rate'):self._episode_sums[name]=torch.zeros(8)
            self.common_step_counter=100
            self._thrust=torch.zeros(8,1,3);self._thrust[1,0,0]=.1
            q=torch.tensor([[1.,0.,0.,0.]]).repeat(8,1);q[4]=torch.tensor([0.,1.,0.,0.])
            spring=torch.zeros(8,1);spring[7]=.003
            p=torch.zeros(8,3);p[:,2]=.8
            foot=torch.zeros(8,1,3);foot[:,:,2]=.812;foot[5,:,2]=.162;foot[7,:,2]=.012
            v=torch.zeros(8,3);v[:,2]=1.;v[5,2]=-1.;v[7,2]=-1.
            foot[3,0,0]=.12;v[3,0]=.3  # Drift on a zero-edge moving-route hold.
            if touchdown_projection:p[3,0]=.12  # Body drift, not just transient leg rotation.
            omega=torch.zeros(8,3);omega[7,:2]=torch.tensor([2.4,1.8])
            self._spring_joint_id=0;self._foot_body_id=0
            self._robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=p,root_quat_w=q,
                joint_pos=spring,joint_vel=torch.zeros_like(spring),body_link_pos_w=foot,root_link_lin_vel_w=v,root_com_lin_vel_w=v.clone(),root_ang_vel_b=omega))
            self._terrain=SimpleNamespace(env_origins=torch.zeros(8,3))
            self._foot_event=SimpleNamespace(phase=torch.tensor([3,3,3,3,3,3,1,3]),DROP=0,FLIGHT=3,
                air_steps=torch.full((8,),10),apex_seen=torch.ones(8,dtype=torch.bool))
            points=torch.zeros(8,4,2);points[:,1,0]=.4;points[:,2,0]=.6;points[3,1]=0.
            self.omni_route=SimpleNamespace(index=torch.tensor([1,1,0,1,1,1,1,1]),points=points,
                stationary_mask=torch.tensor([True,False,False,False,False,False,False,False]),evaluation=False)
            self._two_ground_pt_xy_w=points[torch.arange(8),self.omni_route.index].clone()
            self._two_ground_pt1_xy_w=points[torch.arange(8),self.omni_route.index+1].clone()
            self._desired_pos_w=torch.zeros(8,3);self._desired_pos_w[:,2]=1.
            self.sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0.,0.,-9.81)))
            self.original={name:-(j+1)*torch.ones(8)*.01 for j,name in enumerate(REWEIGHTED)}

        def _get_rewards(self):
            d=self._robot.data
            costs,_,_=yaw_costs(d.root_quat_w,d.root_ang_vel_b[:,2],self._foot_event.phase,
                d.root_quat_w[:,1:3].square().sum(-1)>.5,.01,c['vertical_pretraining']['yaw_refinement'])
            self.original.update(costs)
            for k,v in self.original.items():self._episode_sums[k]+=v
            self.omni_route.index[1]=2  # Parent advances a hit to the next target.
            self._two_ground_pt_xy_w[1,0]=100.  # Must not leak into old-target guidance.
            return torch.stack(list(self.original.values())).sum(0)

        def _reset_idx(self,ids):pass

    env=wrap_env(Parent)();reward=env._get_rewards();a=env._moving_phase_last_audit
    if body_rate:
        # Exercise the production wrapper before its parent advances targets.
        # Hardware flight22 had |omega_xy|=2.77rad/s while returning to HOLD.
        # Both travel and route holds need a cost before late descent starts.
        probe=wrap_env(Parent)();d=probe._robot.data
        d.root_ang_vel_b[:,:2]=torch.tensor([-1.0377210379,2.5670540333])
        probe._get_rewards();audit=probe._moving_phase_last_audit
        value=audit['deltas'].get('moving_transport_body_rate',torch.zeros(8))
        assert value[1]<0 and value[3]<0,'Fast free-flight travel/hold has no body-rate cost'
        assert not value[torch.tensor([0,2,4,6,7])].any(),'Stationary/first-hop/terminal/contact changed'
        for omega in ((0.,0.,0.),(.36,.48,0.),(0.,0.,2.)):
            probe=wrap_env(Parent)();probe._robot.data.root_ang_vel_b[:]=torch.tensor(omega)
            probe._get_rewards()
            assert not probe._moving_phase_last_audit['deltas']['moving_transport_body_rate'].any()
        for spring,spring_rate in ((.0011,0.),(0.,.051)):
            probe=wrap_env(Parent)();probe._robot.data.root_ang_vel_b[:,0]=2.
            probe._robot.data.joint_pos[:]=spring;probe._robot.data.joint_vel[:]=spring_rate
            probe._get_rewards()
            assert not probe._moving_phase_last_audit['deltas']['moving_transport_body_rate'].any()
        # Complementary weight must not double-charge late descent or restore
        # an upright pose target: equal rate gets equal cost at a steady lean.
        from ..moving_phase_reward import transport_body_rate_cost
        def cost(weight,omega=(1.,0.,0.)):
            return transport_body_rate_cost(torch.tensor([omega]),torch.tensor([3]),
                torch.zeros(1),torch.zeros(1),torch.ones(1),torch.ones(1,dtype=torch.bool),
                torch.tensor([weight]),.01,body_rate)[0]
        torch.testing.assert_close(cost(0.),torch.tensor([-4*.4**2*.01]))
        torch.testing.assert_close(cost(.5),cost(0.)*.5)
        assert cost(1.)==0
    if rotation_prediction:
        from ..moving_phase_reward import predicted_max_tilt,predicted_rotation_cost
        from ..t_frame import quaternion_from_rpy,quaternion_product
        for key in ('moving_navigation','moving_phase_reward','moving_landing_guidance',
                    'rebound_transition','moving_yaw_prediction','moving_training_sequences'):
            if key!='moving_training_sequences' or not straight_focus:assert c[key]==old[key],key
        recorded=json.loads((ROOT/rotation_prediction['diagnosis_source']).read_text())
        assert recorded['optimizer_updates']==0 and recorded['case_id']==98
        sample=next(row for row in recorded['trace'] if row['step']==951)
        probe=wrap_env(Parent)();d=probe._robot.data;k=5
        d.root_quat_w[k]=torch.tensor(sample['q']);d.root_ang_vel_b[k]=torch.tensor(sample['omega'])
        d.root_link_pos_w[k]=torch.tensor(sample['root_xy']+[sample['root_z']])
        d.body_link_pos_w[k,0]=torch.tensor(sample['foot_xy']+[sample['foot_clearance']+.012])
        d.root_link_lin_vel_w[k]=torch.tensor(sample['velocity'])
        d.joint_pos[k]=sample['spring'];d.joint_vel[k]=sample['spring_rate'];probe._foot_event.phase[k]=2
        probe._get_rewards();audit=probe._moving_phase_last_audit
        assert sample['reward_terms']['moving_excess_tilt']==0
        cost=audit['deltas'].get('moving_predicted_rotation',torch.zeros(8))
        assert cost[k]<0,'Real launch tipping risk has no anticipatory cone signal before25deg'
        guard=audit['rotation_prediction']
        assert torch.rad2deg(guard['current_tilt'][k])<15.1
        assert torch.rad2deg(guard['predicted_max_tilt'][k])>25
        assert not cost[~guard['active']].any()
        # A finite-grid integration independently checks the analytic interior
        # extrema, including rotations that cross upright or go past pi.
        generator=torch.Generator().manual_seed(7001)
        q=quaternion_from_rpy((torch.rand(24,3,generator=generator)-.5)*1.2)
        omega=(torch.rand(24,3,generator=generator)-.5)*16
        horizon=rotation_prediction['response_horizon_s'];times=torch.linspace(0,horizon,513)
        speed=omega.norm(dim=-1);angle=speed[:,None]*times[None,:]
        vector=omega[:,None,:]*(times[None,:,None]/2)*torch.sinc(angle[:,:,None]/(2*math.pi))
        delta=torch.cat((torch.cos(angle[:,:,None]/2),vector),-1)
        future=quaternion_product(q[:,None,:],delta)
        numeric=torch.acos((1-2*future[:,:,1:3].square().sum(-1)).clamp(-1,1)).max(-1).values
        analytic=predicted_max_tilt(q,omega,horizon)
        assert (analytic>=numeric-1e-4).all()
        assert (analytic-numeric<=speed*horizon/512+1e-4).all()
        identity=torch.tensor([[1.,0.,0.,0.]])
        torch.testing.assert_close(predicted_max_tilt(identity,torch.tensor([[0.,40.,0.]]),horizon),torch.tensor([math.pi]))
        # Stable useful lean, pure body yaw, and gentle inward correction are
        # free; rotation through upright toward the opposite unsafe cone is not.
        pose=quaternion_from_rpy(torch.tensor([[0.,math.radians(15),0.]]).repeat(4,1))
        rates=torch.tensor([[0.,0.,0.],[0.,0.,8.],[0.,-1.5,0.],[0.,-6.,0.]])
        args=(pose,rates,torch.full((4,),3),torch.zeros(4),torch.zeros(4),
            torch.ones(4),torch.ones(4,dtype=torch.bool),25.,.01,rotation_prediction)
        values,_=predicted_rotation_cost(*args)
        torch.testing.assert_close(values[:3],torch.zeros(3))
        assert values[3]<0
        neg,_=predicted_rotation_cost(-pose,*args[1:]);torch.testing.assert_close(neg,values)
        for slot,value in ((2,torch.ones(4,dtype=torch.long)),(3,torch.full((4,),.003)),
                           (4,torch.full((4,),.1)),(5,torch.zeros(4)),(6,torch.zeros(4,dtype=torch.bool))):
            excluded=list(args);excluded[slot]=value
            assert not predicted_rotation_cost(*excluded)[0].any()
    torch.testing.assert_close(a['moving'],torch.tensor([False,True,False,include_holds,False,True,True,True]))
    expected_late=1-(math.sqrt(1+2*9.81*.15)-1)/9.81/anticipation if anticipation else .5
    torch.testing.assert_close(a['weight'][[1,5,6,7]],torch.tensor([0.,expected_late,1.,1.]))
    base_reward=torch.stack(list(env.original.values())).sum(0)
    torch.testing.assert_close(reward[~a['moving']],base_reward[~a['moving']])
    for key in REWEIGHTED:
        factor=a['weight'] if key in PHASE_COSTS else torch.zeros(8)
        expected=torch.where(a['moving'],factor*env.original[key],env.original[key])
        if rebound and key in ('attitude_penalty','vertical_tilt'):
            expected=torch.where(a['rebound']['preparation_active'],0.,expected)
        torch.testing.assert_close(env._episode_sums[key],expected)
    if touchdown_projection:
        # Actual reward wrapper, same A-position/velocity/next goal, different
        # airborne leg orientation. A reference-touchdown forecast must not
        # mistake rotation of the current foot for progress of the body.
        from ..ring_geometry import rotate
        forecasts=[];touchdown_errors=[]
        for angle in (-20.,20.):
            probe=wrap_env(Parent)();d=probe._robot.data;k=5
            half=math.radians(angle)/2
            d.root_quat_w[k]=torch.tensor([math.cos(half),0.,math.sin(half),0.])
            d.root_link_lin_vel_w[k]=torch.tensor([.4,0.,-.5])
            d.body_link_pos_w[k,0]=d.root_link_pos_w[k]-rotate(d.root_quat_w[k],torch.tensor([0.,0.,.285]))
            probe._get_rewards();g=probe._moving_phase_last_audit['guidance']
            forecasts.append(g['projected_foot_xy'][k]);touchdown_errors.append(g['foot_error_m'][k])
        torch.testing.assert_close(forecasts[0],forecasts[1],atol=1e-6,rtol=1e-6)
        assert abs(float(touchdown_errors[0]-touchdown_errors[1]))>.1,'Measured foot scorer was replaced'
        from ..moving_phase_reward import touchdown_projection_origin
        import csv
        observed=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(
            (ROOT/touchdown_projection['diagnosis_source']).open())]
        bad=next(row for row in observed if row['step']==340)
        assert bad['strict']==0 and bad['root_z']<.3
        at=next(row for row in observed if row['step']==260)
        root=torch.tensor([[at[k] for k in ('root_x','root_y','root_z')]])
        foot=torch.tensor([[at[k] for k in ('foot_x','foot_y','foot_z')]])
        velocity=torch.tensor([[at[k] for k in ('vx','vy','vz')]])
        normal,_=rebound_approach(root,foot,velocity,torch.tensor([[.4,0.]]),torch.tensor([.06]),rebound)
        origin=touchdown_projection_origin(root,foot,torch.zeros(1),normal)
        assert origin[0,0]-foot[0,0]>.1,'Real overshoot precursor no longer reproduced'
        mask=torch.tensor([True])
        args=(foot[:,:2],foot[:,2]-.012,velocity,root[:,2],
            torch.tensor([[at['target_x'],at['target_y']]]),torch.ones(1),
            torch.tensor([3]),mask,~mask,9.81,.01,c['moving_landing_guidance'],torch.tensor([.9]))
        old_terms,old_forecast=landing_guidance(*args)
        new_terms,new_forecast=landing_guidance(*args,projection_xy=origin)
        assert new_forecast['projected_error_m']>old_forecast['projected_error_m']
        assert new_terms['moving_velocity_guidance']<old_terms['moving_velocity_guidance']
        torch.testing.assert_close(new_forecast['foot_error_m'],old_forecast['foot_error_m'])
        torch.testing.assert_close(new_terms['moving_landing_error'],old_terms['moving_landing_error'])
        # Compression changes the instantaneous link length, not its restored
        # length for the following flight; neither term changes observed foot.
        compressed=foot+.03*(root-foot)/(root-foot).norm(dim=-1,keepdim=True)
        torch.testing.assert_close(touchdown_projection_origin(root,compressed,torch.tensor([.03]),normal),origin)
    if include_holds:
        legacy=wrap_env(Parent)()
        legacy.moving_phase_config=dict(legacy.moving_phase_config,include_route_holds=False)
        legacy_reward=legacy._get_rewards()
        assert not legacy._moving_phase_last_audit['moving'][3] and a['moving'][3]
        torch.testing.assert_close(reward[[0,1,2,4,5,6,7]],legacy_reward[[0,1,2,4,5,6,7]])
        assert a['guidance']['desired_velocity_xy'][3,0]<0
        assert env._episode_sums['moving_velocity_guidance'][3]<0
        # A subsequent nonzero goal stays in the same task scope. A pure
        # stationary member at the same hold still receives exactly zero deltas.
        transition=wrap_env(Parent)();transition.omni_route.index[3]=2
        transition._get_rewards();assert transition._moving_phase_last_audit['moving'][3]
        stationary=wrap_env(Parent)();stationary.omni_route.stationary_mask[3]=True
        stationary._get_rewards()
        assert not any(v[3] for v in stationary._moving_phase_last_audit['deltas'].values())
    # Cost is gradual on descent, never applied just because near the ground
    # during ascent; stance/contact keep the original pose stabilization.
    w=late_weight(torch.full((4,),3),torch.zeros(4),torch.tensor([.31,.3,.15,0.]),-torch.ones(4),.3)
    torch.testing.assert_close(w,torch.tensor([0.,0.,.5,1.]))
    assert not late_weight(torch.full((4,),3),torch.zeros(4),torch.zeros(4),torch.ones(4),.3).any()
    if anticipation:
        # At45cm and3m/s downward, the old30cm rule gives zero cost
        # weight. Time anticipation begins before that legacy window.
        old_weight=late_weight(torch.tensor([3]),torch.zeros(1),torch.tensor([.45]),torch.tensor([-3.]),.3)
        new_weight=late_weight(torch.tensor([3]),torch.zeros(1),torch.tensor([.45]),torch.tensor([-3.]),.3,anticipation)
        assert old_weight.item()==0 and 0<new_weight.item()<1
        # Same time-to-contact, different descent speed/clearance: same
        # anticipation, which fixed height cannot express. Test all phases.
        time=torch.tensor([.25,.18,.09,0.]);speed=torch.tensor([2.,3.,4.,3.])
        height=speed*time+.5*9.81*time.square()
        weight=late_weight(torch.full((4,),3),torch.zeros(4),height,-speed,.3,anticipation)
        torch.testing.assert_close(weight,torch.tensor([0.,0.,.5,1.]),atol=1e-6,rtol=1e-6)
        assert not late_weight(torch.full((4,),3),torch.zeros(4),height,speed,.3,anticipation).any()
        for phase in (0,1,2,3):
            weight=late_weight(torch.full((4,),phase),torch.zeros(4),height,-speed,.3,anticipation)
            assert torch.isfinite(weight).all() and ((weight>=0)&(weight<=1)).all()
            if phase==0:assert not weight.any()
            if phase==1:assert (weight==1).all()
        assert torch.isfinite(late_weight(torch.tensor([3]),torch.zeros(1),torch.zeros(1),torch.zeros(1),.3,anticipation)).all()
    # Quantify the conflicting original objectives using the real old kernel:
    # 15deg lean for0.8s and0.5m/0.8s translation, not a simulated trajectory.
    q=torch.tensor([[math.cos(math.radians(15)/2),0.,math.sin(math.radians(15)/2),0.]])
    old_tilt=attitude_costs(q,torch.zeros(1,3),torch.tensor([3]),torch.tensor([False]),.8,
        old['vertical_pretraining']['attitude_refinement'])['vertical_tilt']
    old_velocity=torch.tensor([-10*(.5/.8)**2*.8])
    d=reward_deltas({'vertical_tilt':old_tilt,'lateral_vel_p':old_velocity},torch.tensor([True]),
        torch.tensor([0.]),torch.tensor([math.radians(15)]),.8,c['moving_phase_reward'])
    torch.testing.assert_close(old_tilt+d['vertical_tilt'],torch.zeros(1))
    torch.testing.assert_close(old_velocity+d['lateral_vel_p'],torch.zeros(1))
    large=reward_deltas({},torch.tensor([True,False]),torch.zeros(2),
        torch.tensor([math.radians(35),math.radians(35)]),.01,c['moving_phase_reward'])
    assert large['moving_excess_tilt'][0]<0 and large['moving_excess_tilt'][1]==0
    if outward:
        # Replay the actual failure-state reward seam: at15.8deg cone tilt,
        # the original outward kernel is negative but the legacy phase rule
        # cancels it. Earlier outward braking retains a fraction of that cost.
        trace=json.loads((ROOT/c['diagnosis_source']).read_text())['trace']
        snap=next(row for row in trace if row['step']==1014)
        q=torch.tensor([snap['q']]);omega=torch.tensor([snap['omega']])
        args=(q,omega,torch.tensor([snap['phase']]),torch.tensor([snap['spring']]),
            torch.tensor([snap['spring_rate']]),torch.tensor([False]),
            torch.tensor([snap['foot_clearance']]),torch.tensor([snap['velocity'][2]]),.01,
            c['vertical_pretraining']['flight_braking'])
        original=flight_braking_costs(*args)['vertical_outward_rotation']
        tilt=torch.acos(1-2*q[:,1:3].square().sum(-1));zero=torch.zeros(1);mask=torch.tensor([True])
        assert original.item()<0 and snap['guidance']['stabilization_weight']==0
        legacy_config=dict(old['moving_phase_reward']);legacy_config.pop('transport_outward_braking',None)
        legacy=reward_deltas({'vertical_outward_rotation':original},mask,zero,tilt,.01,legacy_config)
        updated=reward_deltas({'vertical_outward_rotation':original},mask,zero,tilt,.01,c['moving_phase_reward'])
        torch.testing.assert_close(original+legacy['vertical_outward_rotation'],zero)
        assert (original+updated['vertical_outward_rotation']).item()<0
        inward_args=list(args);inward_args[1]=-omega
        assert flight_braking_costs(*inward_args)['vertical_outward_rotation'].item()==0
        fixed_args=list(args);fixed_args[1]=torch.zeros_like(omega)
        assert flight_braking_costs(*fixed_args)['vertical_outward_rotation'].item()==0
        torch.testing.assert_close(outward_weight(torch.tensor([0.,.2,.8,0.]),
            torch.deg2rad(torch.tensor([9.,15.,15.,25.])),c['moving_phase_reward']),torch.tensor([0.,.5,.8,1.]))
    guidance_checked=False
    if c.get('moving_landing_guidance'):
        g=c['moving_landing_guidance']
        assert env._moving_phase_last_audit['guidance']['desired_velocity_xy'][1,0]<1.2
        assert env._episode_sums['moving_landing_error'][7]<0
        assert not env._episode_sums['moving_landing_error'][:7].any()
        # A physically consistent launch-time reference: target0.5m in the
        # nominal ballistic cycle. Correct translation is preferred to zero
        # velocity; this directly tests the old anti-motion objective's seam.
        foot=torch.zeros(3,2);goal=torch.tensor([[.5,0.]]).repeat(3,1)
        root=torch.full((3,),.285);target=torch.ones(3)
        cycle=2*math.sqrt(2*(1.+g['apex_margin_m']-.285)/9.81)
        velocity=torch.zeros(3,3);velocity[0,0]=.5/cycle;velocity[2,0]=-.5/cycle
        mask=torch.ones(3,dtype=torch.bool);valid=torch.zeros(3,dtype=torch.bool)
        costs,info=landing_guidance(foot,torch.zeros(3),velocity,root,goal,target,
            torch.ones(3,dtype=torch.long),mask,valid,9.81,.01,g)
        assert abs(costs['moving_velocity_guidance'][0])<1e-7
        assert costs['moving_velocity_guidance'][0]>costs['moving_velocity_guidance'][1]>costs['moving_velocity_guidance'][2]
        torch.testing.assert_close(info['remaining_time_s'],torch.full((3,),cycle))
        # On a retry that overshot the target, reverse the desired velocity
        # using actual position, not the original0.5m edge or target index.
        retry_foot=goal+torch.tensor([[.2,0.]]).repeat(3,1)
        _,retry=landing_guidance(retry_foot,torch.zeros(3),velocity,root,goal,target,
            torch.ones(3,dtype=torch.long),mask,valid,9.81,.01,g)
        assert (retry['desired_velocity_xy'][:,0]<0).all()
        # Vector norm clipping and end-of-descent fade prevent a singular
        # late-flight request; the event penalty still sees large inaccuracies.
        costs,last=landing_guidance(foot,torch.zeros(3),torch.tensor([[0.,0.,-3.]]).repeat(3,1),root,
            torch.tensor([[10.,10.]]).repeat(3,1),target,torch.full((3,),3),mask,
            torch.tensor([True,False,True]),9.81,.01,g)
        torch.testing.assert_close(last['desired_velocity_xy'].norm(dim=-1),torch.full((3,),g['speed_limit_m_s']))
        assert torch.isfinite(last['remaining_time_s']).all() and not costs['moving_velocity_guidance'].any()
        assert costs['moving_landing_error'][0]<0 and costs['moving_landing_error'][1]==0
        costs,_=landing_guidance(foot,torch.ones(3),velocity,root,goal,target,torch.full((3,),3),
            torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),9.81,.01,g)
        assert not any(v.any() for v in costs.values())
        guidance_checked=True
        if projection:
            legacy_guide=dict(old['moving_landing_guidance'])
            legacy_guide.pop('objective',None);legacy_guide.pop('projected_error_width_m',None)
            # Equal forecast position misses must not be discounted by1/T²
            # early in flight. Exercise the real helper at two horizons.
            time=torch.tensor([.3,.6]);height=.5*9.81*time.square()
            args=(torch.zeros(2,2),height,torch.zeros(2,3),torch.ones(2),
                torch.tensor([[.1,0.],[.1,0.]]),torch.ones(2),torch.full((2,),3),
                torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),9.81,.01)
            now,forecast=landing_guidance(*args,g)
            previous,_=landing_guidance(*args,legacy_guide)
            torch.testing.assert_close(now['moving_velocity_guidance'][0],now['moving_velocity_guidance'][1])
            torch.testing.assert_close(previous['moving_velocity_guidance'][0],4*previous['moving_velocity_guidance'][1])
            # Actual formal failure, exactly reproduced by the bounded trace:
            # atstep255 the foot is still before the goal, but momentum already
            # forecasts an overshoot. A corrected translating velocity costs0;
            # stopping is not required. This does not simulate a corrected hop.
            trace=json.loads((ROOT/projection['diagnosis_source']).read_text())
            assert trace['optimizer_updates']==0 and trace['case_id']==76
            snap=next(row for row in trace['trace'] if row['step']==255)
            foot=torch.tensor([snap['foot_xy']]);goal=torch.tensor([snap['target_xy']])
            velocity=torch.tensor([snap['velocity']]);mask=torch.tensor([True])
            args=(foot,torch.tensor([snap['foot_clearance']]),velocity,torch.tensor([snap['root_z']]),
                goal,torch.ones(1),torch.tensor([snap['phase']]),mask,~mask,9.81,.01)
            now,forecast=landing_guidance(*args,g)
            previous,_=landing_guidance(*args,legacy_guide)
            assert forecast['projected_foot_xy'][0,0]<goal[0,0]<foot[0,0]
            assert now['moving_velocity_guidance'].item()<previous['moving_velocity_guidance'].item()<0
            corrected=velocity.clone();corrected[:,:2]=(goal-foot)/forecast['remaining_time_s'][:,None]
            corrected_args=list(args);corrected_args[2]=corrected
            aligned,_=landing_guidance(*corrected_args,g)
            assert corrected[:,:2].norm()>0 and aligned['moving_velocity_guidance'].abs().item()<1e-7
            torch.testing.assert_close(now['moving_landing_error'],previous['moving_landing_error'])
            # Runtime-relevant bad configuration must fail before simulation.
            for bad in ({'objective':'typo'}, {'projected_error_width_m':0.},
                        {'projected_error_width_m':float('nan')}):
                class Invalid(Parent):
                    def __init__(self):
                        super().__init__();self.course=dict(c,moving_landing_guidance=dict(g,**bad))
                try:wrap_env(Invalid)()
                except ValueError:pass
                else:raise AssertionError('Invalid projected guidance accepted')
        if precision:
            # Exercise the actual guide at inaccurate/underheight touchdown:
            # narrower width strengthens its event cost without changing the
            # desired velocity, timing, continuous cost or nonmoving rewards.
            errors=torch.tensor([.0,.05,.1,.2,.5])
            goal=torch.stack((errors,torch.zeros_like(errors)),-1)
            zero=torch.zeros(5);mask=torch.ones(5,dtype=torch.bool)
            args=(torch.zeros(5,2),zero,torch.zeros(5,3),torch.full((5,),.285),goal,
                torch.ones(5),torch.full((5,),3),mask,mask,9.81,.01)
            new,info=landing_guidance(*args,g)
            previous,old_info=landing_guidance(*args,old['moving_landing_guidance'])
            torch.testing.assert_close(new['moving_velocity_guidance'],previous['moving_velocity_guidance'])
            for key in info:torch.testing.assert_close(info[key],old_info[key])
            assert new['moving_landing_error'][0]==0
            assert (new['moving_landing_error'][1:]<previous['moving_landing_error'][1:]).all()
            torch.testing.assert_close(new['moving_landing_error'][1:4],torch.tensor([-.8284271,-2.472136,-6.246211]),atol=1e-5,rtol=1e-5)
            assert env._episode_sums['moving_landing_error'][7]<-14.
    if stability:
        assert (env._episode_sums['moving_touchdown_rate'][:7]==0).all()
        assert env._episode_sums['moving_touchdown_rate'][7]<-3.
        omega=torch.tensor([[0.,0.,20.],[.2,0.,0.],[.35,0.,0.],[1.,0.,0.],[0.,1.,0.],[3.,0.,0.],[3.,0.,0.],[3.,0.,0.]])
        moving=torch.tensor([True]*7+[False]);valid=torch.tensor([True]*6+[False,True])
        cost=touchdown_rate_cost(omega,moving,valid,stability)
        assert not cost[:3].any() and not cost[6:].any()
        assert cost[3]==cost[4] and cost[5]<cost[3]<0
        assert torch.isfinite(cost).all() and (cost<=0).all()
    if impact:
        # Replay the actual precontact state: modest angular rate is followed
        # by an impact spike, while the new proxy detects transverse approach
        # before that impulse. This is not a corrected-trajectory simulation.
        trace=json.loads((ROOT/c['impact_alignment_refinement']['diagnosis_source']).read_text())
        assert trace['optimizer_updates']==0 and trace['case_id']==32
        pre=next(r for r in trace['trace'] if r['step']==756)
        hit=next(r for r in trace['trace'] if r['step']==757)
        assert math.hypot(*pre['omega'][:2])<.5 and math.hypot(*hit['omega'][:2])>4.
        root=torch.tensor([pre['root_xy']+[pre['root_z']]])
        foot=torch.tensor([pre['foot_xy']+[pre['foot_clearance']+.012]])
        velocity=torch.tensor([pre['velocity']]);mask=torch.tensor([True])
        args=(root,foot,velocity,torch.tensor([pre['phase']]),torch.tensor([pre['spring']]),
            torch.tensor([pre['spring_rate']]),torch.tensor([pre['foot_clearance']]),mask,
            torch.tensor([pre['guidance']['stabilization_weight']]),.01,impact)
        cost,info=impact_alignment_cost(*args)
        assert info['active'].item() and info['transverse_speed_m_s'].item()>1.2 and cost.item()<0
        # A leaned leg aligned with the same descending/nonzero horizontal
        # velocity has no transverse cost; zero lean is not the objective.
        aligned_args=list(args)
        aligned_args[1]=root+velocity/velocity.norm()* .285
        aligned,info=impact_alignment_cost(*aligned_args)
        assert velocity[0,:2].norm()>.7 and aligned.abs().item()<1e-7
        for slot,value in [(3,torch.tensor([0])),(3,torch.tensor([1])),
                           (4,torch.tensor([.003])),(5,torch.tensor([.1])),
                           (6,torch.tensor([0.])),(7,~mask),(8,torch.tensor([0.]))]:
            excluded=list(args);excluded[slot]=value
            assert not impact_alignment_cost(*excluded)[0].any()
        ascending=list(args);ascending[2]=velocity.clone();ascending[2][:,2].abs_()
        assert not impact_alignment_cost(*ascending)[0].any()
        assert not env._episode_sums['moving_impact_alignment'][~a['moving']].any()
    if rebound:
        from ..t_frame import quaternion_from_rpy,quaternion_product
        from ..stationary_retention import RetentionRoute
        assert not impact and c['moving_phase_reward']==old['moving_phase_reward']
        expected_guide=dict(old['moving_landing_guidance'])
        if touchdown_projection:
            expected_guide['projection_origin']='root_A_with_reference_touchdown_leg_v1'
        assert c['moving_landing_guidance']==expected_guide
        # Actual camera observation is reproducible in the saved trace; timing
        # after contact cannot beat a motor response longer than the stance.
        import csv
        trace=[{k:float(v) for k,v in row.items()} for row in csv.DictReader((ROOT/rebound['diagnosis_source']).open())]
        touch=next(row for row in trace if row['step']==672)
        release=next(row for row in trace if row['step']>672 and row['spring_q']<=.001 and row['phase']>=2)
        assert .01*(release['step']-touch['step'])<c['moving_yaw_prediction']['response_horizon_s']
        assert touch['vx']>.35 and release['vx']<0
        # Neutral foot placement carries desired horizontal speed; requesting
        # acceleration shifts the foot back. It is not radial incoming reversal.
        root=torch.tensor([[0.,0.,.285]]);foot=torch.zeros(1,3)
        incoming=torch.tensor([[.4,0.,-3.4]]);stance=torch.tensor([.06])
        normal,lead=rebound_approach(root,foot,incoming,torch.tensor([[.4,0.]]),stance,rebound)
        torch.testing.assert_close(lead,torch.tensor([[.012,0.]]))
        assert normal[0,0]<0 and normal[0,2]>.99
        _,faster=rebound_approach(root,foot,incoming,torch.tensor([[.5,0.]]),stance,rebound)
        _,slower=rebound_approach(root,foot,incoming,torch.tensor([[.3,0.]]),stance,rebound)
        assert faster[0,0]<lead[0,0]<slower[0,0]
        normal,_=rebound_approach(root,foot,incoming,torch.tensor([[-100.,100.]]),stance,rebound)
        torch.testing.assert_close(normal.norm(dim=-1),torch.ones(1))
        assert torch.rad2deg(torch.acos(normal[0,2]))<=rebound['touchdown_tilt_limit_deg']+1e-4
        # Fixed body wz=0 can still change world heading during roll/pitch.
        q=quaternion_from_rpy(torch.tensor([[.3,.2,0.]]));omega=torch.tensor([[0.,1.,0.]])
        heading,rate=heading_motion(q,omega);eps=1e-4
        dq=torch.cat((torch.zeros(1,1),omega),-1)
        q2=q+.5*eps*quaternion_product(q,dq);q2/=q2.norm(dim=-1,keepdim=True)
        h2,_=heading_motion(q2,omega)
        torch.testing.assert_close((h2-heading)/eps,rate,atol=5e-4,rtol=5e-4)
        assert omega[0,2]==0 and rate[0]>.25
        hneg,rneg=heading_motion(-q,omega)
        torch.testing.assert_close(hneg,heading);torch.testing.assert_close(rneg,rate)
        # Retry launches target the missed CURRENT point; successful contact
        # latches the old NEXT point even after the parent's queue has changed.
        torch.testing.assert_close(env.rebound_plan.velocity[7,0],torch.tensor(.4)/env.rebound_plan.flight[7])
        class Hit(Parent):
            def _get_rewards(self):
                value=super()._get_rewards();self.omni_route.index[7]=2
                self._two_ground_pt_xy_w[7,0]=100.
                return value
        hit=wrap_env(Hit)();hit._robot.data.body_link_pos_w[7,0,0]=.4
        hit._get_rewards();target=hit.rebound_plan.velocity[7].clone()
        torch.testing.assert_close(target[0],torch.tensor(.2)/hit.rebound_plan.flight[7])
        assert hit.rebound_plan.pending[7]
        hit._foot_event.phase[7]=2;hit._robot.data.joint_pos[7]=0.
        hit._robot.data.body_link_pos_w[7,0,2]=.05;hit._robot.data.root_link_lin_vel_w[7,2]=2.
        hit.common_step_counter=106;hit._get_rewards()
        assert hit._moving_phase_last_audit['rebound']['launch'][7]
        assert hit._moving_phase_last_audit['deltas']['moving_launch_velocity'][7]<0
        if launch_endpoint:
            # Exercise the real contact->free-liftoff call site; historical
            # velocity-squared cost cannot enforce a distance-scaled objective.
            distance=(hit._robot.data.root_link_lin_vel_w[7,:2]-target).norm()*hit.rebound_plan.flight[7]
            expected=-rebound['launch_weight']*(torch.sqrt(1+(distance/launch_endpoint['width_m'])**2)-1)
            torch.testing.assert_close(hit._moving_phase_last_audit['deltas']['moving_launch_velocity'][7],expected)
            from ..moving_phase_reward import launch_velocity_cost
            # A moving launch at the commanded velocity must remain free;
            # opposite/cardinal errors and identical endpoint errors have
            # identical costs, including zero (finite derivative).
            target_xy=torch.tensor([[.5,0.]]).expand(6,-1)
            delta=torch.tensor([[0.,0.],[.1,0.],[-.1,0.],[0.,.1],[.2,0.],[10.,0.]],requires_grad=True)
            times=torch.tensor([1.,1.,1.,1.,.5,1.])
            active=torch.tensor([True,True,True,True,True,False])
            costs=launch_velocity_cost(target_xy+delta,target_xy,times,active,4.,launch_endpoint)
            assert costs[0]==costs[5]==0
            torch.testing.assert_close(costs[1:5],costs[1].expand(4))
            costs.sum().backward();assert torch.isfinite(delta.grad).all()
            assert not delta.grad[[0,5]].any()
            legacy=launch_velocity_cost(target_xy+delta,target_xy,times,active,4.)
            torch.testing.assert_close(legacy,torch.where(active,-4*delta.square().sum(-1),0.))
        if launch_rotation:
            cost=hit._moving_phase_last_audit['deltas'].get('moving_launch_rotation',torch.zeros(8))
            assert cost[7]<0,'Fast contact-to-liftoff rotation has no event cost'
            assert not cost[:7].any(),'Launch cost leaked into nonlaunch lifecycle samples'
            from ..moving_phase_reward import launch_rotation_cost
            mask=torch.tensor([True,True,True,True,False])
            omega=torch.tensor([[0.,0.,10.],[.36,.48,0.],[1.,0.,0.],[0.,-1.,0.],[2.,0.,0.]])
            value=launch_rotation_cost(omega,mask,launch_rotation)
            assert not value[[0,1,4]].any()
            torch.testing.assert_close(value[2:4],torch.full((2,),-4*.4**2))
        hit._get_rewards()
        assert not hit._moving_phase_last_audit['rebound']['launch'][7]
        assert not hit._moving_phase_last_audit['deltas']['moving_launch_velocity'][7]
        if launch_rotation and not response_window:
            assert not hit._moving_phase_last_audit['deltas']['moving_launch_rotation'].any(),'Same launch charged twice'
        if response_window:
            # Real case127 peaked80ms AFTER release; the original event-only
            # charge misses this actual delayed free-flight rotation pattern.
            hit.common_step_counter=114
            hit._robot.data.root_ang_vel_b[7]=torch.tensor(response_window['rate_peak_omega_body_rad_s'])
            hit._get_rewards()
            assert not hit._moving_phase_last_audit['rebound']['launch'][7]
            assert hit._moving_phase_last_audit['deltas']['moving_launch_rotation'][7]<0,'Delayed post-launch rate peak has no cost'
            peak=math.hypot(*response_window['rate_peak_omega_body_rad_s'][:2])
            torch.testing.assert_close(hit._moving_phase_last_audit['deltas']['moving_launch_rotation'][7],
                torch.tensor(-4*(peak-.6)**2*.01/.18))
            hit._foot_event.phase[7]=1;hit._get_rewards()
            assert not hit._moving_phase_last_audit['deltas']['moving_launch_rotation'].any(),'Contact receives free-flight window cost'
            hit._foot_event.phase[7]=2
            hit.common_step_counter=123;hit._get_rewards()
            assert hit._moving_phase_last_audit['deltas']['moving_launch_rotation'][7]<0
            hit.common_step_counter=124;hit._get_rewards()
            assert not hit._moving_phase_last_audit['deltas']['moving_launch_rotation'].any(),'Launch response window never expires'
        before=hit.rebound_plan.flight.clone();hit._reset_idx(torch.tensor([7]))
        assert not hit.rebound_plan.pending[7] and not hit.rebound_plan.totals[7].any()
        assert not hit.body_rate_totals[7].any()
        assert hit.rebound_plan.launch_step[7]==-1
        torch.testing.assert_close(hit.rebound_plan.flight[:7],before[:7])
        # The physical clock floor addresses the actual under-estimated early
        # flight clock without changing legacy behavior when omitted.
        snap=next(row for row in trace if row['step']==260)
        mask=torch.tensor([True]);args=(torch.tensor([[snap['foot_x'],snap['foot_y']]]),
            torch.tensor([snap['foot_z']-.012]),torch.tensor([[snap[k] for k in ('vx','vy','vz')]]),
            torch.tensor([snap['root_z']]),torch.tensor([[snap['target_x'],snap['target_y']]]),
            torch.ones(1),torch.tensor([3]),mask,~mask,9.81,.01,c['moving_landing_guidance'])
        _,early=landing_guidance(*args);_,corrected=landing_guidance(*args,torch.tensor([.9]))
        assert corrected['remaining_time_s'][0]>=.9>early['remaining_time_s'][0]
        route=RetentionRoute(768,'cpu',c['omni'],42,training_mixture=c['moving_training_sequences'])
        route.reset(torch.arange(768));stationary=route.stationary_mask
        assert stationary.sum()==384 and not route.points[stationary].any()
        kinds=route.sequence_kind[~stationary]
        assert all((kinds==k).sum()>(10 if straight_focus else 40) for k in (0,1,2))
        if straight_focus and not launch_rotation:
            assert (kinds==1).sum()>270
            assert .35<((route.sequence_kind==1)&~stationary).float().mean()<.45
        edges=route.points.diff(dim=1)
        for kind in (1,2):
            selected=(route.sequence_kind==kind)&~stationary
            torch.testing.assert_close(edges[selected].norm(dim=-1),torch.full_like(edges[selected,:,0],c['moving_training_sequences']['spacing_m']),atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(edges[selected],edges[selected,:1].expand_as(edges[selected]),atol=2e-6,rtol=1e-5)
            if kind==1:assert not route.points[selected,:,1].any() and (edges[selected,:,0]>0).all()
        if launch_rotation:
            selected=(route.sequence_kind==3)&~stationary
            assert 100<selected.sum()<200
            assert not route.points[selected,:3].any()
            torch.testing.assert_close(edges[selected,2].norm(dim=-1),torch.full_like(edges[selected,2,0],.4))
            assert not edges[selected,:2].any() and not edges[selected,3:].any()
            targets=route.points[selected,3]
            assert ((targets.abs()<1e-6).sum(-1)==1).all()
            for axis in (0,1):
                assert (targets[:,axis]>.39).any() and (targets[:,axis]<-.39).any()
        old_eval=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True)
        new_eval=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True,training_mixture=c['moving_training_sequences'])
        old_eval.reset(torch.arange(128));new_eval.reset(torch.arange(128))
        assert torch.equal(old_eval.points,new_eval.points),'Formal eval distribution changed'
        plan=ReboundPlan(64,'cpu',42,rebound)
        assert plan.horizon.min()>=.30 and plan.horizon.max()<=.45 and plan.horizon.std()>0
        plan.reset(torch.arange(64),True)
        torch.testing.assert_close(plan.horizon,torch.full((64,),.375))
        report_rows=[dict(rebound_totals=[10.,.5,.01,10.,math.radians(2)**2*10,.4,10.,math.radians(8),1.]) for _ in range(128)]
        report=rebound_motion_metrics(report_rows,c['moving_yaw_prediction'])
        assert report['yaw_gate']['passed'] and abs(report['moving_heading_rms_deg']-2)<1e-6
        assert report['launch_events']==1280 and abs(report['mean_air_horizontal_force_N']-.05)<1e-6
        report_rows[0]['rebound_totals'][7]=math.radians(14)
        assert not rebound_motion_metrics(report_rows,c['moving_yaw_prediction'])['yaw_gate']['passed']
        report_rows[0]['rebound_totals']=[0.]*9
        assert not rebound_motion_metrics(report_rows,c['moving_yaw_prediction'])['metric_coverage_complete']
        assert not rebound_motion_metrics([],c['moving_yaw_prediction'])['yaw_gate']['passed']
    # Repeated contact/retries, failed holds, missing prefixes and later
    # deaths must not masquerade as continuous successful moving jumps.
    def event(index,strict=True,moving=True,first=True):
        return dict(target_index=index,jump_index=index+1,strict=strict,moving=moving,
            first_attempt_for_target=first,foot_error_m=.01,touchdown_tilt_deg=2.)
    rows=[dict(events=[event(0,moving=False)]+[event(i) for i in range(1,11)],physical_death=False),
          dict(events=[event(1),event(2),event(3,False),event(3,first=False)]+[event(i) for i in range(4,11)],physical_death=False),
          dict(events=[event(1),event(2,moving=False),event(3),event(4)],physical_death=True),
          dict(events=[event(1),event(2,False,moving=False),event(3),event(4)],physical_death=False)]
    continuity=continuity_metrics(rows)
    assert continuity['prefix']['3']['strict_episodes']==2
    assert continuity['prefix']['3']['strict_and_survived_episodes']==1
    assert continuity['prefix']['5']['strict_episodes']==1
    assert continuity['prefix']['10']['strict_episodes']==1
    assert continuity['prefix']['10']['missing_or_failed_episodes']==3
    assert continuity['physical_deaths']==1
    clean=dict(events=[event(0,moving=False),event(1),event(2)],physical_death=False)
    clean['events'][1]['foot_error_m']=.1
    assert landing_precision_metrics([clean])['passed']
    clean['events'][1]['foot_error_m']=.100001
    result=landing_precision_metrics([clean])
    assert not result['passed'] and result['moving_first_attempt']['outside_goal_count']==1
    clean['events'][1]['foot_error_m']=.05
    clean['events'][2].update(moving=False,foot_error_m=.11)
    assert not landing_precision_metrics([clean])['passed'],'Failed hold cannot pass10cm goal'
    clean['events'][2].update(moving=True,first_attempt_for_target=False)
    assert not landing_precision_metrics([clean])['passed'],'Retry cannot hide a large miss'
    clean['events'][2]['foot_error_m']=.01;clean['physical_death']=True
    assert not landing_precision_metrics([clean])['passed'],'Deaths cannot pass10cm goal'
    clean['physical_death']=False;clean['events']=[]
    assert not landing_precision_metrics([clean])['passed'],'Missing exposure cannot pass10cm goal'
    # A fixed+X batch has its own task scope; retries, omitted cases, height-only
    # navigation, late errors/deaths and yaw failures must never be accepted.
    from copy import deepcopy
    from ..obs73_omni import straight_eval_spec
    before=deepcopy(c);spec=straight_eval_spec(c,.4,10)
    assert c==before and spec['evaluation_scope']=='straight_x_step0.4_jumps10'
    for spacing,jumps in ((float('nan'),10),(.51,10),(.4,32),(.4,0)):
        try:straight_eval_spec(c,spacing,jumps)
        except ValueError:pass
        else:raise AssertionError('Invalid straight evaluation specification accepted')
    rows=[]
    for case_id in range(128):
        events=[dict(event(index,moving=index>0),target_local_xy_m=[index*.4,0.],
            apex_root_link_world_m=1.02,attempt_for_target=1) for index in range(11)]
        rows.append(dict(case_id=case_id,events=events,physical_death=False,
            rebound_totals=[10.,.5,.01,10.,math.radians(2)**2*10,.4,10.,math.radians(8),1.]))
    yaw=dict(rms_heading_goal_deg=3.,peak_heading_goal_deg=10.)
    assert straight_batch_metrics(rows,spec,yaw)['passed']
    for change in ('missing','height_nav','retry','initial_retry','late_death','late_foot','yaw','wrong_route'):
        bad=deepcopy(rows);r=bad[64]
        if change=='missing':r['events'].pop()
        elif change=='height_nav':r['events'][5].update(strict=False,apex_root_link_world_m=.999,navigation_advanced=True)
        elif change=='retry':
            r['events'].insert(6,dict(r['events'][5],first_attempt_for_target=False,attempt_for_target=2))
        elif change=='initial_retry':
            r['events'].insert(1,dict(r['events'][0],first_attempt_for_target=False,attempt_for_target=2))
        elif change=='late_death':r['physical_death']=True
        elif change=='late_foot':r['events'].append(dict(r['events'][10],target_index=11,target_local_xy_m=[4.4,0.],foot_error_m=.10001))
        elif change=='yaw':r['rebound_totals'][7]=math.radians(10.01)
        else:r['events'][4]['target_local_xy_m']=[1.6,.001]
        if change=='wrong_route':
            try:straight_batch_metrics(bad,spec,yaw)
            except ValueError:continue
            raise AssertionError('Wrong straight route accepted')
        result=straight_batch_metrics(bad,spec,yaw)
        assert not result['passed'],change
        if change in ('missing','height_nav','retry','initial_retry','late_death'):
            assert result['response_groups']['weaker']['strict_sequence_and_survived']==63,change
    try:straight_batch_metrics(rows[:-1],spec,yaw)
    except ValueError:pass
    else:raise AssertionError('127-case batch accepted')
    # Additional scope mirrors a single WASD step followed by a stable hold.
    from ..step_hold import step_hold_eval_spec,step_hold_points,step_hold_batch_metrics
    step_spec=step_hold_eval_spec(c);rows=[]
    for case_id in range(128):
        points=step_hold_points(step_spec['route_points'],.4,3,case_id%4)
        events=[dict(event(i,moving=i==3),target_local_xy_m=points[i],apex_root_link_world_m=1.02)
            for i in range(10)]
        rows.append(dict(case_id=case_id,events=events,physical_death=False,
            rebound_totals=[10.,.5,.01,10.,math.radians(2)**2*10,.4,10.,math.radians(8),1.],
            body_rate_totals=[10.,.9,.5,.1,.4,0.,10.,.4,.5]))
    good=step_hold_batch_metrics(rows,step_spec,yaw)
    assert good['passed'] and good['counts']['enough_later_holds']==128
    assert all(v['episodes']==32 for v in good['per_direction'].values())
    assert abs(good['body_rates']['roll_pitch_rate_rms_rad_s']-.3)<1e-6
    for change in ('missing_holds','height','retry','death','late_foot','yaw','rate_coverage','wrong_route'):
        bad=deepcopy(rows);r=bad[64]
        if change=='missing_holds':r['events'].pop()
        elif change=='height':r['events'][5].update(strict=False,apex_root_link_world_m=.999,navigation_advanced=True)
        elif change=='retry':r['events'].insert(4,dict(r['events'][3],first_attempt_for_target=False))
        elif change=='death':r['physical_death']=True
        elif change=='late_foot':r['events'].append(dict(r['events'][-1],target_index=10,foot_error_m=.10001))
        elif change=='yaw':r['rebound_totals'][7]=math.radians(10.01)
        elif change=='rate_coverage':r.pop('body_rate_totals')
        else:r['events'][3]['target_local_xy_m']=[.2,0.]
        if change=='wrong_route':
            try:step_hold_batch_metrics(bad,step_spec,yaw)
            except ValueError:continue
            raise AssertionError('Wrong step/hold route accepted')
        assert not step_hold_batch_metrics(bad,step_spec,yaw)['passed'],change
    # Test the real case-restore route override and guard against accidentally
    # evaluating a saved step/hold payload as the original random omni task.
    from ..step_hold import restore_step_hold_route
    route=GroundRoute(128,'cpu',c['omni'],7001,evaluation=True);route.reset(torch.arange(128))
    bound=[];fake=SimpleNamespace(_robot=SimpleNamespace(_ALL_INDICES=torch.arange(128)),
        step_hold_evaluation_spec=step_spec,omni_route=route,course=c,response_envelope=None,
        ring=SimpleNamespace(center=torch.zeros(128,3)),device='cpu',_bind_targets=lambda ids:bound.append(ids))
    payload=dict(cases=[dict(ring_center_world=[0.,0.,1.]) for _ in range(128)])
    restore_step_hold_route(fake,payload,apply=True)
    for i in range(128):torch.testing.assert_close(route.points[i],torch.tensor(step_hold_points(32,.4,3,i%4)))
    payload['step_hold_evaluation']=step_spec
    del fake.step_hold_evaluation_spec
    try:restore_step_hold_route(fake,payload)
    except ValueError:pass
    else:raise AssertionError('Step/hold cases reused without matching scope')
    return dict(status='passed',source=str(source),optimizer_updates=0,simulation_started=False,
        actual_wrapper_and_parent_queue_advance_checked=True,stationary_first_hop_and_terminals_unchanged=True,
        route_holds_keep_legacy_reward=not include_holds,
        landing_weights=[0.,.5,1.],observation_physics_ppo_and_existing_gates_unchanged=True,
        moving_yaw_gate_added=bool(rebound),
        previous_15deg_tilt_cost_over_08s=float(old_tilt[0]),
        previous_05m_transport_velocity_cost_over_08s=float(old_velocity[0]),
        phase_motion_costs_removed_for_transport=True,large_tilt_envelope_retained=True,
        velocity_guidance_and_retry_direction_checked=guidance_checked,
        old_target_before_queue_advance_checked=True,
        time_based_landing_anticipation_checked=bool(anticipation),continuous_prefix_failure_and_survival_checked=True,
        precision_width_only_checked=bool(precision),maximum_10cm_acceptance_including_holds_retries_and_missing_exposure_checked=True,
        touchdown_rate_event_mask_symmetry_yaw_and_free_allowance_checked=bool(stability),
        same_v12_reward_and_equal_v13_training_budget_checked=bool(control),
        route_hold_recovery_and_nonzero_transition_scope_checked=bool(include_holds),
        real_failure_outward_braking_and_inward_steady_tilt_checked=bool(outward),
        real_overshoot_early_projection_cost_and_nonzero_translation_checked=bool(projection),
        precontact_impact_replay_leaned_translation_and_exclusion_masks_checked=bool(impact),
        rebound_contact_goal_latch_once_liftoff_retry_and_partial_reset_checked=bool(rebound),
        world_heading_derivative_and_training_sequence_exposure_checked=bool(rebound),
        reference_touchdown_projection_rotation_invariance_checked=bool(touchdown_projection),
        reached_underheight_navigation_separate_from_strict_scoring_checked=bool(navigation),
        real_launch_predicted_cone_extrema_masks_and_stable_lean_checked=bool(rotation_prediction),
        straight128_fixed_scope_retries_missing_height_navigation_late_failures_and_yaw_checked=True,
        straight_route_exposure_only_and_half_stationary_preserved_checked=bool(straight_focus),
        free_flight_excess_body_rate_travel_holds_contact_exclusions_and_no_double_charge_checked=bool(body_rate),
        launch_rotation_once_only_and_step_hold_exposure_checked=bool(launch_rotation) and not response_window,
        launch_response_window_delayed_peak_boundary_reset_and_continuous05_checked=bool(response_window),
        launch_endpoint_distance_scale_event_lifecycle_symmetry_and_stationary_masks_checked=bool(launch_endpoint),
        cardinal_step_hold_scope_restore_missing_retries_late_failures_and_rate_coverage_checked=True,
        limitation='Reward semantics and lifecycle checked; one-hop precision,smoothness and survival improvement require manual training/evaluation')


def queued_repair_check():
    """V32 real-failure kernels, route exposure and long-scope scorer."""
    from copy import deepcopy
    from ..obs73_omni import GroundRoute
    from ..stationary_retention import RetentionRoute
    from ..cardinal_queue import queued_points,evaluation_spec,batch_metrics,restore_route
    from ..moving_phase_reward import transport_body_rate_cost,launch_direction_cost
    c=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_cardinal_queue_repair_course_v32.json').read_text())
    old=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_clock_only_course_v29.json').read_text())
    for key in ('vertical_pretraining','stationary_pretraining','stationary_retention','observation_version',
                'envelope','runner_config_source','rebound_transition','moving_yaw_prediction'):
        assert c[key]==old[key],key
    assert c['long_moving_horizon']['stationary_s']==15. and c['long_moving_horizon']['moving_s']==60.
    source=ROOT/c['omni']['initial_checkpoint'];assert source.is_file()
    ids=torch.arange(768);route=RetentionRoute(768,'cpu',c['omni'],42,training_mixture=c['moving_training_sequences'])
    route.reset(ids);assert not route.points[route.stationary_mask].any()
    assert int(route.stationary_mask.sum())==384
    for kind in (3,4,5):assert ((route.sequence_kind==kind)&~route.stationary_mask).any()
    for kind in (4,5):
        points=route.points[route.sequence_kind==kind]
        assert ((points[:,:,0].abs()<1e-6)|(points[:,:,1].abs()<1e-6)).all()
        distances=points.diff(dim=1).norm(dim=-1)
        assert ((distances<1e-6)|((distances-.5).abs()<1e-6)).all()
        assert (points[:,:,0]>.4).any() and (points[:,:,0]<-.4).any()
        assert (points[:,:,1]>.4).any() and (points[:,:,1]<-.4).any()
    before=route.points.clone();route.reset(torch.tensor([1]));torch.testing.assert_close(route.points[2:],before[2:])
    # Full-flight rate cost stays active even while the airborne spring moves;
    # no costs for stationary/first hop/impact or a static useful lean.
    rate=c['moving_body_rate']
    for active,phase,clearance in [(True,3,.5),(False,3,.5),(True,1,.01)]:
        value,audit=transport_body_rate_cost(torch.tensor([[1.5,0.,0.]]),torch.tensor([phase]),
            torch.tensor([.004]),torch.tensor([.3]),torch.tensor([clearance]),torch.tensor([active]),
            torch.zeros(1),.01,rate)
        assert bool(value[0]<0)==(active and phase==3)
    value,_=transport_body_rate_cost(torch.zeros(1,3),torch.tensor([3]),torch.zeros(1),torch.zeros(1),
        torch.ones(1),torch.ones(1,dtype=torch.bool),torch.zeros(1),.01,rate)
    assert not value.any()
    direction=c['launch_direction_refinement'];target=torch.tensor([[0.,-.5]]).repeat(5,1)
    velocity=torch.tensor([[0.,-.5],[0.,.86],[.4,-.5],[0.,.86],[0.,0.]])
    launch=torch.tensor([True,True,True,False,True]);target[4]=0.
    value=launch_direction_cost(velocity,target,launch,direction)
    torch.testing.assert_close(value,torch.tensor([0.,-4*.86**2,-4*.4**2,0.,0.]))
    # Exercise the actual reward wrapper, including its first-hop/stationary/
    # physical-terminal masks. Additional terms must remain exactly zero there.
    env=wrap_env(ContactRewardParent)(c);env._get_rewards()
    for name in ('moving_transport_body_rate','moving_launch_direction'):
        assert not env._episode_sums[name][1:].any(),name
    spec=evaluation_spec(c);rows=[]
    for case_id in range(128):
        points=queued_points(96,.5,case_id%4);events=[]
        for i in range(48):
            events.append(dict(target_index=i,target_local_xy_m=points[i],strict=True,
                first_attempt_for_target=True,moving=i>0 and points[i]!=points[i-1],
                foot_error_m=.02,apex_root_link_world_m=1.02,touchdown_tilt_deg=2.))
        rows.append(dict(case_id=case_id,events=events,physical_death=False,
            rebound_totals=[30.,.5,.01,30.,math.radians(2)**2*30,.4,30.,math.radians(8),1.],
            body_rate_totals=[30.,2.7,.5,.3,.4,0.,30.,1.2,.5],seconds=60.,timed_out=True))
    result=batch_metrics(rows,spec,c['moving_yaw_prediction']);assert result['passed']
    assert all(v['episodes']==32 for v in result['per_direction'].values())
    for change in ('late_miss','late_yaw','death','missing','retry','short_horizon'):
        bad=deepcopy(rows);r=bad[127]
        if change=='late_miss':r['events'][-1]['foot_error_m']=.3747;r['events'][-1]['strict']=False
        elif change=='late_yaw':r['rebound_totals'][7]=math.radians(35.)
        elif change=='death':r['physical_death']=True
        elif change=='missing':r['events']=r['events'][:12]
        elif change=='short_horizon':r['seconds']=45.
        else:r['events'][9]['first_attempt_for_target']=False
        assert not batch_metrics(bad,spec,c['moving_yaw_prediction'])['passed'],change
    fake=SimpleNamespace(cardinal_queue_evaluation_spec=spec,omni_route=SimpleNamespace(
        points=torch.zeros(128,96,2),index=torch.ones(128,dtype=torch.long)))
    restore_route(fake,dict(cases=[{}]*128),apply=True)
    for i in range(128):torch.testing.assert_close(fake.omni_route.points[i],torch.tensor(queued_points(96,.5,i%4)))
    payload=dict(cases=[{}]*128,cardinal_queue_evaluation=spec)
    del fake.cardinal_queue_evaluation_spec
    try:restore_route(fake,payload)
    except ValueError:pass
    else:raise AssertionError('60s queue cases accepted without the matching explicit scope')
    return dict(status='passed',source=str(source),balanced_cardinal_and_queued_routes=True,
        half_stationary_membership_and_rewards_preserved=True,full_air_rate_gap_closed=True,
        real_wrong_sign_liftoff_and_cross_axis_cost_checked=True,actual_reward_wrapper_masks_checked=True,
        cardinal128_scope_restore_and_late_failures_checked=True,optimizer_updates=0,simulation_started=False)


def continuous_velocity_check():
    """V33 physical-release error must retain credit over motor response."""
    from copy import deepcopy
    from ..moving_phase_reward import launch_velocity_window_cost
    c=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_rebound_velocity_window_course_v33.json').read_text())
    old=json.loads((ROOT/c['omni']['initial_checkpoint']).parent.joinpath('ring_config.json').read_text())
    allowed={'version','output_series','training_iterations','initialization','reward','termination','omni','launch_velocity_window'}
    assert set(c)==set(old)|{'launch_velocity_window'}
    for key in set(c)-allowed:assert c[key]==old[key],key
    assert c['omni']==dict(old['omni'],initial_checkpoint=c['omni']['initial_checkpoint'])
    assert c['stationary_retention']['training_stationary_fraction']==.5
    validate_checkpoint_course(ROOT/c['omni']['initial_checkpoint'],c)
    cfg=c['launch_velocity_window'];weight=c['rebound_transition']['launch_weight']
    # Actual frozen v32 case71,step427. Positive/negative and X/Y versions
    # must receive equal costs. A correct speed and all nonmoving masks free.
    velocity=torch.tensor([[.007,-.150],[-.007,.150],[-.150,.007],[.150,-.007],[0.,-.5]])
    target=torch.tensor([[-.020,-.513],[.020,.513],[-.513,-.020],[.513,.020],[0.,-.5]])
    duration=torch.ones(5);launch_step=torch.full((5,),100,dtype=torch.long)
    total=torch.zeros(5)
    for now in range(100,118):
        value,audit=launch_velocity_window_cost(velocity,target,duration,launch_step,now,
            torch.ones(5,dtype=torch.bool),torch.full((5,),2),torch.ones(5),.01,weight,cfg)
        assert audit['active'].all() and (value<=0).all()
        torch.testing.assert_close(value[:4],value[0].expand(4));assert value[4]==0
        total+=value
    scaled=(velocity-target).norm(dim=-1)/cfg['width_m']
    expected=-weight*scaled.square()/(torch.sqrt(1+scaled.square())+1)
    torch.testing.assert_close(total,expected)
    for now,step,moving,phase,clearance in [(118,100,True,2,1.),(99,100,True,2,1.),
            (100,-1,True,2,1.),(100,100,False,2,1.),(100,100,True,1,.02),(100,100,True,3,.01)]:
        value,audit=launch_velocity_window_cost(velocity,target,duration,torch.full((5,),step),now,
            torch.full((5,),moving),torch.full((5,),phase),torch.full((5,),clearance),.01,weight,cfg)
        assert not value.any() and not audit['active'].any()
    # Real wrapper: latch once, keep target through the response interval,
    # no second launch event, then expire. Airborne spring rate must not mute it.
    env=wrap_env(ContactRewardParent)(c);d=env._robot.data;k=2
    d.root_link_lin_vel_w[k]=torch.tensor([.007,-.150,3.]);env._foot_event.phase[k]=2
    env.rebound_plan.pending[k]=True;env.rebound_plan.velocity[k]=target[0]
    target_before=env.rebound_plan.velocity[k].clone();env.common_step_counter=100
    costs=[]
    for now in range(100,119):
        env.common_step_counter=now
        if now>100:d.joint_vel[k,0]=.3
        env._get_rewards();a=env._moving_phase_last_audit
        costs.append(a['deltas']['moving_launch_velocity'][k].clone())
        assert bool(a['rebound']['launch'][k])==(now==100)
        assert bool(a['rebound']['launch_velocity_active'][k])==(now<118)
        assert not a['deltas']['moving_launch_velocity'][[0,1,3]].any()
        torch.testing.assert_close(env.rebound_plan.velocity[k],target_before)
    assert all(v<0 for v in costs[:18]) and costs[18]==0
    # A real new impact cancels the old window; it may not bleed into a
    # subsequent flight before the next physical free-liftoff event.
    env.common_step_counter=110;env._foot_event.phase[k]=3
    d.root_link_lin_vel_w[k,2]=-2.;d.joint_pos[k,0]=.004
    env._get_rewards();assert env.rebound_plan.launch_step[k]==-1
    d.joint_pos[k,0]=0.;d.joint_vel[k,0]=.3;env._foot_event.phase[k]=2
    env.common_step_counter=111;env._get_rewards()
    assert not env._moving_phase_last_audit['rebound']['launch_velocity_active'][k]
    # Legacy v32 must retain its original once-only velocity loss exactly.
    legacy=wrap_env(ContactRewardParent)(old);legacy._robot.data.root_link_lin_vel_w[k]=torch.tensor([.007,-.150,3.])
    legacy._foot_event.phase[k]=2;legacy.rebound_plan.pending[k]=True;legacy.rebound_plan.velocity[k]=target[0]
    legacy._get_rewards();a=legacy._moving_phase_last_audit
    torch.testing.assert_close(a['deltas']['moving_launch_velocity'][k],-weight*(velocity[0]-target[0]).square().sum())
    legacy._get_rewards();assert legacy._moving_phase_last_audit['deltas']['moving_launch_velocity'][k]==0
    return dict(status='passed',source=c['omni']['initial_checkpoint'],optimizer_updates=0,
        real_case71_release_replayed=True,actual_wrapper_response_window_and_contact_reset_checked=True,
        sign_axis_symmetry_and_full_window_budget_checked=True,stationary_first_hop_and_legacy_v32_preserved=True,
        unchanged_observation_physics_routes_body_rate_and_yaw=True,
        real_case71_full_window_cost=float(total[0]))


def outward_tilt_work_check(course_path):
    """Replay the early and late outward-torque samples through the reward wrapper."""
    course = json.loads(Path(course_path).read_text())
    source = ROOT / course['omni']['initial_checkpoint']
    previous = json.loads(source.with_name('ring_config.json').read_text())
    allowed = {'version', 'output_series', 'training_iterations', 'initialization',
               'reward', 'omni', 'moving_outward_tilt_work'}
    assert set(course) == set(previous) | {'moving_outward_tilt_work'}
    for key in set(course) - allowed:
        assert course[key] == previous[key], key
    assert course['omni'] == dict(previous['omni'], initial_checkpoint=course['omni']['initial_checkpoint'])
    validate_checkpoint_course(source, course)
    samples = (
        # case47 step3303, immediately after launch
        ([.99317569, .04311747, -.00677605, .10815190],
         [2.30873942, .00269216, .26894861], [.04128329, .01415533, .00086731], .08269345, 2),
        # case126 step3786, 970ms after launch, close to the next touchdown
        ([.99171215, .04721161, .04252394, -.11166822],
         [.80782783, 1.04291594, -1.47248244], [-.01568021, .02293660, -.00873778], .59789282, 3),
        # case7 step560, a normal rebound during a strict three-hop sequence
        ([.99883771, .01448403, .02763692, .03673553],
         [.09100053, -.08211080, -.58505851], [-.04319571, -.01468490, -.00176410], .06305658, 2),
    )

    def evaluate(spec, sample, torque_scale=1., clearance=None):
        env = wrap_env(ContactRewardParent)(spec)
        d = env._robot.data
        q, omega, moment, height, phase = sample
        # An equivalent copy in the pure-stationary member must cost zero.
        d.root_quat_w[:3] = torch.tensor(q)
        d.root_ang_vel_b[:3] = torch.tensor(omega)
        env._moment = torch.tensor(moment).repeat(4, 1, 1) * torque_scale
        d.body_link_pos_w[:3, 0, 2] = .012 + (height if clearance is None else clearance)
        env._foot_event.phase[:3] = phase
        d.root_link_lin_vel_w[:3, 2] = 2. if phase == 2 else -2.
        env.rebound_plan.launch_step[:] = 3
        env._get_rewards()
        return env._moving_phase_last_audit['deltas']

    key = 'moving_outward_tilt_work'
    costs = []
    for sample in samples:
        current = evaluate(course, sample)
        legacy = evaluate(previous, sample)
        assert set(current) == set(legacy) | {key}
        for other in legacy:
            torch.testing.assert_close(current[other], legacy[other], atol=0., rtol=0.)
        assert not current[key][[0, 1, 3]].any()
        costs.append(float(current[key][2]))
    assert costs[0] < 0 and costs[1] < 0 and costs[2] == 0
    for sample in samples[:2]:
        assert not evaluate(course, sample, torque_scale=-1.)[key].any()
        assert not evaluate(course, sample, clearance=.02)[key].any()
    return dict(status='passed', case47_early_cost=costs[0], case126_late_cost=costs[1],
                case7_clean_cost=costs[2], stationary_and_v33_objectives_preserved=True)


def omni_focus_check(course_path):
    from ..stationary_retention import RetentionRoute

    course = json.loads(Path(course_path).read_text())
    source = ROOT / course['omni']['initial_checkpoint']
    previous = json.loads(source.with_name('ring_config.json').read_text())
    changed = {'version', 'output_series', 'training_iterations', 'initialization', 'reward',
               'omni', 'omni_focus', 'moving_training_sequences', 'straight_sequence_focus'}
    assert {k:v for k,v in course.items() if k not in changed} == {
        k:v for k,v in previous.items() if k not in changed}
    assert not any(key in course for key in ('moving_training_sequences', 'straight_sequence_focus'))
    validate_checkpoint_course(source, course)
    config = course['omni']
    ids = torch.arange(768)
    route = RetentionRoute(768, 'cpu', config, seed=42)
    route.reset(ids)
    assert int(route.stationary_mask.sum()) == 384
    assert not route.points[route.stationary_mask].any() and not route.sequence_kind.any()
    points = route.points[~route.stationary_mask]
    assert float(points.norm(dim=-1).max()) <= config['workspace_radius_m'] + 1e-5
    edges = points.diff(dim=1)
    lengths = edges.norm(dim=-1)
    moving = lengths > .1
    low, high = config['moving_distance_range_m']
    assert ((lengths < 1e-6) | ((lengths >= low - 1e-6) & (lengths <= high + 1e-6))).all()
    hold_fraction = float((~moving).float().mean())
    assert abs(hold_fraction - config['zero_probability']) < .01
    long_min = config['training_long_steps']['min_distance_m']
    fraction = config['training_long_steps']['fraction']
    long_fraction = float((lengths[moving] >= long_min).float().mean())
    expected = fraction + (1 - fraction) * (high - long_min) / (high - low)
    assert abs(long_fraction - expected) < .015
    angles = torch.atan2(edges[..., 1], edges[..., 0])
    sectors = ((angles[moving] / (math.pi / 4) + .5).floor().long() % 8)
    consecutive = moving[:, 1:] & moving[:, :-1]
    change = angles.diff(dim=1)
    turns = torch.atan2(change.sin(), change.cos()).abs()[consecutive]
    assert torch.bincount(sectors, minlength=8).min() > .09 * sectors.numel()
    assert ((turns >= math.pi / 2).float().mean() > .4)
    eval_ids = torch.arange(128)
    old_eval = RetentionRoute(128, 'cpu', previous['omni'], 7001, evaluation=True,
                              training_mixture=previous['moving_training_sequences'])
    new_eval = RetentionRoute(128, 'cpu', config, 7001, evaluation=True)
    old_eval.reset(eval_ids)
    new_eval.reset(eval_ids)
    assert torch.equal(new_eval.points, old_eval.points)
    return dict(status='passed', stationary_envs=384, free_heading_envs=384,
                hold_fraction=hold_fraction, steps_at_least_30cm_fraction=long_fraction,
                heading_counts=torch.bincount(sectors, minlength=8).tolist(),
                turns_at_least_90deg_fraction=float((turns >= math.pi / 2).float().mean()),
                original_eval_route_identical=True, previous_rewards_preserved=True)


def omni_rebound_handoff_check(course_path):
    """Exercise descent, confirmed contact, unloading, and free flight in order."""
    course = json.loads(Path(course_path).read_text())
    source = ROOT / course['omni']['initial_checkpoint']
    previous = json.loads(source.with_name('ring_config.json').read_text())
    changed = {'version', 'output_series', 'training_iterations', 'initialization',
               'reward', 'omni', 'omni_rebound_handoff'}
    assert set(course) == set(previous) | {'omni_rebound_handoff'}
    for key in set(course) - changed:
        assert course[key] == previous[key], key
    assert course['omni'] == dict(previous['omni'], initial_checkpoint=course['omni']['initial_checkpoint'])
    validate_checkpoint_course(source, course)
    key = 'moving_contact_handoff_velocity'
    k = 2
    report = {}
    for case_id, misses, touchdown_offset, incoming, unloading in (
            (118, False, [-.035, -.085], [.133, .335], [.526, .264]),
            (99, True, [.170, -.055], [-.362, -.232], [-1.473, -.173]),
            (118, True, [.048, -.121], [-.033, .163], [-.327, 1.245])):
        env = wrap_env(ContactRewardParent)(course)
        d = env._robot.data
        env._moment = torch.zeros(4, 1, 3)
        # Incoming edge north and outgoing edge east. The recovery cases
        # use the frozen replay's foot offset and unloading velocity.
        env.omni_route.points[k, 1:3] = torch.tensor([[0., .45], [.46, .45]])
        env.bind()
        old_goal = env._two_ground_pt_xy_w[k].clone()
        d.body_link_pos_w[k, 0, :2] = old_goal + torch.tensor(touchdown_offset)
        d.body_link_pos_w[k, 0, 2] = .02
        d.root_link_pos_w[k] = d.body_link_pos_w[k, 0] + torch.tensor([0., 0., .273])
        d.root_link_lin_vel_w[k] = torch.tensor([*incoming, -2.4])
        env._get_rewards()
        a = env._moving_phase_last_audit
        assert a['rebound']['handoff_preparation_active'][k]
        assert not a['deltas'][key].any()
        expected_goal = old_goal if misses else env._two_ground_pt1_xy_w[k]
        forecast = a['rebound']['handoff_predicted_landing_xy'][k]
        torch.testing.assert_close(a['rebound']['handoff_outgoing_velocity'][k],
                                   (expected_goal-forecast)/env.rebound_plan.flight[k])
        assert not a['rebound']['handoff_preparation_active'][[0, 1, 3]].any()

        env.common_step_counter += 1
        d.joint_pos[k, 0] = .014
        d.joint_vel[k, 0] = 1.9
        d.body_link_pos_w[k, 0, 2] = .012
        env._get_rewards()
        a = env._moving_phase_last_audit
        assert not a['deltas'][key].any()
        assert env.omni_route.index[k] == (1 if misses else 2)
        target_velocity = env.rebound_plan.velocity[k].clone()
        torch.testing.assert_close(target_velocity,
            (expected_goal-d.body_link_pos_w[k, 0, :2])/env.rebound_plan.flight[k])

        env.common_step_counter += 3
        d.joint_pos[k, 0] = .018
        d.joint_vel[k, 0] = -1.5
        d.root_link_lin_vel_w[k] = torch.tensor([*unloading, 1.2])
        env._get_rewards()
        a = env._moving_phase_last_audit
        loss = float(a['deltas'][key][k])
        assert loss < 0 and not a['deltas'][key][[0, 1, 3]].any()
        assert bool(a['rebound']['turn_unloading_active'][k]) == (not misses)
        assert bool(a['rebound']['retry_unloading_active'][k]) == misses
        assert not a['deltas']['moving_launch_velocity'].any()
        report[f"{'retry' if misses else 'turn'}_{case_id}_loss"] = loss

        # Moving rapidly along the turn's destination is not transverse.
        # A retry instead gets zero only at its full recovery vector.
        d.root_link_lin_vel_w[k, :2] = target_velocity * (1. if misses else 2.)
        env._get_rewards()
        assert abs(float(env._moving_phase_last_audit['deltas'][key][k])) < 1e-7
        d.root_link_lin_vel_w[k, :2] = torch.tensor(unloading)
        env.common_step_counter += 10
        env._get_rewards()
        assert not env._moving_phase_last_audit['deltas'][key].any()
        d.joint_pos[k, 0] = 0.
        d.joint_vel[k, 0] = 0.
        d.body_link_pos_w[k, 0, 2] = .1
        env._foot_event.phase[k] = 2
        env._get_rewards()
        a = env._moving_phase_last_audit
        assert a['rebound']['launch'][k] and a['deltas']['moving_launch_velocity'][k] < 0
        assert not a['deltas'][key].any()
    return dict(status='passed', frozen_unloading_cases=report,
                score_and_route_unchanged=True, free_liftoff_cost_preserved=True)


def omni_reversal_check(course_path):
    from ..obs73_omni import DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND
    from ..stationary_retention import RetentionRoute

    course = json.loads(Path(course_path).read_text())
    source = ROOT / course['omni']['initial_checkpoint']
    old = json.loads(source.with_name('ring_config.json').read_text())
    changed = {'version', 'output_series', 'training_iterations', 'initialization',
               'reward', 'omni', 'omni_reversal_baseline'}
    assert {k:v for k,v in course.items() if k not in changed} == {
        k:v for k,v in old.items() if k not in changed}
    config = course['omni']
    assert {k:v for k,v in config.items() if k not in ('initial_checkpoint', 'sampling', 'training_reversals')} == {
        k:v for k,v in old['omni'].items() if k not in ('initial_checkpoint', 'sampling')}
    validate_checkpoint_course(source, course)
    ids = torch.arange(768)
    route = RetentionRoute(768, 'cpu', config, 42)
    route.reset(ids)
    stationary = route.stationary_mask
    assert int(stationary.sum()) == 384 and not route.points[stationary].any()
    kinds = route.sequence_kind[~stationary]
    count = {kind:int((kinds == kind).sum()) for kind in (0, DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND)}
    assert sum(count.values()) == len(kinds) and count[0] > .6 * len(kinds)
    assert all(.09 < count[k] / len(kinds) < .21 for k in (DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND))
    assert float(route.points.norm(dim=-1).max()) <= config['workspace_radius_m'] + 1e-5
    headings = []
    for kind in (DIRECT_REVERSAL_KIND, PAUSED_REVERSAL_KIND):
        xy = route.points[route.sequence_kind == kind]
        edges = xy.diff(dim=1)
        distances = edges.norm(dim=-1)
        moving = distances > .1
        assert (moving.sum(dim=1) >= 36).all()
        torch.testing.assert_close(distances[moving], torch.full_like(distances[moving], .5))
        first = moving.long().argmax(dim=1)
        assert ((first >= 1) & (first <= 3)).all()
        outgoing = edges[torch.arange(len(xy)), first] / .5
        angle = torch.atan2(outgoing[:, 1], outgoing[:, 0])
        headings.extend(((angle / (math.pi / 4) + .5).floor().long() % 8).tolist())
        along = (edges * outgoing[:, None]).sum(dim=-1)
        torch.testing.assert_close(edges, along[:, :, None] * outgoing[:, None], atol=1e-6, rtol=1e-5)
        for edge, i in zip(along, first):
            sign = (edge[int(i):] / .5).round().long()
            assert (sign == 1).any() and (sign == -1).any()
            in_motion = sign[sign != 0]
            assert int((in_motion[1:] != in_motion[:-1]).sum()) >= 10
            if kind == DIRECT_REVERSAL_KIND:
                assert not (sign == 0).any()
            else:
                jumps = (sign != 0).nonzero().flatten()
                reversal_gaps = (jumps[1:] - jumps[:-1])[in_motion[1:] != in_motion[:-1]]
                assert ((reversal_gaps >= 2) & (reversal_gaps <= 4)).all()
    assert set(headings) == set(range(8))
    before = route.points.clone()
    route.reset(torch.tensor([0, 1, 23]))
    keep = ids[(ids != 0) & (ids != 1) & (ids != 23)]
    assert torch.equal(route.points[keep], before[keep]) and not route.points[0].any()

    eval_ids = torch.arange(128)
    prior = RetentionRoute(128, 'cpu', old['omni'], 7001, evaluation=True)
    candidate = RetentionRoute(128, 'cpu', config, 7001, evaluation=True)
    prior.reset(eval_ids)
    candidate.reset(eval_ids)
    assert torch.equal(candidate.points, prior.points) and not candidate.sequence_kind.any()
    return dict(status='passed', stationary=384, unrestricted=count[0],
                direct=count[DIRECT_REVERSAL_KIND], paused=count[PAUSED_REVERSAL_KIND],
                all_eight_heading_sectors=True, old_eval_routes_identical=True,
                reward_and_physics_configs_preserved=True)


if __name__=='__main__':
    import sys
    print(json.dumps(continuous_velocity_check() if '--continuous-velocity' in sys.argv else
        queued_repair_check() if '--queued-repair' in sys.argv else run(),indent=2))


def guarded_keys_check(course):
    """Drive the new torch queue and independent deployed 0.5m FIFO through identical events."""
    import torch, importlib.util, sys
    from types import SimpleNamespace
    from pathlib import Path
    from benchmark37.guarded_keys import GuardedKeys,FIELDS
    deploy=Path(__file__).resolve().parents[2]/'deployment'
    sys.path.insert(0,str(deploy))
    try:
        spec=importlib.util.spec_from_file_location('observed_step_reference',deploy/'obs73_step_runtime.py')
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    finally:sys.path.pop(0)
    ref.STEP_M=.5; q=ref.StationaryWaypointQueue()
    r=SimpleNamespace(index=torch.zeros(1,dtype=torch.long),points=torch.zeros(1,96,2))
    g=GuardedKeys(r,.01,True);g.reset(g.ids);g.start=1;g.period=350
    def t(x):return torch.tensor([x])
    states=set();transitions=set()
    for tick in range(1,701):
        sec=tick*.01
        if tick in (1,351):
            for _ in range(3):q.request_step(sec,b'w' if tick==1 else b's')
        # Cross both yaw guards separately, hide a preview, refuse an early launch,
        # then miss in STEP, cancel pending and hold at measured foot until recovered.
        landing=tick in (20,40,62,90,148,208,265,320,401,455,512,570,630)
        yaw=11. if tick==20 else 8. if tick==148 else 0.;yr=30. if tick in (147,148) else 0.
        rp=1.4 if tick in (264,265) else .4
        offset=.14 if tick==265 else .02
        foot=q.targets_xy[0]+[0.,offset];event=dict(strict=offset<=.1,foot_error_m=offset,tilt_deg=3.,
            heading_error_deg=yaw,predicted_heading_error_deg=yaw+.14*yr,roll_pitch_rate_rad_s=rp,foot_world_xy_m=foot)
        tr=q.on_landing(event if landing else None,sec)
        if tr:transitions.add(tr)
        g.update(t(landing),t(landing and offset<=.1),t(offset<=.1),torch.tensor(foot).float()[None],t(offset),
                 t(yaw),t(yr),t(rp),t(not landing),t(not landing),t(False))
        assert ('HOLD','PREVIEW','STEP','PAUSED','RECOVER')[int(g.state[0])]==q.state,(tick,q.state)
        torch.testing.assert_close(torch.stack((g.xy[0],g.next[0])),torch.tensor(q.targets_xy).float())
        assert g.count[0,-1].item()==q.remaining_steps
        states.add(q.state)
    assert len(states)==5 and {'STEP_MISS_CANCEL_LOCAL_HOLD','STEP_PREVIEW_RESUMED','STEP_RECOVERED_HOLD'}<=transitions
    assert course['stationary_retention']['training_stationary_fraction']==.5 and course['guarded_keys']['launch_multiplier']>=1
    return dict(states=sorted(states),ticks=700,queue= dict(zip(FIELDS,g.count[0].tolist())))
