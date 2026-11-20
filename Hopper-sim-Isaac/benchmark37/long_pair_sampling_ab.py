"""Stage4h: episode mixture for consecutive long hops, unchanged rewards/robot."""
import contextlib
import hashlib
import json
import math
from types import SimpleNamespace

import torch

from . import first_hop_margin_ab as source_task
from . import range08_curriculum as range_task
from .artifacts import ROOT, sha
from .calibration_20261028_v2 import REVISION, physics_contract
from .two_ground_targets import _ensure, _set_active_goal

SOURCE=ROOT/'outputs/benchmark37_stage4g_first_hop_margin_control/train_seed42_20261029_073211_802030Z/model_49.pt'
SOURCE_SHA256='b7fcb7297b7a0ae2243dcf3e0e6bcb7ae463b4f4a8c642f07f5fd66cce7500fd'
ARMS=('control','longmix')
VARIANTS={a:'real_release_two_ground_range08_long_pair_'+a for a in ARMS}
SERIES={a:'benchmark37_stage4h_long_pair_'+a for a in ARMS}
ITERATIONS=50
HARD_PROBABILITY={'control':0.,'longmix':.5}
HARD_MIN_M=.6
MAX_M=.8
MODE_RNG_OFFSET=4102941
PAIRED_RNG_OFFSET=4102943
ROUTE_EDGES=512


def contract(arm):
    return dict(name='long_pair_sampling_ab_v1',arm=arm,variant=VARIANTS[arm],
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_task.contract('control'),
        intervention='Only train episode mixture: ordinary full-range or persistent0.6..0.8m consecutive-long mode',
        hard_episode_probability=HARD_PROBABILITY[arm],hard_edge_range_m=[HARD_MIN_M,MAX_M],
        ordinary='each edge10% exact zero; otherwise radius uniform0..0.8m; independent full-circle heading',
        mixture='sample mode once per episode, keep through strict advances; misses retain both points and mode; partial resets change selected rows only',
        randomization='separate mode generator; candidate edge sampler consumes exactly the source three random arrays; global torch RNG unchanged; control source sampler bitwise preserved',
        evaluation='both arms ordinary full range; primary paired272x1x15s seed7001 with identical precomputed per-env512-edge schedules and initial DR/state snapshots, no training hard mode',
        observation='same41D, five4-motor action histories, both ground XY point slots/frame/scale; hard mode is not an actor input',
        unchanged='all rewards including K50 deficit reference1m on all attempts and energy target1.04m; strict1m/10cm/15deg, v2 physics/motor/DR, release0.8..1.2m and immediate policy control, PPO/horizon/queue rules',
        initialization='Exact selected Stage4g control41D policy/critic/normalizers; fresh optimizer and iteration0, same seed42 and768x50x256 per arm',
        training_iterations=ITERATIONS,code_sha256=sha(__file__))


def reward_contract(arm):
    if arm not in ARMS:raise ValueError('Unknown long-pair arm')
    return source_task.reward_contract('control')


def configure(cfg,arm):
    base=source_task.configure(cfg,'control')
    return dict(arm=arm,hard_episode_probability=HARD_PROBABILITY[arm],hard_edge_range_m=[HARD_MIN_M,MAX_M],
        sampling_max_m=MAX_M,obs_dim=41,actual_apex_deficit_scale=50.,reference_world_m_by_slot=[1.,1.,1.],
        planned_updates=ITERATIONS,source_reward_config=base)


def validate_source(path,source_lock,physics_lock):
    path=path.resolve(strict=True)
    if path!=SOURCE.resolve() or sha(path)!=SOURCE_SHA256 or source_task.validate_trained_checkpoint(path)!='control':
        raise ValueError('Long-pair sampling requires the pinned formal Stage4g control model_49')
    m=json.loads((path.parent/'manifest.json').read_text())
    if m.get('source_lock')!=source_lock or m.get('physics_lock')!=physics_lock:
        raise ValueError('Long-pair source reference/physics locks changed')
    return m


def validate_trained_checkpoint(path,allow_smoke=False):
    path=path.resolve(strict=True);m=json.loads((path.parent/'manifest.json').read_text())
    arm=next((a for a in ARMS if VARIANTS[a]==m.get('variant')),None)
    if arm is None:raise ValueError('Not a Stage4h long-pair checkpoint')
    smoke=bool(m.get('smoke'));updates=2 if smoke else ITERATIONS
    cfg=json.loads((path.parent/'env_config.json').read_text())
    if (smoke and not allow_smoke or m.get('status')!='complete' or m.get('mode')!='train'
            or path.parent.parent!=ROOT/'outputs'/SERIES[arm] or path.name!=f'model_{updates-1}.pt'
            or m.get('requested_iterations')!=updates or m.get('num_envs')!=(16 if smoke else 768)
            or m.get('task_intervention')!=contract(arm) or m.get('reward_intervention')!=reward_contract(arm)
            or m.get('initial_checkpoint_sha256')!=SOURCE_SHA256 or m.get('checkpoint_hashes',{}).get(path.name)!=sha(path)
            or m.get('physics_revision')!=REVISION or m.get('calibration_lock')!=physics_contract()
            or m.get('long_pair_runtime_config')!=configure(SimpleNamespace(**cfg),arm)):
        raise ValueError('Incomplete or mismatched Stage4h checkpoint provenance')
    return arm


def _ensure_mode(env,seed,arm,training,paired):
    _ensure(env,seed)
    env._range08_training=False
    if not hasattr(env,'_long_pair_hard'):
        env._long_pair_hard=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        env._long_pair_mode_rng=torch.Generator(device=env.device).manual_seed(int(seed)+MODE_RNG_OFFSET)
        env._long_pair_reset_counts=torch.zeros(2,device=env.device,dtype=torch.long)
        env._long_pair_edge_counts=torch.zeros(4,device=env.device,dtype=torch.long)  # all, zero, short, long
    env._long_pair_arm,env._long_pair_training,env._long_pair_paired=arm,training,paired
    if paired and not hasattr(env,'_paired_route_offsets'):
        rng=torch.Generator(device=env.device).manual_seed(int(seed)+PAIRED_RNG_OFFSET)
        shape=(env.num_envs,ROUTE_EDGES)
        radius=MAX_M*torch.rand(shape,device=env.device,generator=rng)
        angle=2*math.pi*torch.rand(shape,device=env.device,generator=rng)
        radius=torch.where(torch.rand(shape,device=env.device,generator=rng)<.1,0.,radius)
        edges=radius[...,None]*torch.stack((torch.cos(angle),torch.sin(angle)),-1)
        env._paired_route_offsets=edges.cumsum(1)
        env._paired_route_origin=torch.zeros(env.num_envs,2,device=env.device)
        env._paired_route_edges_hash=hashlib.sha256(edges.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        env._paired_route_offsets_hash=hashlib.sha256(env._paired_route_offsets.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _record_edges(env,offset):
    r=torch.linalg.vector_norm(offset,dim=-1)
    env._range08_max_sampled_edge=torch.maximum(getattr(env,'_range08_max_sampled_edge',r.new_zeros(())),r.max())
    env._long_pair_edge_counts+=torch.stack((r.new_tensor(len(r),dtype=torch.long),
        (r<1e-6).sum(),(r<=.4+1e-6).sum(),(r>.4+1e-6).sum()))


def _mixed_offset(env,ids):
    n=len(ids)
    # Same draw order/count as the frozen full-range _offset; ordinary rows
    # remain identical. The independent mode draw never shifts this stream.
    uniform=torch.rand(n,device=env.device,generator=env._two_ground_rng)
    angle=2*math.pi*torch.rand(n,device=env.device,generator=env._two_ground_rng)
    zero=torch.rand(n,device=env.device,generator=env._two_ground_rng)<.1
    ordinary=torch.where(zero,0.,MAX_M*uniform)
    radius=torch.where(env._long_pair_hard[ids],HARD_MIN_M+(MAX_M-HARD_MIN_M)*uniform,ordinary)
    env._range08_max_sampled_edge=torch.maximum(getattr(env,'_range08_max_sampled_edge',radius.new_zeros(())),radius.max())
    return radius[:,None]*torch.stack((torch.cos(angle),torch.sin(angle)),-1)


def reset_pair(env,ids,seed,arm,training,paired=False):
    if not len(ids):return
    _ensure_mode(env,seed,arm,training,paired)
    env._long_pair_hard[ids]=False
    if paired:
        env._paired_route_origin[ids]=env._robot.data.root_link_pos_w[ids,:2]
        env._two_ground_pt_xy_w[ids]=env._paired_route_origin[ids]+env._paired_route_offsets[ids,0]
        env._two_ground_pt1_xy_w[ids]=env._paired_route_origin[ids]+env._paired_route_offsets[ids,1]
        env._two_ground_route_index[ids]=0;env._two_ground_total_resets+=len(ids)
        _set_active_goal(env,ids)
    elif training and arm=='longmix':
        env._long_pair_hard[ids]=torch.rand(len(ids),device=env.device,generator=env._long_pair_mode_rng)<HARD_PROBABILITY[arm]
        env._two_ground_pt_xy_w[ids]=env._robot.data.root_link_pos_w[ids,:2]+_mixed_offset(env,ids)
        env._two_ground_pt1_xy_w[ids]=env._two_ground_pt_xy_w[ids]+_mixed_offset(env,ids)
        env._two_ground_route_index[ids]=0;env._two_ground_total_resets+=len(ids)
        _set_active_goal(env,ids)
    else:
        source_task.api('control').reset_pair(env,ids,seed)
    hard=env._long_pair_hard[ids].sum()
    env._long_pair_reset_counts+=torch.stack((hard.new_tensor(len(ids))-hard,hard))
    _record_edges(env,env._two_ground_pt_xy_w[ids]-env._robot.data.root_link_pos_w[ids,:2])
    _record_edges(env,env._two_ground_pt1_xy_w[ids]-env._two_ground_pt_xy_w[ids])


def advance_on_hit(env,hit):
    ids=torch.nonzero(hit,as_tuple=False).flatten()
    if not len(ids):return
    if env._long_pair_paired:
        index=env._two_ground_route_index[ids]+2
        if torch.any(index>=ROUTE_EDGES):raise RuntimeError('Paired target schedule exhausted; no wrap or random fallback')
        env._two_ground_pt_xy_w[ids]=env._two_ground_pt1_xy_w[ids].clone()
        env._two_ground_pt1_xy_w[ids]=env._paired_route_origin[ids]+env._paired_route_offsets[ids,index]
        env._two_ground_route_index[ids]+=1;env._two_ground_total_advances+=len(ids)
        _set_active_goal(env,ids)
    elif env._long_pair_training and env._long_pair_arm=='longmix':
        env._two_ground_pt_xy_w[ids]=env._two_ground_pt1_xy_w[ids].clone()
        env._two_ground_pt1_xy_w[ids]=env._two_ground_pt_xy_w[ids]+_mixed_offset(env,ids)
        env._two_ground_route_index[ids]+=1;env._two_ground_total_advances+=len(ids)
        _set_active_goal(env,ids)
    else:source_task.api('control').advance_on_hit(env,hit)
    _record_edges(env,env._two_ground_pt1_xy_w[ids]-env._two_ground_pt_xy_w[ids])


def audit_pair(env):
    result=source_task.api('control').audit_pair(env)
    result.update(long_pair_arm=env._long_pair_arm,training_mode=env._long_pair_training,
        training_mixture_enabled=env._long_pair_training and env._long_pair_arm=='longmix',
        configured_hard_probability=HARD_PROBABILITY[env._long_pair_arm],hard_edge_range_m=[HARD_MIN_M,MAX_M],
        ordinary_hard_reset_counts=env._long_pair_reset_counts.tolist(),edge_all_zero_short_long=env._long_pair_edge_counts.tolist(),
        active_hard_rows=int(env._long_pair_hard.sum()),paired_route_schedule=env._long_pair_paired,
        paired_route_edges_sha256=getattr(env,'_paired_route_edges_hash',None),
        paired_route_offsets_sha256=getattr(env,'_paired_route_offsets_hash',None))
    radius=torch.linalg.vector_norm(env._two_ground_pt1_xy_w-env._two_ground_pt_xy_w,dim=-1)
    if torch.any(radius[env._long_pair_hard]<HARD_MIN_M-1e-5):raise RuntimeError('Hard mode edge below0.6m')
    if not env._long_pair_training and env._long_pair_hard.any():raise RuntimeError('Training hard mode leaked into eval')
    return result


def api(arm,training=False,paired=False):
    return SimpleNamespace(VARIANT=VARIANTS[arm],SERIES=SERIES[arm],contract=lambda:contract(arm),validate_source=validate_source,
        reset_pair=lambda e,ids,seed:reset_pair(e,ids,seed,arm,training,paired),advance_on_hit=advance_on_hit,audit_pair=audit_pair)


@contextlib.contextmanager
def paired_reset_context(env,ids):
    """Discarded post-quota resets must not shift active rows' noise/delay RNG."""
    collector=getattr(env,'collector',None)
    selected=range(env.num_envs) if ids is None else ids.detach().cpu().tolist()
    discard=collector is not None and all(collector.counts[i]>=collector.p['episodes_per_env'] for i in selected)
    if not discard:
        yield;return
    from .fixed_case_rewards import reset_rng_context
    with reset_rng_context(env.device):yield


def instrument_paired_collector(base):
    class PairedStateCollector(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.paired_initial_state=None
        def record_reset_states(self):
            super().record_reset_states()
            if self.paired_initial_state is not None:return
            e,d=self.env,self.env._robot.data
            tensors=dict(root_link_pos_w=d.root_link_pos_w,root_quat_w=d.root_quat_w,
                root_com_velocity_w=d.root_com_lin_vel_w,root_angular_velocity_w=d.root_ang_vel_w,
                joint_pos=d.joint_pos,joint_vel=d.joint_vel,mass_multiplier=e.dr_mass_multi,
                inertia_multiplier=e.dr_inertia_multi,motor_tau=e.dr_tm,motor_state=e._motor_u,
                motor_power_history=e._motor_u_history,soc_tracker=e._soc_tracker,
                action_history=torch.stack(tuple(e._action_history),dim=1),P_t=e._two_ground_pt_xy_w,P_next=e._two_ground_pt1_xy_w)
            self.paired_initial_state={k:v.detach().cpu().tolist() for k,v in tensors.items()}
            self.paired_initial_state['route_edges_sha256']=e._paired_route_edges_hash
            self.paired_initial_state['route_offsets_sha256']=e._paired_route_offsets_hash
            self.paired_initial_state['torch_cpu_rng_sha256']=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            device=torch.device(e.device)
            if device.type=='cuda':
                self.paired_initial_state['torch_device_rng_sha256']=hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
            from .artifacts import write_json
            write_json(self.folder/'paired_initial_states.json',self.paired_initial_state)
        def result(self):
            result=super().result()
            result['paired_evaluation']=dict(version='stage4h_paired_schedule_v1',num_envs=self.env.num_envs,
                episodes_per_env=1,route_edges=ROUTE_EDGES,initial_state_sha256=sha(self.folder/'paired_initial_states.json'),
                route_edges_sha256=self.env._paired_route_edges_hash,
                route_offsets_sha256=self.env._paired_route_offsets_hash,
                scope='same initial DR/state and same ground-point schedule by route index; missed attempts retry, later states/route indices can diverge; finite15s simulation only')
            return result
    return PairedStateCollector
