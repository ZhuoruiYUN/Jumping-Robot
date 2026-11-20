"""Frozen Stage4i reward contract and tracker; one coefficient K only."""
import json
from types import SimpleNamespace

import torch

from .artifacts import ROOT, sha
from . import long_pair_sampling_ab as inherited
from .first_hop_margin_ab import FirstHopMarginTracker

SOURCE=ROOT/'outputs/benchmark37_stage4h_long_pair_control/train_seed42_20261029_093043_308415Z/model_49.pt'
SOURCE_SHA256='93dd4d5d664e4aa6ebb35a20e3b1a047c25eaed83fc7bad139dcb6ecb6b4fdd4'
SCALES=(0.,50.)
SERIES={k:'benchmark37_stage4i_foot_excess_'+('control' if k==0 else 'penalty') for k in SCALES}
VARIANTS={k:'real_release_two_ground_range08_foot_excess_'+('control' if k==0 else 'penalty') for k in SCALES}


def contract(scale):
    if scale not in SCALES:raise ValueError('Foot excess scale must be0 or50')
    return dict(name='foot_excess_event_v1',scale_per_m=scale,variant=VARIANTS[scale],
        source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA256,
        formula='-K*relu(foot_center_XY_error_to_P_t-0.10m)',
        timing='Once at inherited valid alive post-drop touchdown; no dt/attempt multiplier; independent of apex/tilt gates',
        initialization='Same exact41D actor/critic/normalizers, fresh optimizer/iteration0;768x50x256 per arm',
        unchanged='Stage4h control full0..0.8m sampler,41D next target,K50 apex deficit1m,all old rewards/PPO,v2 physics/DR,release0.8..1.2m immediate control,15s,strict10cm/1m/15deg,retry/advance/history',
        code_sha256=sha(__file__))


def reward_contract(scale):
    return dict(inherited=inherited.reward_contract('control'),term='foot_excess',
                scale_per_m=scale,limit_m=.10,formula='-K*relu(e_foot-0.10)',code_sha256=sha(__file__))


def configure(cfg,scale):
    inherited.configure(cfg,'control')
    return dict(scale_per_m=scale,limit_m=.10,obs_dim=41,apex_deficit_scale=50.,apex_reference_world_m=1.,no_dt_or_attempt_multiplier=True)


def validate_source(path,source_lock,physics_lock):
    if path.resolve()!=SOURCE.resolve() or sha(path)!=SOURCE_SHA256 or inherited.validate_trained_checkpoint(path)!='control':
        raise ValueError('Foot excess requires pinned formal Stage4h control')
    m=json.loads((path.parent/'manifest.json').read_text())
    if m['source_lock']!=source_lock or m['physics_lock']!=physics_lock:raise ValueError('Source reference/physics lock differs')
    return m


def validate_checkpoint(path,allow_smoke=False):
    path=path.resolve(strict=True);m=json.loads((path.parent/'manifest.json').read_text())
    scale=next((k for k,v in VARIANTS.items() if m.get('variant')==v),None)
    if scale is None:raise ValueError('Not a foot-excess checkpoint')
    smoke=bool(m.get('smoke'));updates=2 if smoke else 50
    if (smoke and not allow_smoke or m.get('status')!='complete' or m.get('mode')!='train'
            or path.parent.parent!=ROOT/'outputs'/SERIES[scale] or path.name!=f'model_{updates-1}.pt'
            or m.get('requested_iterations')!=updates or m.get('num_envs')!=(16 if smoke else 768)
            or m.get('task_intervention')!=contract(scale) or m.get('reward_intervention')!=reward_contract(scale)
            or m.get('initial_checkpoint_sha256')!=SOURCE_SHA256 or m.get('checkpoint_hashes',{}).get(path.name)!=sha(path)
            or m.get('physics_revision')!=inherited.REVISION or m.get('calibration_lock')!=inherited.physics_contract()
            or m.get('foot_excess_runtime_config')!=configure(SimpleNamespace(**json.loads((path.parent/'env_config.json').read_text())),scale)):
        raise ValueError('Incomplete/wrong foot-excess model provenance')
    return scale


def api(scale,training=False,paired=False):
    base=inherited.api('control',training,paired)
    return SimpleNamespace(**{**vars(base),'VARIANT':VARIANTS[scale],'SERIES':SERIES[scale],
        'contract':lambda:contract(scale),'validate_source':validate_source})


class FootExcessTracker(FirstHopMarginTracker):
    def __init__(self,n,device,scale):
        super().__init__(n,device,'control')
        if scale not in SCALES:raise ValueError('Unsupported foot-excess scale')
        self.foot_excess_scale=scale
        self.last_foot_excess_reward=torch.zeros(n,device=device)
        self.foot_excess_events=torch.zeros(3,dtype=torch.long,device=device)
        self.foot_excess_m=torch.zeros(3,device=device)
        self.foot_excess_reward=torch.zeros(3,device=device)

    def reset(self,ids):
        super().reset(ids);self.last_foot_excess_reward[ids]=0

    def step(self,q,z,vz,foot_xy,target_xy,tilt_deg,died):
        base=super().step(q,z,vz,foot_xy,target_xy,tilt_deg,died)
        excess=torch.where(self.last_deficit_event,
            (torch.linalg.vector_norm(foot_xy-target_xy,dim=-1)-.10).clamp_min(0),0.)
        self.last_foot_excess_reward=-self.foot_excess_scale*excess
        for i in range(3):
            selected=self.last_deficit_event & ((self.attempt_index==i+1) if i<2 else (self.attempt_index>=3))
            self.foot_excess_events[i]+=(selected & (excess>0)).sum()
            self.foot_excess_m[i]+=torch.where(selected,excess,0.).sum()
            self.foot_excess_reward[i]+=torch.where(selected,self.last_foot_excess_reward,0.).sum()
        return base  # Preserve positive foot reward and strict qualification.

    def audit(self):
        return dict(super().audit(),foot_excess_scale=self.foot_excess_scale,
            foot_excess_events_by_slot=self.foot_excess_events.tolist(),
            foot_excess_m_by_slot=self.foot_excess_m.tolist(),foot_excess_reward_by_slot=self.foot_excess_reward.tolist())
