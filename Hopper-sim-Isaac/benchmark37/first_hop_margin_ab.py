"""Stage4g: only the first confirmed hop's deficit reference gets a margin."""
import json
from types import SimpleNamespace

import torch

from . import apex_deficit_ab as source_task
from .artifacts import ROOT, sha
from .calibration_20261028_v2 import REVISION, physics_contract

SOURCE = ROOT / 'outputs/benchmark37_stage4f_apex_deficit_penalty/train_seed42_20261029_055902_461395Z/model_49.pt'
SOURCE_SHA256 = '324f88cd04a180ab18844ce5db4ca4b81f1118d3399fa5c107bbb5f866a5d639'
ARMS = ('control', 'margin3cm')
MARGINS = {'control': 0., 'margin3cm': .03}
SCALE = 50.
ITERATIONS = 50
VARIANTS = {a: 'real_release_two_ground_range08_first_hop_margin_'+a for a in ARMS}
SERIES = {a: 'benchmark37_stage4g_first_hop_margin_'+a for a in ARMS}


def references(arm):
    return [1.+MARGINS[arm], 1., 1.]


def contract(arm):
    return dict(name='first_hop_deficit_margin_v1',arm=arm,variant=VARIANTS[arm],
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_task.contract('penalty'),scale=SCALE,
        deficit_reference_world_m_by_slot=references(arm),
        slots=['first','second','later'],
        intervention='Only first confirmed post-drop hop deficit reference 1.00 vs1.03m; K50 in both arms, subsequent attempts1.00m',
        timing='Same final-flight root-link world peak at valid touchdown; no dt or first-two multiplier; no initial-drop/unconfirmed/short/no-apex/fatal payment',
        first_definition='Attempt index1 per reset, consumed on confirmed-flight touchdown even on foot/tilt/height miss; retry is attempt2 and has no extra margin',
        unchanged='41D actor/critic, four motors, five action histories, both ground targets and frames, v2 physics/DR/PPO, release0.8..1.2m with immediate control, full0..0.8m sampler, energy_tracking target1.04m, other rewards, strict foot<=.1m/apex>=1m/tilt<=15deg and target advance',
        initialization='Exact Stage4f K50 actor/critic/normalizers; fresh optimizer and iteration0; same seed42 and768x50x256 transitions per arm',
        training_iterations=ITERATIONS,code_sha256=sha(__file__))


def reward_contract(arm):
    return dict(inherited=source_task.reward_contract('penalty'),term='apex_deficit',scale=SCALE,
        reference_world_m_by_slot=references(arm),score_apex_world_m=1.,
        formula='-50*relu(reference_for_attempt-final_flight_peak_root_link_world_z)',
        timing='once at valid touchdown; no dt or early-attempt multiplier',code_sha256=sha(__file__))


def configure(cfg,arm):
    if arm not in ARMS or cfg.flight_power_penalty_scale != -.6 or cfg.action_rate_reward_scale != -.5:
        raise ValueError('First-hop margin requires unchanged Stage4f reward configuration')
    return dict(arm=arm,actual_apex_deficit_scale=SCALE,reference_world_m_by_slot=references(arm),
        score_apex_world_m=1.,obs_dim=41,planned_updates=ITERATIONS,
        settlement='final-flight peak at valid touchdown; first attempt only gets margin')


def validate_source(path,source_lock,physics_lock):
    path=path.resolve(strict=True)
    if path != SOURCE.resolve() or sha(path) != SOURCE_SHA256 or source_task.validate_trained_checkpoint(path) != 'penalty':
        raise ValueError('First-hop margin requires pinned formal Stage4f K50 model_49')
    m=json.loads((path.parent/'manifest.json').read_text())
    if m.get('source_lock') != source_lock or m.get('physics_lock') != physics_lock:
        raise ValueError('First-hop margin source reference/physics locks changed')
    return m


def validate_trained_checkpoint(path,allow_smoke=False):
    path=path.resolve(strict=True);m=json.loads((path.parent/'manifest.json').read_text())
    arm=next((a for a in ARMS if VARIANTS[a]==m.get('variant')),None)
    if arm is None:raise ValueError('Not a Stage4g first-hop margin checkpoint')
    smoke=bool(m.get('smoke'));updates=2 if smoke else ITERATIONS
    cfg=json.loads((path.parent/'env_config.json').read_text())
    if (smoke and not allow_smoke or m.get('status')!='complete' or m.get('mode')!='train'
            or path.parent.parent != ROOT/'outputs'/SERIES[arm] or path.name != f'model_{updates-1}.pt'
            or m.get('requested_iterations')!=updates or m.get('num_envs')!=(16 if smoke else 768)
            or m.get('task_intervention')!=contract(arm) or m.get('reward_intervention')!=reward_contract(arm)
            or m.get('initial_checkpoint_sha256')!=SOURCE_SHA256 or m.get('checkpoint_hashes',{}).get(path.name)!=sha(path)
            or m.get('physics_revision')!=REVISION or m.get('calibration_lock')!=physics_contract()
            or m.get('first_hop_margin_runtime_config')!=configure(SimpleNamespace(**cfg),arm)):
        raise ValueError('Incomplete or mismatched Stage4g checkpoint provenance')
    return arm


def api(arm):
    base=source_task.api('penalty')
    return SimpleNamespace(VARIANT=VARIANTS[arm],SERIES=SERIES[arm],contract=lambda:contract(arm),
        validate_source=validate_source,reset_pair=base.reset_pair,advance_on_hit=base.advance_on_hit,audit_pair=base.audit_pair)


class FirstHopMarginTracker(source_task.ApexDeficitTracker):
    """Add the first-hop margin delta to K50; preserve all foot/queue predicates."""
    def __init__(self,num_envs,device,arm):
        if arm not in ARMS:raise ValueError('Unknown first-hop margin arm')
        super().__init__(num_envs,device,'penalty')
        self.margin_arm=arm
        self.margin_extra_reward_by_slot=torch.zeros(3,device=device)
        self.reference_shortfall_events_by_slot=torch.zeros(3,device=device,dtype=torch.long)

    def step(self,q,z,vz,foot_xy,target_xy,tilt_deg,died):
        # Capture before source step resets peak/phase at touchdown. The source
        # last_deficit_event is authoritative for eligibility, including death.
        peak=torch.maximum(self.peak,z)
        foot_reward=super().step(q,z,vz,foot_xy,target_xy,tilt_deg,died)
        first=self.last_deficit_event & (self.attempt_index==1)
        extra=torch.where(first,((1.+MARGINS[self.margin_arm]-peak).clamp_min(0)
                                -(1.-peak).clamp_min(0)),torch.zeros_like(peak))
        self.last_deficit_m=self.last_deficit_m+extra
        delta=-SCALE*extra
        self.last_deficit_reward=self.last_deficit_reward+delta
        self.deficit_m_by_slot[0]+=extra.sum()
        self.deficit_reward_by_slot[0]+=delta.sum()
        self.margin_extra_reward_by_slot[0]+=delta.sum()
        for slot in range(3):
            selected=self.last_deficit_event & ((self.attempt_index==slot+1) if slot<2 else (self.attempt_index>=3))
            self.reference_shortfall_events_by_slot[slot]+=(selected & (self.last_deficit_m>0)).sum()
        return foot_reward

    def audit(self):
        result=super().audit()
        result.update(first_hop_margin_arm=self.margin_arm,reference_world_m_by_slot=references(self.margin_arm),
            score_apex_world_m=1.,score_shortfall_events_by_slot=result['shortfall_events_by_slot'],
            reward_reference_shortfall_events_by_slot=self.reference_shortfall_events_by_slot.tolist(),
            margin_extra_reward_by_slot=self.margin_extra_reward_by_slot.tolist(),
            accounting_note='deficit_m/reward use reward references; shortfall_events/score_shortfall_events still count below strict1m, not below margin reference')
        return result
