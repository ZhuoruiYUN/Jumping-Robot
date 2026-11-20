"""Stage4f: one finalized flight-apex deficit penalty, scored at valid touchdown."""
import json
from types import SimpleNamespace

import torch

from . import flight_power_ab as source_task
from .artifacts import ROOT, sha
from .calibration_20261028_v2 import REVISION, physics_contract
from .two_ground_targets import TwoGroundTracker

SOURCE = ROOT / "outputs/benchmark37_stage4e_flight_power_control/train_seed42_20261029_034615_105001Z/model_49.pt"
SOURCE_SHA256 = "5fc7d38618813ef2f8bf789eed99843a135aee9a79f3ae8f5fa247abbd27e106"
ARMS = ("control", "penalty")
SCALES = {"control": 0., "penalty": 50.}
VARIANTS = {a: f"real_release_two_ground_range08_apex_deficit_{a}" for a in ARMS}
SERIES = {a: f"benchmark37_stage4f_apex_deficit_{a}" for a in ARMS}
ITERATIONS = 50
HEIGHT_M = 1.0


def contract(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown apex deficit arm: {arm}")
    return dict(name="range08_apex_deficit_ab_v1",arm=arm,variant=VARIANTS[arm],
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_task.contract("control"),
        intervention="Only add separate apex_deficit event reward; control K=0, deficit K=50 per metre",
        scale=SCALES[arm],height_world_m=HEIGHT_M,
        formula="-K*max(1.0-final_flight_peak_root_link_world_z,0); once at confirmed valid post-drop touchdown, no dt or first-two multiplier",
        timing="Finalize full-flight maximum at touchdown, not first local apex. Initial drop, unconfirmed bounce, <5 air samples, no apex seen and fatal touch excluded; repeated compressed contact cannot pay twice",
        gating="No foot-error or tilt gate on deficit penalty. Existing positive foot reward and strict target advancement remain unchanged",
        observation="Same 41D actor/critic, 4 motor actions, root-Body target transforms and five action histories; simulator peak truth used only in reward/evaluation",
        initialization="Same exact source actor/critic/normalizers, fresh optimizer, iteration0; formal 768x50x256 transitions per arm; smoke16x2 only diagnostic",
        sampling="Full 0..0.8m, 10% exact stationary edges, full-circle from update0; no repeated ramp",
        unchanged="v2 physics/motor/DR, release0.8..1.2m with immediate policy control, flight-power -0.6, action-rate -0.5 and all old reward terms/PPO; strict foot<=10cm/apex>=1m/tilt<=15deg and queue retries",
        evaluation="One development seed7001, random two_ground15 plus complete motion/energy diagnostics; fixed grid available separately for selected candidates",
        training_iterations=ITERATIONS,code_sha256=sha(__file__))


def reward_contract(arm):
    return dict(inherited=source_task.reward_contract("control"),term="apex_deficit",scale=SCALES[arm],
                height_world_m=HEIGHT_M,formula="-K*relu(1.0-final_flight_peak)",
                timing="once at valid touchdown; no dt or early-attempt multiplier",code_sha256=sha(__file__))


def configure(cfg, arm):
    if arm not in ARMS or cfg.flight_power_penalty_scale != -.6 or cfg.action_rate_reward_scale != -.5:
        raise ValueError("Apex deficit A/B requires reviewed source flight/action-rate coefficients")
    return dict(arm=arm,actual_apex_deficit_scale=SCALES[arm],height_world_m=HEIGHT_M,
                settlement="full-flight maximum finalized at valid touchdown",obs_dim=41,
                flight_power_penalty_scale=-.6,planned_updates=ITERATIONS)


def validate_source(path, source_lock, physics_lock):
    path=path.resolve(strict=True)
    if path != SOURCE.resolve() or sha(path) != SOURCE_SHA256 or source_task.validate_trained_checkpoint(path) != "control":
        raise ValueError("Apex deficit A/B requires the pinned completed Stage4e control model_49")
    m=json.loads((path.parent/"manifest.json").read_text())
    if m.get("source_lock") != source_lock or m.get("physics_lock") != physics_lock:
        raise ValueError("Apex deficit source physics/reference locks changed")
    return m


def validate_trained_checkpoint(path, allow_smoke=False):
    path=path.resolve(strict=True)
    m=json.loads((path.parent/"manifest.json").read_text())
    arm=next((a for a in ARMS if VARIANTS[a]==m.get("variant")),None)
    if arm is None:
        raise ValueError("Not a Stage4f apex deficit checkpoint")
    smoke=bool(m.get("smoke"));updates=2 if smoke else ITERATIONS
    runtime=m.get("apex_deficit_runtime_config",{})
    if (smoke and not allow_smoke or m.get("status") != "complete" or m.get("mode") != "train"
            or path.parent.parent != ROOT/"outputs"/SERIES[arm] or path.name != f"model_{updates-1}.pt"
            or m.get("requested_iterations") != updates or m.get("num_envs") != (16 if smoke else 768)
            or m.get("task_intervention") != contract(arm) or m.get("reward_intervention") != reward_contract(arm)
            or m.get("initial_checkpoint_sha256") != SOURCE_SHA256
            or m.get("checkpoint_hashes",{}).get(path.name) != sha(path)
            or m.get("physics_revision") != REVISION or m.get("calibration_lock") != physics_contract()
            or runtime.get("actual_apex_deficit_scale") != SCALES[arm]):
        raise ValueError("Incomplete or mismatched Stage4f training provenance/checkpoint")
    cfg=json.loads((path.parent/"env_config.json").read_text())
    if cfg.get("flight_power_penalty_scale") != -.6 or cfg.get("action_rate_reward_scale") != -.5:
        raise ValueError("Stage4f inherited reward coefficients changed")
    return arm


def api(arm):
    base=source_task.api("control")
    return SimpleNamespace(VARIANT=VARIANTS[arm],SERIES=SERIES[arm],contract=lambda:contract(arm),
        validate_source=validate_source,reset_pair=base.reset_pair,advance_on_hit=base.advance_on_hit,audit_pair=base.audit_pair)


class ApexDeficitTracker(TwoGroundTracker):
    """Preserve the original foot tracker return/gate and expose a separate loss."""
    def __init__(self, num_envs, device, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown apex deficit arm: {arm}")
        super().__init__(num_envs,device)
        self.deficit_arm=arm
        self.last_deficit_reward=torch.zeros(num_envs,device=device)
        self.last_deficit_event=torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.last_deficit_m=torch.zeros(num_envs,device=device)
        self.deficit_events_by_slot=torch.zeros(3,device=device,dtype=torch.long)
        self.shortfall_events_by_slot=torch.zeros_like(self.deficit_events_by_slot)
        self.deficit_m_by_slot=torch.zeros(3,device=device)
        self.deficit_reward_by_slot=torch.zeros(3,device=device)

    def reset(self, ids):
        super().reset(ids)
        self.last_deficit_reward[ids]=0
        self.last_deficit_event[ids]=False
        self.last_deficit_m[ids]=0

    def step(self,q,z,vz,foot_xy,target_xy,tilt_deg,died):
        # Match the frozen valid-touch predicate, before its phase/peak reset.
        airborne=(self.phase==self.CANDIDATE)|(self.phase==self.FLIGHT)
        peak=torch.where(airborne,torch.maximum(self.peak,z),self.peak)
        apex_seen=self.apex_seen | (airborne & (vz<=0))
        event=((self.phase==self.FLIGHT)&(q>.002)&(self.air_steps>=5)&apex_seen&~died)
        deficit=torch.where(event,(HEIGHT_M-peak).clamp_min(0),torch.zeros_like(peak))
        self.last_deficit_reward=-SCALES[self.deficit_arm]*deficit
        self.last_deficit_event=event
        self.last_deficit_m=deficit
        base=super().step(q,z,vz,foot_xy,target_xy,tilt_deg,died)
        for i in range(3):
            slot=event & ((self.attempt_index==i+1) if i<2 else (self.attempt_index>=3))
            self.deficit_events_by_slot[i]+=slot.sum()
            self.shortfall_events_by_slot[i]+=(slot&(deficit>0)).sum()
            self.deficit_m_by_slot[i]+=torch.where(slot,deficit,0.).sum()
            self.deficit_reward_by_slot[i]+=torch.where(slot,self.last_deficit_reward,0.).sum()
        # Keep base reward_sum/last_qualified strictly about the old foot term.
        return base

    def audit(self):
        result=super().audit()
        result.update(deficit_arm=self.deficit_arm,scale=SCALES[self.deficit_arm],
            settlement="full-flight maximum at valid touchdown; independent of foot and tilt gates",
            deficit_events_by_slot=self.deficit_events_by_slot.tolist(),
            shortfall_events_by_slot=self.shortfall_events_by_slot.tolist(),
            deficit_m_by_slot=self.deficit_m_by_slot.tolist(),
            deficit_reward_by_slot=self.deficit_reward_by_slot.tolist())
        return result
