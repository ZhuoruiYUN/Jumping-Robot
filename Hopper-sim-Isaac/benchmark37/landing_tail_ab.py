"""Stage4d: remove only the foot reward's 10cm cutoff; strict gates unchanged."""

import json
from types import SimpleNamespace

import torch

from .artifacts import ROOT, sha
from . import smoothness_ab as source_task
from .calibration_20261028_v2 import REVISION, physics_contract
from .two_ground_targets import TwoGroundTracker
from .foot_touchdown_reward import REWARD_SCALE, FOOT_WIDTH_M, FOOT_LIMIT_M, APEX_MIN_M, TILT_LIMIT_DEG

SOURCE = ROOT / "outputs/benchmark37_stage4c_smooth_control/train_seed42_20261028_112425_724647Z/model_49.pt"
SOURCE_SHA256 = "955be180780eb9e0f6faf0727febbb73b0ea6ce8900876b73e7c2810dfb9d9c1"
ARMS = ("control", "continuous")
VARIANTS = {arm:f"real_release_two_ground_range08_tail_{arm}" for arm in ARMS}
SERIES = {arm:f"benchmark37_stage4d_foot_tail_{arm}" for arm in ARMS}
ITERATIONS = 50


def contract(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown landing-tail arm: {arm}")
    return dict(name="range08_continuous_foot_event_ab_v1", arm=arm, variant=VARIANTS[arm],
                source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
                source_task_contract=source_task.contract("control"),
                intervention="Only reward outside 10cm changes: control is unchanged; continuous extends the identical 5cm Gaussian beyond 10cm for otherwise-qualified touchdown events",
                reward="w_j*4*exp(-0.5*(foot_error/0.05)^2); w_j=2 for attempts1/2, else1. Control additionally gates foot_error<=0.10; continuous does not. Same valid-flight/apex>=1m/tilt<=15deg/alive gates",
                strict_scoring_and_route="foot<=0.10m plus existing height/tilt/valid-flight/alive gates; partial reward never advances target queue or counts as a strict hit",
                initialization="Same complete 41D weights/normalizers from Stage4c A; fresh optimizer, iteration0; 768 envs x50 updates x256 steps each; smoke weights not continued",
                sampling="Fixed full 0..0.8m both arms from update0; existing 10% exact-zero/full-circle sampler, queue and retries",
                unchanged="Source v2 physics, -0.5 action-rate coefficient, other rewards and PPO, 41D actor/critic, motor DR, root release0.8..1.2m with immediate policy control, event/reset/history and evaluation protocol",
                training_iterations=ITERATIONS,
                code_sha256=sha(ROOT / "benchmark37/landing_tail_ab.py"))


def reward_contract(arm):
    return dict(inherited=source_task.reward_contract("control"),
                term="foot_touchdown", tail_enabled=arm == "continuous",
                unchanged_inside_10cm=True, width_m=FOOT_WIDTH_M, scale=REWARD_SCALE,
                strict_foot_limit_m=FOOT_LIMIT_M,
                delta="existing attempt multiplier * 4*exp(-0.5*(error/0.05)^2) only for valid, height/tilt/alive-qualified error>0.10m events; control delta=0",
                code_sha256=sha(ROOT / "benchmark37/landing_tail_ab.py"))


def configure(cfg, arm):
    if arm not in ARMS or cfg.action_rate_reward_scale != -.5:
        raise ValueError("Landing-tail A/B must retain reviewed A's action-rate coefficient -0.5")
    return dict(arm=arm, tail_enabled=arm == "continuous", reward_width_m=FOOT_WIDTH_M,
                reward_scale=REWARD_SCALE, strict_foot_limit_m=FOOT_LIMIT_M,
                action_rate_reward_scale=cfg.action_rate_reward_scale, sampling_max_m=.8,
                training_curriculum=False, obs_dim=41, formal_updates=ITERATIONS)


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    m = json.loads((path.parent / "manifest.json").read_text())
    cfg = json.loads((path.parent / "env_config.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or m.get("mode") != "train" or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != source_task.VARIANTS["control"]
            or m.get("source_lock") != source_lock or m.get("physics_lock") != physics_lock
            or m.get("physics_revision") != REVISION or m.get("calibration_lock") != physics_contract()
            or m.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or m.get("task_intervention") != source_task.contract("control")
            or m.get("reward_intervention") != source_task.reward_contract("control")
            or m.get("requested_iterations") != 50 or cfg.get("action_rate_reward_scale") != -.5
            or m.get("two_ground_runtime_end", {}).get("current_sampling_max_m") != .8):
        raise ValueError("Landing-tail A/B requires the pinned completed Stage4c A model_49 and matching v2 contracts")
    return m


def api(arm):
    return SimpleNamespace(VARIANT=VARIANTS[arm], SERIES=SERIES[arm],
                           contract=lambda:contract(arm), validate_source=validate_source,
                           reset_pair=source_task.reset_pair, advance_on_hit=source_task.task.advance_on_hit,
                           audit_pair=source_task.audit_pair)


class LandingTailTracker(TwoGroundTracker):
    """Reuse the frozen state machine and strict mask, then add miss-only reward."""
    def __init__(self, num_envs, device, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown landing-tail arm: {arm}")
        super().__init__(num_envs, device)
        self.tail_arm = arm
        self.tail_eligible_counts = torch.zeros(3, dtype=torch.long, device=device)
        self.tail_rewarded_counts = torch.zeros_like(self.tail_eligible_counts)
        self.tail_reward_by_slot = torch.zeros(3, device=device)

    def step(self, q, z, vz, foot_xy, target_xy, tilt_deg, died):
        # Read boundaries before super updates phase/peak/attempt index.
        airborne = (self.phase == self.CANDIDATE) | (self.phase == self.FLIGHT)
        peak = torch.where(airborne, torch.maximum(self.peak, z), self.peak)
        apex_seen = self.apex_seen | (airborne & (vz <= 0))
        valid = ((self.phase == self.FLIGHT) & (q > .002) & (self.air_steps >= 5)
                 & apex_seen & ~died)
        error = torch.linalg.vector_norm(foot_xy-target_xy, dim=-1)
        eligible = valid & (peak >= APEX_MIN_M) & (tilt_deg <= TILT_LIMIT_DEG) & (error > FOOT_LIMIT_M)
        base = super().step(q,z,vz,foot_xy,target_xy,tilt_deg,died)
        if self.tail_arm == "continuous":
            factor = torch.where(self.attempt_index <= 2, 2., 1.)
            delta = torch.where(eligible, factor*REWARD_SCALE*torch.exp(-.5*(error/FOOT_WIDTH_M).square()),
                                torch.zeros_like(error))
        else:
            delta = torch.zeros_like(error)
        for i in range(3):
            slot = eligible & ((self.attempt_index == i+1) if i < 2 else (self.attempt_index >= 3))
            self.tail_eligible_counts[i] += slot.sum()
            self.tail_rewarded_counts[i] += (slot & (delta > 0)).sum()
            self.tail_reward_by_slot[i] += torch.where(slot,delta,0.).sum()
        self.reward_sum += delta.sum()
        # Route progression reads last_qualified, NOT returned reward>0.
        return base if self.tail_arm == "control" else base+delta

    def audit(self):
        result = super().audit()
        result.update(tail_arm=self.tail_arm, tail_enabled=self.tail_arm == "continuous",
                      tail_eligible_counts=self.tail_eligible_counts.tolist(),
                      tail_rewarded_counts=self.tail_rewarded_counts.tolist(),
                      tail_reward_by_slot=self.tail_reward_by_slot.tolist(),
                      total_reward_sum=float(self.reward_sum),
                      counter_note="base rewarded_counts and last_qualified remain strict hits; tail counts/rewards are separate and do not advance targets")
        return result
