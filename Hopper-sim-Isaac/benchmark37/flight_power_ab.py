"""Stage4e: matched continuation; only the airborne power reward coefficient changes."""
import json
from types import SimpleNamespace

from . import landing_tail_ab as source_task
from .artifacts import ROOT, sha
from .calibration_20261028_v2 import REVISION, physics_contract

SOURCE = ROOT / "outputs/benchmark37_stage4d_foot_tail_control/train_seed42_20261028_134843_882013Z/model_49.pt"
SOURCE_SHA256 = "97ff6287b9c831cfe035e92cf997926b81d2c87147b085c519f0e0a1c68aac65"
ARMS = ("control", "relaxed")
VARIANTS = {arm: f"real_release_two_ground_range08_power_{arm}" for arm in ARMS}
SERIES = {arm: f"benchmark37_stage4e_flight_power_{arm}" for arm in ARMS}
ITERATIONS = 50
SCALES = {"control": -.6, "relaxed": -.3}


def contract(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown flight power arm: {arm}")
    return dict(name="range08_flight_power_ab_v1",arm=arm,variant=VARIANTS[arm],
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA256,
        source_task_contract=source_task.contract("control"),
        intervention="Only flight_power_penalty_scale changes: control -0.6; relaxed -0.3; no phase/event/physics changes",
        flight_power_penalty_scale=SCALES[arm],
        reward="scale * where(spring_q<=0.002, frozen_NN_power_proxy, 0) * policy_dt; original unclipped proxy retained",
        initialization="Exact same complete 41D policy/critic/normalizer tensors from pinned Stage4d control; fresh optimizer, iteration0; formal 768x50x256 transitions each",
        unchanged="41D observations/LSTM/action histories and 4 motor actions; v2 physics/DR; 0.8..1.2m release with immediate policy control; strict 10cm/1m/15deg events and route retries; action-rate -0.5, lateral velocity -10, attitude -20; foot Gaussian5cm with strict cutoff; PPO",
        sampling="Both arms full 0..0.8m from update0, 10% exact-zero, full-circle headings, no repeated ramp",
        training_iterations=ITERATIONS,code_sha256=sha(__file__))


def reward_contract(arm):
    return dict(inherited=source_task.reward_contract("control"),term="flight_power_p",arm=arm,
                original_scale=-.6,scale=SCALES[arm],
                delta="(scale+0.6)*where(spring_q<=0.002,predicted_power_proxy,0)*policy_dt; no new smoothing or energy model",
                code_sha256=sha(__file__))


def configure(cfg, arm):
    if arm not in ARMS or cfg.flight_power_penalty_scale != -.6 or cfg.action_rate_reward_scale != -.5:
        raise ValueError("Flight power A/B requires original flight=-0.6 and action-rate=-0.5 configuration")
    cfg.flight_power_penalty_scale = SCALES[arm]
    return dict(arm=arm,actual_flight_power_penalty_scale=cfg.flight_power_penalty_scale,
                sampling_max_m=.8,training_curriculum=False,obs_dim=41,planned_updates=ITERATIONS)


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    m = json.loads((path.parent/"manifest.json").read_text())
    cfg = json.loads((path.parent/"env_config.json").read_text())
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256 or m.get("mode") != "train"
            or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != source_task.VARIANTS["control"]
            or m.get("task_intervention") != source_task.contract("control")
            or m.get("reward_intervention") != source_task.reward_contract("control")
            or m.get("source_lock") != source_lock or m.get("physics_lock") != physics_lock
            or m.get("physics_revision") != REVISION or m.get("calibration_lock") != physics_contract()
            or m.get("checkpoint_hashes",{}).get(path.name) != SOURCE_SHA256
            or m.get("requested_iterations") != 50 or m.get("num_envs") != 768
            or cfg.get("flight_power_penalty_scale") != -.6 or cfg.get("action_rate_reward_scale") != -.5
            or m.get("two_ground_runtime_end",{}).get("current_sampling_max_m") != .8):
        raise ValueError("Flight power A/B requires the pinned completed Stage4d control model_49, full-range v2 contracts")
    return m


def validate_trained_checkpoint(path, allow_smoke=False):
    """Select future results by complete provenance and file hash, never by filename alone."""
    path = path.resolve(strict=True)
    m = json.loads((path.parent/"manifest.json").read_text())
    arm = next((a for a in ARMS if VARIANTS[a] == m.get("variant")),None)
    if arm is None:
        raise ValueError("Not a Stage4e flight power checkpoint")
    smoke = bool(m.get("smoke"))
    updates = 2 if smoke else ITERATIONS
    if (smoke and not allow_smoke or m.get("status") != "complete" or m.get("mode") != "train"
            or path.parent.parent != ROOT/"outputs"/SERIES[arm]
            or path.name != f"model_{updates-1}.pt" or m.get("requested_iterations") != updates
            or m.get("num_envs") != (16 if smoke else 768)
            or m.get("task_intervention") != contract(arm) or m.get("reward_intervention") != reward_contract(arm)
            or m.get("initial_checkpoint_sha256") != SOURCE_SHA256
            or m.get("checkpoint_hashes",{}).get(path.name) != sha(path)
            or m.get("physics_revision") != REVISION or m.get("calibration_lock") != physics_contract()):
        raise ValueError("Incomplete or mismatched Stage4e training provenance/checkpoint")
    cfg=json.loads((path.parent/"env_config.json").read_text())
    if cfg.get("flight_power_penalty_scale") != SCALES[arm] or cfg.get("action_rate_reward_scale") != -.5:
        raise ValueError("Saved Stage4e reward coefficient does not match its arm")
    return arm


def api(arm):
    inherited = source_task.api("control")
    return SimpleNamespace(VARIANT=VARIANTS[arm],SERIES=SERIES[arm],contract=lambda:contract(arm),
        validate_source=validate_source,reset_pair=inherited.reset_pair,
        advance_on_hit=inherited.advance_on_hit,audit_pair=inherited.audit_pair)
