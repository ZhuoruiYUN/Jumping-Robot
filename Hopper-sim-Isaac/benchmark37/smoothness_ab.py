"""Stage4c matched budget: only action_rate_reward_scale differs by arm."""

import json
from types import SimpleNamespace

from . import range08_curriculum as task
from .artifacts import ROOT, sha
from .calibration_20261028_v2 import physics_contract, REVISION
from .point_energy_margin import reward_contract as inherited_reward_contract

SOURCE = ROOT / "outputs/benchmark37_stage4b_range08/train_seed42_20261028_095930_178194Z/model_199.pt"
SOURCE_SHA256 = "1f125d1ece3fd62b05c0bffee65ad51a19af6869cfc6687cffdfbb41ff43f4c9"
ARMS = ("control", "smooth")
VARIANTS = {"control": "real_release_two_ground_range08_smooth_control",
            "smooth": "real_release_two_ground_range08_smooth_penalty"}
SERIES = {"control": "benchmark37_stage4c_smooth_control", "smooth": "benchmark37_stage4c_smooth_penalty"}
ITERATIONS = 50
SCALES = {"control": -.5, "smooth": -1.}


def contract(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown smoothness arm: {arm}")
    return dict(name="range08_action_rate_ab_v1", arm=arm, variant=VARIANTS[arm],
                source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA256,
                source_task_contract=task.contract(),
                intervention="Matched 50-update full-range continuation; only clipped-action delta-square coefficient differs: control -0.5, smooth -1.0",
                action_rate_reward_scale=SCALES[arm],
                reward_formula="scale * sum_i (a_t_i-a_prev_i)^2 * policy_dt, inherited unconditional all-phase term; no new action filtering or phase gate",
                sampling="Both arms use unchanged 0..0.8m radius sampler from update0; 10% exact stationary edges; full-circle headings; no repeated ramp",
                unchanged="41D actor/critic and normalization; LSTM/history; fixed v2 physics/motor/DR; release 0.8..1.2m with immediate policy control; world 1m apex, 10cm foot/15deg tilt gates; queue/retries, all other rewards and PPO",
                initialization="Same complete pinned 41D weights/normalizers, fresh optimizer, iteration0; 768 envs x 50 updates x 256 steps each; no smoke continuation",
                training_iterations=ITERATIONS,
                code_sha256=sha(ROOT / "benchmark37/smoothness_ab.py"))


def reward_contract(arm):
    return dict(inherited=inherited_reward_contract(),
                term="action_rate_p", original_scale=-.5, scale=SCALES[arm],
                delta_formula="(scale - original_scale) * sum_i(delta_action_i^2) * dt",
                arm=arm, code_sha256=sha(ROOT / "benchmark37/smoothness_ab.py"))


def configure(cfg, arm):
    if cfg.action_rate_reward_scale != -.5:
        raise ValueError("Inherited action-rate coefficient drifted from the reviewed -0.5 source")
    cfg.action_rate_reward_scale = SCALES[arm]
    return dict(arm=arm, actual_action_rate_reward_scale=cfg.action_rate_reward_scale,
                sampling_max_m=.8, training_curriculum=False, obs_dim=41,
                planned_updates=ITERATIONS)


def validate_source(path, source_lock, physics_lock):
    path = path.resolve(strict=True)
    m = json.loads((path.parent / "manifest.json").read_text())
    cfg = json.loads((path.parent / "env_config.json").read_text())
    end = m.get("two_ground_runtime_end", {})
    if (path != SOURCE.resolve() or sha(path) != SOURCE_SHA256
            or m.get("mode") != "train" or m.get("status") != "complete" or m.get("smoke")
            or m.get("variant") != task.VARIANT or m.get("task_intervention") != task.contract()
            or m.get("source_lock") != source_lock or m.get("physics_lock") != physics_lock
            or m.get("physics_revision") != REVISION or m.get("calibration_lock") != physics_contract()
            or m.get("checkpoint_hashes", {}).get(path.name) != SOURCE_SHA256
            or m.get("reward_intervention") != inherited_reward_contract()
            or cfg.get("action_rate_reward_scale") != -.5
            or end.get("current_sampling_max_m") != .8 or end.get("curriculum_update") != 200.):
        raise ValueError("Smoothness A/B requires the pinned completed Stage4b model_199 and matching full-range v2 contracts")
    return m


def reset_pair(env, ids, seed):
    env._range08_training = False
    task.reset_pair(env, ids, seed)


def audit_pair(env):
    if env._range08_training:
        raise RuntimeError("Smoothness A/B must remain at full 0..0.8m range")
    return task.audit_pair(env)


def api(arm):
    return SimpleNamespace(VARIANT=VARIANTS[arm], SERIES=SERIES[arm],
                           contract=lambda:contract(arm), validate_source=validate_source,
                           reset_pair=reset_pair, advance_on_hit=task.advance_on_hit, audit_pair=audit_pair)
