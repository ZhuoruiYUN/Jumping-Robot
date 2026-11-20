"""Stage4l: multiply the existing positive foot-touchdown term by 1.5 only."""
import math
from types import SimpleNamespace

import torch

from . import precision_refine_task as base
from .artifacts import ROOT, sha
from .foot_excess_ab import FootExcessTracker

SOURCE, SOURCE_SHA256 = base.SOURCE, base.SOURCE_SHA256
VARIANT = 'real_release_two_ground_range08_landing_gain'
SERIES = 'benchmark37_stage4l_landing_gain'
GAIN = 1.5
read = base.read
audit_runner = base.audit_runner
validate_source = base.validate_source


def configure_ppo(cfg):
    return dict(base.configure_ppo(cfg), positive_foot_gain=GAIN)


def contract():
    return dict(name='landing_gain_v1', variant=VARIANT,
        source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA256,
        intervention='Multiply only the existing positive foot_touchdown reward by1.5; effective Gaussian scale4->6',
        initialization='Exact41D actor/critic/LSTM/normalizers/learned std; fresh optimizer and iteration0; seed42,768x50x256',
        unchanged='5cm Gaussian width,10cm/1m/15deg/valid/alive gates,first-two x2,all other rewards including K50 apex deficit,full-circle0..0.8m sampler,41D observations,release0.8..1.2m immediate control,v2 physics/DR,PPO entropy0.002,15s,strict target advance/retry/history',
        hypothesis='Stronger existing landing objective may improve the10cm tail; unverified, no causal or deployment claim',
        code_sha256=sha(__file__))


def reward_contract():
    return dict(inherited=base.reward_contract(), term='foot_touchdown', multiplier=GAIN,
        old_gaussian_scale=4., new_gaussian_scale=6., width_m=.05,
        first_two_multiplier=2., strict_foot_limit_m=.10,
        formula='1.5 * inherited positive foot_touchdown; apex_deficit/foot_excess and all other terms unchanged',
        code_sha256=sha(__file__))


def api(training=False, paired=False):
    old = base.api(training, paired)
    return SimpleNamespace(**{**vars(old), 'VARIANT': VARIANT, 'SERIES': SERIES,
        'contract': contract, 'validate_source': validate_source})


class LandingGainTracker(FootExcessTracker):
    def __init__(self, n, device):
        super().__init__(n, device, 0.)
        self.gain_extra_by_slot = torch.zeros(3, device=device)

    def step(self, *args):
        positive = super().step(*args)
        extra = (GAIN - 1.) * positive
        for i in range(3):
            slot = (self.attempt_index == i+1) if i < 2 else (self.attempt_index >= 3)
            self.gain_extra_by_slot[i] += torch.where(slot, extra, 0.).sum()
        self.reward_sum += extra.sum()
        # Keep inherited last_qualified, attempt state and deficit settlement.
        return positive + extra

    def audit(self):
        old = super().audit()
        old.update(landing_gain=GAIN, gain_extra_reward_by_slot=self.gain_extra_by_slot.tolist(),
            actual_positive_foot_reward_by_slot=[
                a+b+c for a,b,c in zip(old['base_reward_by_slot'], old['extra_reward_by_slot'],
                                      self.gain_extra_by_slot.tolist())],
            total_positive_foot_reward=float(self.reward_sum))
        return old


def validate_checkpoint(path, allow_smoke=False):
    path = path.resolve(strict=True)
    m = read(path.parent/'manifest.json')
    smoke = bool(m.get('smoke'))
    updates, count = (2, 16) if smoke else (50, 768)
    cfg = read(path.parent/'runner_config.json')
    expected = read(SOURCE.parent/'env_config.json')
    expected['scene']['num_envs'] = count
    expected['csv_log_path'] = str(path.parent/'single_env.csv')
    audit, warm = m.get('landing_gain_runtime', {}), m.get('warm_start_audit', {})
    sampler, rewards = m.get('two_ground_runtime_end', {}), m.get('first_hop_margin_runtime', {})
    std = torch.load(SOURCE, map_location='cpu', weights_only=False)['model_state_dict']['std'].tolist()
    if (smoke and not allow_smoke or m.get('mode') != 'train' or m.get('status') != 'complete'
            or not m.get('simulation_started') or path.parent.parent != ROOT/'outputs'/SERIES
            or path.name != f'model_{updates-1}.pt' or m.get('variant') != VARIANT
            or m.get('requested_iterations') != updates or m.get('num_envs') != count
            or m.get('seed') != 42 or cfg['seed'] != 42 or cfg['max_iterations'] != updates
            or m.get('task_intervention') != contract() or m.get('reward_intervention') != reward_contract()
            or m.get('initial_checkpoint_sha256') != SOURCE_SHA256
            or m.get('checkpoint_hashes', {}).get(path.name) != sha(path)
            or m.get('landing_gain_runtime_config') != configure_ppo(cfg)
            or read(path.parent/'env_config.json') != expected
            or m.get('physics_revision') != base.source_task.inherited.REVISION
            or m.get('calibration_lock') != base.source_task.inherited.physics_contract()
            or audit.get('entropy_coef') != .002 or audit.get('initial_action_std') != std
            or audit.get('optimizer_state_entries') != 0 or audit.get('iteration') != 0
            or warm.get('source_checkpoint') != str(SOURCE) or warm.get('source_sha256') != SOURCE_SHA256
            or warm.get('new_iteration') != 0 or warm.get('optimizer_state_entries') != 0
            or not m.get('arguments', {}).get('landing_gain')
            or m['arguments'].get('precision_refine') or m['arguments'].get('entropy_refine_coef') is not None
            or m['arguments'].get('foot_excess_scale') != 0.
            or sampler.get('training_mode') is not True or sampler.get('training_mixture_enabled')
            or sampler.get('active_hard_rows') != 0 or sampler.get('current_sampling_max_m') != .8
            or rewards.get('landing_gain') != GAIN or rewards.get('scale') != 50.
            or rewards.get('reference_world_m_by_slot') != [1., 1., 1.]
            or rewards.get('foot_excess_scale') != 0.):
        raise ValueError('Incomplete/wrong Stage4l checkpoint or changed source/config/reward')
    data = torch.load(path, map_location='cpu', weights_only=False)
    state = data['model_state_dict']
    if (data['iter'] != updates-1 or state['memory_a.rnn.weight_ih_l0'].shape[1] != 41
            or state['memory_c.rnn.weight_ih_l0'].shape[1] != 41
            or not all(torch.isfinite(v).all() for v in state.values())):
        raise ValueError('Saved iteration/41D policy/finite parameters failed')
    for a,b,extra,actual in zip(rewards['base_reward_by_slot'], rewards['extra_reward_by_slot'],
            rewards['gain_extra_reward_by_slot'], rewards['actual_positive_foot_reward_by_slot']):
        if not (math.isclose(extra, (GAIN-1)*(a+b), rel_tol=2e-5, abs_tol=.01)
                and math.isclose(actual, GAIN*(a+b), rel_tol=2e-5, abs_tol=.01)):
            raise ValueError('Positive landing gain ledger differs from multiplier1.5')
    if not math.isclose(sum(rewards['actual_positive_foot_reward_by_slot']),
                        rewards['total_positive_foot_reward'], rel_tol=1e-4, abs_tol=.01):
        raise ValueError('Positive foot reward total/slot ledger differs')
    return path
