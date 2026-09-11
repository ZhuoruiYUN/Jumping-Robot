"""Train/evaluate current-target vs next-target preview with direct motor PPO."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('train', 'eval'), required=True)
parser.add_argument('--variant', choices=('current', 'preview'), required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=120)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--distance', choices=('range', '20cm'), default='range')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
source = Path(args.checkpoint).expanduser().resolve()
if not source.is_file():
    parser.error(f'Missing checkpoint: {source}')
output = Path(args.output).expanduser().resolve()
if output.exists():
    parser.error(f'Output already exists: {output}; choose a new experiment directory')
if min(args.num_envs, args.iterations, args.steps) <= 0:
    parser.error('Environment count, iterations and steps must be positive')
app = AppLauncher(args).app

import os
import sys
import random
import hashlib
import numpy as np
import torch
import isaaclab

sys.path.insert(0, os.path.join(os.path.dirname(isaaclab.__file__), 'source', 'isaaclab_rl'))
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.tracking_ab_env import TrackingABEnv, TrackingABEnvCfg

ROOT = Path(__file__).resolve().parent
CONTRACT = {
    'version': 2, 'protocol': 'preserved_preview_practical_ablation',
    'variant': args.variant, 'obs_dim': 43, 'actions': 4,
    'command_scale_m': 0.20, 'relative_next': True,
    'route': 'continuous_announced_targets', 'reference': 'landing_xy',
    'learning_rate': 1e-5, 'schedule': 'fixed', 'std': 0.03,
    'source_preview_weights': 'preserved',
    'actor_normalization': False, 'critic_normalization': False,
    'height_m': 1.0, 'height_tolerance_m': 0.15, 'target_tolerance_m': 0.10,
    'motor_tc_s': 0.0674, 'action_delay': 'legacy_indexing_setting_0',
    'source': str(source), 'seed': args.seed,
}


class ABRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos={'tracking_contract': CONTRACT})


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = torch.load(source, map_location='cpu', weights_only=False)
    state = data['model_state_dict']
    if state['memory_a.rnn.weight_ih_l0'].shape[1] != 43:
        raise ValueError('This comparison requires the selected 43-D source checkpoint')
    existing = (data.get('infos') or {}).get('tracking_contract')
    if args.mode == 'eval':
        if not existing or existing['version'] != 2 or existing['variant'] != args.variant:
            raise ValueError('Evaluation must match the embedded tracking A/B contract')
    elif existing:
        raise ValueError('Start both arms from the same legacy source, not a trained A/B arm')
    elif '2026-09-07_22-13-09' not in source.parts or source.name != 'model_10.pt':
        raise ValueError('Source scale not certified: use the selected 10--20 cm model_10.pt')

    cfg = TrackingABEnvCfg()
    cfg.preview = args.variant == 'preview'
    cfg.collect_tracking_metrics = args.mode == 'eval'
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = False
    if args.distance == '20cm':
        cfg.short_hop_radius_min = cfg.short_hop_radius_max = 0.20
        cfg.long_hop_radius_min = cfg.long_hop_radius_max = 0.20
    cfg.power_model_path = str(ROOT / 'Quadhopper_Stable/model/quadhopper_memory_power.pt')
    cfg.csv_log_path = str(output / 'single_env.csv')
    output.mkdir(parents=True)
    env = RslRlVecEnvWrapper(TrackingABEnv(cfg))
    rcfg = PlannerCircularPPORunnerCfg()
    rcfg.experiment_name = 'quadhopper_tracking_ab'
    rcfg.seed = args.seed
    rcfg.num_steps_per_env = 256
    rcfg.save_interval = 10
    rcfg.algorithm.schedule = 'fixed'
    rcfg.algorithm.learning_rate = CONTRACT['learning_rate']
    rcfg.algorithm.num_mini_batches = min(16, args.num_envs)
    rcfg.algorithm.entropy_coef = 0.0
    # The certified source has no serialized normalizer tensors. Use actual
    # identity modules; toggling update flags alone does not bypass forward().
    rcfg.empirical_normalization = False
    rcfg.policy.actor_obs_normalization = False
    rcfg.policy.critic_obs_normalization = False
    runner = ABRunner(env, rcfg.to_dict(), log_dir=str(output), device=args.device)
    if args.mode == 'train':
        # Preserve the source checkpoint's learned next-target pathway. The
        # current arm ablates it only at the observation boundary.
        state['std'].fill_(CONTRACT['std'])
    runner.alg.policy.load_state_dict(state)
    runner.alg.policy.actor_obs_normalization = False
    runner.alg.policy.critic_obs_normalization = False
    if not isinstance(runner.alg.policy.actor_obs_normalizer, torch.nn.Identity):
        raise RuntimeError('Actor must use identity normalization')
    if not isinstance(runner.alg.policy.critic_obs_normalizer, torch.nn.Identity):
        raise RuntimeError('Critic must use identity normalization')
    runner.alg.policy.std.requires_grad_(False)
    runner.current_learning_iteration = 0
    if env.num_actions != 4 or env.get_observations()['policy'].shape[-1] != 43:
        raise RuntimeError('Expected 43 observations and four direct motor actions')
    if runner.alg.schedule != 'fixed' or runner.alg.learning_rate != CONTRACT['learning_rate']:
        raise RuntimeError('Fixed learning rate contract was not applied')
    manifest = dict(CONTRACT, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    cli=vars(args), environment=cfg.to_dict(), runner=rcfg.to_dict())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print('[TRACKING-AB] ' + json.dumps(CONTRACT), flush=True)
    try:
        if args.mode == 'train':
            # This is genuinely pre-update; model_0 from RSL-RL is not.
            torch.save({'model_state_dict': runner.alg.policy.state_dict(), 'iter': 0,
                        'infos': {'tracking_contract': CONTRACT}}, output / 'initial.pt')
            runner.learn(args.iterations, init_at_random_ep_len=False)
            (output / 'train_complete.json').write_text(json.dumps({'iterations': args.iterations}))
        else:
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = env.reset()
            for _ in range(args.steps):
                with torch.inference_mode():
                    obs, _, dones, _ = env.step(policy(obs))
                    runner.alg.policy.reset(dones)
            base = env.unwrapped
            result = base.tracking_metrics.summary()
            result.update(variant=args.variant, checkpoint=str(source), seed=args.seed,
                          distance=args.distance, steps=args.steps, num_envs=args.num_envs,
                          death_rate=(base._physical_death_total > 0).float().mean().item(),
                          death_events=base._physical_death_total.sum().item())
            if result['touchdowns'] == 0:
                result['valid_evaluation'] = False
            else:
                result['valid_evaluation'] = True
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print('[TRACKING-RESULT] ' + json.dumps(result), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
