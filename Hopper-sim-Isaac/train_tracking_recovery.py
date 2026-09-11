"""Train/evaluate continuous direct-motor tracking with untimed spatial guidance."""
import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('train', 'eval'), default='train')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=240)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--distance-min', type=float, default=0.20)
parser.add_argument('--distance-max', type=float, default=0.25)
parser.add_argument('--hard-distance-min', type=float, default=0.25,
                    help='Boundary between replay and hard distance intervals.')
parser.add_argument('--hard-sample-probability', type=float, default=0.0,
                    help='Probability of sampling [hard-distance-min, distance-max].')
parser.add_argument('--max-turn-angle-deg', type=float, default=30.0,
                    help='Maximum signed heading change between consecutive waypoints.')
parser.add_argument('--zero-hop-probability', type=float, default=0.0,
                    help='Probability that an appended waypoint is exactly at the current landing point.')
parser.add_argument('--reverse-turn-probability', type=float, default=0.0,
                    help='Probability of a heading reversal around 180 degrees after a hop.')
parser.add_argument('--reverse-turn-halfwidth-deg', type=float, default=30.0,
                    help='Half-width of the reversal band around +/- 180 degrees.')
parser.add_argument('--landing-precision-width', type=float,
                    help='Gaussian touchdown-precision width in metres; unset preserves the checkpoint curriculum default.')
parser.add_argument('--target-hit-reward-scale', type=float,
                    help='Override sparse touchdown reward; unset preserves the default.')
parser.add_argument('--landing-precision-reward-scale', type=float,
                    help='Override dense touchdown-precision reward; unset preserves the default.')
parser.add_argument('--landing-error-penalty-scale', type=float,
                    help='Override radial touchdown-error penalty; unset preserves the default.')
parser.add_argument('--landing-lateral-error-penalty-scale', type=float,
                    help='Override lateral touchdown-error penalty; unset preserves the default.')
parser.add_argument('--robust-dynamics', action='store_true',
                    help='Enable mass/inertia/motor-delay randomization for sim-to-real robustness.')
parser.add_argument('--observation-noise-std', type=float,
                    help='Override policy observation noise standard deviation.')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--target-tolerance', type=float, default=0.05)
parser.add_argument('--allow-compatible-checkpoint', action='store_true',
                    help='Evaluate an older compatible 52-D recovery checkpoint under this environment.')
parser.add_argument('--visualize', action='store_true',
                    help='Enable Isaac debug visualization during evaluation.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
source = Path(args.checkpoint).expanduser().resolve()
output = Path(args.output).expanduser().resolve()
if not source.is_file():
    parser.error(f'Missing checkpoint: {source}')
if output.exists():
    parser.error(f'Output already exists: {output}')
if min(args.num_envs, args.iterations, args.steps) <= 0:
    parser.error('num_envs, iterations and steps must be positive')
if not (0 <= args.distance_min <= args.distance_max and args.distance_max > 0.0) or args.target_tolerance <= 0:
    parser.error('Require 0 <= distance-min <= distance-max, distance-max > 0, and positive target-tolerance')
if args.learning_rate <= 0.0:
    parser.error('learning-rate must be positive')
if not 0.0 <= args.hard_sample_probability <= 1.0:
    parser.error('hard-sample-probability must be in [0, 1]')
if not 0.0 < args.max_turn_angle_deg <= 180.0:
    parser.error('max-turn-angle-deg must be in (0, 180]')
if not 0.0 <= args.zero_hop_probability <= 1.0:
    parser.error('zero-hop-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_probability <= 1.0:
    parser.error('reverse-turn-probability must be in [0, 1]')
if not 0.0 < args.reverse_turn_halfwidth_deg <= 180.0:
    parser.error('reverse-turn-halfwidth-deg must be in (0, 180]')
if args.reverse_turn_probability > 0.0 and (
    args.max_turn_angle_deg < 180.0 - args.reverse_turn_halfwidth_deg
):
    parser.error('max-turn-angle-deg must include the configured reverse-turn band')
if args.landing_precision_width is not None and args.landing_precision_width <= 0.0:
    parser.error('landing-precision-width must be positive')
if args.observation_noise_std is not None and args.observation_noise_std < 0.0:
    parser.error('observation-noise-std must be non-negative')
if args.hard_sample_probability > 0.0 and not (
    args.distance_min < args.hard_distance_min < args.distance_max
):
    parser.error('hard-distance-min must lie strictly inside the distance range')
if args.mode == 'eval' and args.hard_sample_probability != 0.0:
    parser.error('Evaluation must use uniform distances; set hard-sample-probability to 0')
if args.visualize and args.mode != 'eval':
    parser.error('--visualize is only supported in eval mode')
app = AppLauncher(args).app

import os
import random
import sys
import numpy as np
import torch
import isaaclab

sys.path.insert(0, os.path.join(os.path.dirname(isaaclab.__file__), 'source', 'isaaclab_rl'))
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.spatial_tracking_env import SpatialTrackingEnv, SpatialTrackingEnvCfg
from Quadhopper_Planner_Random.spatial_path import expand_policy_inputs

ROOT = Path(__file__).resolve().parent
CONTRACT = {
    'version': 11,
    'protocol': 'continuous_untimed_hard_replay_pair_precision_curriculum',
    'source': str(source),
    'obs_dim': 52,
    'actions': 4,
    'preview': True,
    'distance_range_m': [args.distance_min, args.distance_max],
    'max_turn_angle_deg': args.max_turn_angle_deg,
    'zero_hop_probability': args.zero_hop_probability,
    'reverse_turn_probability': args.reverse_turn_probability,
    'reverse_turn_halfwidth_deg': args.reverse_turn_halfwidth_deg,
    'distance_sampling': (
        {
            'mode': 'weighted_two_interval',
            'replay_range_m': [args.distance_min, args.hard_distance_min],
            'hard_range_m': [args.hard_distance_min, args.distance_max],
            'hard_probability': args.hard_sample_probability,
        }
        if args.hard_sample_probability > 0.0
        else {'mode': 'uniform', 'range_m': [args.distance_min, args.distance_max]}
    ),
    'target_tolerance_m': args.target_tolerance,
    'waypoint_observation_scale_m': 0.20,
    'path_corridor_m': 0.08,
    'path_lookahead_m': 0.10,
    'spatial_progress_reward_per_hop': 12.0,
    'path_tracking_reward_per_second': 16.0,
    'path_tracking_width_m': 0.08,
    # XY progress weights are populated from the actual cfg below, rather
    # than recording a number that can disagree with inherited defaults.
    'xy_progress_form': 'signed_distance_difference_m',
    'xy_progress_gating': 'same_target_consecutive_airborne_steps',
    'two_hop_pair_definition': 'second_phase_touchdown_and_two_consecutive_5cm_hits',
    'target_hit_reward': 120.0,
    'landing_precision_reward': 60.0,
    'landing_error_penalty': -120.0,
    'landing_lateral_error_penalty': -60.0,
    'projected_landing_reward_per_second': 30.0,
    'projected_landing_error_penalty_per_second': -30.0,
    'stable_bonus_profile': {
        'survival': 2.0, 'distance_to_z': 1.0,
        'energy_tracking': 3.0, 'apex': 3.0,
    },
    'path_penalty_max_per_second': 1.0,
    'landing_hint_penalty_max_per_second': 0.0,
    'target_hit_requires_minimum_apex': False,
    'target_hit_requires_apex_tolerance': False,
    'landing_precision_gated_by_apex': False,
    'initial_drop_height_m': [0.65, 1.05],
    'guidance_clock': False,
    'episode_length_s': 15.0,
    'appended_observations': ['carrot_error_body_xyz_m', 'path_tangent_body_xyz', 'landing_up_hint_body_xyz'],
    'learning_rate': args.learning_rate,
    'schedule': 'fixed',
    'std': 0.04,
    'motor_tc_s': 0.0674,
    'action_delay': 'legacy_indexing_setting_0',
    'seed': args.seed,
}


class RecoveryRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos={'tracking_recovery_contract': CONTRACT})


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = torch.load(source, map_location='cpu', weights_only=False)
    state = data['model_state_dict']
    infos = data.get('infos') or {}
    prior = infos.get('tracking_contract')
    recovery = infos.get('tracking_recovery_contract', {})
    if args.mode == 'train':
        if prior and prior.get('version') == 2 and prior.get('variant') == 'preview':
            state = expand_policy_inputs(state)
        elif recovery.get('version') not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
            raise ValueError('Curriculum training requires a compatible 52-D spatial checkpoint')
        state['std'].fill_(CONTRACT['std'])
    else:
        spatial = (data.get('infos') or {}).get('tracking_recovery_contract', {})
        current_contract = (
            spatial.get('protocol') == CONTRACT['protocol']
            and spatial.get('version') == CONTRACT['version']
        )
        compatible_prior = spatial.get('version') in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        if not current_contract and not (args.allow_compatible_checkpoint and compatible_prior):
            raise ValueError('Evaluation requires a hard-replay pair-precision v11 checkpoint; '
                             'use --allow-compatible-checkpoint for an older spatial policy')

    cfg = SpatialTrackingEnvCfg()
    cfg.short_hop_radius_min = args.distance_min
    cfg.short_hop_radius_max = args.distance_max
    cfg.long_hop_radius_min = args.distance_min
    cfg.long_hop_radius_max = args.distance_max
    cfg.hard_radius_split = args.hard_distance_min
    cfg.hard_radius_probability = args.hard_sample_probability
    cfg.max_turn_angle_deg = args.max_turn_angle_deg
    cfg.zero_hop_probability = args.zero_hop_probability
    cfg.reverse_turn_probability = args.reverse_turn_probability
    cfg.reverse_turn_halfwidth_deg = args.reverse_turn_halfwidth_deg
    reward_overrides = {
        'landing_precision_width': args.landing_precision_width,
        'target_hit_reward_scale': args.target_hit_reward_scale,
        'landing_precision_reward_scale': args.landing_precision_reward_scale,
        'landing_error_penalty_scale': args.landing_error_penalty_scale,
        'landing_lateral_error_penalty_scale': args.landing_lateral_error_penalty_scale,
    }
    for name, value in reward_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.robust_dynamics:
        cfg.randomize_dynamics = True
        cfg.randomize_action_delay = True
    if args.observation_noise_std is not None:
        cfg.observation_noise_std = args.observation_noise_std
    cfg.target_tolerance = args.target_tolerance
    cfg.collect_tracking_metrics = args.mode == 'eval'
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = args.visualize
    cfg.power_model_path = str(ROOT / 'Quadhopper_Stable/model/quadhopper_memory_power.pt')
    cfg.csv_log_path = str(output / 'single_env.csv')
    if cfg.xy_progress_reward_scale != 0.0:
        raise ValueError('Legacy XY progress must stay disabled to avoid duplicate/queue-jump rewards')
    CONTRACT.update(
        xy_potential_progress_scale=cfg.spatial_xy_progress_reward_scale,
        legacy_xy_progress_reward_scale=cfg.xy_progress_reward_scale,
        two_hop_pair_reward=cfg.two_hop_pair_reward_scale,
        landing_precision_width_m=cfg.landing_precision_width,
        target_hit_reward=cfg.target_hit_reward_scale,
        landing_precision_reward=cfg.landing_precision_reward_scale,
        landing_error_penalty=cfg.landing_error_penalty_scale,
        landing_lateral_error_penalty=cfg.landing_lateral_error_penalty_scale,
        robust_dynamics=cfg.randomize_dynamics,
        observation_noise_std=cfg.observation_noise_std,
    )
    output.mkdir(parents=True)
    env = RslRlVecEnvWrapper(SpatialTrackingEnv(cfg))

    rcfg = PlannerCircularPPORunnerCfg()
    rcfg.experiment_name = 'quadhopper_pair_precision_guidance'
    rcfg.seed = args.seed
    rcfg.num_steps_per_env = 256
    rcfg.save_interval = 10
    rcfg.algorithm.schedule = 'fixed'
    rcfg.algorithm.learning_rate = CONTRACT['learning_rate']
    rcfg.algorithm.num_mini_batches = min(16, args.num_envs)
    rcfg.algorithm.entropy_coef = 0.0
    rcfg.empirical_normalization = False
    rcfg.policy.actor_obs_normalization = False
    rcfg.policy.critic_obs_normalization = False
    runner = RecoveryRunner(env, rcfg.to_dict(), log_dir=str(output), device=args.device)
    runner.alg.policy.load_state_dict(state)
    runner.alg.policy.std.requires_grad_(False)
    if env.num_actions != 4 or env.get_observations()['policy'].shape[-1] != 52:
        raise RuntimeError('Expected 52 observations and four direct motor actions')
    manifest = dict(
        CONTRACT,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cli=vars(args), environment=cfg.to_dict(), runner=rcfg.to_dict(),
    )
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print('[RECOVERY] ' + json.dumps(CONTRACT), flush=True)
    try:
        if args.mode == 'train':
            torch.save({'model_state_dict': runner.alg.policy.state_dict(), 'iter': 0,
                        'infos': {'tracking_recovery_contract': CONTRACT}}, output / 'initial.pt')
            runner.learn(args.iterations, init_at_random_ep_len=False)
            (output / 'train_complete.json').write_text(json.dumps({'iterations': args.iterations}))
        else:
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = env.reset()
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    runner.alg.policy.reset(dones)
            base = env.unwrapped
            result = base.tracking_metrics.summary()
            guide = base._guide_sum
            result.update(checkpoint=str(source), steps=args.steps, num_envs=args.num_envs, seed=args.seed,
                          checkpoint_contract=spatial,
                          evaluation_contract=CONTRACT,
                          death_rate=(base._physical_death_total > 0).float().mean().item(),
                          mean_path_distance_m=(guide[0]/guide[2].clamp_min(1)).item(),
                          path_corridor_rate=(guide[1]/guide[2].clamp_min(1)).item(),
                          mean_spatial_progress=(guide[3]/guide[2].clamp_min(1)).item(),
                          xy_progress_reward_per_env=(base._xy_progress_sum[0]/args.num_envs).item(),
                          xy_approach_reward_per_env=(base._xy_progress_sum[1]/args.num_envs).item(),
                          xy_retreat_penalty_per_env=(base._xy_progress_sum[2]/args.num_envs).item())
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print('[SPATIAL-EVAL] ' + json.dumps(result), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
