"""Train/evaluate continuous direct-motor tracking with untimed spatial guidance."""
import argparse
import hashlib
import json
import time
import traceback
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
                    help='Enable mass/inertia, delay, motor-response, and per-motor thrust randomization.')
parser.add_argument('--hardware-robust', action='store_true',
                    help='Train through the wider low-thrust and slow-motor envelope measured during deployment.')
parser.add_argument('--hardware-randomization-probability', type=float, default=1.0,
                    help='Fraction of reset environments receiving hardware perturbations.')
parser.add_argument('--observation-noise-std', type=float,
                    help='Override policy observation noise standard deviation.')
parser.add_argument('--stationary-hold-radius', type=float,
                    help='Apply stationary stability terms only this close to an exact zero-hop target.')
parser.add_argument('--stationary-tilt-penalty-scale', type=float,
                    help='Soft post-apex tilt penalty for settled exact zero hops.')
parser.add_argument('--stationary-yaw-rate-penalty-scale', type=float,
                    help='Soft post-apex yaw-rate penalty for settled exact zero hops.')
parser.add_argument('--stationary-action-spread-penalty-scale', type=float,
                    help='Soft post-apex motor-spread penalty for settled exact zero hops.')
parser.add_argument('--stationary-apex-width', type=float,
                    help='Apex-quality width for exact zero-hop refinement.')
parser.add_argument('--stationary-apex-reward-scale', type=float,
                    help='Measured-apex quality reward for exact zero hops.')
parser.add_argument('--stationary-apex-error-penalty-scale', type=float,
                    help='Measured-apex absolute-error penalty for exact zero hops.')
parser.add_argument('--stationary-apex-shortfall-penalty-scale', type=float,
                    help='Measured-apex low-height penalty for exact zero hops.')
parser.add_argument('--stationary-apex-progress-scale', type=float,
                    help='Bounded reward for measured maximum-height gains on exact zero hops.')
parser.add_argument('--stationary-apex-progress-width', type=float,
                    help='Shortfall range over which stationary maximum-height progress is shaped.')
parser.add_argument('--stationary-ascent-support-scale', type=float,
                    help='Per-second reward for a stationary hop still ascending below its apex target.')
parser.add_argument('--stationary-ascent-collective-scale', type=float,
                    help='Collective-thrust reward during the valid stationary ascent phase.')
parser.add_argument('--stationary-predicted-apex-width', type=float,
                    help='Ballistic predicted-apex error width during a valid stationary ascent.')
parser.add_argument('--stationary-predicted-apex-reward-scale', type=float,
                    help='Per-second predicted-apex reward during a valid stationary ascent.')
parser.add_argument('--ascent-predicted-apex-width', type=float,
                    help='All-hop ascent predicted-apex reward width.')
parser.add_argument('--ascent-predicted-apex-reward-scale', type=float,
                    help='Per-second all-hop ascent predicted-apex reward.')
parser.add_argument('--apex-event-height-width', type=float,
                    help='All-hop physical apex-event quality width in metres.')
parser.add_argument('--apex-event-height-reward-scale', type=float,
                    help='Small bounded all-hop apex-event quality reward.')
parser.add_argument('--apex-event-shortfall-penalty-scale', type=float,
                    help='All-hop apex-event penalty per metre below the active target.')
parser.add_argument('--simulation-only-phase-collective-baseline', action='store_true',
                    help='Research-only takeoff/ascent collective floor; exported deployment is forbidden.')
parser.add_argument('--phase-contact-collective-floor-pwm', type=float,
                    help='Simulation-only mean-PWM floor while spring contact is active.')
parser.add_argument('--phase-ascent-collective-floor-pwm', type=float,
                    help='Simulation-only mean-PWM floor during valid positive-vz ascent.')
parser.add_argument('--simulation-only-phase-collective-controller', action='store_true',
                    help='Research-only fixed collective plus bounded zero-mean motor residual.')
parser.add_argument('--phase-contact-collective-pwm', type=float)
parser.add_argument('--phase-ascent-collective-pwm', type=float)
parser.add_argument('--phase-descent-collective-pwm', type=float)
parser.add_argument('--phase-contact-differential-limit-pwm', type=float)
parser.add_argument('--phase-ascent-differential-limit-pwm', type=float)
parser.add_argument('--phase-descent-differential-limit-pwm', type=float)
parser.add_argument('--simulation-only-apex-bias-controller', action='store_true',
                    help='Research-only per-hop apex feedback into a symmetric contact PWM bias.')
parser.add_argument('--apex-bias-initial-pwm', type=float)
parser.add_argument('--apex-bias-gain-pwm-per-m', type=float)
parser.add_argument('--apex-bias-max-pwm', type=float)
parser.add_argument('--apex-bias-feedback', choices=('apex', 'liftoff_vz'), default='apex')
parser.add_argument('--apex-bias-liftoff-vz-target', type=float)
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--target-tolerance', type=float, default=0.05)
parser.add_argument('--apex-tolerance', type=float,
                    help='Apex tolerance used by target-relative height gates; unset preserves the task default.')
parser.add_argument('--target-height', type=float, default=1.000,
                    help='Absolute simulation/policy root-Z apex command in metres.')
parser.add_argument('--disable-initial-drop', action='store_true',
                    help='Reset on the ground instead of training-only random aerial drops.')
parser.add_argument('--episode-length-s', type=float,
                    help='Override episode duration; use a value longer than the eval horizon for continuous playback.')
parser.add_argument('--play-motor-time-constant', type=float,
                    help='Fixed nominal motor time constant for deterministic plant identification.')
parser.add_argument('--play-thrust-scale', type=float,
                    help='Fixed nominal per-motor thrust multiplier for plant identification.')
parser.add_argument('--height-curriculum-start', type=float,
                    help='Start a cosine curriculum from this policy/sim apex height to --target-height.')
parser.add_argument('--height-curriculum-iterations', type=float,
                    help='Number of PPO iterations over which the height curriculum reaches --target-height.')
parser.add_argument('--termination-penalty-scale', type=float,
                    help='Override the terminal fall penalty; use a negative value to protect reliability.')
parser.add_argument('--target-relative-apex-gate', action='store_true',
                    help='Gate touchdown success/rewards by current target height minus apex tolerance.')
parser.add_argument('--actuator-mode', choices=('parity', 'corrected_history', 'baseline'),
                    default='parity',
                    help=('Actuator/history contract. parity matches deployment; '
                          'corrected_history disables PWM shaping only; baseline restores '
                          'the source checkpoint\'s legacy delay and raw-action history.'))
parser.add_argument('--action-parameterization', choices=('direct_motor', 'collective_residual_v1'),
                    default='direct_motor')
parser.add_argument('--legacy-v11-contract', action='store_true',
                    help=('Write an explicit v11 direct-motor contract. This is only valid with '
                          '--actuator-mode baseline and is incompatible with the v12 launcher.'))
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
if args.target_height <= 0.38:
    parser.error('target-height must exceed the 0.380 m simulation landing root height')
if args.episode_length_s is not None and args.episode_length_s <= 0.0:
    parser.error('episode-length-s must be positive')
if args.apex_tolerance is not None and not (0.01 <= args.apex_tolerance <= 0.30):
    parser.error('apex-tolerance must be in [0.01, 0.30] metres')
if args.play_motor_time_constant is not None and not (0.01 <= args.play_motor_time_constant <= 0.20):
    parser.error('play-motor-time-constant must be in [0.01, 0.20] seconds')
if args.play_thrust_scale is not None and not (0.50 <= args.play_thrust_scale <= 1.20):
    parser.error('play-thrust-scale must be in [0.50, 1.20]')
if (args.height_curriculum_start is None) != (args.height_curriculum_iterations is None):
    parser.error('height-curriculum-start and height-curriculum-iterations must be supplied together')
if args.height_curriculum_start is not None:
    if not (0.38 < args.height_curriculum_start <= args.target_height):
        parser.error('height-curriculum-start must be in (0.380, target-height]')
    if args.height_curriculum_iterations <= 0.0:
        parser.error('height-curriculum-iterations must be positive')
if not 0.0 <= args.hard_sample_probability <= 1.0:
    parser.error('hard-sample-probability must be in [0, 1]')
if not 0.0 <= args.hardware_randomization_probability <= 1.0:
    parser.error('hardware-randomization-probability must be in [0, 1]')
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
if args.stationary_hold_radius is not None and args.stationary_hold_radius <= 0.0:
    parser.error('stationary-hold-radius must be positive')
if args.stationary_apex_width is not None and args.stationary_apex_width <= 0.0:
    parser.error('stationary-apex-width must be positive')
if args.stationary_apex_progress_width is not None and args.stationary_apex_progress_width <= 0.0:
    parser.error('stationary-apex-progress-width must be positive')
if args.stationary_ascent_support_scale is not None and args.stationary_ascent_support_scale < 0.0:
    parser.error('stationary-ascent-support-scale must be non-negative')
if args.stationary_ascent_collective_scale is not None and args.stationary_ascent_collective_scale < 0.0:
    parser.error('stationary-ascent-collective-scale must be non-negative')
if args.stationary_predicted_apex_width is not None and args.stationary_predicted_apex_width <= 0.0:
    parser.error('stationary-predicted-apex-width must be positive')
if args.ascent_predicted_apex_width is not None and args.ascent_predicted_apex_width <= 0.0:
    parser.error('ascent-predicted-apex-width must be positive')
if (args.stationary_predicted_apex_reward_scale is not None
        and args.stationary_predicted_apex_reward_scale < 0.0):
    parser.error('stationary-predicted-apex-reward-scale must be non-negative')
if (args.ascent_predicted_apex_reward_scale is not None
        and args.ascent_predicted_apex_reward_scale < 0.0):
    parser.error('ascent-predicted-apex-reward-scale must be non-negative')
if args.apex_event_height_width is not None and args.apex_event_height_width <= 0.0:
    parser.error('apex-event-height-width must be positive')
if (args.apex_event_height_reward_scale is not None
        and args.apex_event_height_reward_scale < 0.0):
    parser.error('apex-event-height-reward-scale must be non-negative')
if (args.apex_event_shortfall_penalty_scale is not None
        and args.apex_event_shortfall_penalty_scale > 0.0):
    parser.error('apex-event-shortfall-penalty-scale must be non-positive')
if args.legacy_v11_contract and args.actuator_mode != 'baseline':
    parser.error('--legacy-v11-contract requires --actuator-mode baseline')
if args.action_parameterization == 'collective_residual_v1' and args.actuator_mode != 'baseline':
    parser.error('collective_residual_v1 requires --actuator-mode baseline')
if args.simulation_only_phase_collective_baseline:
    if args.phase_contact_collective_floor_pwm is None or args.phase_ascent_collective_floor_pwm is None:
        parser.error('simulation-only phase baseline requires both phase collective floor PWM values')
    if not (0.0 <= args.phase_contact_collective_floor_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_floor_pwm <= 1000.0):
        parser.error('phase collective floor PWM values must be in [0, 1000]')
elif (args.phase_contact_collective_floor_pwm is not None
      or args.phase_ascent_collective_floor_pwm is not None):
    parser.error('phase collective floor PWM values require --simulation-only-phase-collective-baseline')
if args.simulation_only_phase_collective_baseline and args.simulation_only_phase_collective_controller:
    parser.error('choose either the phase collective floor or the fixed collective controller, not both')
controller_values = (
    args.phase_contact_collective_pwm,
    args.phase_ascent_collective_pwm,
    args.phase_descent_collective_pwm,
    args.phase_contact_differential_limit_pwm,
    args.phase_ascent_differential_limit_pwm,
    args.phase_descent_differential_limit_pwm,
)
if args.simulation_only_phase_collective_controller:
    if any(value is None for value in controller_values):
        parser.error('phase collective controller requires both collective and both differential-limit PWM values')
    if not (0.0 <= args.phase_contact_collective_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_descent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_contact_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_ascent_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_descent_differential_limit_pwm <= 500.0):
        parser.error('phase controller PWM values must be collective [0,1000], differential limit [0,500]')
elif any(value is not None for value in controller_values):
    parser.error('phase controller PWM values require --simulation-only-phase-collective-controller')
apex_bias_values = (args.apex_bias_initial_pwm, args.apex_bias_gain_pwm_per_m, args.apex_bias_max_pwm)
if args.simulation_only_apex_bias_controller:
    if any(value is None for value in apex_bias_values):
        parser.error('apex bias controller requires initial, gain, and max PWM values')
    if not (0.0 <= args.apex_bias_initial_pwm <= args.apex_bias_max_pwm <= 400.0
            and 0.0 < args.apex_bias_gain_pwm_per_m <= 2000.0):
        parser.error('apex bias requires 0 <= initial <= max <= 400 and gain in (0, 2000]')
    if args.apex_bias_feedback == 'liftoff_vz':
        if args.apex_bias_liftoff_vz_target is None or args.apex_bias_liftoff_vz_target <= 0.0:
            parser.error('liftoff_vz feedback requires a positive --apex-bias-liftoff-vz-target')
    elif args.apex_bias_liftoff_vz_target is not None:
        parser.error('apex-bias-liftoff-vz-target requires --apex-bias-feedback liftoff_vz')
elif any(value is not None for value in apex_bias_values):
    parser.error('apex bias values require --simulation-only-apex-bias-controller')
elif args.apex_bias_liftoff_vz_target is not None:
    parser.error('apex-bias-liftoff-vz-target requires --simulation-only-apex-bias-controller')
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
from Quadhopper_Planner_Random.spatial_tracking_env import (
    SpatialTrackingEnv,
    SpatialTrackingPhysical1mEnvCfg,
)
from Quadhopper_Planner_Random.spatial_path import expand_policy_inputs

ROOT = Path(__file__).resolve().parent
CONTRACT = {
    'version': 12,
    'protocol': 'physical_1m_actuator_parity_stationary_precision',
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
    'start_from_random_drop': not args.disable_initial_drop,
    'initial_drop_height_m': [0.65, 1.05] if not args.disable_initial_drop else None,
    'guidance_clock': False,
    'episode_length_s': args.episode_length_s or 15.0,
    'appended_observations': ['carrot_error_body_xyz_m', 'path_tangent_body_xyz', 'landing_up_hint_body_xyz'],
    'learning_rate': args.learning_rate,
    'schedule': 'fixed',
    'std': 0.04,
    'motor_tc_s': args.play_motor_time_constant or 0.0674,
    'nominal_thrust_scale': args.play_thrust_scale or 1.0,
    'policy_target_root_z_m': args.target_height,
    'physical_ground_root_z_m': 0.285,
    # 0.380 m is a spatial-planner landing reference, not a conversion
    # between simulator and Vicon root-Z coordinates.  The native v11
    # observation and physical target use the same root-Z command.
    'spatial_planner_landing_reference_z_m': 0.380,
    'physical_target_root_z_m': args.target_height,
    'action_delay': 'zero_delay_is_newest_command; optional delays count prior 10ms commands',
    'action_history': 'five_executed_post_limit_motor_actions_oldest_to_newest',
    'deployment_action_shaping': {
        'version': 'height_spread_yaw_v1',
        'max_pwm_spread': 600.0,
        'airborne_yaw_diagonal_pwm_diff': 140.0,
        'grounded_yaw_diagonal_pwm_diff': 220.0,
        'constraint_projection_iterations': 2,
        'pwm_quantization': 'floor_to_integer',
    },
    'seed': args.seed,
}
if args.legacy_v11_contract:
    # The accepted 40_v1/mixed family observes raw actions and applies the
    # five-sample legacy delay. Do not label that stream as v12's executed and
    # shaped motor-command history.
    CONTRACT.update(
        version=11,
        protocol='legacy_v11_physical_1m_height_completion',
        action_delay='legacy_indexing_setting_0; five-sample raw-action history selects oldest',
        action_history='five_raw_policy_actions_oldest_to_newest',
        deployment_action_shaping='none',
        v12_launcher_compatible=False,
    )
if args.action_parameterization == 'collective_residual_v1':
    CONTRACT.update(
        version=13,
        protocol='collective_residual_v1_physical_1m',
        action_history='five_raw_collective_residual_actions_oldest_to_newest',
        deployment_action_shaping='none',
        action_parameterization='collective_residual_v1',
        v12_launcher_compatible=False,
    )


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
        elif recovery.get('version') not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            raise ValueError('Curriculum training requires a compatible 52-D spatial checkpoint')
        state['std'].fill_(CONTRACT['std'])
    else:
        spatial = (data.get('infos') or {}).get('tracking_recovery_contract', {})
        current_contract = (
            spatial.get('protocol') == CONTRACT['protocol']
            and spatial.get('version') == CONTRACT['version']
        )
        compatible_prior = spatial.get('version') in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        if not current_contract and not (args.allow_compatible_checkpoint and compatible_prior):
            raise ValueError('Evaluation requires a physical-1m actuator-parity v12 checkpoint; '
                             'use --allow-compatible-checkpoint for an older spatial policy')

    cfg = SpatialTrackingPhysical1mEnvCfg()
    if args.actuator_mode == 'corrected_history':
        cfg.apply_deployment_action_shaping = False
    elif args.actuator_mode == 'baseline':
        cfg.corrected_action_delay_indexing = False
        cfg.record_executed_action_history = False
        cfg.apply_deployment_action_shaping = False
    cfg.target_height = args.target_height
    cfg.deployment_action_target_height = args.target_height
    if args.disable_initial_drop:
        cfg.start_from_random_drop = False
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    cfg.action_parameterization = args.action_parameterization
    if args.apex_tolerance is not None:
        cfg.apex_tolerance = args.apex_tolerance
    if args.play_motor_time_constant is not None:
        cfg.play_motor_time_constant = args.play_motor_time_constant
    if args.play_thrust_scale is not None:
        cfg.play_thrust_scale = args.play_thrust_scale
    cfg.simulation_only_phase_collective_baseline = args.simulation_only_phase_collective_baseline
    if args.simulation_only_phase_collective_baseline:
        cfg.phase_contact_collective_floor_pwm = args.phase_contact_collective_floor_pwm
        cfg.phase_ascent_collective_floor_pwm = args.phase_ascent_collective_floor_pwm
    cfg.simulation_only_phase_collective_controller = args.simulation_only_phase_collective_controller
    if args.simulation_only_phase_collective_controller:
        cfg.phase_contact_collective_pwm = args.phase_contact_collective_pwm
        cfg.phase_ascent_collective_pwm = args.phase_ascent_collective_pwm
        cfg.phase_descent_collective_pwm = args.phase_descent_collective_pwm
        cfg.phase_contact_differential_limit_pwm = args.phase_contact_differential_limit_pwm
        cfg.phase_ascent_differential_limit_pwm = args.phase_ascent_differential_limit_pwm
        cfg.phase_descent_differential_limit_pwm = args.phase_descent_differential_limit_pwm
    cfg.simulation_only_apex_bias_controller = args.simulation_only_apex_bias_controller
    if args.simulation_only_apex_bias_controller:
        cfg.apex_bias_initial_pwm = args.apex_bias_initial_pwm
        cfg.apex_bias_gain_pwm_per_m = args.apex_bias_gain_pwm_per_m
        cfg.apex_bias_max_pwm = args.apex_bias_max_pwm
        cfg.apex_bias_feedback_mode = args.apex_bias_feedback
        if args.apex_bias_feedback == 'liftoff_vz':
            cfg.apex_bias_liftoff_vz_target_mps = args.apex_bias_liftoff_vz_target
    if args.height_curriculum_start is not None:
        cfg.fixed_height_curriculum = True
        cfg.height_curriculum_start = args.height_curriculum_start
        cfg.height_curriculum_end = args.target_height
        cfg.height_curriculum_iterations = args.height_curriculum_iterations
        cfg.height_curriculum_iteration_offset = 0.0
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
    if args.robust_dynamics or args.hardware_robust:
        # Real hops arrive with per-motor thrust, bandwidth, and timing
        # differences.  Keep these ranges mild enough to retain the measured
        # calibrated nominal plant in the curriculum centre, while ensuring a
        # policy cannot rely on perfectly matched rotors.
        cfg.randomize_dynamics = True
        cfg.randomize_action_delay = True
        cfg.randomize_motor_time_constant = True
        cfg.randomize_thrust_curve = True
        cfg.train_thrust_scale_min = 0.93
        cfg.train_thrust_scale_max = 1.07
        cfg.train_thrust_curve_shape_min = 0.96
        cfg.train_thrust_curve_shape_max = 1.04
    if args.hardware_robust:
        # Start from the flight-data nominal plant and cover the remaining
        # hardware uncertainty without asking every rollout to solve an
        # implausible worst-case combination.  The previous 0.78 thrust / 110
        # ms / 1.25 inertia envelope destroyed the already-good 1 m hopping
        # behaviour, rather than making it robust.  This range still includes
        # lower rebound authority and slower motors seen after impact.
        cfg.train_mass_multi_min = 0.97
        cfg.train_mass_multi_max = 1.07
        cfg.train_inertia_multi_min = 0.88
        cfg.train_inertia_multi_max = 1.15
        cfg.train_motor_time_constant_min = 0.054
        cfg.train_motor_time_constant_max = 0.090
        cfg.train_thrust_scale_min = 0.87
        cfg.train_thrust_scale_max = 1.06
        cfg.train_thrust_curve_shape_min = 0.94
        cfg.train_thrust_curve_shape_max = 1.06
        # Flight logs show no persistent multi-step command delay, so do not
        # trade away nominal takeoff height for an artificial delay model.
        cfg.randomize_action_delay = False
    cfg.hardware_randomization_probability = args.hardware_randomization_probability
    if args.observation_noise_std is not None:
        cfg.observation_noise_std = args.observation_noise_std
    if args.termination_penalty_scale is not None:
        cfg.termination_penalty_scale = args.termination_penalty_scale
    if args.target_relative_apex_gate:
        cfg.target_relative_valid_apex = True
        cfg.require_minimum_apex_for_hit = True
        cfg.gate_touchdown_rewards_by_apex = True
    stationary_overrides = {
        'stationary_hold_radius_m': args.stationary_hold_radius,
        'stationary_tilt_penalty_scale': args.stationary_tilt_penalty_scale,
        'stationary_yaw_rate_penalty_scale': args.stationary_yaw_rate_penalty_scale,
        'stationary_action_spread_penalty_scale': args.stationary_action_spread_penalty_scale,
        'stationary_apex_width_m': args.stationary_apex_width,
        'stationary_apex_reward_scale': args.stationary_apex_reward_scale,
        'stationary_apex_error_penalty_scale': args.stationary_apex_error_penalty_scale,
        'stationary_apex_shortfall_penalty_scale': args.stationary_apex_shortfall_penalty_scale,
        'stationary_apex_progress_scale': args.stationary_apex_progress_scale,
        'stationary_apex_progress_width_m': args.stationary_apex_progress_width,
        'stationary_ascent_support_scale': args.stationary_ascent_support_scale,
        'stationary_ascent_collective_scale': args.stationary_ascent_collective_scale,
        'stationary_predicted_apex_width_m': args.stationary_predicted_apex_width,
        'stationary_predicted_apex_reward_scale': args.stationary_predicted_apex_reward_scale,
        'ascent_predicted_apex_width_m': args.ascent_predicted_apex_width,
        'ascent_predicted_apex_reward_scale': args.ascent_predicted_apex_reward_scale,
        'apex_event_height_width_m': args.apex_event_height_width,
        'apex_event_height_reward_scale': args.apex_event_height_reward_scale,
        'apex_event_shortfall_penalty_scale': args.apex_event_shortfall_penalty_scale,
    }
    for name, value in stationary_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
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
        actuator_mode=args.actuator_mode,
        action_parameterization=cfg.action_parameterization,
        xy_potential_progress_scale=cfg.spatial_xy_progress_reward_scale,
        legacy_xy_progress_reward_scale=cfg.xy_progress_reward_scale,
        two_hop_pair_reward=cfg.two_hop_pair_reward_scale,
        landing_precision_width_m=cfg.landing_precision_width,
        target_hit_reward=cfg.target_hit_reward_scale,
        landing_precision_reward=cfg.landing_precision_reward_scale,
        landing_error_penalty=cfg.landing_error_penalty_scale,
        landing_lateral_error_penalty=cfg.landing_lateral_error_penalty_scale,
        target_hit_requires_minimum_apex=cfg.require_minimum_apex_for_hit,
        target_hit_requires_apex_tolerance=cfg.require_apex_tolerance_for_hit,
        landing_precision_gated_by_apex=cfg.gate_touchdown_rewards_by_apex,
        robust_dynamics=cfg.randomize_dynamics,
        hardware_robust=args.hardware_robust,
        hardware_randomization_probability=cfg.hardware_randomization_probability,
        randomize_action_delay=cfg.randomize_action_delay,
        randomize_motor_time_constant=cfg.randomize_motor_time_constant,
        motor_time_constant_range_s=[cfg.train_motor_time_constant_min, cfg.train_motor_time_constant_max],
        mass_multiplier_range=[cfg.train_mass_multi_min, cfg.train_mass_multi_max],
        inertia_multiplier_range=[cfg.train_inertia_multi_min, cfg.train_inertia_multi_max],
        randomize_thrust_curve=cfg.randomize_thrust_curve,
        thrust_scale_range=[cfg.train_thrust_scale_min, cfg.train_thrust_scale_max],
        thrust_curve_shape_range=[cfg.train_thrust_curve_shape_min, cfg.train_thrust_curve_shape_max],
        observation_noise_std=cfg.observation_noise_std,
        corrected_action_delay_indexing=cfg.corrected_action_delay_indexing,
        simulation_only_phase_collective_baseline=cfg.simulation_only_phase_collective_baseline,
        phase_contact_collective_floor_pwm=cfg.phase_contact_collective_floor_pwm,
        phase_ascent_collective_floor_pwm=cfg.phase_ascent_collective_floor_pwm,
        simulation_only_phase_collective_controller=cfg.simulation_only_phase_collective_controller,
        phase_contact_collective_pwm=cfg.phase_contact_collective_pwm,
        phase_ascent_collective_pwm=cfg.phase_ascent_collective_pwm,
        phase_descent_collective_pwm=cfg.phase_descent_collective_pwm,
        phase_contact_differential_limit_pwm=cfg.phase_contact_differential_limit_pwm,
        phase_ascent_differential_limit_pwm=cfg.phase_ascent_differential_limit_pwm,
        phase_descent_differential_limit_pwm=cfg.phase_descent_differential_limit_pwm,
        simulation_only_apex_bias_controller=cfg.simulation_only_apex_bias_controller,
        apex_bias_initial_pwm=cfg.apex_bias_initial_pwm,
        apex_bias_gain_pwm_per_m=cfg.apex_bias_gain_pwm_per_m,
        apex_bias_max_pwm=cfg.apex_bias_max_pwm,
        apex_bias_feedback_mode=cfg.apex_bias_feedback_mode,
        apex_bias_liftoff_vz_target_mps=cfg.apex_bias_liftoff_vz_target_mps,
        fixed_height_curriculum=cfg.fixed_height_curriculum,
        height_curriculum_start_m=cfg.height_curriculum_start,
        height_curriculum_end_m=cfg.height_curriculum_end,
        height_curriculum_iterations=cfg.height_curriculum_iterations,
        termination_penalty_scale=cfg.termination_penalty_scale,
        target_relative_apex_gate=cfg.target_relative_valid_apex,
        require_minimum_apex_for_hit=cfg.require_minimum_apex_for_hit,
        gate_touchdown_rewards_by_apex=cfg.gate_touchdown_rewards_by_apex,
        apex_tolerance_m=cfg.apex_tolerance,
        stationary_hold_radius_m=cfg.stationary_hold_radius_m,
        stationary_tilt_penalty_scale=cfg.stationary_tilt_penalty_scale,
        stationary_yaw_rate_penalty_scale=cfg.stationary_yaw_rate_penalty_scale,
        stationary_action_spread_penalty_scale=cfg.stationary_action_spread_penalty_scale,
        stationary_apex_width_m=cfg.stationary_apex_width_m,
        stationary_apex_reward_scale=cfg.stationary_apex_reward_scale,
        stationary_apex_error_penalty_scale=cfg.stationary_apex_error_penalty_scale,
        stationary_apex_shortfall_penalty_scale=cfg.stationary_apex_shortfall_penalty_scale,
        stationary_apex_progress_scale=cfg.stationary_apex_progress_scale,
        stationary_apex_progress_width_m=cfg.stationary_apex_progress_width_m,
        stationary_ascent_support_scale=cfg.stationary_ascent_support_scale,
        stationary_ascent_collective_scale=cfg.stationary_ascent_collective_scale,
        stationary_predicted_apex_width_m=cfg.stationary_predicted_apex_width_m,
        stationary_predicted_apex_reward_scale=cfg.stationary_predicted_apex_reward_scale,
        ascent_predicted_apex_width_m=cfg.ascent_predicted_apex_width_m,
        ascent_predicted_apex_reward_scale=cfg.ascent_predicted_apex_reward_scale,
        apex_event_height_width_m=cfg.apex_event_height_width_m,
        apex_event_height_reward_scale=cfg.apex_event_height_reward_scale,
        apex_event_shortfall_penalty_scale=cfg.apex_event_shortfall_penalty_scale,
    )
    output.mkdir(parents=True)
    print('[RECOVERY] constructing vector environment', flush=True)
    env = RslRlVecEnvWrapper(SpatialTrackingEnv(cfg))
    print('[RECOVERY] vector environment ready', flush=True)

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
    print('[RECOVERY] PPO runner ready', flush=True)
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
            progress_interval = max(1, min(1000, args.steps // 10))
            started_at = time.monotonic()
            with torch.inference_mode():
                for step in range(args.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    runner.alg.policy.reset(dones)
                    if (step + 1) % progress_interval == 0 or step + 1 == args.steps:
                        print(
                            f'[SPATIAL-EVAL] progress={step + 1}/{args.steps} '
                            f'elapsed_s={time.monotonic() - started_at:.1f}',
                            flush=True,
                        )
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
except BaseException:
    failure_path = output / 'startup_exception.txt'
    failure_path.write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    app.close()
