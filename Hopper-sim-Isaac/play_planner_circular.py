"""Play a planner-conditioned circular or random-route checkpoint."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play planner-conditioned circular hopping")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Fixed evaluation seed so checkpoints see the same waypoint sequence.",
)
parser.add_argument("--height_stage", choices=("low", "high", "alternate"), default="alternate")
parser.add_argument("--height_high", type=float, default=1.0)
parser.add_argument("--height_low", type=float, default=0.7)
parser.add_argument("--adaptive_apex_height", action="store_true")
parser.add_argument("--adaptive_height_min", type=float, default=1.00)
parser.add_argument("--adaptive_height_max", type=float, default=1.25)
parser.add_argument("--adaptive_height_distance_start", type=float, default=0.10)
parser.add_argument("--adaptive_height_distance_end", type=float, default=0.35)
parser.add_argument("--adaptive_height_noise", type=float, default=0.0)
parser.add_argument(
    "--route", choices=("circle", "random_two_hop"), default="circle"
)
parser.add_argument(
    "--distance_stage",
    choices=("direction", "medium", "short", "bridge", "full", "custom"),
    default="full",
)
parser.add_argument("--short_radius_min", type=float, default=0.5)
parser.add_argument("--short_radius_max", type=float, default=0.8)
parser.add_argument("--long_radius_min", type=float, default=0.8)
parser.add_argument("--long_radius_max", type=float, default=1.0)
parser.add_argument(
    "--start_from_random_drop",
    action="store_true",
    help="Evaluate from a baseline-style random high release before the first commanded hop.",
)
parser.add_argument("--initial_drop_height_min", type=float, default=0.8)
parser.add_argument("--initial_drop_height_max", type=float, default=1.5)
parser.add_argument("--landing_compensation", type=float, default=0.0)
parser.add_argument("--relative_next_hop", action="store_true")
parser.add_argument("--planner_velocity_observation", action="store_true")
parser.add_argument("--max_turn_angle", type=float, default=180.0)
parser.add_argument("--landing_velocity_scale", type=float, default=1.0)
parser.add_argument("--takeoff_xy_bias_gain", type=float, default=0.0)
parser.add_argument("--takeoff_xy_bias_max", type=float, default=0.08)
parser.add_argument(
    "--planner_reference_blend",
    type=float,
    default=None,
    help="Fixed blend from stationary apex (0.0) to full planner trajectory (1.0).",
)
parser.add_argument(
    "--direct_landing_reference",
    action="store_true",
    help=(
        "Use the physical touchdown target XY as the stationary flight reference; "
        "keeps the same planner observations, state machine, and rewards."
    ),
)
parser.add_argument(
    "--compact_eval",
    action="store_true",
    help="Print only the metrics needed for a waypoint-tracking comparison.",
)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--deploy_xy_tolerance", type=float, default=1.5)
parser.add_argument("--deploy_min_touchdowns", type=int, default=3)
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument(
    "--play_motor_time_constant",
    type=float,
    default=None,
    help="Override the first-order motor lag time constant for single-env playback.",
)
parser.add_argument(
    "--action_lpf_tau",
    type=float,
    default=0.0,
    help="Extra first-order low-pass filter (s) applied to policy actions at eval time; "
    "simulates a deployment-side action filter that slows the effective fast-loop response.",
)
parser.add_argument("--thrust_rand_min", type=float, default=1.0)
parser.add_argument("--thrust_rand_max", type=float, default=1.0)
parser.add_argument("--thrust_shape_rand_min", type=float, default=1.0)
parser.add_argument("--thrust_shape_rand_max", type=float, default=1.0)
parser.add_argument(
    "--allow_single_env_thrust_rand",
    action="store_true",
    help="Apply thrust-curve randomization even with num_envs=1 (for mismatch experiments).",
)
parser.add_argument(
    "--static_apex_reference",
    action="store_true",
    help="Track the stationary per-hop apex instead of the time-parameterized planner trajectory.",
)
parser.add_argument(
    "--baseline_compatible_stance_reference",
    action="store_true",
    help="Keep the inherited 37-D target height at the apex command during stance.",
)
parser.add_argument("--anticipatory", action="store_true")
parser.add_argument("--phase_tilt_finetune", action="store_true")
parser.add_argument("--takeoff_tilt_deg", type=float, default=8.0)
parser.add_argument("--takeoff_tilt_phase_end", type=float, default=0.40)
parser.add_argument("--takeoff_tilt_reward_scale", type=float, default=1.0)
parser.add_argument("--phase_takeoff_finetune", action="store_true")
parser.add_argument("--takeoff_velocity_target", type=float, default=0.30)
parser.add_argument("--takeoff_velocity_reward_scale", type=float, default=1.0)
parser.add_argument("--takeoff_velocity_from_planner", action="store_true")
parser.add_argument("--takeoff_velocity_event_reward", action="store_true")
parser.add_argument("--takeoff_phase_attitude_relax", type=float, default=0.50)
parser.add_argument("--takeoff_phase_angvel_relax", type=float, default=0.35)
parser.add_argument("--takeoff_phase_tilt_barrier_relax", type=float, default=0.60)
parser.add_argument(
    "--continuity",
    action="store_true",
    help="Use the settled-touchdown two-hop continuity references from v32 training.",
)
parser.add_argument("--anticipatory_speed", type=float, default=0.30)
parser.add_argument("--anticipatory_tilt_deg", type=float, default=6.0)
parser.add_argument("--landing_correction_gain", type=float, default=0.0)
parser.add_argument(
    "--continuous_queue",
    action="store_true",
    help="Keep the waypoint queue continuous; this is now the default for random two-hop routes.",
)
parser.add_argument(
    "--pair_restart_queue",
    action="store_true",
    help="Legacy mode: after a long hop, restart a fresh short/long pair around the measured touchdown.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps and print the latest episode metrics.",
)
parser.add_argument(
    "--reset_rnn_on_touchdown",
    action="store_true",
    help=(
        "Diagnostic ablation: clear the recurrent actor/critic state after every "
        "touchdown before the next hop. Episode dones are always reset."
    ),
)
parser.add_argument(
    "--lenient_apex_hit",
    action="store_true",
    help="Match curricula that count a spatial hit without requiring apex tolerance.",
)
parser.add_argument("--tilt_death_sq_threshold", type=float, default=0.5)
parser.add_argument("--terminate_angvel_norm", type=float, default=0.0)
parser.add_argument("--no_debug_vis", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if (args_cli.anticipatory or args_cli.continuity) and not args_cli.relative_next_hop:
    parser.error("--anticipatory/--continuity requires --relative_next_hop")
if args_cli.anticipatory and args_cli.continuity:
    parser.error("--anticipatory and --continuity are separate reference modes")
if args_cli.continuous_queue and args_cli.pair_restart_queue:
    parser.error("--continuous_queue and --pair_restart_queue are mutually exclusive")
args_cli.continuous_queue = not args_cli.pair_restart_queue
DISTANCE_STAGES = {
    "direction": (0.22, 0.30, 0.22, 0.30),
    "medium": (0.30, 0.50, 0.30, 0.50),
    "short": (0.50, 0.80, 0.50, 0.80),
    "bridge": (0.50, 0.80, 0.65, 0.85),
    "full": (0.50, 0.80, 0.80, 1.00),
}
if args_cli.route == "random_two_hop" and args_cli.distance_stage != "custom":
    (
        args_cli.short_radius_min,
        args_cli.short_radius_max,
        args_cli.long_radius_min,
        args_cli.long_radius_max,
    ) = DISTANCE_STAGES[args_cli.distance_stage]
if not (0.0 <= args_cli.initial_drop_height_min <= args_cli.initial_drop_height_max):
    parser.error("--initial_drop_height_min/max must satisfy 0 <= min <= max")
if not (0.1 <= args_cli.thrust_rand_min <= args_cli.thrust_rand_max):
    parser.error("--thrust_rand_min/max must satisfy 0.1 <= min <= max")
if not (0.1 <= args_cli.thrust_shape_rand_min <= args_cli.thrust_shape_rand_max):
    parser.error("--thrust_shape_rand_min/max must satisfy 0.1 <= min <= max")
if args_cli.planner_reference_blend is not None and not (
    0.0 <= args_cli.planner_reference_blend <= 1.0
):
    parser.error("--planner_reference_blend must be in [0, 1]")
if args_cli.takeoff_xy_bias_max < 0.0:
    parser.error("--takeoff_xy_bias_max must be non-negative")
if args_cli.adaptive_height_min <= 0.0:
    parser.error("--adaptive_height_min must be positive")
if args_cli.adaptive_height_max < args_cli.adaptive_height_min:
    parser.error("--adaptive_height_max must be >= --adaptive_height_min")
if args_cli.adaptive_height_distance_end <= args_cli.adaptive_height_distance_start:
    parser.error("--adaptive_height_distance_end must be greater than --adaptive_height_distance_start")
if args_cli.adaptive_height_noise < 0.0:
    parser.error("--adaptive_height_noise must be non-negative")
if not (0.0 <= args_cli.takeoff_tilt_phase_end <= 1.0):
    parser.error("--takeoff_tilt_phase_end must be in [0, 1]")
if args_cli.takeoff_tilt_reward_scale < 0.0:
    parser.error("--takeoff_tilt_reward_scale must be non-negative")
if args_cli.takeoff_velocity_target < 0.0:
    parser.error("--takeoff_velocity_target must be non-negative")
if args_cli.takeoff_velocity_reward_scale < 0.0:
    parser.error("--takeoff_velocity_reward_scale must be non-negative")
if not (0.0 < args_cli.tilt_death_sq_threshold <= 1.0):
    parser.error("--tilt_death_sq_threshold must be in (0, 1]")
if args_cli.terminate_angvel_norm < 0.0:
    parser.error("--terminate_angvel_norm must be non-negative")
for name in (
    "takeoff_phase_attitude_relax",
    "takeoff_phase_angvel_relax",
    "takeoff_phase_tilt_barrier_relax",
):
    if not (0.0 <= getattr(args_cli, name) <= 1.0):
        parser.error(f"--{name} must be in [0, 1]")

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "low": "quadhopper_planner_circular_v15_descend_to_070",
    "high": "quadhopper_planner_circular_v15_high_100",
    "alternate": "quadhopper_planner_circular_v15_alternate_070_100",
}
if args_cli.route == "random_two_hop":
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    height_label = (
        f"alternate_{low_cm:03d}_{high_cm:03d}"
        if args_cli.height_stage == "alternate"
        else f"fixed_{round((args_cli.height_low if args_cli.height_stage == 'low' else args_cli.height_high) * 100.0):03d}"
    )
    experiment = (
        f"quadhopper_planner_random_two_hop_v24_{args_cli.distance_stage}_{height_label}"
    )
else:
    experiment = EXPERIMENTS[args_cli.height_stage]
LOG_ROOT = PROJECT_DIR / "logs/rsl_rl" / experiment


def latest_checkpoint() -> Path | None:
    candidates = list(LOG_ROOT.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


checkpoint = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else latest_checkpoint()
if checkpoint is None:
    parser.error(f"No planner circular checkpoint found under {LOG_ROOT}")
checkpoint = checkpoint.resolve()
if not checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Planner_Circular  # noqa: F401
import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.checkpoint_migration import migrate_stable_checkpoint
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg


def infer_checkpoint_input_width(checkpoint_data, checkpoint_path: Path) -> int:
    state = checkpoint_data.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    weight = state.get("memory_a.rnn.weight_ih_l0")
    if weight is not None:
        return int(weight.shape[1])
    actor_weight = state.get("actor.0.weight")
    if actor_weight is not None:
        return int(actor_weight.shape[1])
    raise ValueError(f"Cannot infer checkpoint observation width: {checkpoint_path}")


def main():
    env_cfg = (
        PlannerRandomTwoHopEnvCfg()
        if args_cli.route == "random_two_hop"
        else PlannerCircularEnvCfg()
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = not args_cli.no_debug_vis
    env_cfg.force_full_planner = (
        not args_cli.static_apex_reference and args_cli.planner_reference_blend is None
    )
    if args_cli.planner_reference_blend is not None:
        env_cfg.planner_reference_blend = args_cli.planner_reference_blend
    env_cfg.flight_reference_uses_landing_xy = args_cli.direct_landing_reference
    env_cfg.stance_reference_uses_apex_height = (
        args_cli.baseline_compatible_stance_reference
    )
    # Evaluation must expose policy quality rather than observation RNG.
    env_cfg.observation_noise_std = 0.0
    env_cfg.observation_space = 47 if args_cli.planner_velocity_observation else 43
    env_cfg.planner_velocity_observation = args_cli.planner_velocity_observation
    env_cfg.alternate_target_heights = args_cli.height_stage == "alternate"
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.adaptive_apex_height = args_cli.adaptive_apex_height
    env_cfg.adaptive_height_min = args_cli.adaptive_height_min
    env_cfg.adaptive_height_max = args_cli.adaptive_height_max
    env_cfg.adaptive_height_distance_start = args_cli.adaptive_height_distance_start
    env_cfg.adaptive_height_distance_end = args_cli.adaptive_height_distance_end
    env_cfg.adaptive_height_noise = args_cli.adaptive_height_noise
    env_cfg.target_height = (
        args_cli.height_low if args_cli.height_stage == "low" else args_cli.height_high
    )
    env_cfg.start_from_random_drop = args_cli.start_from_random_drop
    env_cfg.initial_drop_height_min = args_cli.initial_drop_height_min
    env_cfg.initial_drop_height_max = args_cli.initial_drop_height_max
    env_cfg.randomize_thrust_curve = (
        args_cli.thrust_rand_min != 1.0
        or args_cli.thrust_rand_max != 1.0
        or args_cli.thrust_shape_rand_min != 1.0
        or args_cli.thrust_shape_rand_max != 1.0
    )
    env_cfg.allow_single_env_thrust_rand = args_cli.allow_single_env_thrust_rand
    if args_cli.play_motor_time_constant is not None:
        env_cfg.play_motor_time_constant = args_cli.play_motor_time_constant
    env_cfg.train_thrust_scale_min = args_cli.thrust_rand_min
    env_cfg.train_thrust_scale_max = args_cli.thrust_rand_max
    env_cfg.train_thrust_curve_shape_min = args_cli.thrust_shape_rand_min
    env_cfg.train_thrust_curve_shape_max = args_cli.thrust_shape_rand_max
    env_cfg.target_tolerance = args_cli.target_tolerance
    # Evaluation is always the final fixed command, never the training schedule.
    env_cfg.fixed_height_curriculum = False
    env_cfg.symmetric_height_tracking = True
    env_cfg.require_apex_tolerance_for_hit = not args_cli.lenient_apex_hit
    env_cfg.tilt_death_sq_threshold = args_cli.tilt_death_sq_threshold
    env_cfg.terminate_angvel_norm = args_cli.terminate_angvel_norm
    if args_cli.route == "random_two_hop":
        env_cfg.short_hop_radius_min = args_cli.short_radius_min
        env_cfg.short_hop_radius_max = args_cli.short_radius_max
        env_cfg.long_hop_radius_min = args_cli.long_radius_min
        env_cfg.long_hop_radius_max = args_cli.long_radius_max
        env_cfg.hop_distance = max(
            args_cli.short_radius_max, args_cli.long_radius_max, 0.05
        )
        env_cfg.planner_landing_compensation_m = args_cli.landing_compensation
        env_cfg.takeoff_xy_bias_gain = args_cli.takeoff_xy_bias_gain
        env_cfg.takeoff_xy_bias_max_m = args_cli.takeoff_xy_bias_max
        env_cfg.relative_next_hop_observation = args_cli.relative_next_hop
        env_cfg.max_turn_angle_deg = args_cli.max_turn_angle
        env_cfg.planner_landing_xy_velocity_scale = args_cli.landing_velocity_scale
        env_cfg.restart_two_hop_pair = not args_cli.continuous_queue
        # Landing feedback is an independent receding-horizon reference and
        # can be evaluated without enabling either terminal-velocity mode.
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        if args_cli.anticipatory:
            env_cfg.anticipatory_velocity_blend = 1.0
            env_cfg.anticipatory_speed_max = args_cli.anticipatory_speed
            env_cfg.anticipatory_tilt_rad = math.radians(
                args_cli.anticipatory_tilt_deg
            )
            env_cfg.anticipation_start_phase = 0.50
            env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
            env_cfg.prepared_velocity_tolerance = max(
                0.35, args_cli.anticipatory_speed + 0.25
            )
        if args_cli.phase_tilt_finetune:
            env_cfg.takeoff_tilt_rad = math.radians(args_cli.takeoff_tilt_deg)
            env_cfg.takeoff_tilt_phase_end = args_cli.takeoff_tilt_phase_end
            env_cfg.takeoff_tilt_reward_scale = (
                35.0 * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.takeoff_tilt_penalty_scale = (
                -8.0 * args_cli.takeoff_tilt_reward_scale
            )
        if args_cli.phase_takeoff_finetune:
            env_cfg.takeoff_tilt_rad = math.radians(args_cli.takeoff_tilt_deg)
            env_cfg.takeoff_tilt_phase_end = args_cli.takeoff_tilt_phase_end
            env_cfg.takeoff_phase_attitude_relax = args_cli.takeoff_phase_attitude_relax
            env_cfg.takeoff_phase_angvel_relax = args_cli.takeoff_phase_angvel_relax
            env_cfg.takeoff_phase_tilt_barrier_relax = (
                args_cli.takeoff_phase_tilt_barrier_relax
            )
            env_cfg.takeoff_velocity_target_mps = args_cli.takeoff_velocity_target
            env_cfg.takeoff_velocity_from_planner = args_cli.takeoff_velocity_from_planner
            env_cfg.takeoff_velocity_event_reward = args_cli.takeoff_velocity_event_reward
            env_cfg.takeoff_velocity_reward_scale = (
                55.0 * args_cli.takeoff_velocity_reward_scale
            )
            env_cfg.takeoff_velocity_penalty_scale = (
                -18.0 * args_cli.takeoff_velocity_reward_scale
            )
            if args_cli.takeoff_tilt_reward_scale > 0.0:
                env_cfg.takeoff_tilt_reward_scale = (
                    15.0 * args_cli.takeoff_tilt_reward_scale
                )
                env_cfg.takeoff_tilt_penalty_scale = (
                    -4.0 * args_cli.takeoff_tilt_reward_scale
                )
        if args_cli.continuity:
            env_cfg.planner_landing_xy_velocity_scale = 0.0
            env_cfg.anticipatory_velocity_blend = 0.0
            env_cfg.anticipatory_tilt_rad = math.radians(
                args_cli.anticipatory_tilt_deg
            )
            env_cfg.anticipation_start_phase = 0.60
            env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
            env_cfg.prepared_velocity_tolerance = 0.55
        print(
            "[PLAY] Random two-hop geometry: "
            f"first=[{args_cli.short_radius_min:.2f}, {args_cli.short_radius_max:.2f}] m "
            "about the current position; "
            f"second=[{args_cli.long_radius_min:.2f}, {args_cli.long_radius_max:.2f}] m "
            f"about P_t; max turn={args_cli.max_turn_angle:.1f} deg; "
            f"planner compensation={args_cli.landing_compensation:.3f} m; "
            f"takeoff bias gain={args_cli.takeoff_xy_bias_gain:.2f}, "
            f"max={args_cli.takeoff_xy_bias_max:.3f} m; "
            f"landing-v scale={args_cli.landing_velocity_scale:.2f}; "
            f"success tolerance={args_cli.target_tolerance:.2f} m; "
            f"queue={'continuous' if args_cli.continuous_queue else 'pair-restart'}"
        )
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    output_name = "planner_random_two_hop" if args_cli.route == "random_two_hop" else "planner_circular"
    env_cfg.csv_log_path = str(PROJECT_DIR / f"outputs/{output_name}/on_quadhopper_sim.csv")
    task_id = (
        "Quadhopper-Planner-Random-Two-Hop-Direct-v0"
        if args_cli.route == "random_two_hop"
        else "Quadhopper-Planner-Circular-Direct-v0"
    )
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg))
    runner = OnPolicyRunner(env, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args_cli.device)
    target_obs_dim = 47 if args_cli.planner_velocity_observation else 43
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    input_width = infer_checkpoint_input_width(checkpoint_data, checkpoint)
    load_checkpoint = checkpoint
    if input_width != target_obs_dim:
        if input_width in (37, 42, 43) and input_width < target_obs_dim:
            load_checkpoint = (
                PROJECT_DIR
                / "outputs"
                / "planner_migrations"
                / f"play_initial_policy_{input_width}d_to_{target_obs_dim}d.pt"
            )
            load_checkpoint = migrate_stable_checkpoint(
                checkpoint,
                load_checkpoint,
                target_obs_dim=target_obs_dim,
            )
            print(
                f"[PLAY] Migrated checkpoint observation width {input_width} -> "
                f"{target_obs_dim}: {load_checkpoint}"
            )
        else:
            raise ValueError(
                f"Checkpoint observation width {input_width} is incompatible with "
                f"play observation width {target_obs_dim}"
            )
    print(f"[PLAY] Loading planner {args_cli.route} checkpoint: {load_checkpoint}")
    runner.load(str(load_checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args_cli.device)
    obs, _ = env.reset()
    base_env = env.unwrapped
    local_episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=base_env.device)
    eval_episodes = torch.zeros_like(local_episode_steps)
    eval_deaths = torch.zeros_like(local_episode_steps)
    eval_timeouts = torch.zeros_like(local_episode_steps)
    eval_max_xy_error = torch.zeros(args_cli.num_envs, device=base_env.device)
    eval_touchdowns = 0
    eval_short_touchdowns = 0
    eval_long_touchdowns = 0
    eval_hits = 0
    if args_cli.route == "random_two_hop":
        p_t, p_t1 = base_env.commands.lookahead()
        first_distance = torch.linalg.norm(
            p_t - base_env.commands.anchor_w, dim=1
        )
        second_distance = torch.linalg.norm(p_t1 - p_t, dim=1)
        print(
            "[PLAY] Sampled distances: "
            f"first={first_distance.min().item():.3f}--{first_distance.max().item():.3f} m, "
            f"second={second_distance.min().item():.3f}--{second_distance.max().item():.3f} m"
        )
    step_count = 0
    latest_log = {}
    action_filter_state = None
    while simulation_app.is_running():
        local_episode_steps += 1
        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.action_scale != 1.0:
                actions = (actions * args_cli.action_scale).clamp(-1.0, 1.0)
            if args_cli.action_lpf_tau > 0.0:
                alpha = env.unwrapped.step_dt / (args_cli.action_lpf_tau + env.unwrapped.step_dt)
                if action_filter_state is None:
                    action_filter_state = actions.clone()
                else:
                    action_filter_state = (
                        alpha * actions + (1.0 - alpha) * action_filter_state
                    )
                actions = action_filter_state
            obs, _, dones, extras = env.step(actions)
            # get_inference_policy() only performs the actor forward pass. In
            # contrast to OnPolicyRunner.learn(), it does not clear recurrent
            # state for environments that just reset. Do that explicitly so
            # multi-episode evaluation matches training. The optional
            # touchdown mask isolates whether memory carried from hop 1 causes
            # the hop-2 failure without changing physics or observations.
            touchdown_mask = base_env._touchdown_event.clone()
            recurrent_reset_mask = dones.to(base_env.device).bool()
            if args_cli.reset_rnn_on_touchdown:
                recurrent_reset_mask = recurrent_reset_mask | touchdown_mask
            runner.alg.policy.reset(recurrent_reset_mask)
        step_count += 1
        eval_touchdowns += int(torch.sum(touchdown_mask).item())
        eval_short_touchdowns += int(
            torch.sum(base_env._setup_touchdown_event).item()
        )
        eval_long_touchdowns += int(
            torch.sum(base_env._final_touchdown_event).item()
        )
        eval_hits += int(torch.sum(base_env._target_hit_event).item())
        # Deployment error must use the physical waypoint.  Using
        # _desired_pos_w here would give the two reference ablations different
        # measurement targets (midpoint for A, touchdown point for B).
        physical_target_xy, _ = base_env.commands.lookahead()
        xy_error = torch.linalg.norm(
            base_env._robot.data.root_pos_w[:, :2] - physical_target_xy,
            dim=1,
        )
        eval_max_xy_error = torch.maximum(eval_max_xy_error, xy_error)
        done_mask = dones.to(base_env.device).bool()
        if torch.any(done_mask):
            timeout_mask = local_episode_steps >= base_env.max_episode_length - 1
            eval_timeouts[done_mask & timeout_mask] += 1
            eval_deaths[done_mask & ~timeout_mask] += 1
            eval_episodes[done_mask] += 1
            local_episode_steps[done_mask] = 0
        if "log" in extras:
            latest_log = extras["log"]
        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            break
    if latest_log and not args_cli.compact_eval:
        print(f"[EVAL] steps={step_count}")
        for key in sorted(latest_log):
            if key.startswith(("Metrics/", "Diagnostics/")):
                value = latest_log[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()
                print(f"[EVAL] {key}={value}")
    if args_cli.max_steps is not None and args_cli.route == "random_two_hop":
        base_env = env.unwrapped
        touchdown_count = torch.sum(base_env._touchdown_count).float()
        short_touchdown_count = torch.sum(base_env._short_touchdown_count).float()
        long_touchdown_count = torch.sum(base_env._long_touchdown_count).float()
        per_env_hit_rate = base_env._target_hit_count.float() / torch.clamp(
            base_env._touchdown_count.float(), min=1.0
        )
        per_env_max_streak = base_env._max_consecutive_hits.float()
        valid_apex = base_env._settled_apex_valid
        direct_metrics = {
            "episode_touchdown_error_m": (
                torch.sum(base_env._touchdown_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_along_error_m": (
                torch.sum(base_env._touchdown_along_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_lateral_abs_error_m": (
                torch.sum(base_env._touchdown_lateral_abs_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_attitude_error_rad": (
                torch.sum(base_env._touchdown_attitude_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_error_mps": (
                torch.sum(base_env._touchdown_next_velocity_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_projection_mps": (
                torch.sum(base_env._touchdown_next_velocity_projection_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_lateral_abs_mps": (
                torch.sum(base_env._touchdown_next_velocity_lateral_abs_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "prepared_landing_rate": (
                torch.sum(base_env._prepared_landing_count).float()
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "target_hit_rate": (
                torch.sum(base_env._target_hit_count).float()
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "target_hit_rate_min": torch.min(per_env_hit_rate),
            "target_hit_rate_p10": torch.quantile(per_env_hit_rate, 0.10),
            "target_hit_rate_median": torch.median(per_env_hit_rate),
            "short_touchdown_error_m": (
                torch.sum(base_env._short_touchdown_error_sum)
                / torch.clamp(short_touchdown_count, min=1.0)
            ),
            "short_target_hit_rate": (
                torch.sum(base_env._short_target_hit_count).float()
                / torch.clamp(short_touchdown_count, min=1.0)
            ),
            "long_touchdown_error_m": (
                torch.sum(base_env._long_touchdown_error_sum)
                / torch.clamp(long_touchdown_count, min=1.0)
            ),
            "long_target_hit_rate": (
                torch.sum(base_env._long_target_hit_count).float()
                / torch.clamp(long_touchdown_count, min=1.0)
            ),
            "last_apex_height_m": (
                torch.mean(base_env._settled_apex_height[valid_apex])
                if torch.any(valid_apex)
                else torch.tensor(0.0, device=base_env.device)
            ),
            "last_apex_error_m": (
                torch.mean(base_env._settled_apex_error[valid_apex])
                if torch.any(valid_apex)
                else torch.tensor(0.0, device=base_env.device)
            ),
            "max_consecutive_hits": torch.mean(
                per_env_max_streak
            ),
            "max_consecutive_hits_min": torch.min(per_env_max_streak),
            "max_consecutive_hits_median": torch.median(per_env_max_streak),
            "route_completion": torch.mean(
                (
                    base_env._consecutive_hits
                    >= base_env.commands.steps_per_revolution
                ).float()
            ),
            "successful_waypoints": torch.mean(
                base_env._successful_cycles.float()
            ),
        }
        if hasattr(base_env, "_pair_attempts"):
            pair_attempts = torch.sum(base_env._pair_attempts).float()
            conditional_attempts = torch.sum(
                base_env._conditional_second_attempts
            ).float()
            direct_metrics["two_hop_pair_success_rate"] = (
                torch.sum(base_env._pair_hits).float()
                / torch.clamp(pair_attempts, min=1.0)
            )
            direct_metrics["conditional_second_hit_rate"] = (
                torch.sum(base_env._conditional_second_hits).float()
                / torch.clamp(conditional_attempts, min=1.0)
            )
        compact_direct_keys = {
            "episode_touchdown_error_m",
            "touchdown_along_error_m",
            "touchdown_lateral_abs_error_m",
            "target_hit_rate",
            "short_touchdown_error_m",
            "short_target_hit_rate",
            "long_touchdown_error_m",
            "long_target_hit_rate",
            "last_apex_height_m",
            "last_apex_error_m",
            "max_consecutive_hits",
            "max_consecutive_hits_median",
            "successful_waypoints",
            "two_hop_pair_success_rate",
            "conditional_second_hit_rate",
        }
        reference_mode = (
            "landing_xy" if args_cli.direct_landing_reference else "planner_midpoint"
        )
        print(f"[EVAL-DIRECT] reference_mode={reference_mode}", flush=True)
        print(f"[EVAL-DIRECT] steps={step_count}", flush=True)
        for key, value in direct_metrics.items():
            if args_cli.compact_eval and key not in compact_direct_keys:
                continue
            print(f"[EVAL-DIRECT] {key}={value.item()}", flush=True)
        deploy_success = (
            (eval_deaths == 0)
            & (base_env._touchdown_count >= args_cli.deploy_min_touchdowns)
            & (eval_max_xy_error <= args_cli.deploy_xy_tolerance)
        )
        print(f"[EVAL-DEPLOY] steps={step_count}", flush=True)
        completed_episodes = int(torch.sum(eval_episodes).item())
        if not args_cli.compact_eval:
            print(f"[EVAL-DEPLOY] episodes={completed_episodes}", flush=True)
            print(f"[EVAL-DEPLOY] touchdowns={eval_touchdowns}", flush=True)
            print(f"[EVAL-DEPLOY] short_touchdowns={eval_short_touchdowns}", flush=True)
            print(f"[EVAL-DEPLOY] long_touchdowns={eval_long_touchdowns}", flush=True)
            print(f"[EVAL-DEPLOY] target_hits={eval_hits}", flush=True)
        print(
            f"[EVAL-DEPLOY] touchdowns_per_env={eval_touchdowns / args_cli.num_envs:.6f}",
            flush=True,
        )
        if completed_episodes > 0 and not args_cli.compact_eval:
            print(
                "[EVAL-DEPLOY] touchdowns_per_episode="
                f"{eval_touchdowns / completed_episodes:.6f}",
                flush=True,
            )
        print(
            f"[EVAL-DEPLOY] death_rate={torch.mean((eval_deaths > 0).float()).item()}",
            flush=True,
        )
        print(
            f"[EVAL-DEPLOY] timeout_rate={torch.mean((eval_timeouts > 0).float()).item()}",
            flush=True,
        )
        print(
            f"[EVAL-DEPLOY] deploy_success_rate={torch.mean(deploy_success.float()).item()}",
            flush=True,
        )
        print(
            f"[EVAL-DEPLOY] max_xy_error_mean_m={torch.mean(eval_max_xy_error).item()}",
            flush=True,
        )
        if hasattr(base_env, "_physical_death_total") and not args_cli.compact_eval:
            physical_deaths = torch.sum(base_env._physical_death_total).item()
            low_deaths = torch.sum(base_env._low_height_death_total).item()
            tilt_deaths = torch.sum(base_env._tilt_death_total).item()
            print(f"[EVAL-DEPLOY] physical_death_events={physical_deaths}", flush=True)
            print(f"[EVAL-DEPLOY] low_height_death_events={low_deaths}", flush=True)
            print(f"[EVAL-DEPLOY] tilt_death_events={tilt_deaths}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
