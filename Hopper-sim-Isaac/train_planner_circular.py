"""Train two-cycle planner-conditioned circular or random-route hopping."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Quadhopper planner circular task")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="37-D stable, 42-D legacy planner, 43-D planner, or 47-D planner checkpoint"
)
parser.add_argument(
    "--height_stage",
    choices=("low", "high", "alternate"),
    default="low",
    help="Curriculum stage: fixed 0.70 m, fixed 1.00 m, or alternating commands.",
)
parser.add_argument("--resume_optimizer", action="store_true")
parser.add_argument(
    "--accuracy_finetune",
    action="store_true",
    help="Fine-tune first-attempt landing accuracy; terminate an episode on each miss.",
)
parser.add_argument(
    "--streak_finetune",
    action="store_true",
    help="Fine-tune consecutive hits with balanced phases while retaining recovery trajectories.",
)
parser.add_argument(
    "--anticipatory_finetune",
    action="store_true",
    help="Fine-tune a two-hop plan whose Pt terminal velocity and attitude prepare Pt+1.",
)
parser.add_argument(
    "--continuity_finetune",
    action="store_true",
    help="Curriculum for consecutive two-hop hits with next-target landing preparation.",
)
parser.add_argument(
    "--first_attempt_finetune",
    action="store_true",
    help="Train a rolling route where every touchdown advances, including misses.",
)
parser.add_argument(
    "--pair_polish",
    action="store_true",
    help="Low-noise final polish emphasizing consecutive first-attempt pairs.",
)
parser.add_argument(
    "--safety_finetune",
    action="store_true",
    help="Fine-tune a random-drop tracking policy to prioritize not falling over.",
)
parser.add_argument(
    "--robust_finetune",
    action="store_true",
    help="Gentle deployment robustness fine-tune with small perturbations and anti-saturation shaping.",
)
parser.add_argument(
    "--static_apex_finetune",
    action="store_true",
    help="Fine-tune short two-hop behavior against a stationary per-hop apex reference instead of the full planner trajectory.",
)
parser.add_argument(
    "--trajectory_landing_finetune",
    action="store_true",
    help="Fine-tune random two-hop behavior with full trajectory tracking and touchdown accuracy as the primary objectives.",
)
parser.add_argument(
    "--baseline_warmstart_finetune",
    action="store_true",
    help="Warm-start random two-hop from the stable baseline while preserving the baseline jump reward contract.",
)
parser.add_argument(
    "--freeze_baseline_actor",
    action="store_true",
    help=(
        "During a 37-D to planner warm-start, freeze the actor's original "
        "pathway and train only the newly appended recurrent input columns."
    ),
)
parser.add_argument(
    "--static_apex_height_probe",
    action="store_true",
    help="Static-apex diagnostic stage: preserve 1 m hopping on near-stationary commands before learning XY landing.",
)
parser.add_argument(
    "--baseline_compatible_stance_reference",
    action="store_true",
    help="Keep the inherited 37-D target height at the apex command during stance.",
)
parser.add_argument("--smooth_distance_curriculum", action="store_true")
parser.add_argument("--distance_curriculum_iterations", type=float, default=100.0)
parser.add_argument(
    "--retry_target_on_miss",
    action="store_true",
    help="Keep the current waypoint after a miss during the initial accuracy curriculum.",
)
parser.add_argument(
    "--long_hop_curriculum",
    action="store_true",
    help="Randomize the initial short/long phase, expand only the long-hop distance, "
    "and enable long-phase flight-stability shaping.",
)
parser.add_argument("--long_curriculum_radius_min", type=float, default=0.30)
parser.add_argument("--long_curriculum_radius_max", type=float, default=0.38)
parser.add_argument("--long_curriculum_turn_deg", type=float, default=30.0)
parser.add_argument(
    "--two_hop_transition_curriculum",
    action="store_true",
    help="Start every episode on the short phase and shape its descent/touchdown "
    "as the preparation state for the following long hop.",
)
parser.add_argument(
    "--direct_variable_height",
    action="store_true",
    help="Train periodic alternating heights directly from a stable baseline without a fixed-height specialist.",
)
parser.add_argument(
    "--expand_variable_height",
    action="store_true",
    help="Expand an existing alternating-height policy to a wider periodic height pair.",
)
parser.add_argument("--height_high", type=float, default=1.0)
parser.add_argument("--height_low", type=float, default=0.7)
parser.add_argument("--adaptive_apex_height", action="store_true")
parser.add_argument("--adaptive_height_min", type=float, default=1.00)
parser.add_argument("--adaptive_height_max", type=float, default=1.25)
parser.add_argument("--adaptive_height_distance_start", type=float, default=0.10)
parser.add_argument("--adaptive_height_distance_end", type=float, default=0.35)
parser.add_argument("--adaptive_height_noise", type=float, default=0.0)
parser.add_argument(
    "--route",
    choices=("circle", "random_two_hop"),
    default="circle",
    help="Ground waypoint generator. random_two_hop alternates 0.5--0.8 m and 0.8--1.0 m hops.",
)
parser.add_argument(
    "--distance_stage",
    choices=("direction", "medium", "short", "bridge", "full", "custom"),
    default="full",
    help="Random-route distance curriculum stage; custom uses the four explicit radius flags.",
)
parser.add_argument("--short_radius_min", type=float, default=0.5)
parser.add_argument("--short_radius_max", type=float, default=0.8)
parser.add_argument("--long_radius_min", type=float, default=0.8)
parser.add_argument("--long_radius_max", type=float, default=1.0)
parser.add_argument(
    "--start_from_random_drop",
    action="store_true",
    help="Reset from a baseline-style random high release before the first commanded hop.",
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
parser.add_argument("--takeoff_xy_bias_curriculum_iterations", type=float, default=0.0)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--anticipatory_speed", type=float, default=0.30)
parser.add_argument("--anticipatory_tilt_deg", type=float, default=6.0)
parser.add_argument("--anticipatory_reward_scale", type=float, default=1.0)
parser.add_argument("--airborne_setup_finetune", action="store_true")
parser.add_argument("--airborne_setup_start_phase", type=float, default=0.52)
parser.add_argument("--airborne_setup_tilt_deg", type=float, default=4.0)
parser.add_argument("--airborne_setup_reward_scale", type=float, default=1.0)
parser.add_argument("--airborne_setup_landing_gate_width", type=float, default=0.14)
parser.add_argument(
    "--phase_tilt_finetune",
    action="store_true",
    help="Reward target-direction tilt only in early ascent, then recover upright for touchdown.",
)
parser.add_argument("--takeoff_tilt_deg", type=float, default=8.0)
parser.add_argument("--takeoff_tilt_phase_end", type=float, default=0.40)
parser.add_argument("--takeoff_tilt_reward_scale", type=float, default=1.0)
parser.add_argument(
    "--phase_takeoff_finetune",
    action="store_true",
    help="Relax early-ascent attitude penalties and reward target-direction takeoff velocity.",
)
parser.add_argument(
    "--takeoff_impulse_finetune",
    action="store_true",
    help="Static-apex substage: emphasize liftoff horizontal impulse before touchdown accuracy.",
)
parser.add_argument(
    "--setup_hop_finetune",
    action="store_true",
    help="Treat the short hop touchdown as a preparation state for the following long hop.",
)
parser.add_argument("--takeoff_velocity_target", type=float, default=0.30)
parser.add_argument("--takeoff_velocity_reward_scale", type=float, default=1.0)
parser.add_argument("--takeoff_velocity_from_planner", action="store_true")
parser.add_argument("--takeoff_velocity_event_reward", action="store_true")
parser.add_argument("--takeoff_long_hop_only", action="store_true")
parser.add_argument("--takeoff_phase_attitude_relax", type=float, default=0.50)
parser.add_argument("--takeoff_phase_angvel_relax", type=float, default=0.35)
parser.add_argument("--takeoff_phase_tilt_barrier_relax", type=float, default=0.60)
parser.add_argument("--setup_velocity_target", type=float, default=0.25)
parser.add_argument("--setup_velocity_width", type=float, default=0.18)
parser.add_argument("--setup_velocity_reward_scale", type=float, default=60.0)
parser.add_argument("--setup_lateral_velocity_penalty", type=float, default=35.0)
parser.add_argument("--setup_touchdown_attitude_deadband", type=float, default=0.06)
parser.add_argument("--final_touchdown_attitude_penalty", type=float, default=60.0)
parser.add_argument("--final_touchdown_velocity_penalty", type=float, default=25.0)
parser.add_argument("--landing_correction_gain", type=float, default=0.0)
parser.add_argument(
    "--planner_reference_blend",
    type=float,
    default=None,
    help="Fixed blend from stationary apex (0.0) to full planner trajectory (1.0); disables the automatic blend curriculum.",
)
parser.add_argument(
    "--direct_landing_reference",
    action="store_true",
    help=(
        "Use the physical touchdown target XY as the stationary flight reference "
        "while retaining the planner observation and two-hop state machine."
    ),
)
parser.add_argument(
    "--pair_restart_queue",
    action="store_true",
    help="Restart a new short/long pair around the measured second touchdown.",
)
parser.add_argument(
    "--two_hop_episode",
    action="store_true",
    help="Terminate and reset each environment after the second touchdown.",
)
parser.add_argument(
    "--safety_attitude_scale",
    type=float,
    default=1.0,
    help="Multiplier for safety_finetune attitude/touchdown/tilt-barrier penalties.",
)
parser.add_argument(
    "--safety_angular_vel_scale",
    type=float,
    default=1.0,
    help="Multiplier for safety_finetune angular velocity penalty.",
)
parser.add_argument(
    "--safety_xy_scale",
    type=float,
    default=1.0,
    help="Multiplier for safety_finetune XY tracking, landing, and lateral-velocity terms.",
)
parser.add_argument(
    "--safety_xy_dense_scale",
    type=float,
    default=1.0,
    help="Extra multiplier for dense in-flight XY shaping; keep touchdown hit rewards at safety_xy_scale.",
)
parser.add_argument(
    "--safety_yaw_scale",
    type=float,
    default=1.0,
    help="Multiplier for safety_finetune yaw penalty.",
)
parser.add_argument(
    "--safety_height_scale",
    type=float,
    default=1.0,
    help="Multiplier for safety_finetune apex/overshoot height penalties.",
)
parser.add_argument(
    "--attitude_damping_polish",
    action="store_true",
    help="Overlay a conservative damping polish: rate/action smoothing without stronger angle tracking.",
)
parser.add_argument("--damping_airborne_angvel_start", type=float, default=1.05)
parser.add_argument("--damping_airborne_angvel_scale", type=float, default=2.2)
parser.add_argument("--damping_action_spread_scale", type=float, default=24.0)
parser.add_argument("--damping_action_rate_scale", type=float, default=1.8)
parser.add_argument("--damping_yaw_rate_start", type=float, default=0.80)
parser.add_argument("--damping_yaw_rate_scale", type=float, default=0.18)
parser.add_argument(
    "--free_airborne_roll_pitch",
    action="store_true",
    help="Disable airborne roll/pitch angle and tilt-barrier penalties while keeping touchdown attitude penalties.",
)
parser.add_argument(
    "--static_apex_min_valid_apex",
    type=float,
    default=0.88,
    help="Minimum cycle apex required for static-apex touchdown positive rewards.",
)
parser.add_argument(
    "--low_apex_event_penalty",
    type=float,
    default=0.0,
    help="Positive magnitude for the low-apex touchdown event penalty in random_two_hop "
    "from-zero training (0 = disabled, the env default). Fixes the flat-hop local optimum "
    "where long hops fly below minimum_valid_apex and can never hit.",
)
parser.add_argument(
    "--lenient_apex_hit",
    action="store_true",
    help="Disable the apex-tolerance hit gate (require_apex_tolerance_for_hit=False) in "
    "random_two_hop from-zero training. The trainer's general setup leaves the gate ON "
    "(apex must be within apex_tolerance of the 1 m command), which long hops structurally "
    "violate, so they can never hit no matter how precisely they land.",
)
parser.add_argument(
    "--tilt_death_sq_threshold",
    type=float,
    default=0.5,
    help="roll_pitch_sq (qx^2+qy^2) tilt-death threshold; 0.5 ~= 90 deg roll/pitch. "
    "From-zero long hops flip over ~0.3 s after takeoff and die mid-flight before the "
    "policy can ever learn them; raise this (e.g. 0.95) during from-zero training so "
    "long hops can complete and land.",
)
parser.add_argument(
    "--terminate_far_xy",
    type=float,
    default=2.0,
    help="random_two_hop route: terminate when the root drifts farther than this from the "
    "current target. Long hops die mid-flight at the default 2.0, so the policy never "
    "collects long-hop touchdowns. Relax (e.g. 6.0) during from-zero long-hop learning.",
)
parser.add_argument(
    "--terminate_angvel_norm",
    type=float,
    default=18.0,
    help="random_two_hop route: terminate when the body angular-velocity norm exceeds this. "
    "Relax (e.g. 30.0) during from-zero long-hop learning for the same reason as above.",
)
parser.add_argument(
    "--static_apex_low_apex_penalty",
    type=float,
    default=180.0,
    help="Positive magnitude for penalizing static-apex touchdown below the valid apex gate.",
)
parser.add_argument(
    "--static_apex_curriculum_iterations",
    type=float,
    default=100.0,
    help="Equivalent iterations to keep a pure stationary-apex reference before blending the planner trajectory.",
)
parser.add_argument(
    "--full_planner_curriculum_iterations",
    type=float,
    default=400.0,
    help="Equivalent iteration at which the trajectory-reference blend reaches full strength.",
)
parser.add_argument(
    "--airborne_stability_scale",
    type=float,
    default=1.0,
    help="Extra in-air roll/pitch, roll/pitch-rate, and motor-spread penalties for smoother real deployment.",
)
parser.add_argument(
    "--safety_mass_min",
    type=float,
    default=0.95,
    help="Minimum mass multiplier for safety_finetune dynamics randomization; lower means stronger effective thrust.",
)
parser.add_argument(
    "--safety_mass_max",
    type=float,
    default=1.05,
    help="Maximum mass multiplier for safety_finetune dynamics randomization.",
)
parser.add_argument(
    "--thrust_rand_min",
    type=float,
    default=1.0,
    help="Minimum per-motor thrust multiplier for sim-to-real thrust randomization.",
)
parser.add_argument(
    "--thrust_rand_max",
    type=float,
    default=1.0,
    help="Maximum per-motor thrust multiplier for sim-to-real thrust randomization.",
)
parser.add_argument(
    "--thrust_shape_rand_min",
    type=float,
    default=1.0,
    help="Minimum multiplier for the quadratic thrust-curve term.",
)
parser.add_argument(
    "--thrust_shape_rand_max",
    type=float,
    default=1.0,
    help="Maximum multiplier for the quadratic thrust-curve term.",
)
parser.add_argument(
    "--motor_time_constant_min",
    type=float,
    default=None,
    help="Enable per-motor time-constant randomization with this lower bound (s); "
    "sim2real: real motors appear ~0.03s vs the 0.125s sim assumption.",
)
parser.add_argument(
    "--motor_time_constant_max",
    type=float,
    default=None,
    help="Upper bound (s) for per-motor time-constant randomization; "
    "must be provided together with --motor_time_constant_min.",
)
parser.add_argument(
    "--fine_tune_lr",
    type=float,
    default=None,
    help="Override the PPO learning rate for conservative sim-to-real fine-tunes.",
)
parser.add_argument(
    "--fixed_action_std",
    type=float,
    default=None,
    help="Override the random-route fixed Gaussian action std; useful for low-noise sim-to-real fine-tunes.",
)
parser.add_argument("--initial_roll_pitch_perturb_deg", type=float, default=0.0)
parser.add_argument("--initial_yaw_perturb_deg", type=float, default=0.0)
parser.add_argument("--initial_xy_velocity_perturb", type=float, default=0.0)
parser.add_argument("--initial_z_velocity_perturb", type=float, default=0.0)
parser.add_argument("--initial_angular_velocity_perturb", type=float, default=0.0)
parser.add_argument("--airborne_force_disturbance", type=float, default=0.0)
parser.add_argument("--airborne_torque_disturbance", type=float, default=0.0)
parser.add_argument("--disturbance_resample_time", type=float, default=0.15)
parser.add_argument("--tolerance_curriculum_start", type=float, default=0.15)
parser.add_argument("--tolerance_curriculum_iterations", type=float, default=60.0)
parser.add_argument(
    "--low_curriculum_iterations",
    type=float,
    default=300.0,
    help="Deprecated compatibility option; fixed 0.70 m training does not use a height curriculum.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pair_polish and not args_cli.first_attempt_finetune:
    parser.error("--pair_polish requires --first_attempt_finetune")
if args_cli.static_apex_height_probe and not args_cli.static_apex_finetune:
    parser.error("--static_apex_height_probe requires --static_apex_finetune")
if args_cli.takeoff_impulse_finetune and not args_cli.static_apex_finetune:
    parser.error("--takeoff_impulse_finetune requires --static_apex_finetune")
if args_cli.smooth_distance_curriculum:
    smooth_mode = (
        args_cli.first_attempt_finetune
        or args_cli.safety_finetune
        or args_cli.robust_finetune
        or args_cli.static_apex_finetune
        or args_cli.trajectory_landing_finetune
        or args_cli.baseline_warmstart_finetune
    )
    if not smooth_mode:
        parser.error(
            "--smooth_distance_curriculum requires --first_attempt_finetune, --safety_finetune, --robust_finetune, --static_apex_finetune, --trajectory_landing_finetune, or --baseline_warmstart_finetune"
        )
    if args_cli.distance_stage != "full" and not (
        args_cli.static_apex_finetune
        or args_cli.trajectory_landing_finetune
        or args_cli.baseline_warmstart_finetune
    ):
        parser.error(
            "--smooth_distance_curriculum with non-static modes requires --distance_stage full"
        )
if args_cli.long_hop_curriculum:
    if args_cli.route != "random_two_hop":
        parser.error("--long_hop_curriculum requires --route random_two_hop")
    if args_cli.distance_curriculum_iterations <= 0.0:
        parser.error("--long_hop_curriculum requires --distance_curriculum_iterations > 0")
    if not (
        0.0 <= args_cli.long_curriculum_radius_min
        <= args_cli.long_curriculum_radius_max
        <= args_cli.long_radius_max
    ):
        parser.error(
            "long curriculum radii must satisfy 0 <= min <= max <= --long_radius_max"
        )
    if not (0.0 < args_cli.long_curriculum_turn_deg <= args_cli.max_turn_angle):
        parser.error(
            "--long_curriculum_turn_deg must be in (0, --max_turn_angle]"
        )
if args_cli.two_hop_transition_curriculum:
    if not args_cli.long_hop_curriculum:
        parser.error(
            "--two_hop_transition_curriculum requires --long_hop_curriculum"
        )
    if not args_cli.relative_next_hop:
        parser.error(
            "--two_hop_transition_curriculum requires --relative_next_hop"
        )
if args_cli.freeze_baseline_actor and not args_cli.baseline_warmstart_finetune:
    parser.error("--freeze_baseline_actor requires --baseline_warmstart_finetune")
if args_cli.pair_restart_queue and args_cli.route != "random_two_hop":
    parser.error("--pair_restart_queue requires --route random_two_hop")
if args_cli.two_hop_episode and args_cli.route != "random_two_hop":
    parser.error("--two_hop_episode requires --route random_two_hop")
if args_cli.two_hop_episode and args_cli.pair_restart_queue:
    parser.error("--two_hop_episode and --pair_restart_queue are mutually exclusive")
if sum(
    int(flag)
    for flag in (
        args_cli.accuracy_finetune,
        args_cli.streak_finetune,
        args_cli.anticipatory_finetune,
        args_cli.continuity_finetune,
        args_cli.first_attempt_finetune,
        args_cli.safety_finetune,
        args_cli.robust_finetune,
        args_cli.static_apex_finetune,
        args_cli.trajectory_landing_finetune,
        args_cli.baseline_warmstart_finetune,
    )
) > 1:
    parser.error("fine-tuning modes are separate stages")
DISTANCE_STAGES = {
    "direction": (0.22, 0.30, 0.22, 0.30, 0.12),
    "medium": (0.30, 0.50, 0.30, 0.50, 0.10),
    "short": (0.50, 0.80, 0.50, 0.80, 0.08),
    "bridge": (0.50, 0.80, 0.65, 0.85, 0.06),
    "full": (0.50, 0.80, 0.80, 1.00, 0.06),
}
if args_cli.route == "random_two_hop" and args_cli.distance_stage != "custom":
    (
        args_cli.short_radius_min,
        args_cli.short_radius_max,
        args_cli.long_radius_min,
        args_cli.long_radius_max,
        _,
    ) = DISTANCE_STAGES[args_cli.distance_stage]
if not (0.0 <= args_cli.initial_drop_height_min <= args_cli.initial_drop_height_max):
    parser.error("--initial_drop_height_min/max must satisfy 0 <= min <= max")
if not (0.1 <= args_cli.safety_mass_min <= args_cli.safety_mass_max):
    parser.error("--safety_mass_min/max must satisfy 0.1 <= min <= max")
if not (0.1 <= args_cli.thrust_rand_min <= args_cli.thrust_rand_max):
    parser.error("--thrust_rand_min/max must satisfy 0.1 <= min <= max")
if not (0.1 <= args_cli.thrust_shape_rand_min <= args_cli.thrust_shape_rand_max):
    parser.error("--thrust_shape_rand_min/max must satisfy 0.1 <= min <= max")
if (args_cli.motor_time_constant_min is None) != (args_cli.motor_time_constant_max is None):
    parser.error("--motor_time_constant_min/max must be provided together")
if args_cli.motor_time_constant_min is not None and not (
    0.0 <= args_cli.motor_time_constant_min <= args_cli.motor_time_constant_max
):
    parser.error("--motor_time_constant_min/max must satisfy 0 <= min <= max")
if args_cli.fine_tune_lr is not None and not (1e-7 <= args_cli.fine_tune_lr <= 1e-3):
    parser.error("--fine_tune_lr must be within [1e-7, 1e-3]")
if args_cli.fixed_action_std is not None and args_cli.fixed_action_std < 0.0:
    parser.error("--fixed_action_std must be non-negative")
if args_cli.airborne_stability_scale < 0.0:
    parser.error("--airborne_stability_scale must be non-negative")
if args_cli.safety_xy_dense_scale < 0.0:
    parser.error("--safety_xy_dense_scale must be non-negative")
if args_cli.damping_airborne_angvel_start < 0.0:
    parser.error("--damping_airborne_angvel_start must be non-negative")
if args_cli.damping_airborne_angvel_scale < 0.0:
    parser.error("--damping_airborne_angvel_scale must be non-negative")
if args_cli.damping_action_spread_scale < 0.0:
    parser.error("--damping_action_spread_scale must be non-negative")
if args_cli.damping_action_rate_scale < 0.0:
    parser.error("--damping_action_rate_scale must be non-negative")
if args_cli.damping_yaw_rate_start < 0.0:
    parser.error("--damping_yaw_rate_start must be non-negative")
if args_cli.damping_yaw_rate_scale < 0.0:
    parser.error("--damping_yaw_rate_scale must be non-negative")
if args_cli.static_apex_min_valid_apex <= 0.0:
    parser.error("--static_apex_min_valid_apex must be positive")
if args_cli.static_apex_low_apex_penalty < 0.0:
    parser.error("--static_apex_low_apex_penalty must be non-negative")
if args_cli.static_apex_curriculum_iterations < 0.0:
    parser.error("--static_apex_curriculum_iterations must be non-negative")
if args_cli.full_planner_curriculum_iterations <= args_cli.static_apex_curriculum_iterations:
    parser.error("--full_planner_curriculum_iterations must be greater than --static_apex_curriculum_iterations")
if args_cli.planner_reference_blend is not None and not (
    0.0 <= args_cli.planner_reference_blend <= 1.0
):
    parser.error("--planner_reference_blend must be in [0, 1]")
if args_cli.takeoff_xy_bias_max < 0.0:
    parser.error("--takeoff_xy_bias_max must be non-negative")
if args_cli.takeoff_xy_bias_curriculum_iterations < 0.0:
    parser.error("--takeoff_xy_bias_curriculum_iterations must be non-negative")
if args_cli.adaptive_height_min <= 0.0:
    parser.error("--adaptive_height_min must be positive")
if args_cli.adaptive_height_max < args_cli.adaptive_height_min:
    parser.error("--adaptive_height_max must be >= --adaptive_height_min")
if args_cli.adaptive_height_distance_end <= args_cli.adaptive_height_distance_start:
    parser.error("--adaptive_height_distance_end must be greater than --adaptive_height_distance_start")
if args_cli.adaptive_height_noise < 0.0:
    parser.error("--adaptive_height_noise must be non-negative")
if not (0.0 <= args_cli.airborne_setup_start_phase <= 1.0):
    parser.error("--airborne_setup_start_phase must be in [0, 1]")
if args_cli.airborne_setup_tilt_deg < 0.0:
    parser.error("--airborne_setup_tilt_deg must be non-negative")
if args_cli.airborne_setup_reward_scale < 0.0:
    parser.error("--airborne_setup_reward_scale must be non-negative")
if args_cli.airborne_setup_landing_gate_width < 0.0:
    parser.error("--airborne_setup_landing_gate_width must be non-negative")
if not (0.0 <= args_cli.takeoff_tilt_phase_end <= 1.0):
    parser.error("--takeoff_tilt_phase_end must be in [0, 1]")
if args_cli.takeoff_tilt_reward_scale < 0.0:
    parser.error("--takeoff_tilt_reward_scale must be non-negative")
if args_cli.takeoff_velocity_target < 0.0:
    parser.error("--takeoff_velocity_target must be non-negative")
if args_cli.takeoff_velocity_reward_scale < 0.0:
    parser.error("--takeoff_velocity_reward_scale must be non-negative")
if args_cli.setup_velocity_target < 0.0:
    parser.error("--setup_velocity_target must be non-negative")
if args_cli.setup_velocity_width <= 0.0:
    parser.error("--setup_velocity_width must be positive")
if args_cli.setup_velocity_reward_scale < 0.0:
    parser.error("--setup_velocity_reward_scale must be non-negative")
if args_cli.setup_lateral_velocity_penalty < 0.0:
    parser.error("--setup_lateral_velocity_penalty must be non-negative")
if args_cli.setup_touchdown_attitude_deadband < 0.0:
    parser.error("--setup_touchdown_attitude_deadband must be non-negative")
if args_cli.final_touchdown_attitude_penalty < 0.0:
    parser.error("--final_touchdown_attitude_penalty must be non-negative")
if args_cli.final_touchdown_velocity_penalty < 0.0:
    parser.error("--final_touchdown_velocity_penalty must be non-negative")
for name in (
    "takeoff_phase_attitude_relax",
    "takeoff_phase_angvel_relax",
    "takeoff_phase_tilt_barrier_relax",
):
    if not (0.0 <= getattr(args_cli, name) <= 1.0):
        parser.error(f"--{name} must be in [0, 1]")
if args_cli.disturbance_resample_time <= 0.0:
    parser.error("--disturbance_resample_time must be positive")
args_cli.headless = True
args_cli.rendering_mode = "performance"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime

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

from Quadhopper_Planner_Circular.checkpoint_migration import (
    absolute_next_to_relative_state_dict,
    migrate_stable_checkpoint,
)
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "low": "quadhopper_planner_circular_v17_fixed_070_from_stable",
    "high": "quadhopper_planner_circular_v15_high_100",
    "alternate": "quadhopper_planner_circular_v15_alternate_070_100",
}


def infer_checkpoint_input_width(checkpoint_data, checkpoint_path: Path) -> int:
    state = checkpoint_data.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if "memory_a.rnn.weight_ih_l0" in state:
        return state["memory_a.rnn.weight_ih_l0"].shape[1]
    if "actor.0.weight" in state:
        actor_width = state["actor.0.weight"].shape[1]
        raise ValueError(
            f"Unsupported feed-forward checkpoint ({actor_width}-D actor) for recurrent "
            f"43-D planner training: {checkpoint_path}"
        )
    sample_keys = ", ".join(list(state.keys())[:8])
    raise ValueError(
        f"Cannot infer checkpoint observation width from {checkpoint_path}. "
        f"First keys: {sample_keys}"
    )


EXPERIMENT = EXPERIMENTS[args_cli.height_stage]
if args_cli.direct_variable_height:
    if args_cli.height_stage != "alternate":
        raise ValueError("--direct_variable_height requires --height_stage alternate")
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    EXPERIMENT = (
        f"quadhopper_planner_circular_v21_direct_alternate_{low_cm:03d}_{high_cm:03d}"
    )
if args_cli.expand_variable_height:
    if args_cli.height_stage != "alternate":
        raise ValueError("--expand_variable_height requires --height_stage alternate")
    if args_cli.direct_variable_height or args_cli.accuracy_finetune:
        raise ValueError("--expand_variable_height is a separate curriculum stage")
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    EXPERIMENT = (
        f"quadhopper_planner_circular_v22_expand_alternate_{low_cm:03d}_{high_cm:03d}"
    )
if args_cli.accuracy_finetune:
    if args_cli.direct_variable_height:
        raise ValueError("Use direct variable-height learning before --accuracy_finetune")
    if args_cli.height_stage == "low":
        EXPERIMENT = "quadhopper_planner_circular_v18_fixed_070_accuracy"
    elif args_cli.height_stage == "alternate":
        low_cm = round(args_cli.height_low * 100.0)
        high_cm = round(args_cli.height_high * 100.0)
        EXPERIMENT = (
            f"quadhopper_planner_circular_v20_alternate_{low_cm:03d}_{high_cm:03d}_height_accuracy"
        )
    elif args_cli.height_stage == "high" and args_cli.route == "random_two_hop":
        # The random-route block below replaces this placeholder with its
        # dedicated v25 full/distance-stage namespace.  Fixed 1 m route
        # training has already solved height and may fine-tune XY accuracy.
        EXPERIMENT = "quadhopper_planner_circular_v20_fixed_100_accuracy"
    else:
        raise ValueError(
            "--accuracy_finetune supports low/alternate stages, plus fixed-high random_two_hop"
        )
fixed_high_soft_accuracy = False
fixed_high_streak = False
fixed_high_anticipatory = False
fixed_high_continuity = False
fixed_high_first_attempt = False
fixed_high_pair_polish = False
fixed_high_smooth_distance = False
fixed_high_safety = False
fixed_high_robust = False
fixed_high_trajectory_landing = False
fixed_high_baseline_warmstart = False
if args_cli.route == "random_two_hop":
    if args_cli.direct_variable_height or args_cli.expand_variable_height:
        raise ValueError(
            "random_two_hop is a separate route-transfer stage; do not combine it with another curriculum flag"
        )
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    height_label = (
        f"alternate_{low_cm:03d}_{high_cm:03d}"
        if args_cli.height_stage == "alternate"
        else f"fixed_{round((args_cli.height_low if args_cli.height_stage == 'low' else args_cli.height_high) * 100.0):03d}"
    )
    fixed_high_soft_accuracy = (
        args_cli.accuracy_finetune and args_cli.height_stage == "high"
    )
    fixed_high_streak = args_cli.streak_finetune and args_cli.height_stage == "high"
    fixed_high_anticipatory = (
        args_cli.anticipatory_finetune and args_cli.height_stage == "high"
    )
    fixed_high_continuity = (
        args_cli.continuity_finetune and args_cli.height_stage == "high"
    )
    fixed_high_first_attempt = (
        args_cli.first_attempt_finetune and args_cli.height_stage == "high"
    )
    fixed_high_pair_polish = fixed_high_first_attempt and args_cli.pair_polish
    fixed_high_safety = args_cli.safety_finetune and args_cli.height_stage == "high"
    fixed_high_robust = args_cli.robust_finetune and args_cli.height_stage == "high"
    fixed_high_trajectory_landing = (
        args_cli.trajectory_landing_finetune and args_cli.height_stage == "high"
    )
    fixed_high_baseline_warmstart = (
        args_cli.baseline_warmstart_finetune and args_cli.height_stage == "high"
    )
    fixed_high_static_apex = (
        args_cli.static_apex_finetune and args_cli.height_stage == "high"
    )
    fixed_high_smooth_distance = (
        (
            fixed_high_first_attempt
            or fixed_high_safety
            or fixed_high_robust
            or fixed_high_static_apex
            or fixed_high_trajectory_landing
            or fixed_high_baseline_warmstart
        )
        and args_cli.smooth_distance_curriculum
    )
    route_version = (
        "v67"
        if fixed_high_baseline_warmstart
        else
        "v66"
        if fixed_high_trajectory_landing
        else
        "v65"
        if fixed_high_static_apex and args_cli.baseline_compatible_stance_reference
        else
        "v64"
        if fixed_high_static_apex and args_cli.static_apex_height_probe
        else
        "v63"
        if fixed_high_static_apex
        else
        "v61"
        if fixed_high_robust and args_cli.start_from_random_drop
        else
        "v60"
        if fixed_high_safety and args_cli.start_from_random_drop
        else
        "v59"
        if args_cli.start_from_random_drop
        else
        "v36"
        if fixed_high_smooth_distance
        else "v35"
        if fixed_high_pair_polish
        else "v34"
        if fixed_high_first_attempt
        else "v32"
        if fixed_high_continuity
        else "v31"
        if fixed_high_anticipatory
        else "v30"
        if args_cli.relative_next_hop and fixed_high_streak
        else "v29"
        if args_cli.relative_next_hop
        else "v28"
        if fixed_high_streak
        else "v26"
        if fixed_high_soft_accuracy
        else "v25"
        if args_cli.accuracy_finetune
        else "v24"
    )
    accuracy_suffix = (
        "_relative_next_baseline_warmstart_smooth_distance"
        if fixed_high_baseline_warmstart
        and args_cli.relative_next_hop
        and fixed_high_smooth_distance
        else "_baseline_warmstart_smooth_distance"
        if fixed_high_baseline_warmstart and fixed_high_smooth_distance
        else "_relative_next_baseline_warmstart"
        if fixed_high_baseline_warmstart and args_cli.relative_next_hop
        else "_baseline_warmstart"
        if fixed_high_baseline_warmstart
        else
        "_relative_next_base_stance_trajectory_landing_smooth_distance"
        if fixed_high_trajectory_landing
        and args_cli.relative_next_hop
        and fixed_high_smooth_distance
        else "_base_stance_trajectory_landing_smooth_distance"
        if fixed_high_trajectory_landing and fixed_high_smooth_distance
        else "_relative_next_base_stance_trajectory_landing"
        if fixed_high_trajectory_landing and args_cli.relative_next_hop
        else "_base_stance_trajectory_landing"
        if fixed_high_trajectory_landing
        else
        "_rel_static_base_stance_height_probe"
        if fixed_high_static_apex
        and args_cli.relative_next_hop
        and args_cli.static_apex_height_probe
        and args_cli.baseline_compatible_stance_reference
        else "_static_base_stance_height_probe"
        if fixed_high_static_apex
        and args_cli.static_apex_height_probe
        and args_cli.baseline_compatible_stance_reference
        else "_rel_static_base_stance"
        if fixed_high_static_apex
        and args_cli.relative_next_hop
        and args_cli.baseline_compatible_stance_reference
        else "_static_base_stance"
        if fixed_high_static_apex and args_cli.baseline_compatible_stance_reference
        else "_rel_static_height_probe"
        if fixed_high_static_apex
        and args_cli.relative_next_hop
        and args_cli.static_apex_height_probe
        else "_static_height_probe"
        if fixed_high_static_apex and args_cli.static_apex_height_probe
        else "_rel_static_hgate_smooth"
        if fixed_high_static_apex
        and args_cli.relative_next_hop
        and fixed_high_smooth_distance
        else "_static_hgate_smooth"
        if fixed_high_static_apex and fixed_high_smooth_distance
        else "_rel_static_hgate"
        if fixed_high_static_apex and args_cli.relative_next_hop
        else "_static_hgate"
        if fixed_high_static_apex
        else
        "_relative_next_random_drop_robust_smooth_distance"
        if (
            fixed_high_robust
            and fixed_high_smooth_distance
            and args_cli.start_from_random_drop
            and args_cli.relative_next_hop
        )
        else "_random_drop_robust_smooth_distance"
        if fixed_high_robust and fixed_high_smooth_distance and args_cli.start_from_random_drop
        else
        "_relative_next_random_drop_robust"
        if fixed_high_robust and args_cli.start_from_random_drop and args_cli.relative_next_hop
        else "_random_drop_robust"
        if fixed_high_robust and args_cli.start_from_random_drop
        else
        "_relative_next_random_drop_safety_smooth_distance"
        if (
            fixed_high_safety
            and fixed_high_smooth_distance
            and args_cli.start_from_random_drop
            and args_cli.relative_next_hop
        )
        else "_random_drop_safety_smooth_distance"
        if fixed_high_safety and fixed_high_smooth_distance and args_cli.start_from_random_drop
        else
        "_relative_next_random_drop_safety"
        if fixed_high_safety and args_cli.start_from_random_drop and args_cli.relative_next_hop
        else "_random_drop_safety"
        if fixed_high_safety and args_cli.start_from_random_drop
        else
        "_relative_next_random_drop"
        if args_cli.start_from_random_drop and args_cli.relative_next_hop
        else "_random_drop"
        if args_cli.start_from_random_drop
        else
        "_relative_next_smooth_distance"
        if fixed_high_smooth_distance
        else "_relative_next_pair_polish"
        if fixed_high_pair_polish
        else "_relative_next_first_attempt"
        if fixed_high_first_attempt
        else "_relative_next_continuity"
        if fixed_high_continuity
        else "_relative_next_anticipatory"
        if fixed_high_anticipatory
        else "_relative_next_recovery_streak"
        if args_cli.relative_next_hop and fixed_high_streak
        else "_relative_next_accuracy"
        if args_cli.relative_next_hop and fixed_high_soft_accuracy
        else "_relative_next"
        if args_cli.relative_next_hop
        else "_recovery_streak"
        if fixed_high_streak
        else "_soft_accuracy"
        if fixed_high_soft_accuracy
        else "_accuracy"
        if args_cli.accuracy_finetune
        else ""
    )
    EXPERIMENT = (
        f"quadhopper_planner_random_two_hop_{route_version}_"
        f"{args_cli.distance_stage}{accuracy_suffix}_"
        f"turn_{round(args_cli.max_turn_angle):03d}_"
        f"tol_{round(100 * args_cli.target_tolerance):02d}_"
        f"spd_{round(100 * args_cli.anticipatory_speed):03d}_"
        f"tilt_{round(args_cli.anticipatory_tilt_deg):02d}_"
        f"arw_{round(100 * args_cli.anticipatory_reward_scale):03d}_"
        f"{'_ptilt_' + str(round(args_cli.takeoff_tilt_deg)).zfill(2) + '_' + str(round(100 * args_cli.takeoff_tilt_reward_scale)).zfill(3) if args_cli.phase_tilt_finetune else ''}"
        f"{'_impulse' if args_cli.takeoff_impulse_finetune else ''}"
        f"{'_ptakeoff_' + str(round(100 * args_cli.takeoff_velocity_target)).zfill(3) + '_' + str(round(100 * args_cli.takeoff_velocity_reward_scale)).zfill(3) if args_cli.phase_takeoff_finetune else ''}"
        f"{'_airsetup_' + str(round(args_cli.airborne_setup_tilt_deg)).zfill(2) + '_' + str(round(100 * args_cli.airborne_setup_reward_scale)).zfill(3) if args_cli.airborne_setup_finetune else ''}"
        f"{'_setup_' + str(round(100 * args_cli.setup_velocity_target)).zfill(3) + '_' + str(round(args_cli.setup_velocity_reward_scale)).zfill(3) if args_cli.setup_hop_finetune else ''}"
        f"lcg_{round(100 * args_cli.landing_correction_gain):03d}"
        f"{'_tb_' + str(round(100 * args_cli.takeoff_xy_bias_gain)).zfill(3) + '_' + str(round(100 * args_cli.takeoff_xy_bias_max)).zfill(3) if args_cli.takeoff_xy_bias_gain != 0.0 else ''}"
        f"{'_tbc_' + str(round(args_cli.takeoff_xy_bias_curriculum_iterations)).zfill(3) if args_cli.takeoff_xy_bias_gain != 0.0 and args_cli.takeoff_xy_bias_curriculum_iterations > 0.0 else ''}"
        f"{'_hadapt_' + str(round(100 * args_cli.adaptive_height_min)).zfill(3) + '_' + str(round(100 * args_cli.adaptive_height_max)).zfill(3) if args_cli.adaptive_apex_height else ''}"
        f"{'_pvelobs' if args_cli.planner_velocity_observation else ''}"
        f"{'_attw_' + str(round(100 * args_cli.safety_attitude_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_attitude_scale != 1.0 else ''}"
        f"{'_angw_' + str(round(100 * args_cli.safety_angular_vel_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_angular_vel_scale != 1.0 else ''}"
        f"{'_xyw_' + str(round(100 * args_cli.safety_xy_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_xy_scale != 1.0 else ''}"
        f"{'_xyd_' + str(round(100 * args_cli.safety_xy_dense_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_xy_dense_scale != 1.0 else ''}"
        f"{'_yaww_' + str(round(100 * args_cli.safety_yaw_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_yaw_scale != 1.0 else ''}"
        f"{'_heightw_' + str(round(100 * args_cli.safety_height_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.safety_height_scale != 1.0 else ''}"
        f"{'_damp_' + str(round(100 * args_cli.damping_airborne_angvel_start)).zfill(3) + '_' + str(round(100 * args_cli.damping_airborne_angvel_scale)).zfill(3) if args_cli.attitude_damping_polish else ''}"
        f"{'_freeair' if args_cli.free_airborne_roll_pitch else ''}"
        f"{'_blend_' + str(round(100 * args_cli.planner_reference_blend)).zfill(3) if args_cli.planner_reference_blend is not None else ''}"
        f"{'_landingxy' if args_cli.direct_landing_reference else ''}"
        f"{'_pairrestart' if args_cli.pair_restart_queue else ''}"
        f"{'_twohopepisode' if args_cli.two_hop_episode else ''}"
        f"{'_hg' + str(round(100 * args_cli.static_apex_min_valid_apex)).zfill(3) if args_cli.static_apex_finetune else ''}"
        f"{'_la' + str(round(args_cli.static_apex_low_apex_penalty)).zfill(3) if args_cli.static_apex_finetune and args_cli.static_apex_low_apex_penalty > 0.0 else ''}"
        f"{'_air_' + str(round(100 * args_cli.airborne_stability_scale)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.airborne_stability_scale > 0.0 else ''}"
        f"{'_yawair' if (args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.airborne_stability_scale > 0.0 else ''}"
        f"{'_spring' if (args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and args_cli.airborne_stability_scale > 0.0 else ''}"
        f"{'_mass_' + str(round(100 * args_cli.safety_mass_min)).zfill(3) + '_' + str(round(100 * args_cli.safety_mass_max)).zfill(3) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and (args_cli.safety_mass_min != 0.95 or args_cli.safety_mass_max != 1.05) else ''}"
        f"{'_thr_' + str(round(100 * args_cli.thrust_rand_min)).zfill(3) + '_' + str(round(100 * args_cli.thrust_rand_max)).zfill(3) if (args_cli.thrust_rand_min != 1.0 or args_cli.thrust_rand_max != 1.0) else ''}"
        f"{'_tshape_' + str(round(100 * args_cli.thrust_shape_rand_min)).zfill(3) + '_' + str(round(100 * args_cli.thrust_shape_rand_max)).zfill(3) if (args_cli.thrust_shape_rand_min != 1.0 or args_cli.thrust_shape_rand_max != 1.0) else ''}"
        f"{'_tm_' + str(round(1000 * args_cli.motor_time_constant_min)).zfill(3) + '_' + str(round(1000 * args_cli.motor_time_constant_max)).zfill(3) if args_cli.motor_time_constant_min is not None else ''}"
        f"{'_lax_' + str(round(args_cli.low_apex_event_penalty)).zfill(3) if args_cli.low_apex_event_penalty > 0.0 else ''}"
        f"{'_lenapex' if args_cli.lenient_apex_hit else ''}"
        f"{'_longcur_' + str(round(100 * args_cli.long_curriculum_radius_max)).zfill(3) + '_' + str(round(args_cli.distance_curriculum_iterations)).zfill(3) if args_cli.long_hop_curriculum else ''}"
        f"{'_transition' if args_cli.two_hop_transition_curriculum else ''}"
        f"{'_frozenactor' if args_cli.freeze_baseline_actor else ''}"
        f"{'_tds_' + str(round(100 * args_cli.tilt_death_sq_threshold)).zfill(3) if args_cli.tilt_death_sq_threshold != 0.5 else ''}"
        f"{'_fxy_' + str(round(args_cli.terminate_far_xy)).zfill(3) if args_cli.terminate_far_xy != 2.0 else ''}"
        f"{'_av_' + str(round(args_cli.terminate_angvel_norm)).zfill(3) if args_cli.terminate_angvel_norm != 18.0 else ''}"
        f"{'_pert_' + str(round(args_cli.initial_roll_pitch_perturb_deg)).zfill(2) + '_' + str(round(100 * args_cli.airborne_torque_disturbance)).zfill(2) if (args_cli.safety_finetune or args_cli.robust_finetune or args_cli.static_apex_finetune or args_cli.trajectory_landing_finetune or args_cli.baseline_warmstart_finetune) and (args_cli.initial_roll_pitch_perturb_deg > 0.0 or args_cli.airborne_torque_disturbance > 0.0) else ''}"
        f"_{height_label}"
    )


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs" / "rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = None
    checkpoint_data = None
    input_width = None
    source_uses_relative_next = False
    curriculum_iteration_offset = 0.0
    if args_cli.checkpoint:
        source_checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        checkpoint_data = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        input_width = infer_checkpoint_input_width(checkpoint_data, source_checkpoint)
        source_uses_relative_next = any(
            "relative_next" in part for part in source_checkpoint.parts
        ) or any(
            "rel_static" in part for part in source_checkpoint.parts
        ) or any(
            "quadhopper_planner_random_two_hop_v31" in part
            for part in source_checkpoint.parts
        )
        if input_width == 42 and "quadhopper_planner_circular_v4" in source_checkpoint.parts:
            # The v4 policy has already completed the height/path curriculum.
            # Preserve that behavior while learning the new joint-horizon and
            # touchdown-precision objective.
            curriculum_iteration_offset = 400.0
        elif input_width == 42 and (
            "quadhopper_planner_circular_v5" in source_checkpoint.parts
            or "quadhopper_planner_circular_v6" in source_checkpoint.parts
            or "quadhopper_planner_circular_v7" in source_checkpoint.parts
            or "quadhopper_planner_circular_v8" in source_checkpoint.parts
            or "quadhopper_planner_circular_v9" in source_checkpoint.parts
            or "quadhopper_planner_circular_v10" in source_checkpoint.parts
            or "quadhopper_planner_circular_v11_variable_height" in source_checkpoint.parts
        ):
            # Every v5 run starts from a v4 policy which had already reached
            # the full-planner phase. Never regress to the stationary-apex
            # curriculum when resuming a short v5 fine-tuning run.
            curriculum_iteration_offset = max(400.0, float(checkpoint_data.get("iter", 0)))

    env_cfg = (
        PlannerRandomTwoHopEnvCfg()
        if args_cli.route == "random_two_hop"
        else PlannerCircularEnvCfg()
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    target_obs_dim = 47 if args_cli.planner_velocity_observation else 43
    env_cfg.observation_space = target_obs_dim
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
    env_cfg.initial_roll_pitch_perturb_rad = math.radians(args_cli.initial_roll_pitch_perturb_deg)
    env_cfg.initial_yaw_perturb_rad = math.radians(args_cli.initial_yaw_perturb_deg)
    env_cfg.initial_xy_velocity_perturb_mps = args_cli.initial_xy_velocity_perturb
    env_cfg.initial_z_velocity_perturb_mps = args_cli.initial_z_velocity_perturb
    env_cfg.initial_angular_velocity_perturb_radps = args_cli.initial_angular_velocity_perturb
    env_cfg.airborne_force_disturbance_n = args_cli.airborne_force_disturbance
    env_cfg.airborne_torque_disturbance_nm = args_cli.airborne_torque_disturbance
    env_cfg.disturbance_resample_time_s = args_cli.disturbance_resample_time
    env_cfg.randomize_thrust_curve = (
        args_cli.thrust_rand_min != 1.0
        or args_cli.thrust_rand_max != 1.0
        or args_cli.thrust_shape_rand_min != 1.0
        or args_cli.thrust_shape_rand_max != 1.0
    )
    env_cfg.train_thrust_scale_min = args_cli.thrust_rand_min
    env_cfg.train_thrust_scale_max = args_cli.thrust_rand_max
    env_cfg.train_thrust_curve_shape_min = args_cli.thrust_shape_rand_min
    env_cfg.train_thrust_curve_shape_max = args_cli.thrust_shape_rand_max
    env_cfg.randomize_motor_time_constant = args_cli.motor_time_constant_min is not None
    if env_cfg.randomize_motor_time_constant:
        env_cfg.train_motor_time_constant_min = args_cli.motor_time_constant_min
        env_cfg.train_motor_time_constant_max = args_cli.motor_time_constant_max
    if args_cli.target_tolerance <= 0.0:
        raise ValueError("--target_tolerance must be positive")
    env_cfg.target_tolerance = args_cli.target_tolerance
    # The low stage is a fixed-height specialist: command 0.70 m from the
    # first rollout.  Variable/descending-height curricula are deferred until
    # this fixed target is learned reliably.
    env_cfg.fixed_height_curriculum = False
    env_cfg.height_curriculum_start = 1.30
    env_cfg.height_curriculum_end = args_cli.height_low
    env_cfg.height_curriculum_iterations = args_cli.low_curriculum_iterations
    env_cfg.height_curriculum_iteration_offset = (
        float(checkpoint_data.get("iter", 0))
        if input_width == 43 and EXPERIMENT in source_checkpoint.parts
        else 0.0
    )
    env_cfg.symmetric_height_tracking = True
    env_cfg.require_apex_tolerance_for_hit = True
    env_cfg.force_full_planner = input_width in (42, 43, 47)
    if args_cli.static_apex_finetune:
        env_cfg.force_full_planner = False
    if args_cli.planner_reference_blend is not None:
        env_cfg.force_full_planner = False
        env_cfg.planner_reference_blend = args_cli.planner_reference_blend
    env_cfg.flight_reference_uses_landing_xy = args_cli.direct_landing_reference
    env_cfg.stance_reference_uses_apex_height = (
        args_cli.baseline_compatible_stance_reference
    )
    if args_cli.route == "random_two_hop":
        env_cfg.restart_two_hop_pair = args_cli.pair_restart_queue
        env_cfg.terminate_after_two_hop_pair = args_cli.two_hop_episode
        env_cfg.short_hop_radius_min = args_cli.short_radius_min
        env_cfg.short_hop_radius_max = args_cli.short_radius_max
        env_cfg.long_hop_radius_min = args_cli.long_radius_min
        env_cfg.long_hop_radius_max = args_cli.long_radius_max
        env_cfg.hop_distance = max(
            args_cli.short_radius_max, args_cli.long_radius_max, 0.05
        )
        if args_cli.retry_target_on_miss:
            env_cfg.advance_route_on_miss = False
        env_cfg.planner_landing_compensation_m = args_cli.landing_compensation
        env_cfg.takeoff_xy_bias_gain = args_cli.takeoff_xy_bias_gain
        env_cfg.takeoff_xy_bias_max_m = args_cli.takeoff_xy_bias_max
        env_cfg.takeoff_xy_bias_curriculum_iterations = (
            args_cli.takeoff_xy_bias_curriculum_iterations
        )
        env_cfg.relative_next_hop_observation = args_cli.relative_next_hop
        env_cfg.max_turn_angle_deg = args_cli.max_turn_angle
        env_cfg.planner_landing_xy_velocity_scale = args_cli.landing_velocity_scale
        if args_cli.long_hop_curriculum:
            env_cfg.randomize_route_phase = True
            env_cfg.long_hop_curriculum = True
            env_cfg.distance_curriculum_iterations = (
                args_cli.distance_curriculum_iterations
            )
            env_cfg.long_curriculum_radius_min = (
                args_cli.long_curriculum_radius_min
            )
            env_cfg.long_curriculum_radius_max = (
                args_cli.long_curriculum_radius_max
            )
            env_cfg.long_curriculum_max_turn_angle_deg = (
                args_cli.long_curriculum_turn_deg
            )
            env_cfg.long_hop_stability_only = True
            env_cfg.touchdown_attitude_penalty_scale = -45.0
            env_cfg.tilt_barrier_penalty_scale = -160.0
            env_cfg.tilt_barrier_start = 0.08
            env_cfg.tilt_barrier_ground_height = 1.15
            env_cfg.airborne_tilt_penalty_start = 0.025
            env_cfg.airborne_angvel_xy_penalty_start = 1.0
            env_cfg.airborne_yaw_penalty_start = 0.08
            env_cfg.airborne_yaw_rate_penalty_start = 0.8
            env_cfg.airborne_action_spread_penalty_start = 0.45
            env_cfg.airborne_attitude_penalty_scale = -80.0
            env_cfg.airborne_angvel_xy_penalty_scale = -0.60
            env_cfg.airborne_yaw_penalty_scale = -8.0
            env_cfg.airborne_yaw_rate_penalty_scale = -0.25
            env_cfg.airborne_action_spread_penalty_scale = -10.0
        if args_cli.two_hop_transition_curriculum:
            # Deployment begins short -> long. The transition must first learn
            # to survive the physical spring rebound. Earlier versions tried
            # to carry forward velocity and pre-tilt toward hop 2; with the
            # corrected low Izz that produced large roll/pitch rates at the
            # first touchdown. Stabilize both phases and leave the desired
            # direction in P_t/P_t1 instead of prescribing touchdown momentum.
            env_cfg.randomize_route_phase = False
            env_cfg.long_hop_stability_only = False
            env_cfg.setup_velocity_projection_target_mps = 0.0
            env_cfg.setup_velocity_projection_width_mps = 0.15
            env_cfg.setup_velocity_projection_reward_scale = 0.0
            env_cfg.setup_lateral_velocity_penalty_scale = 0.0
            env_cfg.setup_touchdown_attitude_deadband_rad = 0.04
            env_cfg.setup_touchdown_attitude_penalty_scale = 0.0
            env_cfg.anticipatory_tilt_rad = 0.0
            env_cfg.anticipation_start_phase = 0.55
            env_cfg.anticipatory_short_hop_only = True
            env_cfg.anticipatory_include_stance = False
            env_cfg.anticipatory_landing_gate_width = 0.15
            env_cfg.anticipatory_attitude_reward_scale = 0.0
            env_cfg.anticipatory_attitude_penalty_scale = 0.0
            env_cfg.prepared_landing_reward_scale = 0.0
            env_cfg.prepared_attitude_tolerance_rad = 0.15
            env_cfg.prepared_velocity_tolerance = 0.35
            env_cfg.pair_hit_reward_scale = 120.0
            env_cfg.streak_progress_reward_scale = 80.0
            env_cfg.termination_penalty_scale = -40.0
        if args_cli.smooth_distance_curriculum:
            if args_cli.distance_curriculum_iterations <= 0.0:
                raise ValueError("--distance_curriculum_iterations must be positive")
            env_cfg.distance_curriculum_iterations = (
                args_cli.distance_curriculum_iterations
            )
            if args_cli.static_apex_finetune:
                env_cfg.curriculum_short_radius_min = 0.0
                env_cfg.curriculum_short_radius_max = 0.05
                env_cfg.curriculum_long_radius_min = 0.05
                env_cfg.curriculum_long_radius_max = 0.15
                env_cfg.curriculum_max_turn_angle_deg = min(30.0, args_cli.max_turn_angle)
            elif args_cli.baseline_warmstart_finetune:
                # The corrected-physics baseline already passes 0--0.02 m.
                # Expand continuously from that verified support instead of
                # inheriting the random-task defaults (0.30--0.50 m), which
                # would destroy the intended warm start on the first rollout.
                env_cfg.curriculum_short_radius_min = 0.0
                env_cfg.curriculum_short_radius_max = 0.02
                env_cfg.curriculum_long_radius_min = 0.0
                env_cfg.curriculum_long_radius_max = 0.02
                env_cfg.curriculum_max_turn_angle_deg = min(15.0, args_cli.max_turn_angle)
    if args_cli.baseline_warmstart_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--baseline_warmstart_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.randomize_dynamics = False
        env_cfg.randomize_action_delay = False
        env_cfg.randomize_route_phase = False
        env_cfg.terminate_on_target_miss = False
        env_cfg.force_full_planner = False
        env_cfg.stance_reference_uses_apex_height = True
        if args_cli.planner_reference_blend is None:
            env_cfg.planner_reference_blend = 0.0

        xy_scale = args_cli.safety_xy_scale
        xy_dense_scale = xy_scale * args_cli.safety_xy_dense_scale
        yaw_scale = args_cli.safety_yaw_scale
        height_scale = args_cli.safety_height_scale
        attitude_scale = args_cli.safety_attitude_scale
        angular_vel_scale = args_cli.safety_angular_vel_scale

        # Preserve the stable baseline's dense jump contract. Add only small,
        # non-gated touchdown/trajectory hints so warm-start PPO does not
        # destroy the learned vertical rhythm before XY has signal.
        env_cfg.require_apex_tolerance_for_hit = False
        env_cfg.gate_touchdown_rewards_by_apex = False
        env_cfg.gate_dense_xy_rewards_by_apex = False
        env_cfg.distance_to_xy_reward_scale = 6.0 * xy_dense_scale
        env_cfg.xy_progress_reward_scale = 4.0 * xy_dense_scale
        env_cfg.goal_bonus_scale = 2.0 * xy_scale
        env_cfg.planner_position_reward_scale = 4.0 * xy_dense_scale
        env_cfg.planner_velocity_reward_scale = 1.5 * xy_dense_scale
        env_cfg.planner_xy_reward_scale = 6.0 * xy_dense_scale
        env_cfg.xy_error_penalty_scale = -0.10 * xy_dense_scale
        env_cfg.lateral_vel_penalty_scale = -0.25 * xy_dense_scale
        env_cfg.projected_landing_reward_scale = 4.0 * xy_dense_scale
        env_cfg.projected_landing_penalty_scale = -3.0 * xy_dense_scale

        env_cfg.target_hit_reward_scale = 18.0 * xy_scale
        env_cfg.target_miss_penalty_scale = 0.0
        env_cfg.landing_error_penalty_scale = -12.0 * xy_scale
        env_cfg.landing_precision_reward_scale = 10.0 * xy_scale
        env_cfg.landing_precision_width = 0.12
        env_cfg.require_prepared_landing_for_hit = False
        env_cfg.prepared_landing_reward_scale = 8.0 * xy_scale
        env_cfg.prepared_attitude_tolerance_rad = 0.18
        env_cfg.prepared_velocity_tolerance = 0.55

        env_cfg.apex_tolerance = 0.20
        env_cfg.apex_event_reward_scale = 8.0 * height_scale
        env_cfg.apex_error_penalty_scale = -18.0 * height_scale
        env_cfg.apex_shortfall_penalty_scale = -18.0 * height_scale
        env_cfg.airborne_overshoot_penalty_scale = -8.0 * height_scale
        env_cfg.airborne_overshoot_quadratic_gain = 0.0
        env_cfg.height_progress_reward_scale = 3.0 * height_scale
        env_cfg.low_apex_touchdown_penalty_scale = 0.0
        env_cfg.low_apex_touchdown_event_penalty_scale = 0.0

        env_cfg.touchdown_attitude_penalty_scale = -4.0 * attitude_scale
        env_cfg.attitude_penalty_scale = -2.0 * attitude_scale
        env_cfg.angular_vel_penalty_scale = -0.02 * angular_vel_scale
        env_cfg.yaw_penalty_scale = -0.4 * yaw_scale
        env_cfg.tilt_barrier_penalty_scale = -20.0 * attitude_scale
        env_cfg.tilt_barrier_start = 0.22
        env_cfg.tilt_barrier_ground_height = 1.15
        env_cfg.airborne_attitude_penalty_scale = 0.0
        env_cfg.airborne_angvel_xy_penalty_scale = 0.0
        env_cfg.airborne_yaw_penalty_scale = 0.0
        env_cfg.airborne_yaw_rate_penalty_scale = 0.0
        env_cfg.airborne_action_spread_penalty_scale = 0.0
        env_cfg.airborne_spring_pos_penalty_scale = 0.0
        env_cfg.airborne_spring_vel_penalty_scale = 0.0

        env_cfg.pair_hit_reward_scale = 0.0
        env_cfg.streak_progress_reward_scale = 0.0
        env_cfg.circle_complete_reward_scale = 0.0
        env_cfg.termination_penalty_scale = -10.0

        if args_cli.two_hop_episode:
            # A fixed two-touchdown horizon otherwise admits a low, fast-hop
            # shortcut: the policy can collect landing rewards and reset the
            # episode without executing the commanded 1 m jump. Count hits
            # and landing quality only near the requested apex and explicitly
            # reward completing both valid hops.
            env_cfg.require_apex_tolerance_for_hit = True
            env_cfg.apex_tolerance = 0.20
            env_cfg.gate_touchdown_rewards_by_apex = True
            env_cfg.apex_event_reward_scale = 50.0 * height_scale
            env_cfg.apex_error_penalty_scale = -120.0 * height_scale
            env_cfg.apex_shortfall_penalty_scale = -120.0 * height_scale
            env_cfg.height_progress_reward_scale = 30.0 * height_scale
            env_cfg.low_apex_touchdown_penalty_scale = -120.0 * height_scale
            env_cfg.low_apex_touchdown_event_penalty_scale = -80.0 * height_scale
            env_cfg.pair_hit_reward_scale = 100.0 * xy_scale
        env_cfg.final_reward_clip_abs = 0.0
        env_cfg.max_touchdown_error_for_stats = 1.0
        env_cfg.max_touchdown_velocity_error_for_stats = 4.0
        env_cfg.terminate_far_xy_distance = 2.0
        env_cfg.terminate_angvel_norm = 18.0
    if args_cli.expand_variable_height:
        if input_width != 43:
            raise ValueError("--expand_variable_height requires a trained 43-D planner checkpoint")
        env_cfg.apex_event_reward_scale = 150.0
        env_cfg.apex_error_penalty_scale = -350.0
        env_cfg.apex_shortfall_penalty_scale = -200.0
        env_cfg.airborne_overshoot_penalty_scale = -200.0
        env_cfg.height_progress_reward_scale = 80.0
    if args_cli.accuracy_finetune:
        env_cfg.terminate_on_target_miss = True
        env_cfg.target_miss_penalty_scale = -150.0
        env_cfg.landing_error_penalty_scale = -250.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.04
        env_cfg.projected_landing_penalty_scale = -80.0
        if args_cli.route == "random_two_hop" and args_cli.height_stage == "high":
            # Full-distance fixed-height transfer is already near the target
            # basin.  A terminal miss and very large penalties destroyed that
            # policy within 50 updates, so use a conservative precision stage
            # that preserves recovery data while tightening XY guidance.
            env_cfg.terminate_on_target_miss = False
            env_cfg.target_miss_penalty_scale = -90.0
            env_cfg.landing_error_penalty_scale = -160.0
            env_cfg.landing_precision_reward_scale = 100.0
            env_cfg.landing_precision_width = 0.05
            env_cfg.projected_landing_penalty_scale = -50.0
        if args_cli.height_stage in ("low", "high"):
            # Fixed-height specialists already have good height control, so
            # this stage can devote most of its update budget to XY accuracy.
            env_cfg.apex_event_reward_scale = 50.0
            env_cfg.apex_error_penalty_scale = -120.0
            env_cfg.height_progress_reward_scale = 30.0
        else:
            # Alternating commands must first learn that 0.70 and 0.80 m are
            # distinct tasks.  Strong symmetric tracking prevents the policy
            # from compromising at one average apex while landing accurately.
            env_cfg.apex_event_reward_scale = 150.0
            env_cfg.apex_error_penalty_scale = -350.0
            env_cfg.apex_shortfall_penalty_scale = -200.0
            env_cfg.airborne_overshoot_penalty_scale = -200.0
            env_cfg.height_progress_reward_scale = 80.0
    if args_cli.streak_finetune:
        if args_cli.route != "random_two_hop" or args_cli.height_stage != "high":
            raise ValueError(
                "--streak_finetune currently requires random_two_hop with fixed high height"
            )
        # A terminal miss made the rollout distribution collapse onto fresh
        # failures and removed the post-hit states needed for continuity.
        # Keep recovery trajectories: a miss clears the streak in the event
        # logic, while the policy continues to observe later flight states.
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 160.0
        env_cfg.target_miss_penalty_scale = -100.0
        env_cfg.landing_error_penalty_scale = -180.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.045
        env_cfg.projected_landing_penalty_scale = -60.0
        env_cfg.streak_progress_reward_scale = 400.0
        env_cfg.circle_complete_reward_scale = 4000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.anticipatory_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--anticipatory_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.anticipatory_velocity_blend = 1.0
        env_cfg.anticipatory_speed_max = args_cli.anticipatory_speed
        env_cfg.anticipatory_tilt_rad = math.radians(args_cli.anticipatory_tilt_deg)
        env_cfg.anticipation_start_phase = 0.50
        if args_cli.anticipatory_reward_scale < 0.0:
            raise ValueError("--anticipatory_reward_scale must be non-negative")
        if args_cli.landing_correction_gain < 0.0:
            raise ValueError("--landing_correction_gain must be non-negative")
        anticipation_reward_scale = args_cli.anticipatory_reward_scale
        env_cfg.anticipatory_attitude_reward_scale = (
            40.0 * anticipation_reward_scale
        )
        env_cfg.anticipatory_attitude_penalty_scale = (
            -60.0 * anticipation_reward_scale
        )
        env_cfg.anticipatory_velocity_penalty_scale = (
            -80.0 * anticipation_reward_scale
        )
        env_cfg.prepared_landing_reward_scale = 300.0 * anticipation_reward_scale
        env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
        env_cfg.prepared_velocity_tolerance = max(
            0.35, args_cli.anticipatory_speed + 0.25
        )
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        env_cfg.attitude_penalty_scale = -12.0
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 160.0
        env_cfg.target_miss_penalty_scale = -100.0
        env_cfg.landing_error_penalty_scale = -180.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.05
        env_cfg.projected_landing_penalty_scale = -60.0
        env_cfg.streak_progress_reward_scale = 250.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.continuity_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--continuity_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        if args_cli.tolerance_curriculum_start < args_cli.target_tolerance:
            raise ValueError(
                "--tolerance_curriculum_start must be >= --target_tolerance"
            )
        if args_cli.tolerance_curriculum_iterations <= 0.0:
            raise ValueError("--tolerance_curriculum_iterations must be positive")
        # Keep the current touchdown settled and accurate.  P_(t+1) affects
        # the late-flight attitude reference and the pair-success objective,
        # but does not demand a destabilizing horizontal impact velocity.
        env_cfg.planner_landing_xy_velocity_scale = 0.0
        env_cfg.anticipatory_velocity_blend = 0.0
        env_cfg.anticipatory_tilt_rad = math.radians(args_cli.anticipatory_tilt_deg)
        env_cfg.anticipation_start_phase = 0.60
        env_cfg.anticipatory_attitude_reward_scale = 10.0
        env_cfg.anticipatory_attitude_penalty_scale = -15.0
        env_cfg.anticipatory_velocity_penalty_scale = 0.0
        env_cfg.prepared_landing_reward_scale = 60.0
        env_cfg.pair_hit_reward_scale = 300.0
        env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
        env_cfg.prepared_velocity_tolerance = 0.55
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        env_cfg.target_tolerance_curriculum_start = (
            args_cli.tolerance_curriculum_start
        )
        env_cfg.target_tolerance_curriculum_iterations = (
            args_cli.tolerance_curriculum_iterations
        )
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 180.0
        env_cfg.target_miss_penalty_scale = -120.0
        env_cfg.landing_error_penalty_scale = -220.0
        env_cfg.landing_precision_reward_scale = 140.0
        env_cfg.landing_precision_width = 0.045
        env_cfg.projected_landing_penalty_scale = -75.0
        env_cfg.streak_progress_reward_scale = 1200.0
        env_cfg.circle_complete_reward_scale = 5000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.first_attempt_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--first_attempt_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        # The random-route environment advances after every touchdown.  These
        # terms therefore optimize first-attempt accuracy and two-hop pairs;
        # there is no recovery distribution left to exploit.
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 250.0
        env_cfg.target_miss_penalty_scale = -180.0
        env_cfg.landing_error_penalty_scale = -250.0
        env_cfg.landing_precision_reward_scale = 150.0
        env_cfg.landing_precision_width = 0.05
        env_cfg.projected_landing_penalty_scale = -80.0
        env_cfg.pair_hit_reward_scale = 400.0 if args_cli.pair_polish else 100.0
        env_cfg.streak_progress_reward_scale = (
            1200.0 if args_cli.pair_polish else 300.0
        )
        env_cfg.circle_complete_reward_scale = 5000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.safety_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
            or not args_cli.start_from_random_drop
        ):
            raise ValueError(
                "--safety_finetune requires fixed-high random_two_hop, --relative_next_hop, and --start_from_random_drop"
            )
        env_cfg.randomize_dynamics = False
        env_cfg.randomize_action_delay = False
        env_cfg.train_mass_multi_min = args_cli.safety_mass_min
        env_cfg.train_mass_multi_max = args_cli.safety_mass_max
        env_cfg.terminate_on_target_miss = False
        xy_scale = args_cli.safety_xy_scale
        xy_dense_scale = xy_scale * args_cli.safety_xy_dense_scale
        yaw_scale = args_cli.safety_yaw_scale
        height_scale = args_cli.safety_height_scale
        env_cfg.require_apex_tolerance_for_hit = False
        env_cfg.apex_tolerance = 0.10
        env_cfg.target_hit_reward_scale = 100.0 * xy_scale
        env_cfg.target_miss_penalty_scale = -60.0 * xy_scale
        env_cfg.planner_xy_reward_scale = 25.0 * xy_dense_scale
        env_cfg.landing_error_penalty_scale = -110.0 * xy_scale
        env_cfg.landing_precision_reward_scale = 60.0 * xy_scale
        env_cfg.landing_precision_width = 0.07
        env_cfg.require_prepared_landing_for_hit = False
        env_cfg.prepared_landing_reward_scale = 80.0
        env_cfg.prepared_attitude_tolerance_rad = 0.15
        env_cfg.prepared_velocity_tolerance = 0.40
        attitude_scale = args_cli.safety_attitude_scale
        angular_vel_scale = args_cli.safety_angular_vel_scale
        env_cfg.touchdown_attitude_penalty_scale = -160.0 * attitude_scale
        env_cfg.projected_landing_reward_scale = 25.0 * xy_dense_scale
        env_cfg.projected_landing_penalty_scale = -25.0 * xy_dense_scale
        env_cfg.pair_hit_reward_scale = 0.0
        env_cfg.streak_progress_reward_scale = 25.0
        env_cfg.circle_complete_reward_scale = 1000.0
        env_cfg.apex_event_reward_scale = 40.0
        env_cfg.apex_error_penalty_scale = -220.0 * height_scale
        env_cfg.airborne_overshoot_penalty_scale = -300.0 * height_scale
        env_cfg.airborne_overshoot_quadratic_gain = 4.0
        env_cfg.height_progress_reward_scale = 20.0 * height_scale
        env_cfg.termination_penalty_scale = -100.0
        env_cfg.yaw_penalty_scale = -10.0 * yaw_scale
        env_cfg.xy_error_penalty_scale = -1.5 * xy_dense_scale
        env_cfg.attitude_penalty_scale = -70.0 * attitude_scale
        env_cfg.tilt_barrier_penalty_scale = -1200.0 * attitude_scale
        env_cfg.tilt_barrier_start = 0.06
        env_cfg.tilt_barrier_ground_height = 1.10
        env_cfg.angular_vel_penalty_scale = -1.4 * angular_vel_scale
        airborne_stability_scale = args_cli.airborne_stability_scale
        env_cfg.airborne_tilt_penalty_start = 0.025
        env_cfg.airborne_angvel_xy_penalty_start = 1.4
        env_cfg.airborne_action_spread_penalty_start = 0.45
        env_cfg.airborne_attitude_penalty_scale = -1200.0 * airborne_stability_scale
        env_cfg.airborne_angvel_xy_penalty_scale = -3.0 * airborne_stability_scale
        env_cfg.airborne_action_spread_penalty_scale = -30.0 * airborne_stability_scale
        env_cfg.lateral_vel_penalty_scale = -18.0 * xy_dense_scale
        env_cfg.flight_power_penalty_scale = -1.6
        env_cfg.action_rate_reward_scale = -1.5
    if args_cli.robust_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--robust_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.randomize_dynamics = False
        env_cfg.randomize_action_delay = False
        env_cfg.train_mass_multi_min = args_cli.safety_mass_min
        env_cfg.train_mass_multi_max = args_cli.safety_mass_max
        env_cfg.terminate_on_target_miss = False
        xy_scale = args_cli.safety_xy_scale
        xy_dense_scale = xy_scale * args_cli.safety_xy_dense_scale
        yaw_scale = args_cli.safety_yaw_scale
        height_scale = args_cli.safety_height_scale
        attitude_scale = args_cli.safety_attitude_scale
        angular_vel_scale = args_cli.safety_angular_vel_scale

        env_cfg.require_apex_tolerance_for_hit = False
        env_cfg.apex_tolerance = 0.12
        env_cfg.target_hit_reward_scale = 110.0 * xy_scale
        env_cfg.target_miss_penalty_scale = -45.0 * xy_scale
        env_cfg.planner_xy_reward_scale = 14.0 * xy_dense_scale
        env_cfg.xy_error_penalty_scale = -0.8 * xy_dense_scale
        env_cfg.lateral_vel_penalty_scale = -8.0 * xy_dense_scale
        env_cfg.landing_error_penalty_scale = -85.0 * xy_scale
        env_cfg.landing_precision_reward_scale = 55.0 * xy_scale
        env_cfg.landing_precision_width = 0.08
        env_cfg.projected_landing_reward_scale = 14.0 * xy_dense_scale
        env_cfg.projected_landing_penalty_scale = -12.0 * xy_dense_scale

        env_cfg.require_prepared_landing_for_hit = False
        env_cfg.prepared_landing_reward_scale = 65.0
        env_cfg.prepared_attitude_tolerance_rad = 0.14
        env_cfg.prepared_velocity_tolerance = 0.38
        env_cfg.touchdown_attitude_penalty_scale = -90.0 * attitude_scale
        env_cfg.attitude_penalty_scale = -32.0 * attitude_scale
        env_cfg.angular_vel_penalty_scale = -0.25 * angular_vel_scale
        env_cfg.yaw_penalty_scale = -6.0 * yaw_scale

        env_cfg.apex_event_reward_scale = 42.0
        env_cfg.apex_error_penalty_scale = -130.0 * height_scale
        env_cfg.airborne_overshoot_penalty_scale = -180.0 * height_scale
        env_cfg.airborne_overshoot_quadratic_gain = 1.5
        env_cfg.height_progress_reward_scale = 20.0 * height_scale

        env_cfg.tilt_barrier_penalty_scale = -280.0 * attitude_scale
        env_cfg.tilt_barrier_start = 0.10
        env_cfg.tilt_barrier_ground_height = 1.15
        airborne_stability_scale = args_cli.airborne_stability_scale
        env_cfg.airborne_tilt_penalty_start = 0.030
        env_cfg.airborne_angvel_xy_penalty_start = 1.6
        env_cfg.airborne_yaw_penalty_start = 0.06
        env_cfg.airborne_yaw_rate_penalty_start = 0.6
        env_cfg.airborne_action_spread_penalty_start = 0.50
        env_cfg.airborne_spring_pos_penalty_start = 0.012
        env_cfg.airborne_spring_vel_penalty_start = 0.35
        env_cfg.airborne_attitude_penalty_scale = -180.0 * airborne_stability_scale
        env_cfg.airborne_angvel_xy_penalty_scale = -0.8 * airborne_stability_scale
        env_cfg.airborne_yaw_penalty_scale = -45.0 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_yaw_rate_penalty_scale = -0.8 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_action_spread_penalty_scale = -25.0 * airborne_stability_scale
        env_cfg.airborne_spring_pos_penalty_scale = -60.0 * airborne_stability_scale
        env_cfg.airborne_spring_vel_penalty_scale = -0.8 * airborne_stability_scale

        env_cfg.pair_hit_reward_scale = 0.0
        env_cfg.streak_progress_reward_scale = 18.0
        env_cfg.circle_complete_reward_scale = 800.0
        env_cfg.termination_penalty_scale = -15.0
        env_cfg.flight_power_penalty_scale = -1.4
        env_cfg.action_rate_reward_scale = -1.0
        env_cfg.final_reward_clip_abs = 80.0
        env_cfg.max_touchdown_error_for_stats = 1.0
        env_cfg.max_touchdown_velocity_error_for_stats = 4.0
        env_cfg.terminate_far_xy_distance = 2.0
        env_cfg.terminate_angvel_norm = 18.0
    if args_cli.trajectory_landing_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--trajectory_landing_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.randomize_dynamics = False
        env_cfg.randomize_action_delay = False
        env_cfg.train_mass_multi_min = args_cli.safety_mass_min
        env_cfg.train_mass_multi_max = args_cli.safety_mass_max
        env_cfg.terminate_on_target_miss = False
        env_cfg.force_full_planner = False
        env_cfg.stance_reference_uses_apex_height = True
        if args_cli.planner_reference_blend is None:
            env_cfg.planner_reference_blend = 0.0
        env_cfg.curriculum_static_apex_iterations = 40.0
        env_cfg.curriculum_full_planner_iterations = 160.0

        xy_scale = args_cli.safety_xy_scale
        xy_dense_scale = xy_scale * args_cli.safety_xy_dense_scale
        yaw_scale = args_cli.safety_yaw_scale
        height_scale = args_cli.safety_height_scale
        attitude_scale = args_cli.safety_attitude_scale
        angular_vel_scale = args_cli.safety_angular_vel_scale
        airborne_stability_scale = args_cli.airborne_stability_scale

        # This mode deliberately restores dense trajectory credit. Height is a
        # soft feasibility signal, not a gate that suppresses XY learning.
        env_cfg.require_apex_tolerance_for_hit = False
        env_cfg.gate_touchdown_rewards_by_apex = False
        env_cfg.gate_dense_xy_rewards_by_apex = False
        env_cfg.distance_to_xy_reward_scale = 12.0 * xy_dense_scale
        env_cfg.xy_progress_reward_scale = 18.0 * xy_dense_scale
        env_cfg.goal_bonus_scale = 6.0 * xy_scale
        env_cfg.planner_position_reward_scale = 36.0 * xy_dense_scale
        env_cfg.planner_velocity_reward_scale = 18.0 * xy_dense_scale
        env_cfg.planner_xy_reward_scale = 32.0 * xy_dense_scale
        env_cfg.xy_error_penalty_scale = -1.1 * xy_dense_scale
        env_cfg.lateral_vel_penalty_scale = -4.5 * xy_dense_scale
        env_cfg.projected_landing_reward_scale = 18.0 * xy_dense_scale
        env_cfg.projected_landing_penalty_scale = -16.0 * xy_dense_scale

        env_cfg.target_hit_reward_scale = 130.0 * xy_scale
        env_cfg.target_miss_penalty_scale = -55.0 * xy_scale
        env_cfg.landing_error_penalty_scale = -120.0 * xy_scale
        env_cfg.landing_precision_reward_scale = 75.0 * xy_scale
        env_cfg.landing_precision_width = 0.075
        env_cfg.require_prepared_landing_for_hit = False
        env_cfg.prepared_landing_reward_scale = 45.0 * xy_scale
        env_cfg.prepared_attitude_tolerance_rad = 0.16
        env_cfg.prepared_velocity_tolerance = 0.48

        env_cfg.apex_tolerance = 0.16
        env_cfg.apex_event_reward_scale = 32.0 * height_scale
        env_cfg.apex_error_penalty_scale = -80.0 * height_scale
        env_cfg.apex_shortfall_penalty_scale = -70.0 * height_scale
        env_cfg.airborne_overshoot_penalty_scale = -80.0 * height_scale
        env_cfg.airborne_overshoot_quadratic_gain = 0.5
        env_cfg.height_progress_reward_scale = 12.0 * height_scale
        env_cfg.low_apex_touchdown_penalty_scale = (
            -90.0 * height_scale
            if args_cli.static_apex_low_apex_penalty > 0.0
            else 0.0
        )
        env_cfg.low_apex_touchdown_event_penalty_scale = -40.0 * height_scale

        env_cfg.touchdown_attitude_penalty_scale = -35.0 * attitude_scale
        env_cfg.attitude_penalty_scale = -14.0 * attitude_scale
        env_cfg.angular_vel_penalty_scale = -0.12 * angular_vel_scale
        env_cfg.yaw_penalty_scale = -4.5 * yaw_scale
        env_cfg.tilt_barrier_penalty_scale = -90.0 * attitude_scale
        env_cfg.tilt_barrier_start = 0.16
        env_cfg.tilt_barrier_ground_height = 1.10
        env_cfg.airborne_tilt_penalty_start = 0.055
        env_cfg.airborne_angvel_xy_penalty_start = 1.8
        env_cfg.airborne_yaw_penalty_start = 0.10
        env_cfg.airborne_yaw_rate_penalty_start = 0.8
        env_cfg.airborne_action_spread_penalty_start = 0.55
        env_cfg.airborne_spring_pos_penalty_start = 0.015
        env_cfg.airborne_spring_vel_penalty_start = 0.45
        env_cfg.airborne_attitude_penalty_scale = -45.0 * airborne_stability_scale
        env_cfg.airborne_angvel_xy_penalty_scale = -0.25 * airborne_stability_scale
        env_cfg.airborne_yaw_penalty_scale = -12.0 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_yaw_rate_penalty_scale = -0.20 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_action_spread_penalty_scale = -7.0 * airborne_stability_scale
        env_cfg.airborne_spring_pos_penalty_scale = -20.0 * airborne_stability_scale
        env_cfg.airborne_spring_vel_penalty_scale = -0.25 * airborne_stability_scale

        env_cfg.pair_hit_reward_scale = 0.0
        env_cfg.streak_progress_reward_scale = 8.0
        env_cfg.circle_complete_reward_scale = 600.0
        env_cfg.termination_penalty_scale = -12.0
        env_cfg.flight_power_penalty_scale = -0.9
        env_cfg.action_rate_reward_scale = -0.5
        env_cfg.final_reward_clip_abs = 100.0
        env_cfg.max_touchdown_error_for_stats = 1.0
        env_cfg.max_touchdown_velocity_error_for_stats = 4.0
        env_cfg.terminate_far_xy_distance = 2.0
        env_cfg.terminate_angvel_norm = 18.0
    if args_cli.static_apex_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--static_apex_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.randomize_dynamics = True
        env_cfg.randomize_action_delay = True
        env_cfg.train_mass_multi_min = args_cli.safety_mass_min
        env_cfg.train_mass_multi_max = args_cli.safety_mass_max
        env_cfg.terminate_on_target_miss = False

        xy_scale = args_cli.safety_xy_scale
        xy_dense_scale = xy_scale * args_cli.safety_xy_dense_scale
        yaw_scale = args_cli.safety_yaw_scale
        height_scale = args_cli.safety_height_scale
        attitude_scale = args_cli.safety_attitude_scale
        angular_vel_scale = args_cli.safety_angular_vel_scale
        airborne_stability_scale = args_cli.airborne_stability_scale

        env_cfg.require_apex_tolerance_for_hit = True
        env_cfg.minimum_valid_apex = args_cli.static_apex_min_valid_apex
        apex_gate_reference_height = (
            args_cli.adaptive_height_min
            if args_cli.adaptive_apex_height
            else args_cli.height_high
        )
        env_cfg.apex_tolerance = max(
            apex_gate_reference_height - args_cli.static_apex_min_valid_apex, 0.05
        )
        # The inherited stable reward uses dense XY/goal bonuses for a fixed
        # target.  In two-hop fine-tuning those bonuses let the policy earn
        # reward by staying low and sliding toward the waypoint, so keep
        # horizontal credit in the planner touchdown terms instead.
        env_cfg.distance_to_xy_reward_scale = 0.0
        env_cfg.xy_progress_reward_scale = 0.0
        env_cfg.goal_bonus_scale = 0.0
        env_cfg.gate_touchdown_rewards_by_apex = True
        env_cfg.gate_dense_xy_rewards_by_apex = True
        env_cfg.low_apex_touchdown_penalty_scale = (
            -args_cli.static_apex_low_apex_penalty * height_scale
        )
        env_cfg.low_apex_touchdown_event_penalty_scale = -240.0 * height_scale
        env_cfg.curriculum_static_apex_iterations = (
            args_cli.static_apex_curriculum_iterations
        )
        env_cfg.curriculum_full_planner_iterations = (
            args_cli.full_planner_curriculum_iterations
        )
        if args_cli.planner_reference_blend is not None:
            env_cfg.planner_reference_blend = args_cli.planner_reference_blend
        env_cfg.target_hit_reward_scale = 115.0 * xy_scale
        env_cfg.target_miss_penalty_scale = -55.0 * xy_scale
        env_cfg.planner_xy_reward_scale = 8.0 * xy_dense_scale
        env_cfg.xy_error_penalty_scale = -0.45 * xy_dense_scale
        env_cfg.lateral_vel_penalty_scale = -5.0 * xy_dense_scale
        env_cfg.landing_error_penalty_scale = -90.0 * xy_scale
        env_cfg.landing_precision_reward_scale = 58.0 * xy_scale
        env_cfg.landing_precision_width = 0.08
        env_cfg.projected_landing_reward_scale = 6.0 * xy_dense_scale
        env_cfg.projected_landing_penalty_scale = -7.0 * xy_dense_scale

        env_cfg.require_prepared_landing_for_hit = False
        env_cfg.prepared_landing_reward_scale = 55.0
        env_cfg.prepared_attitude_tolerance_rad = 0.13
        env_cfg.prepared_velocity_tolerance = 0.42
        if args_cli.phase_tilt_finetune:
            env_cfg.takeoff_tilt_rad = math.radians(args_cli.takeoff_tilt_deg)
            env_cfg.takeoff_tilt_phase_end = args_cli.takeoff_tilt_phase_end
            env_cfg.takeoff_tilt_reward_scale = (
                35.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.takeoff_tilt_penalty_scale = (
                -8.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
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
                55.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
            )
            env_cfg.takeoff_velocity_penalty_scale = (
                -18.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
            )
            if args_cli.takeoff_tilt_reward_scale > 0.0:
                env_cfg.takeoff_tilt_reward_scale = (
                    15.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
                )
                env_cfg.takeoff_tilt_penalty_scale = (
                    -4.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
                )
        env_cfg.touchdown_attitude_penalty_scale = -70.0 * attitude_scale
        env_cfg.attitude_penalty_scale = -24.0 * attitude_scale
        env_cfg.angular_vel_penalty_scale = -0.18 * angular_vel_scale
        env_cfg.yaw_penalty_scale = -8.0 * yaw_scale

        env_cfg.apex_event_reward_scale = 70.0
        env_cfg.apex_error_penalty_scale = -320.0 * height_scale
        env_cfg.apex_shortfall_penalty_scale = -260.0 * height_scale
        env_cfg.airborne_overshoot_penalty_scale = -220.0 * height_scale
        env_cfg.airborne_overshoot_quadratic_gain = 1.5
        env_cfg.height_progress_reward_scale = 8.0 * height_scale

        env_cfg.tilt_barrier_penalty_scale = -220.0 * attitude_scale
        env_cfg.tilt_barrier_start = 0.12
        env_cfg.tilt_barrier_ground_height = 1.15
        env_cfg.airborne_tilt_penalty_start = 0.035
        env_cfg.airborne_angvel_xy_penalty_start = 1.4
        env_cfg.airborne_yaw_penalty_start = 0.045
        env_cfg.airborne_yaw_rate_penalty_start = 0.5
        env_cfg.airborne_action_spread_penalty_start = 0.45
        env_cfg.airborne_spring_pos_penalty_start = 0.010
        env_cfg.airborne_spring_vel_penalty_start = 0.30
        env_cfg.airborne_attitude_penalty_scale = -120.0 * airborne_stability_scale
        env_cfg.airborne_angvel_xy_penalty_scale = -0.45 * airborne_stability_scale
        env_cfg.airborne_yaw_penalty_scale = -35.0 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_yaw_rate_penalty_scale = -0.5 * yaw_scale * airborne_stability_scale
        env_cfg.airborne_action_spread_penalty_scale = -18.0 * airborne_stability_scale
        env_cfg.airborne_spring_pos_penalty_scale = -45.0 * airborne_stability_scale
        env_cfg.airborne_spring_vel_penalty_scale = -0.5 * airborne_stability_scale

        env_cfg.pair_hit_reward_scale = 0.0
        env_cfg.streak_progress_reward_scale = 16.0
        env_cfg.circle_complete_reward_scale = 800.0
        env_cfg.termination_penalty_scale = -12.0
        env_cfg.flight_power_penalty_scale = -1.2
        env_cfg.action_rate_reward_scale = -0.8
        env_cfg.final_reward_clip_abs = 80.0
        env_cfg.max_touchdown_error_for_stats = 1.0
        env_cfg.max_touchdown_velocity_error_for_stats = 4.0
        env_cfg.terminate_far_xy_distance = args_cli.terminate_far_xy
        env_cfg.terminate_angvel_norm = args_cli.terminate_angvel_norm
        if args_cli.takeoff_impulse_finetune:
            env_cfg.takeoff_tilt_rad = math.radians(args_cli.takeoff_tilt_deg)
            env_cfg.takeoff_tilt_phase_end = args_cli.takeoff_tilt_phase_end
            env_cfg.takeoff_phase_attitude_relax = args_cli.takeoff_phase_attitude_relax
            env_cfg.takeoff_phase_angvel_relax = args_cli.takeoff_phase_angvel_relax
            env_cfg.takeoff_phase_tilt_barrier_relax = (
                args_cli.takeoff_phase_tilt_barrier_relax
            )
            env_cfg.takeoff_velocity_target_mps = args_cli.takeoff_velocity_target
            env_cfg.takeoff_velocity_from_planner = True
            env_cfg.takeoff_velocity_event_reward = True
            env_cfg.takeoff_velocity_reward_scale = (
                150.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
            )
            env_cfg.takeoff_velocity_penalty_scale = (
                -55.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
            )
            env_cfg.anticipatory_tilt_rad = math.radians(args_cli.takeoff_tilt_deg)
            env_cfg.anticipatory_attitude_reward_scale = (
                85.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.anticipatory_attitude_penalty_scale = (
                -18.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.takeoff_tilt_reward_scale = (
                12.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.takeoff_tilt_penalty_scale = (
                -3.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )

            env_cfg.target_hit_reward_scale = 12.0 * xy_scale
            env_cfg.target_miss_penalty_scale = -4.0 * xy_scale
            env_cfg.landing_error_penalty_scale = -14.0 * xy_scale
            env_cfg.landing_precision_reward_scale = 8.0 * xy_scale
            env_cfg.prepared_landing_reward_scale = 0.0
            env_cfg.planner_xy_reward_scale = 2.0 * xy_dense_scale
            env_cfg.xy_error_penalty_scale = -0.12 * xy_dense_scale
            env_cfg.lateral_vel_penalty_scale = -1.5 * xy_dense_scale
            env_cfg.projected_landing_reward_scale = 2.0 * xy_dense_scale
            env_cfg.projected_landing_penalty_scale = -1.8 * xy_dense_scale

            env_cfg.apex_event_reward_scale = 80.0
            env_cfg.apex_error_penalty_scale = -360.0 * height_scale
            env_cfg.apex_shortfall_penalty_scale = -340.0 * height_scale
            env_cfg.airborne_overshoot_penalty_scale = -180.0 * height_scale
            env_cfg.height_progress_reward_scale = 6.0 * height_scale
            env_cfg.low_apex_touchdown_penalty_scale = (
                -300.0 * height_scale
                if args_cli.static_apex_low_apex_penalty > 0.0
                else 0.0
            )
            env_cfg.low_apex_touchdown_event_penalty_scale = -120.0 * height_scale

            env_cfg.touchdown_attitude_penalty_scale = -28.0 * attitude_scale
            env_cfg.attitude_penalty_scale = -12.0 * attitude_scale
            env_cfg.angular_vel_penalty_scale = -0.08 * angular_vel_scale
            env_cfg.yaw_penalty_scale = -4.0 * yaw_scale
            env_cfg.tilt_barrier_penalty_scale = -80.0 * attitude_scale
            env_cfg.airborne_attitude_penalty_scale = -35.0 * airborne_stability_scale
            env_cfg.airborne_angvel_xy_penalty_scale = -0.20 * airborne_stability_scale
            env_cfg.airborne_yaw_penalty_scale = -12.0 * yaw_scale * airborne_stability_scale
            env_cfg.airborne_yaw_rate_penalty_scale = -0.18 * yaw_scale * airborne_stability_scale
            env_cfg.airborne_action_spread_penalty_scale = -8.0 * airborne_stability_scale
            env_cfg.airborne_spring_pos_penalty_scale = -20.0 * airborne_stability_scale
            env_cfg.airborne_spring_vel_penalty_scale = -0.25 * airborne_stability_scale

            env_cfg.streak_progress_reward_scale = 0.0
            env_cfg.circle_complete_reward_scale = 0.0
            env_cfg.flight_power_penalty_scale = -0.8
            env_cfg.action_rate_reward_scale = -0.4
            env_cfg.final_reward_clip_abs = 120.0
        if args_cli.low_apex_event_penalty > 0.0:
            # From-zero fix: the env default has zero penalty for touchdowns below
            # minimum_valid_apex, so the policy settles into flat hops (apex < 0.58)
            # that land precisely but can never satisfy valid_apex -> long-hop hit
            # rate is exactly 0 no matter how small the landing error gets.
            env_cfg.low_apex_touchdown_event_penalty_scale = (
                -args_cli.low_apex_event_penalty
            )
        if args_cli.lenient_apex_hit:
            # From-zero fix: the general setup leaves require_apex_tolerance_for_hit
            # True (hit needs apex within apex_tolerance of the 1 m command).  Long
            # hops fly just outside that window, so they can never hit even at
            # 0.03 m landing error.  The low-apex penalty above keeps the floor.
            env_cfg.require_apex_tolerance_for_hit = False
        if args_cli.tilt_death_sq_threshold != 0.5:
            env_cfg.tilt_death_sq_threshold = args_cli.tilt_death_sq_threshold
        if args_cli.static_apex_height_probe:
            env_cfg.randomize_dynamics = False
            env_cfg.randomize_action_delay = False
            env_cfg.randomize_route_phase = False
            env_cfg.target_hit_reward_scale = 0.0
            env_cfg.target_miss_penalty_scale = 0.0
            env_cfg.planner_position_reward_scale = 0.0
            env_cfg.planner_velocity_reward_scale = 0.0
            env_cfg.planner_xy_reward_scale = 0.0
            env_cfg.xy_error_penalty_scale = 0.0
            env_cfg.xy_progress_reward_scale = 0.0
            env_cfg.lateral_vel_penalty_scale = 0.0
            env_cfg.landing_error_penalty_scale = 0.0
            env_cfg.landing_precision_reward_scale = 0.0
            env_cfg.projected_landing_reward_scale = 0.0
            env_cfg.projected_landing_penalty_scale = 0.0
            env_cfg.prepared_landing_reward_scale = 0.0
            env_cfg.pair_hit_reward_scale = 0.0
            env_cfg.streak_progress_reward_scale = 0.0
            env_cfg.circle_complete_reward_scale = 0.0
            env_cfg.apex_event_reward_scale = 100.0
            env_cfg.apex_error_penalty_scale = -420.0 * height_scale
            env_cfg.apex_shortfall_penalty_scale = -420.0 * height_scale
            env_cfg.height_progress_reward_scale = 12.0 * height_scale
            env_cfg.low_apex_touchdown_penalty_scale = (
                -250.0 * height_scale
                if args_cli.static_apex_low_apex_penalty > 0.0
                else 0.0
            )
            env_cfg.flight_power_penalty_scale = -0.8
            env_cfg.action_rate_reward_scale = -0.4
            env_cfg.final_reward_clip_abs = 120.0
    if args_cli.setup_hop_finetune:
        if args_cli.route != "random_two_hop":
            raise ValueError("--setup_hop_finetune requires --route random_two_hop")
        env_cfg.setup_velocity_projection_target_mps = args_cli.setup_velocity_target
        env_cfg.setup_velocity_projection_width_mps = args_cli.setup_velocity_width
        env_cfg.setup_velocity_projection_reward_scale = (
            args_cli.setup_velocity_reward_scale
        )
        env_cfg.setup_lateral_velocity_penalty_scale = (
            -args_cli.setup_lateral_velocity_penalty
        )
        env_cfg.setup_touchdown_attitude_deadband_rad = (
            args_cli.setup_touchdown_attitude_deadband
        )
        env_cfg.final_touchdown_attitude_penalty_scale = (
            -args_cli.final_touchdown_attitude_penalty
        )
        env_cfg.final_touchdown_velocity_penalty_scale = (
            -args_cli.final_touchdown_velocity_penalty
        )
        env_cfg.gate_touchdown_rewards_by_apex = True
        env_cfg.low_apex_touchdown_penalty_scale = -160.0 * args_cli.safety_height_scale
        env_cfg.low_apex_touchdown_event_penalty_scale = (
            -80.0 * args_cli.safety_height_scale
        )
    if args_cli.airborne_setup_finetune:
        if args_cli.route != "random_two_hop" or not args_cli.relative_next_hop:
            raise ValueError(
                "--airborne_setup_finetune requires random_two_hop and --relative_next_hop"
            )
        setup_scale = args_cli.airborne_setup_reward_scale
        env_cfg.anticipatory_tilt_rad = math.radians(args_cli.airborne_setup_tilt_deg)
        env_cfg.anticipation_start_phase = args_cli.airborne_setup_start_phase
        env_cfg.anticipatory_short_hop_only = True
        env_cfg.anticipatory_include_stance = False
        env_cfg.anticipatory_landing_gate_width = (
            args_cli.airborne_setup_landing_gate_width
        )
        env_cfg.anticipatory_attitude_reward_scale = 28.0 * setup_scale
        env_cfg.anticipatory_attitude_penalty_scale = -20.0 * setup_scale
        env_cfg.anticipatory_velocity_penalty_scale = 0.0
    if args_cli.phase_takeoff_finetune and not args_cli.takeoff_impulse_finetune:
        xy_scale = args_cli.safety_xy_scale
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
        env_cfg.takeoff_long_hop_only = args_cli.takeoff_long_hop_only
        env_cfg.takeoff_velocity_reward_scale = (
            55.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
        )
        env_cfg.takeoff_velocity_penalty_scale = (
            -18.0 * xy_scale * args_cli.takeoff_velocity_reward_scale
        )
        if args_cli.takeoff_tilt_reward_scale > 0.0:
            env_cfg.takeoff_tilt_reward_scale = (
                15.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
            env_cfg.takeoff_tilt_penalty_scale = (
                -4.0 * xy_scale * args_cli.takeoff_tilt_reward_scale
            )
    if args_cli.attitude_damping_polish:
        env_cfg.airborne_angvel_xy_penalty_start = (
            args_cli.damping_airborne_angvel_start
        )
        env_cfg.airborne_angvel_xy_penalty_scale = (
            -args_cli.damping_airborne_angvel_scale
        )
        env_cfg.airborne_action_spread_penalty_scale = (
            -args_cli.damping_action_spread_scale
        )
        env_cfg.action_rate_reward_scale = -args_cli.damping_action_rate_scale
        env_cfg.airborne_yaw_rate_penalty_start = args_cli.damping_yaw_rate_start
        env_cfg.airborne_yaw_rate_penalty_scale = -args_cli.damping_yaw_rate_scale
        if args_cli.free_airborne_roll_pitch:
            env_cfg.attitude_penalty_scale = 0.0
            env_cfg.tilt_barrier_penalty_scale = 0.0
            env_cfg.airborne_attitude_penalty_scale = 0.0
    env_cfg.curriculum_iteration_offset = curriculum_iteration_offset
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    output_name = "planner_random_two_hop" if args_cli.route == "random_two_hop" else "planner_circular"
    env_cfg.csv_log_path = str(PROJECT_DIR / f"outputs/{output_name}/on_quadhopper_sim.csv")
    task_id = (
        "Quadhopper-Planner-Random-Two-Hop-Direct-v0"
        if args_cli.route == "random_two_hop"
        else "Quadhopper-Planner-Circular-Direct-v0"
    )
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg))

    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    if args_cli.direct_variable_height:
        # The 37-D stable policy has no circular or height-conditioned behavior
        # to preserve.  Use the normal planner learning rate and save densely
        # enough to select the point before any late-stage regression.
        runner_cfg.save_interval = 25
    if args_cli.expand_variable_height:
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-4
        runner_cfg.save_interval = 25
    if args_cli.height_stage == "low" and input_width in (42, 43, 47):
        # Preserve a transferred planner policy with conservative updates.
        # A 37-D stable baseline has no circular behavior to protect and uses
        # the normal planner PPO settings to learn the task from scratch.
        runner_cfg.algorithm.learning_rate = 1.0e-4
        runner_cfg.algorithm.entropy_coef = 2.0e-4
        runner_cfg.save_interval = 50
    if args_cli.accuracy_finetune:
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 1.0e-4
        runner_cfg.save_interval = 25
    if args_cli.route == "random_two_hop":
        # Route transfer should preserve the accepted v22 height controller
        # while learning the much broader direction/distance distribution.
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 1.0e-4
        runner_cfg.save_interval = 25
    if fixed_high_soft_accuracy:
        # Balanced phase adaptation is a distribution-transfer step, not a
        # fresh route-learning run.  Use conservative updates and dense
        # checkpoints so deterministic validation can select pre-regression.
        runner_cfg.algorithm.learning_rate = 2.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.streak_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.anticipatory_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.continuity_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.fine_tune_lr is not None:
        runner_cfg.algorithm.learning_rate = args_cli.fine_tune_lr
        runner_cfg.save_interval = min(runner_cfg.save_interval, 10)
    if args_cli.first_attempt_finetune:
        runner_cfg.algorithm.learning_rate = 5.0e-6 if args_cli.pair_polish else 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.safety_finetune:
        runner_cfg.num_steps_per_env = 128
        runner_cfg.algorithm.learning_rate = 2.0e-6
        runner_cfg.algorithm.entropy_coef = 2.0e-5
        runner_cfg.algorithm.num_mini_batches = 16
        runner_cfg.save_interval = 5
    if args_cli.robust_finetune:
        runner_cfg.num_steps_per_env = 96
        runner_cfg.algorithm.learning_rate = 1.0e-6
        runner_cfg.algorithm.entropy_coef = 1.0e-5
        runner_cfg.algorithm.num_mini_batches = 12
        runner_cfg.save_interval = 5
    if args_cli.trajectory_landing_finetune:
        runner_cfg.num_steps_per_env = 96
        runner_cfg.algorithm.learning_rate = 8.0e-7 if input_width in (43, 47) else 1.0e-5
        runner_cfg.algorithm.entropy_coef = 8.0e-6 if input_width in (43, 47) else 5.0e-5
        runner_cfg.algorithm.num_mini_batches = 12
        runner_cfg.save_interval = 5
    if args_cli.baseline_warmstart_finetune:
        runner_cfg.num_steps_per_env = 256
        runner_cfg.algorithm.learning_rate = (
            args_cli.fine_tune_lr
            if args_cli.fine_tune_lr is not None
            else 2.0e-6
        )
        runner_cfg.algorithm.entropy_coef = 1.0e-6
        runner_cfg.algorithm.num_mini_batches = 16
        runner_cfg.save_interval = 5
    if args_cli.static_apex_finetune:
        runner_cfg.num_steps_per_env = 128 if input_width == 37 else 96
        runner_cfg.algorithm.learning_rate = 2.0e-5 if input_width == 37 else 8.0e-7
        runner_cfg.algorithm.entropy_coef = 5.0e-5 if input_width == 37 else 1.0e-5
        runner_cfg.algorithm.num_mini_batches = 16 if input_width == 37 else 12
        runner_cfg.save_interval = 5
    if args_cli.takeoff_impulse_finetune:
        runner_cfg.num_steps_per_env = 128
        runner_cfg.algorithm.learning_rate = 3.0e-5 if input_width == 37 else 2.0e-6
        runner_cfg.algorithm.entropy_coef = 8.0e-5 if input_width == 37 else 2.0e-5
        runner_cfg.algorithm.num_mini_batches = 16
        runner_cfg.save_interval = 5
    if args_cli.static_apex_height_probe:
        runner_cfg.num_steps_per_env = 128
        runner_cfg.algorithm.learning_rate = 5.0e-6 if input_width == 37 else 5.0e-7
        runner_cfg.algorithm.entropy_coef = 1.0e-5
        runner_cfg.algorithm.num_mini_batches = 16
        runner_cfg.save_interval = 5
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device)
    if args_cli.checkpoint:
        if input_width != target_obs_dim and input_width in (37, 42, 43):
            migrated = migrate_stable_checkpoint(
                source_checkpoint,
                log_dir / f"initial_policy_{target_obs_dim}d.pt",
                target_obs_dim=target_obs_dim,
            )
            print(f"[INFO] Migrating {input_width}-D policy to {target_obs_dim}-D: {migrated}")
            runner.load(str(migrated), load_optimizer=False)
        elif input_width == target_obs_dim:
            if args_cli.resume_optimizer and EXPERIMENT in source_checkpoint.parts:
                print(f"[INFO] Exactly resuming {target_obs_dim}-D {args_cli.height_stage} stage")
                runner.load(str(source_checkpoint), load_optimizer=True)
            else:
                transfer_checkpoint = log_dir / f"initial_policy_{target_obs_dim}d_transfer.pt"
                transferred_state = checkpoint_data["model_state_dict"].copy()
                if args_cli.relative_next_hop and not source_uses_relative_next:
                    transferred_state = absolute_next_to_relative_state_dict(
                        transferred_state
                    )
                elif source_uses_relative_next and not args_cli.relative_next_hop:
                    raise ValueError(
                        "A relative-next checkpoint requires --relative_next_hop"
                    )
                transferred_state["std"] = torch.full_like(transferred_state["std"], 0.15)
                torch.save(
                    {
                        "model_state_dict": transferred_state,
                        "iter": 0,
                        "infos": {
                            "source_checkpoint": str(source_checkpoint),
                            "height_stage": args_cli.height_stage,
                            "route": args_cli.route,
                            "relative_next_hop": args_cli.relative_next_hop,
                            "optimizer_reset": True,
                        },
                    },
                    transfer_checkpoint,
                )
                print(f"[INFO] Transferring {target_obs_dim}-D policy with optimizer reset: {source_checkpoint}")
                runner.load(str(transfer_checkpoint), load_optimizer=False)
        else:
            raise ValueError(f"Unsupported checkpoint observation width: {input_width}")
    if args_cli.baseline_warmstart_finetune:
        # The legacy 37-D checkpoint has no serialized normalizer state and
        # was trained/evaluated on raw observations. Updating normalization
        # changes every inherited input immediately, even before useful XY
        # behavior has been learned.
        runner.alg.policy.actor_obs_normalization = False
        print("[INFO] Keeping identity actor observation normalization for baseline warm-start")
    if args_cli.freeze_baseline_actor:
        if input_width != 37 or target_obs_dim <= input_width:
            raise ValueError(
                "--freeze_baseline_actor requires a 37-D source migrated to a wider planner observation"
            )
        trainable_actor_input = None
        for name, parameter in runner.alg.policy.named_parameters():
            is_actor = (
                name.startswith("memory_a.")
                or name.startswith("actor.")
                or name in ("std", "log_std")
            )
            if not is_actor:
                continue
            if name == "memory_a.rnn.weight_ih_l0":
                gradient_mask = torch.zeros_like(parameter)
                gradient_mask[:, input_width:target_obs_dim] = 1.0
                parameter.register_hook(
                    lambda gradient, mask=gradient_mask: gradient * mask
                )
                trainable_actor_input = name
            else:
                parameter.requires_grad_(False)
        if trainable_actor_input is None:
            raise RuntimeError("Could not find actor recurrent input weight for frozen warm-start")
        # The old checkpoint predates normalizer state serialization and is
        # evaluated with identity normalization. Updating actor statistics on
        # the first planner rollout changes all 37 baseline inputs at once and
        # was a second source of catastrophic forgetting in the old run.
        print(
            "[INFO] Frozen baseline actor pathway; training only "
            f"{trainable_actor_input} columns [{input_width}:{target_obs_dim}] "
            "with fixed identity actor normalization"
        )
    if args_cli.route == "random_two_hop":
        fixed_action_std = (
            args_cli.fixed_action_std
            if args_cli.fixed_action_std is not None
            else
            # The rolling first-attempt task changes the post-touchdown state
            # distribution substantially and needs enough exploration to
            # discover the longer-hop action profile.  Other precision stages
            # retain the converged 0.01 checkpoint noise.
            0.01
            if args_cli.pair_polish
            else 0.005
            if args_cli.static_apex_height_probe
            else 0.05
            if args_cli.takeoff_impulse_finetune and input_width == 37
            else 0.015
            if args_cli.takeoff_impulse_finetune
            else 0.008
            if args_cli.trajectory_landing_finetune
            else 0.003
            if args_cli.baseline_warmstart_finetune
            else 0.03
            if args_cli.static_apex_finetune and input_width == 37
            else 0.008
            if (
                args_cli.robust_finetune
                or args_cli.static_apex_finetune
                or args_cli.trajectory_landing_finetune
                or args_cli.baseline_warmstart_finetune
            )
            else 0.01
            if args_cli.safety_finetune
            else 0.02
            if args_cli.smooth_distance_curriculum
            else 0.04
            if args_cli.first_attempt_finetune
            else 0.01
            if args_cli.streak_finetune or args_cli.anticipatory_finetune or args_cli.continuity_finetune
            else 0.01
            if args_cli.accuracy_finetune
            else DISTANCE_STAGES[args_cli.distance_stage][4]
            if args_cli.distance_stage != "custom"
            else 0.08
        )
        if not hasattr(runner.alg.policy, "std"):
            raise RuntimeError("The random-route curriculum requires scalar action std")
        with torch.no_grad():
            runner.alg.policy.std.fill_(fixed_action_std)
        runner.alg.policy.std.requires_grad_(False)
        print(
            f"[INFO] Distance stage {args_cli.distance_stage}: "
            f"short=[{args_cli.short_radius_min:.2f}, {args_cli.short_radius_max:.2f}] m, "
            f"long=[{args_cli.long_radius_min:.2f}, {args_cli.long_radius_max:.2f}] m, "
            f"fixed action std={fixed_action_std:.2f}"
        )
    iterations = args_cli.iterations if args_cli.iterations is not None else runner_cfg.max_iterations
    init_at_random_ep_len = not (
        args_cli.safety_finetune
        or args_cli.robust_finetune
        or args_cli.trajectory_landing_finetune
        or args_cli.baseline_warmstart_finetune
        or args_cli.static_apex_finetune
    )
    runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=init_at_random_ep_len,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
