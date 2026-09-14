"""Open-loop vertical-apex feasibility scan under the deployment-parity plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps-per-case", type=int, default=1200)
parser.add_argument("--hops-per-case", type=int, default=4)
parser.add_argument("--contact-pwms", type=float, nargs="+", default=[500.0, 700.0, 900.0])
parser.add_argument("--ascent-pwms", type=float, nargs="+", default=[250.0, 500.0])
parser.add_argument("--target-height", type=float, default=1.095)
parser.add_argument("--play-thrust-scale", type=float, default=1.0,
                    help="Fixed per-motor thrust multiplier for plant feasibility identification.")
parser.add_argument("--play-motor-time-constant", type=float, default=0.0674,
                    help="Fixed motor first-order time constant in seconds.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    parser.error(f"Output already exists: {args.output}")
if args.steps_per_case <= 0 or args.hops_per_case <= 0:
    parser.error("steps-per-case and hops-per-case must be positive")
if any(not 0.0 <= value <= 1000.0 for value in args.contact_pwms + args.ascent_pwms):
    parser.error("PWM values must lie in [0, 1000]")
if not 0.50 <= args.play_thrust_scale <= 1.20:
    parser.error("play-thrust-scale must be in [0.50, 1.20]")
if not 0.01 <= args.play_motor_time_constant <= 0.20:
    parser.error("play-motor-time-constant must be in [0.01, 0.20] seconds")

app = AppLauncher(args).app

import os
import sys

import isaaclab
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.join(os.path.dirname(isaaclab.__file__), "source", "isaaclab_rl"))
from Quadhopper_Planner_Random.spatial_tracking_env import (
    SpatialTrackingEnv,
    SpatialTrackingPhysical1mEnvCfg,
)


def command_from_pwm(pwm: float, device: torch.device) -> torch.Tensor:
    return torch.full((1, 4), 2.0 * pwm / 1000.0 - 1.0, device=device)


cfg = SpatialTrackingPhysical1mEnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = args.device
cfg.seed = args.seed
cfg.target_height = args.target_height
cfg.deployment_action_target_height = args.target_height
cfg.randomize_dynamics = False
cfg.randomize_action_delay = False
cfg.randomize_motor_time_constant = False
cfg.play_thrust_scale = args.play_thrust_scale
cfg.play_motor_time_constant = args.play_motor_time_constant
# This scan must use the deployable 40_v1/mixed v11 actuator semantics, not
# v12's post-policy PWM shaping.  The command is open-loop and symmetric.
cfg.corrected_action_delay_indexing = False
cfg.record_executed_action_history = False
cfg.apply_deployment_action_shaping = False
cfg.observation_noise_std = 0.0
cfg.collect_tracking_metrics = False
cfg.debug_vis = False
cfg.power_model_path = str(ROOT / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
cfg.csv_log_path = str(args.output.parent / "unused_single_env.csv")
env = SpatialTrackingEnv(cfg)
device = env.device
results: list[dict[str, object]] = []

try:
    for contact_pwm in args.contact_pwms:
        for ascent_pwm in args.ascent_pwms:
            env.reset()
            prior_contact = True
            ready = False
            in_flight = False
            apex = 0.0
            liftoff_z = 0.0
            liftoff_vz = 0.0
            hops: list[dict[str, float]] = []
            for _ in range(args.steps_per_case):
                root_vz = float(env._robot.data.root_lin_vel_w[0, 2].item())
                contact = bool((env._robot.data.joint_pos[0, env._spring_joint_id] > 0.002).item())
                pending_drop = bool(env._initial_drop_pending[0].item())
                pwm = 0.0 if pending_drop else (contact_pwm if contact else (ascent_pwm if root_vz > 0.0 else 0.0))
                env.step(command_from_pwm(pwm, device))

                contact = bool((env._robot.data.joint_pos[0, env._spring_joint_id] > 0.002).item())
                pending_drop = bool(env._initial_drop_pending[0].item())
                z = float(env._robot.data.root_pos_w[0, 2].item())
                vz = float(env._robot.data.root_lin_vel_w[0, 2].item())
                if pending_drop:
                    prior_contact = contact
                    continue
                if contact:
                    ready = True
                if ready and prior_contact and not contact:
                    in_flight = True
                    apex = z
                    liftoff_z = z
                    liftoff_vz = vz
                if in_flight:
                    apex = max(apex, z)
                    if vz <= 0.0:
                        hops.append({
                            "liftoff_z_m": liftoff_z,
                            "liftoff_vz_mps": liftoff_vz,
                            "apex_m": apex,
                        })
                        in_flight = False
                        if len(hops) >= args.hops_per_case:
                            break
                prior_contact = contact
            apexes = [hop["apex_m"] for hop in hops]
            result: dict[str, object] = {
                "contact_pwm": contact_pwm,
                "ascent_pwm": ascent_pwm,
                "hops": hops,
                "hops_collected": len(hops),
                "target_height_m": args.target_height,
            }
            if apexes:
                result.update(
                    mean_apex_m=fmean(apexes),
                    median_apex_m=median(apexes),
                    max_apex_m=max(apexes),
                    mean_liftoff_vz_mps=fmean(hop["liftoff_vz_mps"] for hop in hops),
                    reaches_target=any(apex >= args.target_height for apex in apexes),
                )
            results.append(result)
            print("[OPEN-LOOP]", json.dumps(result), flush=True)
    payload = {
        "target_height_m": args.target_height,
        "contract": "legacy_v11_direct_motor",
        "play_thrust_scale": args.play_thrust_scale,
        "play_motor_time_constant_s": args.play_motor_time_constant,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
finally:
    env.close()
    app.close()
