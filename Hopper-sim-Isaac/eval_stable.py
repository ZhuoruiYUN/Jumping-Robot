"""Headless evaluation for the canonical stable-jump baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate the canonical stable-jump baseline")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--height_tolerance", type=float, default=0.15)
parser.add_argument("--xy_tolerance", type=float, default=0.35)
parser.add_argument("--randomize_dynamics", action="store_true")
parser.add_argument("--randomize_action_delay", action="store_true")
parser.add_argument("--observation_noise_std", type=float, default=0.0)
parser.add_argument(
    "--motor_time_constant",
    type=float,
    default=None,
    help="Override the deterministic first-order motor lag for calibration checks.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.rendering_mode = "performance"

PROJECT_DIR = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_DIR / "logs" / "rsl_rl" / "quadhopper_stable_baseline"


def latest_checkpoint() -> Path | None:
    candidates = list(LOG_ROOT.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


checkpoint = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else latest_checkpoint()
if checkpoint is None:
    parser.error(f"No stable checkpoint found under {LOG_ROOT}; train it first.")
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

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Stable  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Stable.quadhopper_env import QuadhopperEnvCfg
from Quadhopper_Stable.rsl_rl_ppo_cfg import QuadhopperPPORunnerCfg


def main():
    print("[EVAL_STABLE] creating env", flush=True)
    env_cfg = QuadhopperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_vis = False
    env_cfg.observation_noise_std = args_cli.observation_noise_std
    env_cfg.randomize_dynamics = args_cli.randomize_dynamics
    env_cfg.randomize_action_delay = args_cli.randomize_action_delay
    if args_cli.motor_time_constant is not None:
        env_cfg.play_motor_time_constant = args_cli.motor_time_constant
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable" / "model" / "quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs" / "stable" / "eval_stable.csv")

    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Stable-Direct-v0", cfg=env_cfg))
    base_env = env.unwrapped

    print("[EVAL_STABLE] creating runner", flush=True)
    runner_cfg = QuadhopperPPORunnerCfg()
    runner_cfg.experiment_name = "quadhopper_stable_baseline"
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    print(f"[EVAL_STABLE] checkpoint={checkpoint}", flush=True)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")

    print("[EVAL_STABLE] resetting env", flush=True)
    obs, _ = env.reset()
    num_envs = args_cli.num_envs
    device = base_env.device

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    episodes = torch.zeros(num_envs, dtype=torch.long, device=device)
    full_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    death = torch.zeros(num_envs, dtype=torch.long, device=device)
    jump_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    good_jump_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    apex_error_sum = torch.zeros(num_envs, device=device)
    landing_xy_sum = torch.zeros(num_envs, device=device)
    max_xy = torch.zeros(num_envs, device=device)
    max_tilt_proxy = torch.zeros(num_envs, device=device)
    local_episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    spring_id = base_env._spring_joint_id
    prev_contact = base_env._robot.data.joint_pos[:, spring_id] > 0.002
    in_flight = torch.zeros(num_envs, dtype=torch.bool, device=device)
    flight_apex = base_env._robot.data.root_pos_w[:, 2].clone()

    print("[EVAL_STABLE] stepping", flush=True)
    for _ in range(args_cli.max_steps):
        local_episode_steps[active] += 1
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        pos = base_env._robot.data.root_pos_w
        quat = base_env._robot.data.root_quat_w
        target = base_env._desired_pos_w
        xy_error = torch.linalg.norm(pos[:, :2] - target[:, :2], dim=1)
        tilt_proxy = torch.sqrt(torch.clamp(quat[:, 1] ** 2 + quat[:, 2] ** 2, min=0.0))
        max_xy = torch.maximum(max_xy, xy_error)
        max_tilt_proxy = torch.maximum(max_tilt_proxy, tilt_proxy)

        contact = base_env._robot.data.joint_pos[:, spring_id] > 0.002
        takeoff = prev_contact & ~contact
        in_flight |= takeoff
        flight_apex = torch.where(takeoff, pos[:, 2], flight_apex)
        flight_apex = torch.where(in_flight, torch.maximum(flight_apex, pos[:, 2]), flight_apex)
        landing = in_flight & ~prev_contact & contact
        if torch.any(landing):
            apex_error = torch.abs(flight_apex[landing] - target[landing, 2])
            landing_xy = xy_error[landing]
            jump_count[landing] += 1
            good_jump_count[landing] += (
                (apex_error <= args_cli.height_tolerance) & (landing_xy <= args_cli.xy_tolerance)
            ).long()
            apex_error_sum[landing] += apex_error
            landing_xy_sum[landing] += landing_xy
            in_flight[landing] = False
        prev_contact = contact

        done = dones.to(device).bool()
        if torch.any(done):
            time_out = local_episode_steps >= base_env.max_episode_length - 1
            full_length[done & time_out] += 1
            death[done & ~time_out] += 1
            episodes[done] += 1
            active[done] = False
        if not torch.any(active):
            break

    completed = torch.clamp(episodes, min=1)
    jump_total = torch.sum(jump_count).float()
    good_jump_total = torch.sum(good_jump_count).float()
    mean_apex_error = torch.sum(apex_error_sum) / torch.clamp(jump_total, min=1.0)
    mean_landing_xy = torch.sum(landing_xy_sum) / torch.clamp(jump_total, min=1.0)
    deploy_success = (death == 0) & (jump_count >= 3) & (max_xy <= args_cli.xy_tolerance)

    print(f"[EVAL_STABLE] steps={args_cli.max_steps}", flush=True)
    print(f"[EVAL_STABLE] episodes={int(torch.sum(episodes).item())}", flush=True)
    print(f"[EVAL_STABLE] full_length_rate={torch.mean((full_length > 0).float()).item():.6f}", flush=True)
    print(f"[EVAL_STABLE] death_rate={torch.mean((death > 0).float()).item():.6f}", flush=True)
    print(f"[EVAL_STABLE] deploy_success_rate={torch.mean(deploy_success.float()).item():.6f}", flush=True)
    print(f"[EVAL_STABLE] jumps_per_env={torch.mean((jump_count.float() / completed.float())).item():.6f}", flush=True)
    print(
        f"[EVAL_STABLE] good_jump_rate={torch.where(jump_total > 0, good_jump_total / jump_total, jump_total).item():.6f}",
        flush=True,
    )
    print(f"[EVAL_STABLE] mean_apex_error_m={mean_apex_error.item():.6f}", flush=True)
    print(f"[EVAL_STABLE] mean_landing_xy_error_m={mean_landing_xy.item():.6f}", flush=True)
    print(f"[EVAL_STABLE] max_xy_error_mean_m={torch.mean(max_xy).item():.6f}", flush=True)
    print(f"[EVAL_STABLE] max_tilt_proxy_mean={torch.mean(max_tilt_proxy).item():.6f}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
