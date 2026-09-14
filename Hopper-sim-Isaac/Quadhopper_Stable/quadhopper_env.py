from __future__ import annotations

import gymnasium as gym
import torch
import math
import csv
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import CUBOID_MARKER_CFG

from .my_hopper_cfg import MY_HOPPER_CFG
from .agents.rsl_rl_ppo_cfg import QUADHOPPER_MAX_ITERATIONS, QUADHOPPER_NUM_STEPS_PER_ENV
from .actuator_contract import (
    corrected_delay_history_index,
    shape_deployment_motor_commands,
)
from .action_parameterization import collective_residual_to_motor_u


POWER_MODEL_HISTORY_LEN = 25
POWER_MODEL_SHORT_WIN = 10
POWER_MODEL_MID_WIN = 15
POWER_MODEL_LONG_WIN = 25
POWER_MODEL_SOC_SCALE = 10000.0
LEGACY_POWER_REWARD_MULTIPLIER = 40.0


def scale_legacy_power_reward(value: float) -> float:
    return value * LEGACY_POWER_REWARD_MULTIPLIER


def update_power_model_memory(
    motor_u_history: torch.Tensor,
    soc_tracker: torch.Tensor,
    motor_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep power-model features in the same 0..1 scale used for training."""
    motor_u_history = torch.roll(motor_u_history, shifts=-1, dims=1)
    motor_u_history[:, -1, :] = motor_u
    soc_tracker = soc_tracker + torch.sum(motor_u, dim=1, keepdim=True) / POWER_MODEL_SOC_SCALE
    return motor_u_history, soc_tracker


def build_power_model_inputs(motor_u_history: torch.Tensor, soc_tracker: torch.Tensor) -> torch.Tensor:
    pwm_short = motor_u_history[:, -POWER_MODEL_SHORT_WIN:, :].reshape(
        motor_u_history.shape[0], POWER_MODEL_SHORT_WIN * motor_u_history.shape[2]
    )
    pwm_mid = torch.mean(motor_u_history[:, -POWER_MODEL_MID_WIN:, :], dim=1)
    pwm_long = torch.mean(motor_u_history[:, -POWER_MODEL_LONG_WIN:, :], dim=1)
    return torch.cat([pwm_short, pwm_mid, pwm_long, soc_tracker], dim=-1)


def compute_power_reward_terms(
    predicted_power: torch.Tensor,
    is_contact: torch.Tensor,
    joint_vel: torch.Tensor,
    contact_velocity_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    power_penalty = predicted_power
    synergy = torch.where(
        is_contact & (joint_vel < -contact_velocity_threshold),
        predicted_power,
        torch.zeros_like(predicted_power),
    )
    anti_synergy = torch.where(
        is_contact & (joint_vel > contact_velocity_threshold),
        predicted_power,
        torch.zeros_like(predicted_power),
    )
    return power_penalty, synergy, anti_synergy


# =========================================================
# 1. UI窗口与可视化目标点
# =========================================================
class QuadhopperEnvWindow(BaseEnvWindow):
    def __init__(self, env: QuadhopperEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadhopperEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 1  # 100Hz 同频

    action_space = 4
    observation_space = 37
    # Keep domain noise during training, but allow deterministic evaluation.
    observation_noise_std = 0.01
    randomize_dynamics = True
    randomize_action_delay = True
    # Legacy tasks retain the captured action/history semantics. A task that
    # targets the current deployment opts into these as one coupled contract.
    corrected_action_delay_indexing = False
    record_executed_action_history = False
    apply_deployment_action_shaping = False
    # Research-only training aid: supplies a symmetric collective lower bound
    # during the measured takeoff phases.  It is deliberately off by default
    # and must never be included in an exported deployment contract.
    simulation_only_phase_collective_baseline = False
    phase_contact_collective_floor_pwm = 0.0
    phase_ascent_collective_floor_pwm = 0.0
    phase_ascent_vz_min_mps = 0.10
    phase_ascent_height_margin_m = 0.03
    # Stronger research-only alternative to the floor above.  During the
    # energy-setting phases it owns the collective command; the policy can
    # contribute only a zero-mean, tightly bounded attitude residual.
    simulation_only_phase_collective_controller = False
    phase_contact_collective_pwm = 0.0
    phase_ascent_collective_pwm = 0.0
    phase_descent_collective_pwm = 0.0
    phase_contact_differential_limit_pwm = 0.0
    phase_ascent_differential_limit_pwm = 0.0
    phase_descent_differential_limit_pwm = 0.0
    # Per-hop apex outer loop.  Unlike the rejected fixed waveform, this
    # preserves the policy's complete motor pattern and only adds a symmetric
    # contact-phase energy residual learned from the previous measured apex.
    simulation_only_apex_bias_controller = False
    apex_bias_initial_pwm = 0.0
    apex_bias_gain_pwm_per_m = 0.0
    apex_bias_max_pwm = 0.0
    apex_bias_deadband_m = 0.02
    apex_bias_compression_start_m = 0.006
    apex_bias_compression_full_m = 0.030
    apex_bias_feedback_mode = "apex"
    apex_bias_liftoff_vz_target_mps = 0.0
    deployment_action_target_height = 1.0
    deployment_max_pwm = 1000.0
    deployment_max_pwm_spread = 600.0
    deployment_max_airborne_yaw_diagonal_pwm_diff = 140.0
    deployment_max_grounded_yaw_diagonal_pwm_diff = 220.0
    deployment_height_brake_start_m = 0.03
    deployment_height_brake_full_m = 0.23
    deployment_height_brake_vz_min = 0.05
    deployment_height_brake_pwm_mean_soft = 720.0
    deployment_height_brake_pwm_mean_hard = 520.0
    deployment_constraint_projection_iterations = 2
    state_space = 0
    debug_vis = True  # 开启目标点可视化

    # =========================================================
    # CSV logging for simulation/experiment comparison
    # =========================================================
    # 训练时 num_envs > 1: 不写 CSV，避免生成超大文件。
    # 测试/对比时 num_envs == 1: 自动开启 CSV 数据记录。

    # Power model path
    power_model_path = r"D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadhopper-jump\model\quadhopper_memory_power.pt"
    csv_log_path = r"D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\quadhopper-jump\on_quadhopper_sim.csv"
    csv_flush_interval = 100

    # =========================================================
    # Deterministic test settings, used only when num_envs == 1
    # =========================================================
    # 用于测试/实验对比的固定电机参数。
    # 训练时不会使用这些固定值，训练仍使用域随机化。
    # Flight-data identification (2026-09-07): nominal first-order actuator
    # time constant 0.0674 s. Keep roughly +/-20% coverage for robustness.
    train_motor_time_constant_min = 0.054
    train_motor_time_constant_max = 0.081
    train_mass_multi_min = 0.95
    train_mass_multi_max = 1.05
    train_inertia_multi_min = 0.9
    train_inertia_multi_max = 1.1
    randomize_thrust_curve = False
    # 单环境默认保持推力标称（可视化/CSV 对照）；实验需要时显式打开。
    allow_single_env_thrust_rand = False
    # 独立于 randomize_dynamics 的电机时间常数随机化（sim2real：实机电机
    # 标称值来自最新飞行数据辨识；可在专项微调时独立随机化。
    randomize_motor_time_constant = False
    train_thrust_scale_min = 1.0
    train_thrust_scale_max = 1.0
    train_thrust_curve_shape_min = 1.0
    train_thrust_curve_shape_max = 1.0
    train_drop_height_min = 0.8
    train_drop_height_max = 1.5
    curriculum_num_steps_per_env = QUADHOPPER_NUM_STEPS_PER_ENV
    curriculum_max_iterations = QUADHOPPER_MAX_ITERATIONS
    curriculum_vertical_fraction = 0.4
    curriculum_full_difficulty_fraction = 0.8
    curriculum_initial_xy_radius_max = 5.0
    curriculum_roll_pitch_max = 0.25
    curriculum_yaw_max = 3.14159
    play_motor_time_constant = 0.0674
    # Deterministic nominal actuator parameters for one-environment plant
    # identification.  Defaults retain the calibrated source plant exactly;
    # experiments must opt in through the recovery launcher.
    play_thrust_scale = 1.0
    play_thrust_curve_shape = 1.0
    action_parameterization = "direct_motor"
    # 2026-09-05 修正：实机 FC 指令延迟 ≤1ms，旧值 3 步(30ms) 会压低训练增益
    # （对照 LQR 部署工程：89.3% 零延迟 + 小概率 1/2 步）。
    play_action_delay = 0
    play_target_pos = (0,0, 1.3)
    play_initial_pos = (0, 0, 1)
    ui_window_class_type = QuadhopperEnvWindow

    # === 跳跃机专属奖励系数 ===
    survival_reward_scale = 5.0

    # 👑 1. 重振主线权威：大幅提高高度奖励，让它向往天空
    distance_to_xy_reward_scale = 15.0
    distance_to_z_reward_scale = 2.0
    energy_tracking_reward_scale = 12.0
    apex_reward_scale = 8.0
    vertical_energy_error_scale = 2.0

    yaw_penalty_scale = -7.0
    attitude_penalty_scale = -20.0
    angular_vel_penalty_scale = -0.5

    # 👑 2. 滞地惩罚：赖在地上不跳会被疯狂扣分
    lazy_ground_penalty_scale = -10.0

    power_penalty_scale = scale_legacy_power_reward(-0.002)
    synergy_reward_scale = scale_legacy_power_reward(0.02)
    anti_synergy_penalty_scale = scale_legacy_power_reward(-0.02)
    spring_power_velocity_threshold = 0.5

    # 👑 4. Sim-to-Real: 仅抑制横向移动，允许偏头以防止落地摔倒
    xy_error_penalty_scale = -1.5
    xy_progress_reward_scale = 20.0
    goal_bonus_scale = 8.0
    xy_reward_width = 0.5
    goal_tolerance = 0.35
    lateral_vel_penalty_scale = -10.0
    flight_power_penalty_scale = -0.6
    hover_penalty_scale = -12.0
    action_rate_reward_scale = -0.5
    termination_penalty_scale = -10.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)
    robot: ArticulationCfg = MY_HOPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class QuadhopperEnv(DirectRLEnv):
    cfg: QuadhopperEnvCfg

    def __init__(self, cfg: QuadhopperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.history_len = 5
        self._action_history = deque(maxlen=self.history_len)
        self._command_history = deque(maxlen=self.history_len)
        history_reset_value = -1.0 if self.cfg.record_executed_action_history else 0.0
        for _ in range(self.history_len):
            reset_action = torch.full(
                (self.num_envs, 4), history_reset_value, device=self.device
            )
            self._action_history.append(reset_action.clone())
            self._command_history.append(reset_action.clone())

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_distance_to_goal_xy = torch.zeros(self.num_envs, device=self.device)
        self._motor_u = torch.zeros(self.num_envs, 4, device=self.device)
        # Keep policy requests separate from deployment-shaped commands for
        # one-environment takeoff diagnostics.
        self._raw_target_u = torch.zeros_like(self._motor_u)
        self._shaped_target_u = torch.zeros_like(self._motor_u)
        self._apex_bias_pwm = torch.zeros(self.num_envs, device=self.device)

        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)
        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.069
        self.dr_thrust_scale = torch.ones(self.num_envs, 4, device=self.device)
        self.dr_thrust_curve_shape = torch.ones(self.num_envs, 4, device=self.device)

        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation
        self._body_id = self._robot.find_bodies("Body")[0]
        self._spring_joint_id = self._robot.find_joints("center_spring_joint")[0][0]

        # =========================================================
        # 👑 引入神经网络功耗模型及记忆缓冲区
        # =========================================================
        self.power_model = torch.jit.load(self.cfg.power_model_path).to(self.device)
        self.power_model.eval()

        self.nn_history_len = POWER_MODEL_HISTORY_LEN
        self._motor_u_history = torch.zeros(self.num_envs, self.nn_history_len, 4, device=self.device)
        self._soc_tracker = torch.zeros(self.num_envs, 1, device=self.device)

        sum_keys = [
            "survival",
            "distance_to_xy",
            "distance_to_z",
            "energy_tracking",
            "apex",
            "yaw_p",
            "power_penalty",
            "attitude_penalty",
            "angular_vel_p",
            "xy_error_p",
            "xy_progress",
            "goal_bonus",
            "synergy",
            "anti_synergy",
            "lateral_vel_p",
            "flight_power_p",
            "hover_p",
            "action_rate_p",
            "termination_p",
            "lazy_ground_p",
        ]
        self._episode_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in sum_keys}

        self._csv_logging_enabled = self.num_envs == 1
        self._csv_rows = []
        self._csv_header_written = False
        self._csv_log_path = Path(self.cfg.csv_log_path)
        if self._csv_logging_enabled:
            self._csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._csv_log_path.exists():
                self._csv_log_path.unlink()

        self.set_debug_vis(self.cfg.debug_vis)

    # =========================================================
    # Isaac Lab core callbacks
    # =========================================================
    def _csv_header(self) -> list[str]:
        return [
            "Time_s",
            "X", "Y", "Z",
            "Target_X", "Target_Y", "Target_Z",
            "Vx", "Vy", "Vz",
            "Qw", "Qx", "Qy", "Qz",
            "M1", "M2", "M3", "M4",
            "U1", "U2", "U3", "U4",
            "RawCmd1", "RawCmd2", "RawCmd3", "RawCmd4",
            "ShapedCmd1", "ShapedCmd2", "ShapedCmd3", "ShapedCmd4",
            "Power_W",
            "Spring_Pos", "Spring_Vel", "Is_Contact",
        ]

    def _append_csv_row(self, predicted_power: torch.Tensor):
        if not self._csv_logging_enabled:
            return

        env_id = 0
        time_s = float(self.episode_length_buf[env_id].item() * self.step_dt)

        pos = self._robot.data.root_pos_w[env_id]
        vel = self._robot.data.root_lin_vel_w[env_id]
        quat = self._robot.data.root_quat_w[env_id]
        target = self._desired_pos_w[env_id]
        motor_u = self._motor_u[env_id]
        pwm = motor_u * 1000.0
        raw_cmd_pwm = self._raw_target_u[env_id] * 1000.0
        shaped_cmd_pwm = self._shaped_target_u[env_id] * 1000.0
        joint_pos = self._robot.data.joint_pos[env_id, self._spring_joint_id]
        joint_vel = self._robot.data.joint_vel[env_id, self._spring_joint_id]
        is_contact = joint_pos > 0.002

        row = [
            time_s,
            float(pos[0].detach().cpu()), float(pos[1].detach().cpu()), float(pos[2].detach().cpu()),
            float(target[0].detach().cpu()), float(target[1].detach().cpu()), float(target[2].detach().cpu()),
            float(vel[0].detach().cpu()), float(vel[1].detach().cpu()), float(vel[2].detach().cpu()),
            float(quat[0].detach().cpu()), float(quat[1].detach().cpu()), float(quat[2].detach().cpu()), float(quat[3].detach().cpu()),
            float(pwm[0].detach().cpu()), float(pwm[1].detach().cpu()), float(pwm[2].detach().cpu()), float(pwm[3].detach().cpu()),
            float(motor_u[0].detach().cpu()), float(motor_u[1].detach().cpu()), float(motor_u[2].detach().cpu()), float(motor_u[3].detach().cpu()),
            float(raw_cmd_pwm[0].detach().cpu()), float(raw_cmd_pwm[1].detach().cpu()), float(raw_cmd_pwm[2].detach().cpu()), float(raw_cmd_pwm[3].detach().cpu()),
            float(shaped_cmd_pwm[0].detach().cpu()), float(shaped_cmd_pwm[1].detach().cpu()), float(shaped_cmd_pwm[2].detach().cpu()), float(shaped_cmd_pwm[3].detach().cpu()),
            float(predicted_power[env_id].detach().cpu()),
            float(joint_pos.detach().cpu()),
            float(joint_vel.detach().cpu()),
            int(bool(is_contact.detach().cpu())),
        ]
        self._csv_rows.append(row)

        if len(self._csv_rows) >= self.cfg.csv_flush_interval:
            self._flush_csv_rows()

    def _flush_csv_rows(self):
        if not self._csv_logging_enabled or not self._csv_rows:
            return

        with self._csv_log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not self._csv_header_written:
                writer.writerow(self._csv_header())
                self._csv_header_written = True
            writer.writerows(self._csv_rows)

        self._csv_rows.clear()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # TerrainImporterCfg("plane") resolves to Isaac's remote Grid USD.
        # Training must not block or fail when that external asset endpoint is
        # unavailable.  This local cuboid has the same flat z=0 contact
        # surface and configured friction/restitution, while the grid below
        # exactly reproduces TerrainImporter's plane env-origin layout.
        num_envs = self.scene.cfg.num_envs
        env_spacing = self.scene.cfg.env_spacing
        num_rows = math.ceil(num_envs / int(math.sqrt(num_envs)))
        num_cols = math.ceil(num_envs / num_rows)
        rows, cols = torch.meshgrid(
            torch.arange(num_rows, device=self.device),
            torch.arange(num_cols, device=self.device),
            indexing="ij",
        )
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:, 0] = -(rows.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (cols.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        self._terrain = SimpleNamespace(env_origins=env_origins)
        ground_span = max(num_rows, num_cols) * env_spacing + 2.0 * env_spacing
        ground_cfg = sim_utils.CuboidCfg(
            size=(ground_span, ground_span, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=self.cfg.terrain.physics_material,
        )
        ground_cfg.func("/World/ground/local_plane", ground_cfg, translation=(0.0, 0.0, -0.01))
        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self.cfg.record_executed_action_history:
            self._command_history.append(self._actions.clone())
        else:
            # Preserve the frozen baseline: the visible raw-action history is
            # also the queue used by its legacy delay indexing.
            self._action_history.append(self._actions.clone())

        if self.num_envs == 1 or not self.cfg.randomize_action_delay:
            delay_idx = self.cfg.play_action_delay
        else:
            # 2026-09-05 修正：对照实机（≤1ms）与 LQR 部署工程，延迟分布改为
            # 89.3% 零延迟 + 7.5% 1 步 + 3.2% 2 步（旧值固定 2-4 步）。
            delay_idx = int(
                torch.multinomial(
                    torch.tensor([0.893, 0.075, 0.032], device=self.device),
                    num_samples=1,
                ).item()
            )
        delay_history = (
            self._command_history
            if self.cfg.record_executed_action_history
            else self._action_history
        )
        if self.cfg.corrected_action_delay_indexing:
            delay_index = corrected_delay_history_index(delay_idx, self.history_len)
        else:
            # Historical behavior: delay_idx==0 selects slot 0 because -0 == 0.
            delay_index = -delay_idx
        delayed_actions = delay_history[delay_index]

        if self.cfg.action_parameterization == "collective_residual_v1":
            target_u = collective_residual_to_motor_u(delayed_actions)
        else:
            target_u = (delayed_actions * 0.5 + 0.5).clamp(0.0, 1.0)
        self._raw_target_u.copy_(target_u)
        if self.cfg.simulation_only_apex_bias_controller:
            # Apply only symmetric energy late in spring compression.  The
            # existing policy differential remains untouched unless physical
            # saturation itself clips a motor command.
            spring_pos = self._robot.data.joint_pos[:, self._spring_joint_id]
            compression = ((spring_pos - self.cfg.apex_bias_compression_start_m)
                           / max(self.cfg.apex_bias_compression_full_m
                                 - self.cfg.apex_bias_compression_start_m, 1.0e-6)).clamp(0.0, 1.0)
            contact = spring_pos > 0.002
            bias = self._apex_bias_pwm * compression * contact
            target_u = (target_u + (bias / 1000.0).unsqueeze(-1)).clamp(0.0, 1.0)
        if self.cfg.simulation_only_phase_collective_baseline:
            # This is intentionally a *simulation-only* teacher.  It lifts
            # the collective command only to a phase-specific floor while
            # preserving the policy's four-motor differential.  Executed
            # post-floor actions enter the recurrent history below, so the
            # policy is trained against exactly the plant it sees here.
            contact = self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
            initial_drop = getattr(
                self,
                "_initial_drop_pending",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            cycle_active = getattr(
                self,
                "_cycle_active",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            target_height = getattr(self, "_active_target_height", None)
            if target_height is None:
                target_height = torch.full(
                    (self.num_envs,),
                    self.cfg.deployment_action_target_height,
                    device=self.device,
                    dtype=target_u.dtype,
                )
            ascending = (
                ~contact
                & cycle_active
                & ~initial_drop
                & (self._robot.data.root_lin_vel_w[:, 2] > self.cfg.phase_ascent_vz_min_mps)
                & (self._robot.data.root_pos_w[:, 2] < target_height - self.cfg.phase_ascent_height_margin_m)
            )
            floor_pwm = torch.where(
                contact & ~initial_drop,
                torch.full_like(target_height, self.cfg.phase_contact_collective_floor_pwm),
                torch.zeros_like(target_height),
            )
            floor_pwm = torch.where(
                ascending,
                torch.full_like(target_height, self.cfg.phase_ascent_collective_floor_pwm),
                floor_pwm,
            )
            collective_deficit = (floor_pwm / 1000.0 - target_u.mean(dim=-1)).clamp_min(0.0)
            target_u = (target_u + collective_deficit.unsqueeze(-1)).clamp(0.0, 1.0)
        if self.cfg.simulation_only_phase_collective_controller:
            # Do not confuse a mean-PWM floor with collective control.  The
            # latter fixes takeoff energy to the open-loop validated waveform
            # and exposes policy output only as a zero-mean attitude residual.
            # This remains simulation-only until separately validated on hw.
            contact = self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
            initial_drop = getattr(
                self,
                "_initial_drop_pending",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            cycle_active = getattr(
                self,
                "_cycle_active",
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
            target_height = getattr(self, "_active_target_height", None)
            if target_height is None:
                target_height = torch.full(
                    (self.num_envs,),
                    self.cfg.deployment_action_target_height,
                    device=self.device,
                    dtype=target_u.dtype,
                )
            contact_phase = contact & ~initial_drop
            ascent_phase = (
                ~contact
                & cycle_active
                & ~initial_drop
                & (self._robot.data.root_lin_vel_w[:, 2] > self.cfg.phase_ascent_vz_min_mps)
            )
            descent_phase = cycle_active & ~initial_drop & ~contact & ~ascent_phase
            raw_pwm = target_u * 1000.0
            zero_mean_residual = raw_pwm - raw_pwm.mean(dim=-1, keepdim=True)
            contact_residual = zero_mean_residual.clamp(
                -self.cfg.phase_contact_differential_limit_pwm,
                self.cfg.phase_contact_differential_limit_pwm,
            )
            ascent_residual = zero_mean_residual.clamp(
                -self.cfg.phase_ascent_differential_limit_pwm,
                self.cfg.phase_ascent_differential_limit_pwm,
            )
            descent_residual = zero_mean_residual.clamp(
                -self.cfg.phase_descent_differential_limit_pwm,
                self.cfg.phase_descent_differential_limit_pwm,
            )
            contact_pwm = self.cfg.phase_contact_collective_pwm + contact_residual
            ascent_pwm = self.cfg.phase_ascent_collective_pwm + ascent_residual
            descent_pwm = self.cfg.phase_descent_collective_pwm + descent_residual
            # The feasibility scan starts every reset with zero motor command.
            # Own the initial drop as well; otherwise stale source-policy
            # commands alter motor state before the first controlled contact.
            controlled_pwm = torch.where(initial_drop.unsqueeze(-1), torch.zeros_like(raw_pwm), raw_pwm)
            controlled_pwm = torch.where(contact_phase.unsqueeze(-1), contact_pwm, controlled_pwm)
            controlled_pwm = torch.where(ascent_phase.unsqueeze(-1), ascent_pwm, controlled_pwm)
            controlled_pwm = torch.where(descent_phase.unsqueeze(-1), descent_pwm, controlled_pwm)
            target_u = (controlled_pwm / 1000.0).clamp(0.0, 1.0)
        if self.cfg.apply_deployment_action_shaping:
            contact = self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
            target_height = getattr(self, "_active_target_height", None)
            if target_height is None:
                target_height = torch.full(
                    (self.num_envs,),
                    self.cfg.deployment_action_target_height,
                    device=self.device,
                    dtype=target_u.dtype,
                )
            target_u = shape_deployment_motor_commands(
                target_u,
                is_contact=contact,
                root_z=self._robot.data.root_pos_w[:, 2],
                root_vz=self._robot.data.root_lin_vel_w[:, 2],
                target_z=target_height,
                max_pwm=self.cfg.deployment_max_pwm,
                max_pwm_spread=self.cfg.deployment_max_pwm_spread,
                max_airborne_yaw_diagonal_pwm_diff=(
                    self.cfg.deployment_max_airborne_yaw_diagonal_pwm_diff
                ),
                max_grounded_yaw_diagonal_pwm_diff=(
                    self.cfg.deployment_max_grounded_yaw_diagonal_pwm_diff
                ),
                height_brake_start_m=self.cfg.deployment_height_brake_start_m,
                height_brake_full_m=self.cfg.deployment_height_brake_full_m,
                height_brake_vz_min=self.cfg.deployment_height_brake_vz_min,
                height_brake_pwm_mean_soft=self.cfg.deployment_height_brake_pwm_mean_soft,
                height_brake_pwm_mean_hard=self.cfg.deployment_height_brake_pwm_mean_hard,
                projection_iterations=(
                    self.cfg.deployment_constraint_projection_iterations
                ),
            )
        self._shaped_target_u.copy_(target_u)
        if self.cfg.record_executed_action_history:
            self._action_history.append((2.0 * target_u - 1.0).clone())
        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        # Keep neural power-model features aligned with the training scripts:
        # collected PWM was divided by 1000.0 before inference, so use u in 0..1.
        self._motor_u_history, self._soc_tracker = update_power_model_memory(
            self._motor_u_history, self._soc_tracker, u
        )

        # === 物理推力计算依然使用归一化的 u ===
        F = (
            -0.2371 * self.dr_thrust_curve_shape * u ** 2
            + 0.8130 * u
            + 0.0113
        )
        F = torch.clamp(F, min=0.0)
        F = F * self.dr_thrust_scale
        F = F / self.dr_mass_multi

        L = 0.0813
        # 2026-09-05 修正：yaw 反扭矩系数按实机辨识值 1.72e-2（旧值 5.4e-2 偏大 3.1x，
        # 与旧 I_ZZ 偏大 2.9x 恰好抵消指令增益，掩盖了扰动灵敏度错误）。
        K_tau = 1.720762152765e-02
        F1, F2, F3, F4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        tau_x = L * (F1 + F4 - F2 - F3) / self.dr_inertia_multi[:, 0]
        tau_y = L * (F1 + F2 - F3 - F4) / self.dr_inertia_multi[:, 1]
        tau_z = K_tau * -(u1 ** 2 + u3 ** 2 - u2 ** 2 - u4 ** 2) / self.dr_inertia_multi[:, 2]

        self._thrust[:, 0, 2] = F1 + F2 + F3 + F4
        self._moment[:, 0, 0], self._moment[:, 0, 1], self._moment[:, 0, 2] = tau_x, tau_y, tau_z

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        pos_error, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w
        z_pos = self._robot.data.root_pos_w[:, 2].unsqueeze(1)

        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id].unsqueeze(1)
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id].unsqueeze(1)
        is_contact = (joint_pos > 0.002).float()

        history_actions = torch.cat(list(self._action_history), dim=-1)

        obs = torch.cat(
            [lin_vel, ang_vel, quat, pos_error, z_pos, is_contact, joint_pos, joint_vel, history_actions],
            dim=-1,
        )
        if self.cfg.observation_noise_std > 0.0:
            obs += torch.randn_like(obs) * self.cfg.observation_noise_std
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q = self._robot.data.root_quat_w
        z_pos = self._robot.data.root_pos_w[:, 2]

        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id]
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id]
        is_contact = joint_pos > 0.002

        roll_pitch_sq = q[:, 1] ** 2 + q[:, 2] ** 2
        ground_proximity = torch.clamp(1.0 - z_pos, 0.0, 1.0)
        attitude_penalty = roll_pitch_sq * (1.0 + 5.0 * ground_proximity)

        # 抑制横向移动，防止漂移
        lin_vel_b = self._robot.data.root_lin_vel_b
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        lateral_vel_penalty = torch.sum(torch.square(lin_vel_b[:, :2]), dim=1)
        angular_vel_penalty = torch.sum(torch.square(ang_vel_b), dim=1)
        action_rate_penalty = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # =========================================================
        # 👑 调用神经网络模型推演真实功耗
        # =========================================================
        nn_inputs = build_power_model_inputs(self._motor_u_history, self._soc_tracker)

        with torch.no_grad():
            predicted_power = self.power_model(nn_inputs).squeeze(-1)

        self._append_csv_row(predicted_power)

        power_penalty, synergy, anti_synergy = compute_power_reward_terms(
            predicted_power,
            is_contact,
            joint_vel,
            self.cfg.spring_power_velocity_threshold,
        )
        airborne = torch.logical_not(is_contact)
        flight_power_penalty = torch.where(airborne, predicted_power, torch.zeros_like(predicted_power))
        hover_penalty = torch.where(
            airborne & (z_pos > 0.55) & (torch.abs(lin_vel_w[:, 2]) < 0.25),
            torch.ones_like(z_pos),
            torch.zeros_like(z_pos),
        )

        xy_error = self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2]
        distance_to_goal_xy = torch.linalg.norm(xy_error, dim=1)
        xy_error_penalty = torch.clamp(distance_to_goal_xy, max=1.0)
        xy_progress_reward = torch.clamp(self._prev_distance_to_goal_xy - distance_to_goal_xy, min=-0.1, max=0.1)
        self._prev_distance_to_goal_xy = distance_to_goal_xy.detach()
        goal_bonus = torch.where(
            distance_to_goal_xy < self.cfg.goal_tolerance,
            torch.ones_like(distance_to_goal_xy),
            torch.zeros_like(distance_to_goal_xy),
        )
        dist_reward_xy = torch.exp(
            -distance_to_goal_xy / max(self.cfg.xy_reward_width, 1.0e-6)
        )

        z_error = torch.abs(self._desired_pos_w[:, 2] - self._robot.data.root_pos_w[:, 2])
        dist_reward_z = torch.exp(-z_error / 0.4)

        gravity = abs(self.sim.cfg.gravity[2]) if self.sim.cfg.gravity is not None else 9.81
        target_height = torch.clamp(self._desired_pos_w[:, 2], min=0.0)
        vertical_energy = gravity * torch.clamp(z_pos, min=0.0) + 0.5 * torch.square(lin_vel_w[:, 2])
        target_vertical_energy = gravity * target_height
        energy_error = torch.abs(vertical_energy - target_vertical_energy)
        energy_tracking_reward = torch.exp(-energy_error / self.cfg.vertical_energy_error_scale)
        apex_reward = torch.exp(-z_error / 0.18) * torch.exp(-torch.abs(lin_vel_w[:, 2]) / 0.35)

        # 👑 滞地惩罚：如果在 0.3 米以下徘徊，狠狠扣分
        lazy_ground_penalty = torch.where(z_pos < 0.3, torch.ones_like(z_pos), torch.zeros_like(z_pos))

        _, _, yaw = euler_xyz_from_quat(q)
        yaw_penalty = yaw ** 2
        died_penalty = torch.logical_or(z_pos < 0.05, roll_pitch_sq > 0.5).float()

        rewards = {
            "survival": torch.ones_like(z_pos) * self.cfg.survival_reward_scale * self.step_dt,
            "distance_to_xy": dist_reward_xy * self.cfg.distance_to_xy_reward_scale * self.step_dt,
            "distance_to_z": dist_reward_z * self.cfg.distance_to_z_reward_scale * self.step_dt,
            "energy_tracking": energy_tracking_reward * self.cfg.energy_tracking_reward_scale * self.step_dt,
            "apex": apex_reward * self.cfg.apex_reward_scale * self.step_dt,
            "yaw_p": yaw_penalty * self.cfg.yaw_penalty_scale * self.step_dt,
            "power_penalty": power_penalty * self.cfg.power_penalty_scale * self.step_dt,
            "attitude_penalty": attitude_penalty * self.cfg.attitude_penalty_scale * self.step_dt,
            "angular_vel_p": angular_vel_penalty * self.cfg.angular_vel_penalty_scale * self.step_dt,
            "xy_error_p": xy_error_penalty * self.cfg.xy_error_penalty_scale * self.step_dt,
            "xy_progress": xy_progress_reward * self.cfg.xy_progress_reward_scale,
            "goal_bonus": goal_bonus * self.cfg.goal_bonus_scale * self.step_dt,
            "synergy": synergy * self.cfg.synergy_reward_scale * self.step_dt,
            "anti_synergy": anti_synergy * self.cfg.anti_synergy_penalty_scale * self.step_dt,
            "lateral_vel_p": lateral_vel_penalty * self.cfg.lateral_vel_penalty_scale * self.step_dt,
            "flight_power_p": flight_power_penalty * self.cfg.flight_power_penalty_scale * self.step_dt,
            "hover_p": hover_penalty * self.cfg.hover_penalty_scale * self.step_dt,
            "action_rate_p": action_rate_penalty * self.cfg.action_rate_reward_scale * self.step_dt,
            "termination_p": died_penalty * self.cfg.termination_penalty_scale,
            "lazy_ground_p": lazy_ground_penalty * self.cfg.lazy_ground_penalty_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        q = self._robot.data.root_quat_w
        roll_pitch_sq = q[:, 1] ** 2 + q[:, 2] ** 2
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.05, roll_pitch_sq > 0.5)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras

        self._flush_csv_rows()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        if self.num_envs == 1 or not self.cfg.randomize_dynamics:
            self.dr_mass_multi[env_ids] = 1.0
            self.dr_inertia_multi[env_ids] = 1.0
            self.dr_tm[env_ids] = self.cfg.play_motor_time_constant
        else:
            self.dr_mass_multi[env_ids] = torch.empty(num_resets, 1, device=self.device).uniform_(
                self.cfg.train_mass_multi_min,
                self.cfg.train_mass_multi_max,
            )
            self.dr_inertia_multi[env_ids] = torch.empty(num_resets, 3, device=self.device).uniform_(
                self.cfg.train_inertia_multi_min,
                self.cfg.train_inertia_multi_max,
            )
            self.dr_tm[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(
                self.cfg.train_motor_time_constant_min,
                self.cfg.train_motor_time_constant_max,
            )

        # Motor-response randomization is independent of randomize_dynamics,
        # mirroring the thrust-curve switch (sim2real fine-tunes hold mass/
        # inertia/delay fixed while covering the real motor bandwidth).
        if self.cfg.randomize_motor_time_constant:
            self.dr_tm[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(
                self.cfg.train_motor_time_constant_min,
                self.cfg.train_motor_time_constant_max,
            )

        if (
            self.num_envs == 1
            and not self.cfg.allow_single_env_thrust_rand
        ) or not self.cfg.randomize_thrust_curve:
            self.dr_thrust_scale[env_ids] = self.cfg.play_thrust_scale
            self.dr_thrust_curve_shape[env_ids] = self.cfg.play_thrust_curve_shape
        else:
            self.dr_thrust_scale[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(
                self.cfg.train_thrust_scale_min,
                self.cfg.train_thrust_scale_max,
            )
            self.dr_thrust_curve_shape[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(
                self.cfg.train_thrust_curve_shape_min,
                self.cfg.train_thrust_curve_shape_max,
            )

        self._actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        self._raw_target_u[env_ids] = 0.0
        self._shaped_target_u[env_ids] = 0.0
        self._apex_bias_pwm[env_ids] = self.cfg.apex_bias_initial_pwm
        history_reset_value = -1.0 if self.cfg.record_executed_action_history else 0.0
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = history_reset_value
            self._command_history[i][env_ids] = history_reset_value

        # =========================================================
        # 👑 重置时清空神经网络历史与电量
        # =========================================================
        self._motor_u_history[env_ids] = 0.0
        self._soc_tracker[env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]

        if self.num_envs == 1:
            play_target = torch.tensor(self.cfg.play_target_pos, device=self.device, dtype=default_root_state.dtype)
            play_initial = torch.tensor(self.cfg.play_initial_pos, device=self.device, dtype=default_root_state.dtype)
            self._desired_pos_w[env_ids] = play_target.unsqueeze(0)
            default_root_state[:, :3] = play_initial.unsqueeze(0)
        else:
            self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.empty(num_resets, device=self.device).uniform_(1.0, 1.5)

            total_train_steps = (
                float(self.num_envs)
                * float(self.cfg.curriculum_num_steps_per_env)
                * float(self.cfg.curriculum_max_iterations)
            )
            train_progress = float(getattr(self, "common_step_counter", 0)) / max(total_train_steps, 1.0)
            if train_progress < self.cfg.curriculum_vertical_fraction:
                curriculum = 0.0
            elif train_progress < self.cfg.curriculum_full_difficulty_fraction:
                curriculum = (
                    train_progress - self.cfg.curriculum_vertical_fraction
                ) / (self.cfg.curriculum_full_difficulty_fraction - self.cfg.curriculum_vertical_fraction)
                curriculum = max(0.0, min(1.0, curriculum))
            else:
                curriculum = torch.rand(1, device=self.device).item()

            angle = torch.empty(num_resets, device=self.device).uniform_(0.0, 2.0 * math.pi)
            radius = (
                self.cfg.curriculum_initial_xy_radius_max
                * curriculum
                * torch.sqrt(torch.rand(num_resets, device=self.device))
            )
            initial_xy_offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
            default_root_state[:, :2] = self._desired_pos_w[env_ids, :2] + initial_xy_offset

            roll = torch.empty(num_resets, device=self.device).uniform_(
                -self.cfg.curriculum_roll_pitch_max * curriculum,
                self.cfg.curriculum_roll_pitch_max * curriculum,
            )
            pitch = torch.empty(num_resets, device=self.device).uniform_(
                -self.cfg.curriculum_roll_pitch_max * curriculum,
                self.cfg.curriculum_roll_pitch_max * curriculum,
            )
            yaw = torch.empty(num_resets, device=self.device).uniform_(
                -self.cfg.curriculum_yaw_max * curriculum,
                self.cfg.curriculum_yaw_max * curriculum,
            )
            default_root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

            drop_height = torch.empty(num_resets, device=self.device).uniform_(
                self.cfg.train_drop_height_min,
                self.cfg.train_drop_height_max,
            )
            default_root_state[:, 2] = self._desired_pos_w[env_ids, 2] + drop_height

        default_root_state[:, 7:] = 0.0
        self._prev_distance_to_goal_xy[env_ids] = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - default_root_state[:, :2],
            dim=1,
        )

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    # =========================================================
    # 3. 目标点可视化逻辑
    # =========================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)
