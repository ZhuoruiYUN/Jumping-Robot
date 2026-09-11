import serial
import struct
import time
import math
import msvcrt
import threading
import numpy as np
import csv
from datetime import datetime
from collections import deque
from vicon_dssdk import ViconDataStream
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
from spatial52_stationary_observer import Spatial52StationaryObserver

COM_PORT = 'COM9'
BAUD_RATE = 115200
VICON_HOST = 'localhost:801'
# Export this exact checkpoint first with export_spatial_tracking_policy.py.
# This script accepts only the 52-D spatial LSTM and never applies the old
# 43-D semi-MDP correction layer.
TEACHER_POLICY_PATH = 'deployment/spatial_tracking_export_precision_zero40_v1/quadhopper_spatial_tracking_policy.onnx'
PLANNER_POLICY_PATH = ''
USE_SEMIMDP_PLANNER = False
USE_SHORT_ACTION_OFFSET = False

PAYLOAD_FORMAT = "<BBI6hffHHH"
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)
PACKET_SIZE = 2 + PAYLOAD_SIZE

# The 52-D policy was trained with simulated ground root-Z=0.380 m and
# apex command=1.000 m.  Vicon measures the physical root at 0.285 m on the
# ground.  Observation coordinates carry that 9.5 cm offset explicitly;
# physical safety limits use the corresponding physical apex height.
POLICY_GROUND_ROOT_Z = 0.380
MEASURED_GROUND_ROOT_Z = 0.285
POLICY_Z_OFFSET = POLICY_GROUND_ROOT_Z - MEASURED_GROUND_ROOT_Z
POLICY_TARGET_Z = 1.000
PHYSICAL_TARGET_Z = POLICY_TARGET_Z - POLICY_Z_OFFSET
TARGET_Z = PHYSICAL_TARGET_Z
STATIONARY_RADIUS = 0.0
SHORT_ACTION_OFFSET = np.array([0.24, 0.0, -0.12, 0.0], dtype=np.float32)

LANDING_ROOT_HEIGHT = MEASURED_GROUND_ROOT_Z
MAX_FLIGHT_DURATION = 2.0
NOMINAL_FORWARD_SPEED = 0.08
NOMINAL_FORWARD_TILT_RAD = math.radians(6.0)
MAX_FORWARD_SPEED_RESIDUAL = 0.12
MAX_LATERAL_SPEED = 0.10
MAX_FORWARD_TILT_RESIDUAL_RAD = math.radians(4.0)
MAX_LATERAL_TILT_RAD = math.radians(4.0)
VELOCITY_FEEDBACK_GAIN = 0.020
ATTITUDE_FEEDBACK_GAIN = 0.100
MOTOR_CORRECTION_LIMIT = 0.014
LANDING_CORRECTION_HEIGHT = 0.20
STANCE_PLAN_RATE = 0.10
HANDOFF_CONTACT_HOLD_S = 0.20
HANDOFF_MAX_XY_SPEED = 0.25
HANDOFF_MAX_Z_SPEED = 0.35
HANDOFF_MAX_TILT_RAD = math.radians(12.0)
SOFT_TILT_ABORT_RAD = math.radians(35.0)
MAX_TILT_ABORT_RAD = math.radians(60.0)
# 落地姿态保护：触地瞬间倾斜超过该角度即断油回 IDLE。
# 正常落地实测 5-11°，2026-09-02 失控跳落地时 15-19°，取 15° 可干净分开两者。
TOUCHDOWN_TILT_ABORT_DEG = 15.0

# The spatial policy was trained using the calibrated thrust curve and no
# post-policy gain.  Preserve raw policy actions in both the actuator path
# and five-sample LSTM history; independent hard limits remain below.
DEPLOY_ACTION_SCALE = 1.0
DEPLOY_DIFFERENTIAL_SCALE = 1.0  # 已关闭（差动缩放被实飞证伪）
MAX_PWM_SPREAD = 600.0
MAX_PWM_STEP_PER_POLICY_UPDATE = 180.0
HEIGHT_BRAKE_START_M = 0.03
HEIGHT_BRAKE_FULL_M = 0.23
HEIGHT_BRAKE_VZ_MIN = 0.05
HEIGHT_BRAKE_PWM_MEAN_SOFT = 720.0
HEIGHT_BRAKE_PWM_MEAN_HARD = 520.0
SOFT_TILT_BAD_TICKS_LIMIT = 3
HARD_TILT_BAD_TICKS_LIMIT = 2
AIRBORNE_TILT_SPREAD_START_DEG = 18.0
AIRBORNE_TILT_SPREAD_FULL_DEG = 35.0
AIRBORNE_TILT_MIN_PWM_SPREAD = 340.0
AIRBORNE_XY_SPREAD_START_M = 0.24
AIRBORNE_XY_SPREAD_FULL_M = 0.45
AIRBORNE_XY_MIN_PWM_SPREAD = 380.0
MAX_AIRBORNE_YAW_DIAGONAL_PWM_DIFF = 140.0
MAX_GROUNDED_YAW_DIAGONAL_PWM_DIFF = 220.0
# 松手下坠触发：手拿时 Vz≈0；真正松手后约 50-80 ms 内 Vz 会持续低于该阈值。
# 若操作者快速下放导致误触发，把阈值调低到 -0.8 或增大 FREE_FALL_TRIGGER_TICKS。
FREE_FALL_TRIGGER_VZ = -0.5
FREE_FALL_TRIGGER_TICKS = 5

MAX_PWM = 1000
MIN_PWM = 0
FUSION_HORIZON_S = 0.015
OUTPUT_PREDICTION_S = 0.010
IMU_STALE_TIMEOUT_S = 0.050
EXPECTED_IMU_SEQ_STEP = 2
IMU_SEQ_GAP_WARN = 3
LINK_HEALTH_BAD_SAMPLE_LIMIT = 3
FC_STATUS_IMU_OVERRUN = 1 << 0
FC_STATUS_CMD_TIMEOUT = 1 << 1
FC_STATUS_IMU_READ_FAIL = 1 << 2
FC_STATUS_BAD_RF_CMD = 1 << 3
FC_STATUS_INA260_WARN = 1 << 4
FC_STATUS_IMU_CLIPPING = 1 << 5
FC_STATUS_IMU_VIBRATION = 1 << 6
FC_STATUS_CRITICAL_MASK = FC_STATUS_CMD_TIMEOUT | FC_STATUS_IMU_READ_FAIL | FC_STATUS_BAD_RF_CMD
FC_STATUS_DEGRADED_MASK = FC_STATUS_IMU_CLIPPING | FC_STATUS_IMU_VIBRATION
INNOVATION_POS_GATE_M = 0.12
INNOVATION_VEL_GATE_MPS = 3.0
BIAS_LEARN_RATE = 0.002
INNOVATION_POS_VAR_M2 = 0.12 ** 2
INNOVATION_VEL_VAR_M2PS2 = 3.0 ** 2
IMU_ACCEL_NOISE = 0.35
IMU_GYRO_NOISE = math.radians(2.0)
ACCEL_BIAS_NOISE = 0.03
GYRO_BIAS_NOISE = math.radians(0.2)
VICON_POS_NOISE = 0.004
VICON_VEL_NOISE = 0.08
VICON_ATT_NOISE = math.radians(2.5)
VICON_HISTORY_MAXLEN = 256
VICON_VEL_REJECT_MPS = 6.0
IMU_DEGRADED_NOISE_SCALE = 8.0
ESTIMATOR_RESET_REJECT_LIMIT = 5
RESET_COOLDOWN_S = 0.5
ATTITUDE_GATE_RATIO_LIMIT = 1.0
OUTLIER_RESET_SUPPRESS_RATIO = 16.0
EST_FAULT_POS_INNOV = 1 << 0
EST_FAULT_VEL_INNOV = 1 << 1
EST_FAULT_RESET = 1 << 2
EST_FAULT_ATT_INNOV = 1 << 3
PPO_USE_EKF_POSITION = False
# 2026-09-01 两次实飞日志证实：每次落地冲击 IMU 加速度计削波 (FC flags 32/64)，
# EKF 错过弹跳冲量后整个上升段 Vz 反号（Z 在升、Vz 却 -4~-5 m/s），策略在空中
# 相位判断完全颠倒导致差动过大、逐跳漂移发散翻车。Vicon 速度带 40 Hz 低通且
# 方向正确，先用它切断这条因果链；等 EKF 削波处理做完再切回 True。
PPO_USE_EKF_VELOCITY = False
PPO_USE_OUTPUT_PREDICTION = False
PPO_USE_DELAYED_IMU = False
USE_PC_IMU_LPF = False
USE_VICON_VEL_LPF = True

# 【硬件参数对齐】跳跃机完全静止在地面时，Vicon测得的机身CoM高度(单位: 米)
# 如果你的起落架更换或者动捕球位置变动，请微调此数值
REST_LEG_LENGTH = 0.285


latest_vbat = 0.0
latest_current = 0.0

class PT1Filter:
    def __init__(self, cutoff_freq, dt):
        rc = 1.0 / (2.0 * math.pi * cutoff_freq)
        self.alpha = dt / (rc + dt)
        self.state = None

    def apply(self, sample):
        if self.state is None:
            self.state = np.copy(sample)
        else:
            self.state = self.state + self.alpha * (sample - self.state)
        return self.state


class PosVelKF:
    def __init__(self, init_pos):
        self.X = np.array([[init_pos], [0.0]])
        self.P = np.eye(2) * 0.1
        self.Q = np.array([[1e-4, 0.0], [0.0, 5e-2]])

    def predict(self, dt, accel):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5 * dt ** 2], [dt]])
        self.X = F @ self.X + B * accel
        self.P = F @ self.P @ F.T + self.Q
        return self.X[0, 0], self.X[1, 0]

    def update(self, pos_meas, vel_meas=None):
        if vel_meas is not None:
            H = np.array([[1.0, 0.0], [0.0, 1.0]])
            y = np.array([[pos_meas - self.X[0, 0]], [vel_meas - self.X[1, 0]]])
            R_noise = np.array([[1e-4, 0.0], [0.0, 2e-2]])
        else:
            H = np.array([[1.0, 0.0]])
            y = np.array([[pos_meas - self.X[0, 0]]])
            R_noise = np.array([[1e-4]])
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

    def reset_to_measurement(self, pos_meas, vel_meas=0.0):
        self.X = np.array([[pos_meas], [vel_meas]])
        self.P = np.eye(2) * 0.1


class ErrorStateKalmanFilter:
    def __init__(self, init_pos, init_quat=None):
        self.pos = np.array(init_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if init_quat is not None:
            self.quat = self._normalize_quat(init_quat)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.P = np.eye(15, dtype=np.float64) * 0.05

    def state_vector(self):
        return np.concatenate([self.pos, self.vel, self.accel_bias, self.gyro_bias])

    def set_state_vector(self, state):
        state = np.array(state, dtype=np.float64).reshape(12)
        self.pos = state[0:3]
        self.vel = state[3:6]
        self.accel_bias = state[6:9]
        self.gyro_bias = state[9:12]

    def clone(self):
        other = ErrorStateKalmanFilter(self.pos, self.quat)
        other.vel = self.vel.copy()
        other.accel_bias = self.accel_bias.copy()
        other.gyro_bias = self.gyro_bias.copy()
        other.P = self.P.copy()
        return other

    def copy_from(self, other):
        self.pos = other.pos.copy()
        self.vel = other.vel.copy()
        self.quat = other.quat.copy()
        self.accel_bias = other.accel_bias.copy()
        self.gyro_bias = other.gyro_bias.copy()
        self.P = other.P.copy()

    @staticmethod
    def _normalize_quat(quat):
        quat = np.array(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    @staticmethod
    def _skew(vec):
        x, y, z = np.array(vec, dtype=np.float64)
        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ], dtype=np.float64)

    @staticmethod
    def small_angle_quat(delta_theta):
        delta_theta = np.array(delta_theta, dtype=np.float64)
        angle = np.linalg.norm(delta_theta)
        if angle < 1e-9:
            return np.array([0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2], 1.0])
        return R.from_rotvec(delta_theta).as_quat()

    def inject_error_state(self, delta_x):
        delta_x = np.array(delta_x, dtype=np.float64).reshape(15)
        delta_theta = delta_x[6:9]
        self.pos += delta_x[0:3]
        self.vel += delta_x[3:6]
        self.quat = (R.from_quat(self.quat) * R.from_quat(self.small_angle_quat(delta_theta))).as_quat()
        self.quat = self._normalize_quat(self.quat)
        self.accel_bias += delta_x[9:12]
        self.gyro_bias += delta_x[12:15]

    def process_noise(self, dt, imu_degraded=False):
        scale = IMU_DEGRADED_NOISE_SCALE if imu_degraded else 1.0
        accel_var = (IMU_ACCEL_NOISE * scale) ** 2
        gyro_var = (IMU_GYRO_NOISE * scale) ** 2
        accel_bias_var = ACCEL_BIAS_NOISE ** 2
        gyro_bias_var = GYRO_BIAS_NOISE ** 2

        q = np.zeros((15, 15), dtype=np.float64)
        q[0:3, 0:3] = np.eye(3) * (0.25 * dt ** 4 * accel_var)
        q[0:3, 3:6] = np.eye(3) * (0.5 * dt ** 3 * accel_var)
        q[3:6, 0:3] = q[0:3, 3:6]
        q[3:6, 3:6] = np.eye(3) * (dt ** 2 * accel_var)
        q[6:9, 6:9] = np.eye(3) * (dt ** 2 * gyro_var)
        q[9:12, 9:12] = np.eye(3) * (dt * accel_bias_var)
        q[12:15, 12:15] = np.eye(3) * (dt * gyro_bias_var)
        return q

    def predict(self, dt, accel_b, gyro_b=None, imu_degraded=False):
        dt = float(np.clip(dt, 0.001, 0.05))
        accel_b = np.array(accel_b, dtype=np.float64)
        gyro_b = np.zeros(3, dtype=np.float64) if gyro_b is None else np.array(gyro_b, dtype=np.float64)

        corrected_gyro = gyro_b - self.gyro_bias
        delta_angle = corrected_gyro * dt
        self.quat = (R.from_quat(self.quat) * R.from_rotvec(delta_angle)).as_quat()
        self.quat = self._normalize_quat(self.quat)

        corrected_accel_b = accel_b - self.accel_bias
        r_body_to_world = R.from_quat(self.quat)
        accel_w = r_body_to_world.apply(corrected_accel_b) - np.array([0.0, 0.0, 9.81])
        if imu_degraded:
            accel_w = np.zeros(3)

        self.pos = self.pos + self.vel * dt + 0.5 * accel_w * dt * dt
        self.vel = self.vel + accel_w * dt

        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        accel_skew = self._skew(r_body_to_world.apply(corrected_accel_b))
        F[0:3, 6:9] = -0.5 * accel_skew * dt * dt
        F[3:6, 6:9] = -accel_skew * dt
        F[0:3, 9:12] = -0.5 * r_body_to_world.as_matrix() * dt * dt
        F[3:6, 9:12] = -r_body_to_world.as_matrix() * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        self.P = F @ self.P @ F.T + self.process_noise(dt, imu_degraded=imu_degraded)
        self.P = 0.5 * (self.P + self.P.T)
        return self.pos.copy(), self.vel.copy()

    def fuse_vicon(self, measured_pos, measured_vel, measured_quat=None,
                   pos_noise_scale=1.0, vel_noise_scale=1.0, att_noise_scale=1.0,
                   gate_scale=1.0):
        measured_pos = np.array(measured_pos, dtype=np.float64)
        measured_vel = np.array(measured_vel, dtype=np.float64)
        if measured_quat is None:
            measured_quat = self.quat.copy()
        measured_quat = self._normalize_quat(measured_quat)
        if np.dot(self.quat, measured_quat) < 0.0:
            measured_quat = -measured_quat
        attitude_innov = (R.from_quat(self.quat).inv() * R.from_quat(measured_quat)).as_rotvec()
        z = np.concatenate([
            measured_pos,
            measured_vel,
            attitude_innov,
        ])
        h_x = np.concatenate([self.pos, self.vel, np.zeros(3)])
        H = np.zeros((9, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        H[6:9, 6:9] = np.eye(3)

        R_noise = np.diag([
            *(np.ones(3) * (VICON_POS_NOISE * pos_noise_scale) ** 2),
            *(np.ones(3) * (VICON_VEL_NOISE * vel_noise_scale) ** 2),
            *(np.ones(3) * (VICON_ATT_NOISE * att_noise_scale) ** 2),
        ])
        innovation = z - h_x
        S = H @ self.P @ H.T + R_noise
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)

        S_pos = S[0:3, 0:3]
        S_vel = S[3:6, 3:6]
        S_att = S[6:9, 6:9]
        pos_innov = innovation[0:3]
        vel_innov = innovation[3:6]
        att_innov = innovation[6:9]
        pos_test_ratio = float(pos_innov.T @ np.linalg.inv(S_pos) @ pos_innov) / (9.0 * gate_scale)
        vel_test_ratio = float(vel_innov.T @ np.linalg.inv(S_vel) @ vel_innov) / (9.0 * gate_scale)
        att_test_ratio = float(att_innov.T @ np.linalg.inv(S_att) @ att_innov) / (9.0 * gate_scale)
        accepted = (
            pos_test_ratio <= 1.0 and
            vel_test_ratio <= 1.0 and
            att_test_ratio <= ATTITUDE_GATE_RATIO_LIMIT
        )
        if not accepted:
            return False, pos_test_ratio, vel_test_ratio, att_test_ratio, nis

        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = K @ innovation
        self.inject_error_state(delta_x)
        I = np.eye(15, dtype=np.float64)
        joseph = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_noise @ K.T
        self.P = 0.5 * (joseph + joseph.T)
        return True, pos_test_ratio, vel_test_ratio, att_test_ratio, nis

    def reset_to_measurement(self, pos_meas, vel_meas=None, quat_meas=None):
        self.pos = np.array(pos_meas, dtype=np.float64)
        if vel_meas is not None:
            self.vel = np.array(vel_meas, dtype=np.float64)
        if quat_meas is not None:
            self.quat = self._normalize_quat(quat_meas)
        self.P = np.eye(15, dtype=np.float64) * 0.05

    def inflate_covariance(self, scale):
        self.P = np.clip(scale, 1.0, 100.0) * self.P
        self.P = 0.5 * (self.P + self.P.T)


class IMUReplayBuffer:
    def __init__(self, max_samples=256):
        self.samples = deque(maxlen=max_samples)

    def add(self, fc_timestamp_ms, accel_b, gyro_b, pc_rx_time, degraded=False):
        self.samples.append({
            "fc_time_s": float(fc_timestamp_ms) * 0.001,
            "pc_time_s": float(pc_rx_time),
            "accel_b": np.array(accel_b, dtype=np.float64),
            "gyro_b": np.array(gyro_b, dtype=np.float64),
            "degraded": bool(degraded),
        })

    def sample_at(self, target_pc_time):
        if not self.samples:
            return None
        chosen = self.samples[0]
        for sample in self.samples:
            if sample["pc_time_s"] <= target_pc_time:
                chosen = sample
            else:
                break
        return chosen

    def latest(self):
        return self.samples[-1] if self.samples else None

    def age(self, now):
        latest = self.latest()
        if latest is None:
            return float("inf")
        return now - latest["pc_time_s"]


class EstimatorReplayBuffer:
    def __init__(self, max_states=256):
        self.states = deque(maxlen=max_states)

    def save_state(self, timestamp, eskf):
        self.states.append({
            "pc_time_s": float(timestamp),
            "eskf": eskf.clone(),
        })

    def find_state_before(self, target_pc_time):
        if not self.states:
            return None
        if self.states[0]["pc_time_s"] > target_pc_time:
            return None
        chosen = self.states[0]
        for state in self.states:
            if state["pc_time_s"] <= target_pc_time:
                chosen = state
            else:
                break
        return chosen

    def discard_after(self, target_pc_time):
        kept_states = [state for state in self.states if state["pc_time_s"] <= target_pc_time]
        self.states = deque(kept_states, maxlen=self.states.maxlen)

    def replay_to(self, eskf, start_time, end_time, imu_samples):
        last_time = float(start_time)
        for sample in imu_samples:
            sample_time = sample["pc_time_s"]
            if sample_time <= start_time or sample_time > end_time:
                continue
            dt = sample_time - last_time
            eskf.predict(
                dt,
                sample["accel_b"],
                sample["gyro_b"],
                imu_degraded=sample["degraded"],
            )
            last_time = sample_time
        return last_time


class BiasEstimator:
    def __init__(self):
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def set_biases(self, accel_bias, gyro_bias):
        self.accel_bias = np.array(accel_bias, dtype=np.float64)
        self.gyro_bias = np.array(gyro_bias, dtype=np.float64)

    def correct_accel(self, accel_b):
        return np.array(accel_b, dtype=np.float64) - self.accel_bias

    def correct_gyro(self, gyro_b):
        return np.array(gyro_b, dtype=np.float64) - self.gyro_bias

    def observe_static(self, accel_b, gyro_b, expected_gravity_b, enabled):
        if not enabled:
            return
        if np.linalg.norm(gyro_b - self.gyro_bias) > 0.35:
            return
        accel_residual = np.array(accel_b) - np.array(expected_gravity_b)
        self.accel_bias = (1.0 - BIAS_LEARN_RATE) * self.accel_bias + BIAS_LEARN_RATE * accel_residual
        self.gyro_bias = (1.0 - BIAS_LEARN_RATE) * self.gyro_bias + BIAS_LEARN_RATE * np.array(gyro_b)


class InnovationGate:
    def __init__(self, pos_gate_m, vel_gate_mps):
        self.pos_gate_m = pos_gate_m
        self.vel_gate_mps = vel_gate_mps
        self.accepted = 0
        self.rejected = 0

    def accept(self, predicted_pos, measured_pos, predicted_vel, measured_vel):
        pos_innov = np.linalg.norm(np.array(measured_pos) - np.array(predicted_pos))
        vel_innov = np.linalg.norm(np.array(measured_vel) - np.array(predicted_vel))
        if pos_innov <= self.pos_gate_m and vel_innov <= self.vel_gate_mps:
            self.accepted += 1
            return True
        self.rejected += 1
        return False


class EstimatorManager:
    def __init__(self, pos_gate_m, vel_gate_mps):
        self.innovation_gate = InnovationGate(pos_gate_m, vel_gate_mps)
        self.pos_test_ratio = 0.0
        self.vel_test_ratio = 0.0
        self.att_test_ratio = 0.0
        self.estimator_fault_flags = 0
        self.estimator_reset_count = 0
        self.consecutive_rejects = 0
        self.last_reset_time = -1.0e9
        self.reset_event_latch = 0
        self.output_tracking_error = np.zeros(3)

    def fuse_vicon(self, kf_px, kf_py, kf_pz, predicted_pos, measured_pos, predicted_vel, measured_vel):
        pos_innov = np.array(measured_pos) - np.array(predicted_pos)
        vel_innov = np.array(measured_vel) - np.array(predicted_vel)
        pos_innov_norm = float(np.linalg.norm(pos_innov))
        vel_innov_norm = float(np.linalg.norm(vel_innov))

        self.pos_test_ratio = (pos_innov_norm * pos_innov_norm) / INNOVATION_POS_VAR_M2
        self.vel_test_ratio = (vel_innov_norm * vel_innov_norm) / INNOVATION_VEL_VAR_M2PS2
        self.estimator_fault_flags &= ~(EST_FAULT_POS_INNOV | EST_FAULT_VEL_INNOV)

        if self.pos_test_ratio > 1.0:
            self.estimator_fault_flags |= EST_FAULT_POS_INNOV
        if self.vel_test_ratio > 1.0:
            self.estimator_fault_flags |= EST_FAULT_VEL_INNOV

        accepted = self.innovation_gate.accept(predicted_pos, measured_pos, predicted_vel, measured_vel)
        if accepted:
            self.consecutive_rejects = 0
            kf_px.update(measured_pos[0], measured_vel[0])
            kf_py.update(measured_pos[1], measured_vel[1])
            kf_pz.update(measured_pos[2], measured_vel[2])
            return True

        self.consecutive_rejects += 1
        if self.consecutive_rejects >= ESTIMATOR_RESET_REJECT_LIMIT:
            kf_px.reset_to_measurement(measured_pos[0], measured_vel[0])
            kf_py.reset_to_measurement(measured_pos[1], measured_vel[1])
            kf_pz.reset_to_measurement(measured_pos[2], measured_vel[2])
            self.estimator_reset_count += 1
            self.estimator_fault_flags |= EST_FAULT_RESET
            self.consecutive_rejects = 0
        return False

    def fuse_vicon_covariance(self, eskf, measured_pos, measured_vel, measured_quat=None,
                              imu_degraded=False, now_s=0.0):
        self.estimator_fault_flags = 0
        self.reset_event_latch = 0
        noise_scale = IMU_DEGRADED_NOISE_SCALE if imu_degraded else 1.0
        gate_scale = 4.0 if imu_degraded else 1.0
        accepted, self.pos_test_ratio, self.vel_test_ratio, self.att_test_ratio, _ = eskf.fuse_vicon(
            measured_pos,
            measured_vel,
            measured_quat=measured_quat,
            pos_noise_scale=1.0,
            vel_noise_scale=noise_scale,
            att_noise_scale=noise_scale,
            gate_scale=gate_scale,
        )

        if self.pos_test_ratio > 1.0:
            self.estimator_fault_flags |= EST_FAULT_POS_INNOV
        if self.vel_test_ratio > 1.0:
            self.estimator_fault_flags |= EST_FAULT_VEL_INNOV
        if self.att_test_ratio > ATTITUDE_GATE_RATIO_LIMIT:
            self.estimator_fault_flags |= EST_FAULT_ATT_INNOV

        if accepted:
            self.innovation_gate.accepted += 1
            self.consecutive_rejects = 0
            return True

        self.innovation_gate.rejected += 1
        self.consecutive_rejects += 1
        if imu_degraded:
            eskf.inflate_covariance(1.08)
            return False

        if max(self.pos_test_ratio, self.vel_test_ratio) > OUTLIER_RESET_SUPPRESS_RATIO:
            eskf.inflate_covariance(1.25)
            self.consecutive_rejects = min(
                self.consecutive_rejects,
                ESTIMATOR_RESET_REJECT_LIMIT - 1
            )
            return False

        if self.consecutive_rejects >= ESTIMATOR_RESET_REJECT_LIMIT:
            if now_s - self.last_reset_time >= RESET_COOLDOWN_S:
                eskf.reset_to_measurement(measured_pos, measured_vel, measured_quat)
                self.estimator_reset_count += 1
                self.last_reset_time = now_s
                self.reset_event_latch = 1
                self.estimator_fault_flags |= EST_FAULT_RESET
            else:
                eskf.inflate_covariance(1.15)
            self.consecutive_rejects = 0
        return False

    def update_output_tracking_error(self, reference_gyro_b, output_gyro_b, reference_vel_w, output_vel_w, reference_pos_w, output_pos_w):
        self.output_tracking_error = np.array([
            float(np.linalg.norm(np.array(output_gyro_b) - np.array(reference_gyro_b))),
            float(np.linalg.norm(np.array(output_vel_w) - np.array(reference_vel_w))),
            float(np.linalg.norm(np.array(output_pos_w) - np.array(reference_pos_w))),
        ])
        return self.output_tracking_error


class OutputPredictor:
    def __init__(self):
        self.timestamp = None
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)

    def update(self, timestamp, pos, vel):
        self.timestamp = timestamp
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)

    def predict(self, timestamp, accel_w):
        if self.timestamp is None:
            return self.pos.copy(), self.vel.copy()
        dt = np.clip(timestamp - self.timestamp, 0.0, 0.05)
        accel_w = np.array(accel_w, dtype=np.float64)
        pos = self.pos + self.vel * dt + 0.5 * accel_w * dt * dt
        vel = self.vel + accel_w * dt
        return pos, vel


class ViconReceiver:
    def __init__(self, vicon_host):
        self.running = True
        self.connected = False
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.zeros(3)
        self.frame_time_s = time.perf_counter()
        self.frame_history = deque(maxlen=VICON_HISTORY_MAXLEN)
        self.error_count = 0
        self.last_error = ""
        self.new_frame_event = threading.Event()
        self.client = ViconDataStream.Client()
        self.vicon_host = vicon_host
        self._init_vicon()
        if self.connected:
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True
            self.thread.start()

    def _init_vicon(self):
        try:
            self.client.Connect(self.vicon_host)
            self.client.EnableSegmentData()
            self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
            self.client.SetBufferSize(1)
            self.connected = True
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)

    def _store_frame(self, frame_time_s):
        self.frame_history.append({
            "time_s": frame_time_s,
            "pos": self.pos.copy(),
            "quat": self.quat.copy(),
            "vel": self.vel.copy(),
        })

    def latest_frame(self):
        if not self.frame_history:
            return None
        latest = self.frame_history[-1]
        return {
            "time_s": latest["time_s"],
            "pos": latest["pos"].copy(),
            "quat": latest["quat"].copy(),
            "vel": latest["vel"].copy(),
        }

    def _update_loop(self):
        first_frame = True
        while self.running and self.connected:
            try:
                if self.client.GetFrame():
                    frame_time_s = time.perf_counter()
                    subj = self.client.GetSubjectNames()[0]
                    seg = self.client.GetSegmentNames(subj)[0]
                    rot_data, rot_occ = self.client.GetSegmentGlobalRotationQuaternion(subj, seg)
                    trans_data, trans_occ = self.client.GetSegmentGlobalTranslation(subj, seg)
                    if not trans_occ and not rot_occ:
                        rq = np.array(rot_data)
                        rq_norm = np.linalg.norm(rq)
                        if rq_norm < 1e-9:
                            continue
                        rq = rq / rq_norm
                        rp = np.array([trans_data[0] / 1000.0, trans_data[1] / 1000.0, trans_data[2] / 1000.0])
                        if first_frame:
                            self.quat = rq
                            self.pos = rp
                            self.vel = np.zeros(3)
                            self.frame_time_s = frame_time_s
                            self._store_frame(frame_time_s)
                            first_frame = False
                            self.new_frame_event.set()
                        else:
                            if np.linalg.norm(rp - self.pos) < 0.2:
                                if np.dot(self.quat, rq) < 0.0: rq = -rq
                                dt = frame_time_s - self.frame_time_s
                                if 0.001 <= dt <= 0.05:
                                    new_vel = (rp - self.pos) / dt
                                else:
                                    new_vel = np.zeros(3)
                                if np.linalg.norm(new_vel) > VICON_VEL_REJECT_MPS:
                                    new_vel = np.zeros(3)
                                self.vel = new_vel
                                self.quat = rq
                                self.pos = rp
                                self.frame_time_s = frame_time_s
                                self._store_frame(frame_time_s)
                                self.new_frame_event.set()
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
            time.sleep(0.001)

    def close(self):
        self.running = False


class StatefulOnnxPolicy:
    def __init__(self, path, expected_obs_dim):
        self.session = ort.InferenceSession(path)
        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.obs_name = next(iter(self.inputs))
        self.expected_obs_dim = expected_obs_dim
        self.h = None
        self.c = None
        if "h_in" in self.inputs and "c_in" in self.inputs:
            self.h = np.zeros(tuple(self.inputs["h_in"].shape), dtype=np.float32)
            self.c = np.zeros(tuple(self.inputs["c_in"].shape), dtype=np.float32)

    def reset(self):
        if self.h is not None:
            self.h.fill(0.0)
            self.c.fill(0.0)

    def step(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, self.expected_obs_dim)
        feed = {self.obs_name: obs}
        if self.h is not None:
            feed["h_in"] = self.h
            feed["c_in"] = self.c
        outputs = self.session.run(None, feed)
        action = outputs[0][0].astype(np.float32)
        if self.h is not None and len(outputs) >= 3:
            self.h = outputs[1]
            self.c = outputs[2]
        return np.clip(action, -1.0, 1.0)


class StationaryWaypointQueue:
    def __init__(self):
        self.anchor_xy = np.zeros(2, dtype=np.float32)
        self.targets_xy = np.zeros((2, 2), dtype=np.float32)
        self.route_index = 0
        self.first_hop_hit_for_pair = 0.0

    def lock_here(self, xy):
        self.anchor_xy = np.asarray(xy, dtype=np.float32).copy()
        self.targets_xy[0] = self.anchor_xy + np.array([STATIONARY_RADIUS, 0.0], dtype=np.float32)
        self.targets_xy[1] = self.targets_xy[0]
        self.route_index = 0
        self.first_hop_hit_for_pair = 0.0

    def advance_after_touchdown(self, landed_xy, hit):
        is_short = self.is_short()
        if is_short:
            self.first_hop_hit_for_pair = 1.0 if hit else 0.0
        else:
            self.first_hop_hit_for_pair = 0.0
        self.anchor_xy = self.targets_xy[0].copy()
        self.targets_xy[0] = self.targets_xy[1].copy()
        self.targets_xy[1] = self.targets_xy[0].copy()
        self.route_index += 1

    def is_short(self):
        return (self.route_index % 2) == 0

    def pair_context(self):
        short = 1.0 if self.is_short() else 0.0
        return np.array([short, 1.0 - short, self.first_hop_hit_for_pair], dtype=np.float32)


def yaw_world_to_body_xy(vector_xy_w, quat_xyzw):
    yaw = R.from_quat(quat_xyzw).as_euler("zyx", degrees=False)[0]
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = vector_xy_w[0], vector_xy_w[1]
    return np.array([c * x + s * y, -s * x + c * y], dtype=np.float32)


def body_z_axis_xy_in_yaw_body(quat_xyzw):
    axis_w = R.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
    return yaw_world_to_body_xy(axis_w[:2], quat_xyzw)


def handoff_ready(pos_w, vel_w, quat_xyzw, is_contact):
    if is_contact <= 0.5:
        return False
    xy_speed = float(np.linalg.norm(vel_w[:2]))
    z_speed = abs(float(vel_w[2]))
    body_z_w = R.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
    tilt = math.acos(float(np.clip(body_z_w[2], -1.0, 1.0)))
    return (
        xy_speed <= HANDOFF_MAX_XY_SPEED
        and z_speed <= HANDOFF_MAX_Z_SPEED
        and tilt <= HANDOFF_MAX_TILT_RAD
    )


def body_tilt_rad(quat_xyzw):
    body_z_w = R.from_quat(quat_xyzw).apply([0.0, 0.0, 1.0])
    return math.acos(float(np.clip(body_z_w[2], -1.0, 1.0)))


def limit_pwm_spread(pwm_cmd, max_spread=None):
    pwm_cmd = np.clip(np.asarray(pwm_cmd, dtype=np.float32), MIN_PWM, MAX_PWM)
    spread_limit = MAX_PWM_SPREAD if max_spread is None else float(max_spread)
    if spread_limit <= 0.0:
        return pwm_cmd
    center = float(np.mean(pwm_cmd))
    half_spread = 0.5 * spread_limit
    limited = np.clip(pwm_cmd, center - half_spread, center + half_spread)
    return np.clip(limited, MIN_PWM, MAX_PWM)


def limit_yaw_diagonal_pwm(pwm_cmd, is_contact):
    pwm_cmd = np.clip(np.asarray(pwm_cmd, dtype=np.float32), MIN_PWM, MAX_PWM)
    max_diag_diff = (
        MAX_GROUNDED_YAW_DIAGONAL_PWM_DIFF
        if is_contact > 0.5
        else MAX_AIRBORNE_YAW_DIAGONAL_PWM_DIFF
    )
    diag_13 = 0.5 * float(pwm_cmd[0] + pwm_cmd[2])
    diag_24 = 0.5 * float(pwm_cmd[1] + pwm_cmd[3])
    diff = diag_13 - diag_24
    excess = abs(diff) - max_diag_diff
    if excess <= 0.0:
        return pwm_cmd

    correction = 0.5 * math.copysign(excess, diff)
    limited = pwm_cmd.copy()
    limited[[0, 2]] -= correction
    limited[[1, 3]] += correction
    return np.clip(limited, MIN_PWM, MAX_PWM)


def adaptive_airborne_pwm_spread_limit(tilt_deg, xy_error_m, is_contact):
    if is_contact > 0.5:
        return MAX_PWM_SPREAD

    limit = MAX_PWM_SPREAD
    tilt_alpha = np.clip(
        (tilt_deg - AIRBORNE_TILT_SPREAD_START_DEG)
        / max(AIRBORNE_TILT_SPREAD_FULL_DEG - AIRBORNE_TILT_SPREAD_START_DEG, 1.0e-6),
        0.0,
        1.0,
    )
    limit = min(
        limit,
        (1.0 - tilt_alpha) * MAX_PWM_SPREAD
        + tilt_alpha * AIRBORNE_TILT_MIN_PWM_SPREAD,
    )

    xy_alpha = np.clip(
        (xy_error_m - AIRBORNE_XY_SPREAD_START_M)
        / max(AIRBORNE_XY_SPREAD_FULL_M - AIRBORNE_XY_SPREAD_START_M, 1.0e-6),
        0.0,
        1.0,
    )
    limit = min(
        limit,
        (1.0 - xy_alpha) * MAX_PWM_SPREAD
        + xy_alpha * AIRBORNE_XY_MIN_PWM_SPREAD,
    )
    return limit


def limit_collective_for_height(pwm_cmd, pos_z, vel_z):
    pwm_cmd = np.clip(np.asarray(pwm_cmd, dtype=np.float32), MIN_PWM, MAX_PWM)
    if pos_z <= TARGET_Z + HEIGHT_BRAKE_START_M or vel_z <= HEIGHT_BRAKE_VZ_MIN:
        return pwm_cmd

    overshoot_span = max(HEIGHT_BRAKE_FULL_M - HEIGHT_BRAKE_START_M, 1.0e-6)
    brake = np.clip((pos_z - TARGET_Z - HEIGHT_BRAKE_START_M) / overshoot_span, 0.0, 1.0)
    max_mean = (
        (1.0 - brake) * HEIGHT_BRAKE_PWM_MEAN_SOFT
        + brake * HEIGHT_BRAKE_PWM_MEAN_HARD
    )
    mean_pwm = float(np.mean(pwm_cmd))
    if mean_pwm <= max_mean:
        return pwm_cmd
    return np.clip(pwm_cmd - (mean_pwm - max_mean), MIN_PWM, MAX_PWM)


def limit_pwm_slew(pwm_cmd, previous_pwm_cmd):
    pwm_cmd = np.clip(np.asarray(pwm_cmd, dtype=np.float32), MIN_PWM, MAX_PWM)
    if previous_pwm_cmd is None or MAX_PWM_STEP_PER_POLICY_UPDATE <= 0.0:
        return pwm_cmd
    previous_pwm_cmd = np.asarray(previous_pwm_cmd, dtype=np.float32)
    limited = np.clip(
        pwm_cmd,
        previous_pwm_cmd - MAX_PWM_STEP_PER_POLICY_UPDATE,
        previous_pwm_cmd + MAX_PWM_STEP_PER_POLICY_UPDATE,
    )
    return np.clip(limited, MIN_PWM, MAX_PWM)


def decode_planner_command(action):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.array([
        NOMINAL_FORWARD_SPEED + action[0] * MAX_FORWARD_SPEED_RESIDUAL,
        action[1] * MAX_LATERAL_SPEED,
        NOMINAL_FORWARD_TILT_RAD + action[2] * MAX_FORWARD_TILT_RESIDUAL_RAD,
        action[3] * MAX_LATERAL_TILT_RAD,
    ], dtype=np.float32)


def build_planner_observation(
    teacher_obs,
    teacher_action,
    pos_w,
    vel_w,
    quat_xyzw,
    joint_pos,
    joint_vel,
    is_contact,
    cycle_active,
    latched_command,
    queue,
):
    p_t = queue.targets_xy[0]
    p_t1 = queue.targets_xy[1]
    height = max(0.0, float(pos_w[2] - LANDING_ROOT_HEIGHT))
    vz = float(vel_w[2])
    ttl = (vz + math.sqrt(max(0.0, vz * vz + 2.0 * 9.81 * height))) / 9.81
    projected_xy = pos_w[:2] + vel_w[:2] * ttl
    projected_error_b = np.clip(yaw_world_to_body_xy(p_t - projected_xy, quat_xyzw) / 0.25, -4.0, 4.0)
    vxy_b = yaw_world_to_body_xy(vel_w[:2], quat_xyzw)
    next_w = p_t1 - p_t
    norm = max(1.0e-6, float(np.linalg.norm(next_w)))
    next_b = yaw_world_to_body_xy(next_w / norm, quat_xyzw)
    axis_b = body_z_axis_xy_in_yaw_body(quat_xyzw)
    contact = bool(is_contact > 0.5)
    active = bool(cycle_active)
    phase = np.array([
        1.0 if contact else 0.0,
        1.0 if ((not contact) and (not active)) else 0.0,
        1.0 if ((not contact) and active and vz >= 0.0) else 0.0,
        1.0 if ((not contact) and active and vz < 0.0) else 0.0,
    ], dtype=np.float32)
    return np.concatenate([
        teacher_obs,
        teacher_action,
        phase,
        projected_error_b,
        np.array([ttl / MAX_FLIGHT_DURATION], dtype=np.float32),
        vxy_b,
        next_b,
        axis_b,
        np.array([joint_pos, joint_vel], dtype=np.float32),
        latched_command,
        queue.pair_context(),
    ]).astype(np.float32)


def motor_correction(pos_w, vel_w, quat_xyzw, joint_pos, cycle_active, latched_command, queue):
    if MOTOR_CORRECTION_LIMIT <= 0.0:
        return np.zeros(4, dtype=np.float32)
    descending = bool(cycle_active) and vel_w[2] < 0.0
    near_ground = pos_w[2] < LANDING_ROOT_HEIGHT + LANDING_CORRECTION_HEIGHT
    spring_contact = joint_pos > 0.002
    active_mask = (descending and near_ground) or ((not bool(cycle_active)) and spring_contact)
    if not active_mask:
        return np.zeros(4, dtype=np.float32)
    next_w = queue.targets_xy[1] - queue.targets_xy[0]
    norm = max(1.0e-6, float(np.linalg.norm(next_w)))
    next_b = yaw_world_to_body_xy(next_w / norm, quat_xyzw)
    lateral_b = np.array([-next_b[1], next_b[0]], dtype=np.float32)
    forward, lateral, tilt_forward, tilt_lateral = latched_command
    desired_v_b = forward * next_b + lateral * lateral_b
    desired_axis_b = tilt_forward * next_b + tilt_lateral * lateral_b
    vxy_b = yaw_world_to_body_xy(vel_w[:2], quat_xyzw)
    axis_b = body_z_axis_xy_in_yaw_body(quat_xyzw)
    desired_horizontal_axis = (
        VELOCITY_FEEDBACK_GAIN * (desired_v_b - vxy_b)
        + ATTITUDE_FEEDBACK_GAIN * (desired_axis_b - axis_b)
    )
    roll = -desired_horizontal_axis[1]
    pitch = desired_horizontal_axis[0]
    correction = np.array([roll + pitch, -roll + pitch, -roll - pitch, roll - pitch], dtype=np.float32)
    return np.clip(correction, -MOTOR_CORRECTION_LIMIT, MOTOR_CORRECTION_LIMIT)


def run_autonomous_flight():
    try:
        teacher_policy = StatefulOnnxPolicy(TEACHER_POLICY_PATH, expected_obs_dim=52)
        planner_policy = (
            StatefulOnnxPolicy(PLANNER_POLICY_PATH, expected_obs_dim=69)
            if USE_SEMIMDP_PLANNER
            else None
        )
        print(f"✅ 成功加载 teacher PPO 模型: {TEACHER_POLICY_PATH}")
        if USE_SEMIMDP_PLANNER:
            print(f"✅ 成功加载 stationary Semi-MDP 模型: {PLANNER_POLICY_PATH}")
        else:
            print("✅ 当前为 teacher-only 对照模式：不启用 Semi-MDP landing correction")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    vicon = ViconReceiver(VICON_HOST)
    if not vicon.connected: print("Waiting for Vicon..."); return
    while np.all(vicon.pos == 0.0): time.sleep(0.1)

    initial_frame = vicon.latest_frame()
    if initial_frame is None:
        eskf = ErrorStateKalmanFilter(vicon.pos, vicon.quat)
    else:
        eskf = ErrorStateKalmanFilter(initial_frame["pos"], initial_frame["quat"])

    log_data = []
    is_calibrating = True
    calib_count = 0
    CALIB_SAMPLES = 600
    gyro_sum = np.zeros(3)
    accel_sum = np.zeros(3)
    gyro_bias = np.zeros(3)
    accel_bias = np.zeros(3)

    accel_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005) if USE_PC_IMU_LPF else None
    gyro_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005) if USE_PC_IMU_LPF else None
    vicon_vel_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005) if USE_VICON_VEL_LPF else None
    imu_buffer = IMUReplayBuffer()
    estimator_replay_buffer = EstimatorReplayBuffer()
    bias_estimator = BiasEstimator()
    estimator_manager = EstimatorManager(INNOVATION_POS_GATE_M, INNOVATION_VEL_GATE_MPS)
    innovation_gate = estimator_manager.innovation_gate
    output_predictor = OutputPredictor()
    latest_fusion_sample = None

    flight_mode = 'CALIBRATE'
    waypoint_queue = StationaryWaypointQueue()
    spatial_observer = Spatial52StationaryObserver(
        landing_root_height=POLICY_GROUND_ROOT_Z,
        apex_height=POLICY_TARGET_Z,
        waypoint_scale=0.20,
    )
    current_target_x, current_target_y = 0.0, 0.0
    latched_command = np.zeros(4, dtype=np.float32)
    previous_contact = True
    cycle_active = False
    needs_new_plan = True
    startup_ready_ticks = 0
    freefall_ticks = 0
    previous_pwm_cmd = None
    flight_start_time = None
    soft_tilt_bad_ticks = 0
    hard_tilt_bad_ticks = 0
    touchdown_count = 0
    touchdown_abort_requested = False
    touchdown_abort_msg = ""

    action_history = deque([np.zeros(4, dtype=np.float32) for _ in range(5)], maxlen=5)

    # Keep the exact values used by the 100 Hz policy/actuator pipeline for
    # the 200 Hz CSV logger.  The previous implementation logged the disabled
    # 69-D planner input, which made every Obs_Teacher_* field zero and hid the
    # actual policy inputs during flight.
    last_teacher_obs_log = np.zeros(52, dtype=np.float32)
    last_teacher_action_log = np.zeros(4, dtype=np.float32)
    last_pwm_raw_log = np.zeros(4, dtype=np.float32)
    last_pwm_height_log = np.zeros(4, dtype=np.float32)
    last_pwm_spread_yaw_log = np.zeros(4, dtype=np.float32)
    last_pwm_final_log = np.zeros(4, dtype=np.float32)
    last_policy_vel_w_log = np.zeros(3, dtype=np.float32)
    policy_obs_valid = 0
    policy_update_this_frame = 0

    def reset_policy_state():
        nonlocal latched_command, previous_contact, cycle_active, needs_new_plan, startup_ready_ticks, freefall_ticks
        nonlocal previous_pwm_cmd, flight_start_time, soft_tilt_bad_ticks, hard_tilt_bad_ticks
        nonlocal touchdown_count, touchdown_abort_requested, touchdown_abort_msg
        nonlocal policy_obs_valid, policy_update_this_frame
        latched_command[:] = 0.0
        teacher_policy.reset()
        spatial_observer.reset()
        if planner_policy is not None:
            planner_policy.reset()
        action_history.clear()
        for _ in range(5):
            action_history.append(np.zeros(4, dtype=np.float32))
        last_teacher_obs_log.fill(0.0)
        last_teacher_action_log.fill(0.0)
        last_pwm_raw_log.fill(0.0)
        last_pwm_height_log.fill(0.0)
        last_pwm_spread_yaw_log.fill(0.0)
        last_pwm_final_log.fill(0.0)
        last_policy_vel_w_log.fill(0.0)
        previous_contact = True
        cycle_active = False
        needs_new_plan = True
        startup_ready_ticks = 0
        freefall_ticks = 0
        previous_pwm_cmd = None
        flight_start_time = None
        soft_tilt_bad_ticks = 0
        hard_tilt_bad_ticks = 0
        touchdown_count = 0
        touchdown_abort_requested = False
        touchdown_abort_msg = ""
        policy_obs_valid = 0
        policy_update_this_frame = 0

    def reset_planner_handoff_state(is_contact):
        nonlocal latched_command, previous_contact, cycle_active, needs_new_plan
        latched_command[:] = 0.0
        if planner_policy is not None:
            planner_policy.reset()
        previous_contact = is_contact > 0.5
        cycle_active = False
        needs_new_plan = True

    m1, m2, m3, m4 = 0, 0, 0, 0
    latest_filt_acc = np.zeros(3)
    latest_filt_gyro = np.zeros(3)
    latest_mapped_accel = np.zeros(3)
    latest_mapped_gyro = np.zeros(3)
    latest_raw_g = [0.0, 0.0, 0.0]
    kf_vx, kf_vy, kf_vz = 0.0, 0.0, 0.0
    last_imu_sequence = None
    imu_sequence = 0
    imu_seq_gap = 0
    fc_status_flags = 0
    fc_cmd_age_ms = 0
    link_health_ok = False
    link_health_bad_count = 0
    sample_link_health_ok = False
    imu_degraded = False
    output_tracking_error = np.zeros(3)
    serial_parse_error_count = 0
    delayed_state_miss_count = 0
    estimator_predict_count = 0
    deploy_tilt_deg = 0.0
    deploy_xy_error_m = 0.0
    deploy_pwm_spread = 0.0
    deploy_safety_code = 0

    last_calc_time = initial_frame["time_s"] if initial_frame is not None else time.perf_counter()
    last_imu_predict_time = None
    vicon_tick = 0


    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
        ser.reset_input_buffer()
        print(f"🚀 跳跃机系统就绪: 等待传感器校准后，按下 'H' 键开始跳跃...")

        buffer = bytearray()
        start_time = None

        while True:
            # ==== 1. 键盘监听 ====
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\x00', b'\xe0'):
                    arrow = msvcrt.getch()
                    if flight_mode == 'KEYBOARD':
                        if arrow == b'H':
                            current_target_x += 0.3
                        elif arrow == b'P':
                            current_target_x -= 0.3
                        elif arrow == b'K':
                            current_target_y += 0.3
                        elif arrow == b'M':
                            current_target_y -= 0.3
                else:
                    key_lower = key.lower()
                    if key_lower == b'e':
                        break
                    elif key_lower == b'h':
                        if flight_mode in ('IDLE', 'CALIBRATE'):
                            flight_mode = 'PREHOVER'
                            waypoint_queue.lock_here(vicon.pos[:2])
                            current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                            reset_policy_state()
                            print("==== 预稳定模式：已锁定位置，电机零油门待命；再按 H 进入释放模式 ====")
                        elif flight_mode == 'PREHOVER':
                            flight_mode = 'JUMP_STARTUP'
                            waypoint_queue.lock_here(vicon.pos[:2])
                            current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                            reset_policy_state()
                            print("==== 启动释放模式：电机零油门；松手自动起跳，或放地上稳定后自动起跳 ====")
                        else:
                            flight_mode = 'PREHOVER'
                            waypoint_queue.lock_here(vicon.pos[:2])
                            current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                            reset_policy_state()
                            print("==== 返回预稳定模式：已重新锁定当前位置 ====")
                    elif key_lower == b'k':
                        flight_mode = 'KEYBOARD'
                        waypoint_queue.lock_here(vicon.pos[:2])
                        current_target_x, current_target_y = waypoint_queue.targets_xy[0]

            # ==== 2. 串口数据解析 ====
            waiting = ser.in_waiting
            if waiting > 1024:
                ser.reset_input_buffer()
                buffer.clear()
            elif waiting > 0:
                buffer.extend(ser.read(waiting))

            while len(buffer) >= PACKET_SIZE:
                if buffer[0] == 0xAA and buffer[1] == 0xBB:
                    payload = buffer[2:PACKET_SIZE]
                    del buffer[:PACKET_SIZE]
                    try:
                        data = struct.unpack(PAYLOAD_FORMAT, payload)
                        if data[0] == 0xAA:
                            raw_ax_o, raw_ay_o, raw_az_o = data[3] / 16384 * 9.81, data[4] / 16384 * 9.81, data[
                                5] / 16384 * 9.81
                            raw_gx_o, raw_gy_o, raw_gz_o = math.radians(data[6] / 16.4), math.radians(
                                data[7] / 16.4), math.radians(data[8] / 16.4)

                            mapped_accel = np.array([-raw_ay_o, raw_ax_o, raw_az_o])
                            mapped_gyro = np.array([-raw_gy_o, raw_gx_o, raw_gz_o])
                            latest_mapped_accel = mapped_accel
                            latest_mapped_gyro = mapped_gyro

                            latest_vbat = data[9]
                            latest_current = data[10]
                            imu_sequence = int(data[11])
                            fc_status_flags = int(data[12])
                            fc_cmd_age_ms = int(data[13])

                            if last_imu_sequence is None:
                                imu_seq_gap = 0
                            else:
                                imu_seq_delta = (imu_sequence - last_imu_sequence) & 0xFFFF
                                if imu_seq_delta == 0:
                                    imu_seq_gap = IMU_SEQ_GAP_WARN + 1
                                else:
                                    imu_seq_gap = max(0, imu_seq_delta - EXPECTED_IMU_SEQ_STEP)
                            last_imu_sequence = imu_sequence

                            sample_link_health_ok = (
                                imu_seq_gap <= IMU_SEQ_GAP_WARN and
                                (fc_status_flags & FC_STATUS_CRITICAL_MASK) == 0
                            )
                            if sample_link_health_ok:
                                link_health_bad_count = 0
                            else:
                                link_health_bad_count = min(
                                    link_health_bad_count + 1,
                                    LINK_HEALTH_BAD_SAMPLE_LIMIT
                                )
                            link_health_ok = link_health_bad_count < LINK_HEALTH_BAD_SAMPLE_LIMIT
                            imu_sample_valid = (fc_status_flags & FC_STATUS_IMU_READ_FAIL) == 0
                            imu_degraded = (fc_status_flags & FC_STATUS_DEGRADED_MASK) != 0

                            if is_calibrating:
                                if not imu_sample_valid:
                                    continue
                                gyro_sum += mapped_gyro
                                accel_sum += mapped_accel
                                calib_count += 1
                                if calib_count >= CALIB_SAMPLES:
                                    gyro_bias = gyro_sum / CALIB_SAMPLES
                                    init_r = R.from_quat(vicon.quat)
                                    g_body = init_r.inv().apply([0, 0, 9.81])
                                    accel_bias = (accel_sum / CALIB_SAMPLES) - g_body
                                    bias_estimator.set_biases(accel_bias, gyro_bias)
                                    is_calibrating = False
                                    flight_mode = 'IDLE'
                                    print("✅ 传感器校准成功。请将机器置于平地，按下 'H' 起飞跳跃！")
                                continue

                            if not imu_sample_valid:
                                continue

                            clean_acc = bias_estimator.correct_accel(mapped_accel)
                            clean_gyro = bias_estimator.correct_gyro(mapped_gyro)
                            latest_filt_acc = clean_acc if not USE_PC_IMU_LPF else accel_lpf.apply(clean_acc)
                            latest_filt_gyro = clean_gyro if not USE_PC_IMU_LPF else gyro_lpf.apply(clean_gyro)
                            imu_rx_time = time.perf_counter()
                            imu_buffer.add(data[2], latest_filt_acc, latest_filt_gyro, imu_rx_time, degraded=imu_degraded)
                            if last_imu_predict_time is None:
                                last_imu_predict_time = imu_rx_time
                                estimator_replay_buffer.save_state(imu_rx_time, eskf)
                            else:
                                imu_dt = imu_rx_time - last_imu_predict_time
                                if 0.001 <= imu_dt <= 0.05:
                                    eskf.predict(
                                        imu_dt,
                                        latest_filt_acc,
                                        latest_filt_gyro,
                                        imu_degraded=imu_degraded
                                    )
                                    estimator_predict_count += 1
                                last_imu_predict_time = imu_rx_time
                                estimator_replay_buffer.save_state(imu_rx_time, eskf)
                            latest_raw_g = [raw_gx_o, raw_gy_o, raw_gz_o]
                    except Exception:
                        serial_parse_error_count += 1
                else:
                    del buffer[0]

            # ==== 3. Vicon 心跳触发控制循环 (200Hz) ====
            if not is_calibrating and vicon.new_frame_event.wait(timeout=0.001):
                vicon.new_frame_event.clear()
                vicon_tick += 1

                curr_calc_time = time.perf_counter()
                vicon_frame = vicon.latest_frame()
                if vicon_frame is None:
                    continue

                frame_time_s = vicon_frame["time_s"]
                dt = frame_time_s - last_calc_time
                if dt <= 0.001 or dt > 0.05: dt = 0.005
                last_calc_time = frame_time_s

                if USE_VICON_VEL_LPF:
                    vicon_vel_meas = vicon_vel_lpf.apply(vicon_frame["vel"])
                else:
                    vicon_vel_meas = vicon_frame["vel"]
                vicon_pos_w = np.copy(vicon_frame["pos"])
                latest_pure_q = vicon_frame["quat"]
                r_mat = R.from_quat(latest_pure_q)
                imu_age_s = imu_buffer.age(curr_calc_time)
                link_health_ok = (
                    link_health_bad_count < LINK_HEALTH_BAD_SAMPLE_LIMIT and
                    imu_age_s <= IMU_STALE_TIMEOUT_S and
                    (fc_status_flags & FC_STATUS_CRITICAL_MASK) == 0
                )
                fusion_time_s = vicon_frame["time_s"]
                latest_fusion_sample = imu_buffer.sample_at(fusion_time_s)

                if latest_fusion_sample is None:
                    fusion_acc_b = latest_filt_acc
                    fusion_gyro_b = latest_filt_gyro
                    fusion_imu_degraded = imu_degraded
                else:
                    fusion_acc_b = latest_fusion_sample["accel_b"]
                    fusion_gyro_b = latest_fusion_sample["gyro_b"]
                    fusion_imu_degraded = latest_fusion_sample["degraded"] or imu_degraded
                a_world = r_mat.apply(latest_filt_acc) - np.array([0, 0, 9.81])

                delayed_state = estimator_replay_buffer.find_state_before(fusion_time_s)
                if delayed_state is not None:
                    delayed_eskf = delayed_state["eskf"].clone()
                    estimator_manager.fuse_vicon_covariance(
                        delayed_eskf,
                        vicon_pos_w,
                        vicon_vel_meas,
                        measured_quat=latest_pure_q,
                        imu_degraded=fusion_imu_degraded,
                        now_s=curr_calc_time
                    )
                    estimator_replay_buffer.discard_after(delayed_state["pc_time_s"])
                    replayed_time_s = estimator_replay_buffer.replay_to(
                        delayed_eskf,
                        delayed_state["pc_time_s"],
                        curr_calc_time,
                        imu_buffer.samples
                    )
                    eskf.copy_from(delayed_eskf)
                    estimator_replay_buffer.save_state(replayed_time_s, eskf)
                else:
                    delayed_state_miss_count += 1
                    estimator_manager.innovation_gate.rejected += 1

                fused_pos_w = eskf.pos.copy()
                fused_vel_w = eskf.vel.copy()
                kf_vx, kf_vy, kf_vz = fused_vel_w
                output_state_time_s = last_imu_predict_time if last_imu_predict_time is not None else curr_calc_time
                output_predictor.update(output_state_time_s, fused_pos_w, fused_vel_w)
                predicted_pos_w, predicted_vel_w = output_predictor.predict(curr_calc_time + OUTPUT_PREDICTION_S, a_world)
                output_tracking_error = estimator_manager.update_output_tracking_error(
                    fusion_gyro_b,
                    latest_filt_gyro,
                    fused_vel_w,
                    predicted_vel_w,
                    fused_pos_w,
                    predicted_pos_w
                )

                policy_pos_w = vicon_pos_w
                policy_vel_w = fused_vel_w
                policy_gyro_b = latest_filt_gyro
                if PPO_USE_EKF_POSITION:
                    policy_pos_w = predicted_pos_w
                if not PPO_USE_EKF_VELOCITY:
                    policy_vel_w = vicon_vel_meas
                if PPO_USE_OUTPUT_PREDICTION:
                    policy_vel_w = predicted_vel_w
                if PPO_USE_DELAYED_IMU:
                    policy_gyro_b = fusion_gyro_b
                pos_w = policy_pos_w
                last_policy_vel_w_log[:] = policy_vel_w

                # ==== 防炸机保护：只在 JUMP 模式生效，避免手持 ARM 阶段误触发 ====
                tilt_rad = body_tilt_rad(latest_pure_q)
                deploy_tilt_deg = math.degrees(tilt_rad)
                deploy_xy_error_m = float(np.linalg.norm(pos_w[:2] - waypoint_queue.targets_xy[0]))
                deploy_safety_code = 0
                if flight_mode == 'JUMP':
                    if tilt_rad > MAX_TILT_ABORT_RAD:
                        hard_tilt_bad_ticks += 1
                    else:
                        hard_tilt_bad_ticks = 0

                    if hard_tilt_bad_ticks >= HARD_TILT_BAD_TICKS_LIMIT:
                        m1, m2, m3, m4 = 0, 0, 0, 0
                        previous_pwm_cmd = None
                        deploy_safety_code = 2
                        print(
                            f"!!!! ABORT: tilt={deploy_tilt_deg:.1f}deg > "
                            f"{math.degrees(MAX_TILT_ABORT_RAD):.1f}deg for "
                            f"{HARD_TILT_BAD_TICKS_LIMIT} frames, motors off !!!!"
                        )
                        break

                    if tilt_rad > SOFT_TILT_ABORT_RAD:
                        soft_tilt_bad_ticks += 1
                    else:
                        soft_tilt_bad_ticks = 0

                    if soft_tilt_bad_ticks >= SOFT_TILT_BAD_TICKS_LIMIT:
                        deploy_safety_code = 1
                        if soft_tilt_bad_ticks == SOFT_TILT_BAD_TICKS_LIMIT:
                            print(
                                f"!!!! SOFT WARN: tilt={deploy_tilt_deg:.1f}deg > "
                                f"{math.degrees(SOFT_TILT_ABORT_RAD):.1f}deg, keep control active !!!!"
                            )
                else:
                    soft_tilt_bad_ticks = 0
                    hard_tilt_bad_ticks = 0

                if touchdown_abort_requested:
                    touchdown_abort_requested = False
                    m1, m2, m3, m4 = 0, 0, 0, 0
                    previous_pwm_cmd = None
                    deploy_safety_code = 3
                    flight_mode = 'IDLE'
                    reset_policy_state()
                    print(f"!!!! TOUCHDOWN TILT ABORT: {touchdown_abort_msg}，断油回 IDLE；按 H 重新武装 !!!!")
                    touchdown_abort_msg = ""

                policy_update_this_frame = 0
                if flight_mode in ('CALIBRATE', 'IDLE'):
                    m1, m2, m3, m4 = 0, 0, 0, 0
                    previous_pwm_cmd = None
                elif not link_health_ok:
                    m1, m2, m3, m4 = 0, 0, 0, 0
                    previous_pwm_cmd = None
                else:
                    # 100Hz 策略降频执行
                    if vicon_tick % 2 == 0:
                        policy_update_this_frame = 1
                        hold_zero_this_tick = False
                        current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                        # `pos_w` remains physical for safety and logging.
                        # The policy receives the calibrated Isaac root-Z
                        # convention, so its apex command remains exactly 1m.
                        policy_pos_w = pos_w.copy()
                        policy_pos_w[2] += POLICY_Z_OFFSET
                        target_w = np.array([current_target_x, current_target_y, POLICY_TARGET_Z])
                        vel_w = policy_vel_w

                        R_w2b = r_mat.inv()
                        lin_vel_b = R_w2b.apply(vel_w)
                        ang_vel_b = policy_gyro_b
                        quat_w_isaac = np.array(
                            [latest_pure_q[3], latest_pure_q[0], latest_pure_q[1], latest_pure_q[2]])

                        pos_error_w = target_w - policy_pos_w
                        pos_error_b = R_w2b.apply(pos_error_w)

                        # 👑 【核心修改】：通过动捕高度精确盲估弹簧运动相位
                        z_pos = np.array([policy_pos_w[2]])

                        # 判定触地状态（当高度低于完全伸展高度加1cm滑移空间时，判定触地）
                        is_contact = 1.0 if pos_w[2] < (REST_LEG_LENGTH + 0.01) else 0.0
                        is_contact_arr = np.array([is_contact])
                        bias_estimator.observe_static(
                            latest_mapped_accel,
                            latest_mapped_gyro,
                            R_w2b.apply([0, 0, 9.81]),
                            is_contact > 0.5
                        )

                        if is_contact > 0.5:
                            joint_pos = np.array([max(0.0, REST_LEG_LENGTH - pos_w[2])])
                            # Keep the spring phase consistent with the velocity in
                            # the policy observation.  The EKF commonly misses the
                            # touchdown impulse and can retain a descending velocity
                            # during rebound; using it here reverses the spring phase.
                            joint_vel = np.array([-float(policy_vel_w[2])])
                        else:
                            joint_pos = np.array([0.0])
                            joint_vel = np.array([0.0])

                        history_actions = np.concatenate(action_history).astype(np.float32)
                        teacher_stable_obs = np.concatenate([
                            lin_vel_b, ang_vel_b, quat_w_isaac, pos_error_b,
                            z_pos, is_contact_arr, joint_pos, joint_vel, history_actions
                        ]).astype(np.float32)

                        p_t = waypoint_queue.targets_xy[0]
                        p_t1 = waypoint_queue.targets_xy[1]
                        spatial_observer.update(
                            policy_pos_w, p_t, is_contact > 0.5, float(vel_w[2])
                        )
                        spatial_extra = spatial_observer.observation(
                            policy_pos_w, R_w2b, p_t, p_t1
                        )
                        teacher_obs = np.concatenate([
                            teacher_stable_obs,
                            spatial_extra,
                        ]).astype(np.float32)
                        if teacher_obs.shape != (52,):
                            raise RuntimeError(f'Expected a 52-D spatial observation, got {teacher_obs.shape}')

                        teacher_action = teacher_policy.step(teacher_obs)
                        last_teacher_obs_log[:] = teacher_obs
                        last_teacher_action_log[:] = teacher_action
                        policy_obs_valid = 1

                        if flight_mode in ('PREHOVER', 'JUMP_STARTUP'):
                            # 松手下坠检测：手拿时 Vz≈0；真正松手后 Vz 快速跌破阈值。
                            # 命中后本 tick 保持零油门，下一 tick 从清零的 LSTM 开始起跳。
                            if vel_w[2] < FREE_FALL_TRIGGER_VZ:
                                freefall_ticks += 1
                            else:
                                freefall_ticks = 0
                            if freefall_ticks >= FREE_FALL_TRIGGER_TICKS:
                                flight_mode = 'JUMP'
                                waypoint_queue.lock_here(pos_w[:2])
                                current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                                reset_policy_state()
                                reset_planner_handoff_state(is_contact)
                                flight_start_time = time.perf_counter()
                                hold_zero_this_tick = True
                                print("==== 检测到松手下坠：锁定当前位置，开始起跳 ====")

                        if flight_mode == 'JUMP_STARTUP':
                            # 原有地上放置路径：接触 + 静止 0.2 s 自动起跳
                            if handoff_ready(pos_w, vel_w, latest_pure_q, is_contact):
                                startup_ready_ticks += 1
                            else:
                                startup_ready_ticks = 0
                            required_ticks = max(1, int(HANDOFF_CONTACT_HOLD_S * 100.0))
                            if startup_ready_ticks >= required_ticks:
                                flight_mode = 'JUMP'
                                waypoint_queue.lock_here(pos_w[:2])
                                current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                                reset_planner_handoff_state(is_contact)
                                flight_start_time = time.perf_counter()
                                touchdown_count = 1  # 地上放置路径：进入 JUMP 时已在地面，首次起跳落地即按第 1 跳验收
                                print("==== 已稳定接管：planner 开始追点 ====")

                        if flight_mode == 'JUMP':
                            just_touched_down = (is_contact > 0.5) and (not previous_contact)
                            just_took_off = (is_contact <= 0.5) and previous_contact
                            if just_took_off:
                                cycle_active = True
                            if just_touched_down:
                                touchdown_error = float(np.linalg.norm(pos_w[:2] - waypoint_queue.targets_xy[0]))
                                hit = touchdown_error <= 0.10
                                waypoint_queue.advance_after_touchdown(pos_w[:2], hit)
                                current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                                cycle_active = False
                                needs_new_plan = True
                                touchdown_count += 1
                                # 第 1 次触地是松手投放，不算。仅保留落地姿态
                                # 保护；释放位置或落点误差不再触发断油。
                                if (
                                    touchdown_count >= 2
                                    and deploy_tilt_deg > TOUCHDOWN_TILT_ABORT_DEG
                                ):
                                    touchdown_abort_requested = True
                                    touchdown_abort_msg = f"姿态 {deploy_tilt_deg:.1f}°"

                        correction = np.zeros(4, dtype=np.float32)
                        if flight_mode == 'JUMP' and USE_SEMIMDP_PLANNER and planner_policy is not None:
                            planner_obs = build_planner_observation(
                                teacher_obs,
                                teacher_action,
                                pos_w,
                                vel_w,
                                latest_pure_q,
                                float(joint_pos[0]),
                                float(joint_vel[0]),
                                is_contact,
                                cycle_active,
                                latched_command,
                                waypoint_queue,
                            )
                            planner_action = planner_policy.step(planner_obs)
                        else:
                            planner_obs = np.zeros(69, dtype=np.float32)
                            planner_action = np.zeros(4, dtype=np.float32)
                        if flight_mode == 'JUMP' and USE_SHORT_ACTION_OFFSET and waypoint_queue.is_short():
                            planner_action = np.clip(planner_action + SHORT_ACTION_OFFSET, -1.0, 1.0)

                        if flight_mode == 'JUMP' and USE_SEMIMDP_PLANNER:
                            decoded_command = decode_planner_command(planner_action)
                            if is_contact > 0.5:
                                if needs_new_plan:
                                    latched_command = decoded_command
                                    needs_new_plan = False
                                else:
                                    latched_command = (
                                        (1.0 - STANCE_PLAN_RATE) * latched_command
                                        + STANCE_PLAN_RATE * decoded_command
                                    ).astype(np.float32)

                            correction = motor_correction(
                                pos_w,
                                vel_w,
                                latest_pure_q,
                                float(joint_pos[0]),
                                cycle_active,
                                latched_command,
                                waypoint_queue,
                            )
                        action = np.clip(teacher_action + correction, -1.0, 1.0)
                        action = np.clip(action * DEPLOY_ACTION_SCALE, -1.0, 1.0)
                        if DEPLOY_DIFFERENTIAL_SCALE < 1.0:
                            # 只缩放差动分量：collective（均值）不动，姿态/水平修正按比例压缩
                            action_mean = float(np.mean(action))
                            action = action_mean + DEPLOY_DIFFERENTIAL_SCALE * (action - action_mean)
                            action = np.clip(action, -1.0, 1.0)
                        previous_contact = is_contact > 0.5

                        # Isaac stores raw policy actions, before actuator
                        # mapping and any downstream physical safety shaping.
                        action_history.append(
                            np.zeros(4, dtype=np.float32)
                            if hold_zero_this_tick else teacher_action
                        )

                        target_u = action * 0.5 + 0.5
                        target_u = np.clip(target_u, 0.0, 1.0)

                        pwm_cmd = target_u * MAX_PWM
                        last_pwm_raw_log[:] = pwm_cmd
                        pwm_cmd = limit_collective_for_height(
                            pwm_cmd,
                            float(pos_w[2]),
                            float(vel_w[2]),
                        )
                        last_pwm_height_log[:] = pwm_cmd
                        airborne_spread_limit = adaptive_airborne_pwm_spread_limit(
                            deploy_tilt_deg,
                            deploy_xy_error_m,
                            is_contact,
                        )
                        pwm_cmd = limit_pwm_spread(pwm_cmd, airborne_spread_limit)
                        pwm_cmd = limit_yaw_diagonal_pwm(pwm_cmd, is_contact)
                        last_pwm_spread_yaw_log[:] = pwm_cmd
                        pwm_cmd = limit_pwm_slew(pwm_cmd, previous_pwm_cmd)
                        pwm_cmd = limit_pwm_spread(pwm_cmd, airborne_spread_limit)
                        pwm_cmd = limit_yaw_diagonal_pwm(pwm_cmd, is_contact)
                        last_pwm_final_log[:] = pwm_cmd
                        previous_pwm_cmd = pwm_cmd.copy()
                        deploy_pwm_spread = float(np.max(pwm_cmd) - np.min(pwm_cmd))
                        m1, m2, m3, m4 = int(pwm_cmd[0]), int(pwm_cmd[1]), int(pwm_cmd[2]), int(pwm_cmd[3])
                        # 武装待命阶段零油门：只有 JUMP 模式（以及松手触发当拍之后）才允许电机输出
                        if flight_mode in ('PREHOVER', 'JUMP_STARTUP') or hold_zero_this_tick:
                            m1, m2, m3, m4 = 0, 0, 0, 0
                            previous_pwm_cmd = None

                if flight_mode in ('CALIBRATE', 'IDLE', 'PREHOVER', 'JUMP_STARTUP') or not link_health_ok:
                    deploy_pwm_spread = 0.0

                ser.write(struct.pack('<BBHHHH', 0x55, 0xAA, m1, m2, m3, m4))

                if start_time is None: start_time = time.perf_counter()

                log_row = [
                              time.perf_counter() - start_time, dt,
                              pos_w[0], pos_w[1], pos_w[2],
                              kf_vx, kf_vy, kf_vz,
                              *last_policy_vel_w_log.tolist(),
                              current_target_x, current_target_y, TARGET_Z,
                              m1, m2, m3, m4,
                              policy_obs_valid, policy_update_this_frame,
                          ] + last_teacher_obs_log.tolist() + last_teacher_action_log.tolist() + \
                              last_pwm_raw_log.tolist() + last_pwm_height_log.tolist() + \
                              last_pwm_spread_yaw_log.tolist() + last_pwm_final_log.tolist() + [
                              math.degrees(latest_raw_g[0]),
                              math.degrees(latest_raw_g[1]),
                              math.degrees(latest_raw_g[2]),
                              latest_vbat,
                              latest_current,
                              innovation_gate.accepted,
                              innovation_gate.rejected,
                              imu_age_s,
                              float(np.linalg.norm(bias_estimator.accel_bias)),
                              float(np.linalg.norm(bias_estimator.gyro_bias)),
                              imu_sequence,
                              imu_seq_gap,
                              fc_status_flags,
                              fc_cmd_age_ms,
                              int(link_health_ok),
                              int(imu_degraded),
                              estimator_manager.pos_test_ratio,
                              estimator_manager.vel_test_ratio,
                              estimator_manager.att_test_ratio,
                              estimator_manager.estimator_fault_flags,
                              estimator_manager.estimator_reset_count,
                              output_tracking_error[0],
                              output_tracking_error[1],
                              output_tracking_error[2],
                              estimator_predict_count,
                              delayed_state_miss_count,
                              serial_parse_error_count,
                              vicon.error_count,
                              deploy_tilt_deg,
                              deploy_xy_error_m,
                              deploy_pwm_spread,
                              deploy_safety_code,
                              flight_mode
                          ]

                log_data.append(log_row)

    except KeyboardInterrupt:
        pass
    finally:
        if 'ser' in locals() and ser.is_open:
            for _ in range(5): ser.write(struct.pack('<BBHHHH', 0x55, 0xAA, 0, 0, 0, 0)); time.sleep(0.02)
            ser.close()
        vicon.close()

    if log_data:
        csv_filename = f"quadhopper_deployed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        teacher_headers = [
            'Obs_Teacher_LinVel_bx', 'Obs_Teacher_LinVel_by', 'Obs_Teacher_LinVel_bz',
            'Obs_Teacher_AngVel_bx', 'Obs_Teacher_AngVel_by', 'Obs_Teacher_AngVel_bz',
            'Obs_Teacher_Quat_w', 'Obs_Teacher_Quat_x', 'Obs_Teacher_Quat_y', 'Obs_Teacher_Quat_z',
            'Obs_Teacher_PosErr_bx', 'Obs_Teacher_PosErr_by', 'Obs_Teacher_PosErr_bz',
            'Obs_Teacher_Z_Height', 'Obs_Teacher_IsContact', 'Obs_Teacher_JointPos', 'Obs_Teacher_JointVel',
        ]
        history_headers = [f'Obs_Teacher_ActHist_tminus{5 - i // 4}_m{i % 4 + 1}' for i in range(20)]
        spatial_headers = [
            'Obs_Spatial_CurrentTargetErr_bx_div_0p20',
            'Obs_Spatial_CurrentTargetErr_by_div_0p20',
            'Obs_Spatial_NextDisplacement_bx_div_0p20',
            'Obs_Spatial_NextDisplacement_by_div_0p20',
            'Obs_Spatial_CurrentApexHeight_div_2',
            'Obs_Spatial_NextApexHeight_div_2',
            'Obs_Spatial_CarrotErr_bx_clipped_m',
            'Obs_Spatial_CarrotErr_by_clipped_m',
            'Obs_Spatial_CarrotErr_bz_clipped_m',
            'Obs_Spatial_PathTangent_bx',
            'Obs_Spatial_PathTangent_by',
            'Obs_Spatial_PathTangent_bz',
            'Obs_Spatial_LandingUpHint_bx',
            'Obs_Spatial_LandingUpHint_by',
            'Obs_Spatial_LandingUpHint_bz',
        ]
        obs_headers = teacher_headers + history_headers + spatial_headers
        action_headers = [f'Policy_Action_Raw_m{i}' for i in range(1, 5)]
        pwm_raw_headers = [f'PWM_Raw_m{i}' for i in range(1, 5)]
        pwm_height_headers = [f'PWM_After_Height_m{i}' for i in range(1, 5)]
        pwm_spread_yaw_headers = [f'PWM_After_SpreadYaw_m{i}' for i in range(1, 5)]
        pwm_final_headers = [f'PWM_FinalFloat_m{i}' for i in range(1, 5)]

        headers = [
                      'Time_s', 'loop_dt', 'X', 'Y', 'Z', 'EKF_Vel_X', 'EKF_Vel_Y', 'EKF_Vel_Z',
                      'Policy_Vel_X', 'Policy_Vel_Y', 'Policy_Vel_Z',
                      'Target_X', 'Target_Y', 'Target_Z', 'M1', 'M2', 'M3', 'M4',
                      'Policy_Obs_Valid', 'Policy_Update_This_Frame',
                  ] + obs_headers + action_headers + pwm_raw_headers + pwm_height_headers + \
                      pwm_spread_yaw_headers + pwm_final_headers + [
                      'Raw_IMU_Gx_deg', 'Raw_IMU_Gy_deg', 'Raw_IMU_Gz_deg', 'VBat_V', 'Current_A',
                      'Innov_Accepted', 'Innov_Rejected', 'IMU_Age_s', 'Accel_Bias_Norm', 'Gyro_Bias_Norm',
                      'IMU_Seq', 'IMU_Seq_Gap', 'FC_Status_Flags', 'FC_Cmd_Age_ms', 'Link_Health_OK', 'IMU_Degraded',
                      'Est_Pos_Test_Ratio', 'Est_Vel_Test_Ratio', 'Est_Att_Test_Ratio', 'Est_Fault_Flags', 'Est_Reset_Count',
                      'Output_Track_Ang', 'Output_Track_Vel', 'Output_Track_Pos',
                      'Estimator_Predict_Count', 'Delayed_State_Miss_Count', 'Serial_Parse_Error_Count', 'Vicon_Error_Count',
                      'Deploy_Tilt_Deg', 'Deploy_XY_Error_M', 'Deploy_PWM_Spread', 'Deploy_Safety_Code', 'Flight_Mode'
                  ]

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(log_data)
        print(f"✅ 实飞日志已完美保存: {csv_filename}")


if __name__ == '__main__':
    run_autonomous_flight()
