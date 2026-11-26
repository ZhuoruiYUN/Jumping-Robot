#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V88 OBS73 jump trial using the original two-file deployment layout.

The original argument-free launch transmits motor PWM after arming.
--shadow transmits zero PWM. --self-check uses no hardware.
"""
DEPLOYMENT_VERSION = 'obs73_v88_axis_heading_retry_imu_link'
VERSION = 'direct_motor_crossing_quaternion73_history40_20_v1'
VICON_GROUND_Z_M = 0.025
SESSION_NOTICE_S = 15.0  # Reminder only; never cut motors on an airborne timer.
LANDING_RADIUS_M = 0.05
STEP_M = 0.5
PREVIEW_MIN_S = 0.45
XY_ABORT_M = 0.30
POLICY_GAP_ABORT_S = 0.030
# Match the supplied, working QH_NP_V2 transport:0..999, not1000.
# Keep MAX_PWM=1000 as the normalization scale below.
WIRE_PWM_MAX = 999
OBS73_COURSE = {'observation_version': 'direct_motor_crossing_quaternion73_history40_20_v1',
 'stationary_pretraining': {'apex_target_m': 1.0},
 'task': 'obs73_omni_ground_targets_v1'}

import hashlib
import struct
import time
import math
import threading
import numpy as np
import csv
from datetime import datetime
from collections import deque
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort

MODEL_NAME = "quadhopper_step_x05_obs73.onnx"
MODEL_SHA256 = "b009a4af9f88511bfc1d5908191e5316d29a33be35c54d45d0c05323a857d6bc"
_MEMORY_SHAPE = (1, 1, 256)
_MEMORY = {
    "h_in": "h_out",
    "c_in": "c_out",
    "h_mirror_in": "h_mirror_out",
    "c_mirror_in": "c_mirror_out",
    "use_mirror_in": "use_mirror_out",
}
_INPUTS = {
    "obs": ((1, 73), "tensor(float)"),
    "h_in": (_MEMORY_SHAPE, "tensor(float)"),
    "c_in": (_MEMORY_SHAPE, "tensor(float)"),
    "h_mirror_in": (_MEMORY_SHAPE, "tensor(float)"),
    "c_mirror_in": (_MEMORY_SHAPE, "tensor(float)"),
    "current_edge_xy": ((1, 2), "tensor(float)"),
    "preview_edge_xy": ((1, 2), "tensor(float)"),
    "use_mirror_in": ((1,), "tensor(bool)"),
}
_OUTPUTS = {
    "accepted_actions": ((1, 4), "tensor(float)"),
    "raw_actions": ((1, 4), "tensor(float)"),
    **{out_name: _INPUTS[in_name] for in_name, out_name in _MEMORY.items()},
}
_DTYPES = {"tensor(float)": np.float32, "tensor(bool)": np.bool_}


def _policy_input(values, name):
    shape, _ = _INPUTS[name]
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(values, dtype=np.float32)
    if array.shape not in (shape, shape[1:]) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values with shape {shape} or {shape[1:]}")
    return np.ascontiguousarray(array.reshape(shape))


class V73Controller:
    """Advance the two actor memories and yaw allocator at each accepted OBS73 tick."""

    def __init__(self, model_path=None):
        path = Path(model_path) if model_path is not None else Path(__file__).with_name(MODEL_NAME)
        model = path.read_bytes()
        if hashlib.sha256(model).hexdigest() != MODEL_SHA256:
            raise ValueError(f"V88 ONNX SHA-256 mismatch: {path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
        for metadata, expected in (
            (self.session.get_inputs(), _INPUTS),
            (self.session.get_outputs(), _OUTPUTS),
        ):
            actual = {item.name: (tuple(item.shape), item.type) for item in metadata}
            if actual != expected:
                raise ValueError(f"V88 ONNX tensor interface mismatch: {actual}")
        self.reset()

    def reset(self):
        self._memory = {}
        for name in _MEMORY:
            shape, dtype = _INPUTS[name]
            self._memory[name] = np.zeros(shape, dtype=_DTYPES[dtype])
        self.last_raw = np.zeros(4, dtype=np.float32)
        self.last_mirror = False

    def step(self, obs, current_edge, preview_edge):
        """Return allocated F1–F4 in [-1, 1] and commit valid recurrent outputs."""
        inputs = {
            **self._memory,
            "obs": _policy_input(obs, "obs"),
            "current_edge_xy": _policy_input(current_edge, "current_edge_xy"),
            "preview_edge_xy": _policy_input(preview_edge, "preview_edge_xy"),
        }
        outputs = dict(zip(_OUTPUTS, self.session.run(list(_OUTPUTS), inputs)))
        for name, (shape, dtype) in _OUTPUTS.items():
            value = outputs[name]
            if value.shape != shape or value.dtype != _DTYPES[dtype] or not np.isfinite(value).all():
                raise RuntimeError(f"Invalid V88 ONNX output: {name}")
        accepted = outputs["accepted_actions"][0]
        if (np.abs(accepted) > 1.0).any():
            raise RuntimeError("V88 allocated action exceeds [-1, 1]")

        self._memory = {name: outputs[out_name] for name, out_name in _MEMORY.items()}
        self.last_raw = outputs["raw_actions"][0]
        self.last_mirror = bool(outputs["use_mirror_out"][0])
        return accepted


COM_PORT = 'COM9'
BAUD_RATE = 115200
VICON_HOST = 'localhost:801'
ZERO_POLICY_YAW_AT_RELEASE = True
DISARMED_PWM = (0, 0, 0, 0)
PWM_FRAME = struct.Struct('<BBHHHH')


def send_pwm(port, pwm):
    frame = PWM_FRAME.pack(0x55, 0xAA, *pwm)
    if port.write(frame) != PWM_FRAME.size:
        raise OSError('Incomplete PWM frame write')


def action_to_pwm(action):
    pwm = np.floor(np.clip((action + 1.0) * (MAX_PWM * 0.5), MIN_PWM, WIRE_PWM_MAX))
    return tuple(map(int, pwm))


PAYLOAD_FORMAT = "<BBI6hffHHH"
PAYLOAD_STRUCT = struct.Struct(PAYLOAD_FORMAT)
PAYLOAD_SIZE = PAYLOAD_STRUCT.size
PACKET_SIZE = 2 + PAYLOAD_SIZE
IMU_SYNC = b'\xaa\xbb\xaa'
SERIAL_READ_LIMIT = 4096


class ImuFrameDecoder:
    """Retain incomplete framing across reads and count discarded data."""

    def __init__(self):
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.invalid_frames = 0

    def feed(self, chunk):
        self.buffer.extend(chunk)
        cursor = 0
        frames = []
        while True:
            start = self.buffer.find(IMU_SYNC, cursor)
            if start == -1:
                tail = self.buffer[cursor:]
                keep = 2 if tail.endswith(IMU_SYNC[:2]) else int(tail.endswith(IMU_SYNC[:1]))
                self.discarded_bytes += len(tail) - keep
                cursor = len(self.buffer) - keep
                break
            self.discarded_bytes += start - cursor
            cursor = start
            if len(self.buffer) - start < PACKET_SIZE:
                break
            frame = PAYLOAD_STRUCT.unpack_from(self.buffer, start + 2)
            if not (math.isfinite(frame[9]) and math.isfinite(frame[10])):
                self.invalid_frames += 1
                self.discarded_bytes += 1
                cursor += 1
                continue
            frames.append(frame)
            cursor += PACKET_SIZE
        del self.buffer[:cursor]
        return frames

TARGET_Z = OBS73_COURSE['stationary_pretraining']['apex_target_m']
HANDOFF_CONTACT_HOLD_S = 0.20
HANDOFF_MAX_XY_SPEED = 0.25
HANDOFF_MAX_Z_SPEED = 0.35
HANDOFF_MAX_TILT_RAD = math.radians(12.0)
SOFT_TILT_ABORT_RAD = math.radians(35.0)
MAX_TILT_ABORT_RAD = math.radians(60.0)
# 落地姿态保护：触地瞬间倾斜超过该角度即断油回 IDLE。
# 正常落地实测 5-11°，2026-09-02 失控跳落地时 15-19°，取 15° 可干净分开两者。
TOUCHDOWN_TILT_ABORT_DEG = 15.0

SOFT_TILT_BAD_TICKS_LIMIT = 3
HARD_TILT_BAD_TICKS_LIMIT = 2
# 松手下坠触发：手拿时 Vz≈0；真正松手后约 50-80 ms 内 Vz 会持续低于该阈值。
# 若操作者快速下放导致误触发，把阈值调低到 -0.8 或增大 FREE_FALL_TRIGGER_TICKS。
FREE_FALL_TRIGGER_VZ = -0.5
FREE_FALL_TRIGGER_TICKS = 5

MAX_PWM = 1000
MIN_PWM = 0
OUTPUT_PREDICTION_S = 0.010
IMU_STALE_TIMEOUT_S = 0.050
VICON_STALE_TIMEOUT_S = 0.050
# Keep the old CSV excess-count column for comparisons. IMU sequence advances
# on the FC sampling clock, independently of the number of packets sent.
LEGACY_IMU_SEQ_STEP = 2
SERIAL_DIAGNOSTIC_HEADERS = (
    'FC_Timestamp_ms', 'FC_Critical_Bits_In_Read', 'IMU_Seq_Delta',
    'IMU_Rx_Interval_ms', 'IMU_Rejected_Frames',
    'Serial_Rx_Bytes_Total', 'Serial_Valid_Frames_Total', 'Serial_Resync_Bytes',
    'Serial_Rx_Buffer_Peak_Bytes', 'Serial_Rx_Budget_Exceeded_Polls',
)
FC_STATUS_CMD_TIMEOUT = 1 << 1
FC_STATUS_IMU_READ_FAIL = 1 << 2
FC_STATUS_BAD_RF_CMD = 1 << 3
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
        self.tracking_valid = False
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.zeros(3)
        self.frame_time_s = time.perf_counter()
        self.frame_history = deque(maxlen=VICON_HISTORY_MAXLEN)
        self.error_count = 0
        self.last_error = ""
        self.new_frame_event = threading.Event()
        from vicon_dssdk import ViconDataStream
        self.client = ViconDataStream.Client()
        self.vicon_host = vicon_host
        self._init_vicon()
        if self.connected:
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True
            self.thread.start()

    def _init_vicon(self):
        from vicon_dssdk import ViconDataStream
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
        if not self.frame_history or not self.tracking_valid:
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
                    previous_rx = getattr(self,'receive_clock',(None,None))[0]
                    self.receive_clock = (frame_time_s, (frame_time_s-previous_rx)*1000 if previous_rx is not None else float('nan'))
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
                            first_frame = False
                            self.tracking_valid = False
                        else:
                            if np.linalg.norm(rp - self.pos) < 0.2:
                                if np.dot(self.quat, rq) < 0.0: rq = -rq
                                dt = frame_time_s - self.frame_time_s
                                if 0.001 <= dt <= 0.05:
                                    new_vel = (rp - self.pos) / dt
                                else:
                                    self.pos, self.quat, self.frame_time_s = rp, rq, frame_time_s
                                    self.tracking_valid = False
                                    continue
                                if np.linalg.norm(new_vel) > VICON_VEL_REJECT_MPS:
                                    self.pos, self.quat, self.frame_time_s = rp, rq, frame_time_s
                                    self.tracking_valid = False
                                    continue
                                self.vel = new_vel
                                self.quat = rq
                                self.pos = rp
                                self.frame_time_s = frame_time_s
                                self.tracking_valid = True
                                self._store_frame(frame_time_s)
                                self.new_frame_event.set()
                            else:
                                self.pos, self.quat, self.frame_time_s = rp, rq, frame_time_s
                                self.tracking_valid = False
                    else:
                        self.tracking_valid = False
                        first_frame = True
            except Exception as e:
                self.tracking_valid = False
                first_frame = True
                self.error_count += 1
                self.last_error = str(e)
            time.sleep(0.001)

    def close(self):
        self.running = False


class ConsoleKeyPresses:
    """Nonblocking Windows console input, with WASD key-up/keydown edges.

    Console events preserve rapid separate presses without accepting Windows
    typematic repeats. This reads this console only, not global keyboard state.
    An injected event reader allows checks without Windows or hardware.
    """
    def __init__(self, read_events=None):
        self._pressed = set()
        self._buffer = []
        if read_events is not None:
            self._read_events = read_events
            return
        import ctypes
        from ctypes import wintypes

        class KeyEvent(ctypes.Structure):
            _fields_ = [('down', ctypes.c_int32), ('repeat', ctypes.c_uint16),
                        ('vk', ctypes.c_uint16), ('scan', ctypes.c_uint16),
                        ('char', ctypes.c_uint16), ('control', ctypes.c_uint32)]

        class EventUnion(ctypes.Union):
            _fields_ = [('key', KeyEvent), ('padding', ctypes.c_uint32 * 4)]

        class InputRecord(ctypes.Structure):
            _fields_ = [('kind', ctypes.c_uint16), ('event', EventUnion)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel.GetStdHandle.restype = wintypes.HANDLE
        kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetConsoleMode.restype = wintypes.BOOL
        kernel.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetNumberOfConsoleInputEvents.restype = wintypes.BOOL
        kernel.ReadConsoleInputW.argtypes = [wintypes.HANDLE, ctypes.POINTER(InputRecord),
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        kernel.ReadConsoleInputW.restype = wintypes.BOOL
        handle = kernel.GetStdHandle(wintypes.DWORD(-10).value)
        mode = wintypes.DWORD()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            raise OSError('Run this script in an interactive Windows PowerShell console')
        assert ctypes.sizeof(KeyEvent) == 16 and ctypes.sizeof(InputRecord) == 20

        def read_windows_events():
            count = wintypes.DWORD()
            if not kernel.GetNumberOfConsoleInputEvents(handle, ctypes.byref(count)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                return []
            records = (InputRecord * min(count.value, 64))()
            received = wintypes.DWORD()
            if not kernel.ReadConsoleInputW(handle, records, len(records), ctypes.byref(received)):
                raise ctypes.WinError(ctypes.get_last_error())
            return [(bool(r.event.key.down), r.event.key.vk, r.event.key.char)
                    for r in records[:received.value] if r.kind == 1]

        self._read_events = read_windows_events

    def kbhit(self):
        for down, vk, char in self._read_events():
            if not down:
                self._pressed.discard(vk)
                continue
            repeated = vk in self._pressed
            self._pressed.add(vk)
            if vk in (72, 87, 65, 83, 68):  # H and WASD each need a distinct press.
                if not repeated:
                    self._buffer.append(bytes([vk + 32]))
            elif 0 < char < 128:
                self._buffer.append(bytes([char]))
        return bool(self._buffer)

    def getch(self):
        return self._buffer.pop(0)


class StationaryWaypointQueue:
    """World XY queue with same-target retries before advancing."""
    DIRECTIONS={b'w':(1.,0.),b's':(-1.,0.),b'a':(0.,1.),b'd':(0.,-1.)}
    def __init__(self):
        self.lock_here([0.,0.])

    def lock_here(self,xy):
        self.release_origin_xy=np.asarray(xy,dtype=np.float64).copy()
        self.targets_xy=np.tile(self.release_origin_xy,(2,1))
        self.anchor_xy=self.release_origin_xy.copy()
        self.route_index=0
        self.first_hop_hit_for_pair=0.
        self.first_hop_complete=False
        self.cancel()

    def cancel(self):
        self.state='HOLD'
        self.first_hop_complete=False
        self.targets_xy[1]=self.targets_xy[0]
        self.step_start_xy=self.targets_xy[0].copy()
        self.request_time=None
        self.pending_targets=[]

    @property
    def remaining_steps(self):
        return len(self.pending_targets) + int(self._moving)

    @property
    def _moving(self):
        return self.state in ('STEP', 'RETRY')

    def _publish_next(self,now):
        self.targets_xy[1]=(self.pending_targets[0] if self.pending_targets
                            else self.targets_xy[0])
        self.request_time=now if self.pending_targets else None

    def _launch_next(self,now):
        self.step_start_xy=self.targets_xy[0].copy()
        self.targets_xy[0]=self.pending_targets.pop(0)
        self._publish_next(now)
        self.state='STEP'
        self.route_index+=1

    def request_step(self,now,key=b'w'):
        # Called once per physical press by ConsoleKeyPresses. Accumulate from
        # the last planned point, so measured landing errors do not move the route.
        if key not in self.DIRECTIONS:raise ValueError('Expected a WASD movement key')
        tail=self.pending_targets[-1] if self.pending_targets else self.targets_xy[0]
        self.pending_targets.append(tail+STEP_M*np.array(self.DIRECTIONS[key]))
        if len(self.pending_targets)==1 and self.state not in ('PAUSED', 'RETRY'):
            self._publish_next(now)
        if self.state=='HOLD':self.state='PREVIEW'
        return True

    @staticmethod
    def ready(event):
        # Fail closed if the hardware motion sample is missing. The rate uses
        # the last free descending sample, avoiding the impact gyro impulse.
        fields=('heading_error_deg','predicted_heading_error_deg','roll_pitch_rate_rad_s')
        if any(not math.isfinite(event.get(k,math.inf)) for k in fields):return False
        return (abs(event['heading_error_deg'])<=10.
                and abs(event['predicted_heading_error_deg'])<=10.
                and event['roll_pitch_rate_rad_s']<=1.2)

    def _hide_next(self):
        # Retain the route. Suppress the next edge during the same-target retry.
        self.targets_xy[1] = self.targets_xy[0]
        self.request_time = None

    def _pause(self):
        self.state = 'PAUSED'
        self._hide_next()

    def on_landing(self,event,now):
        if event is None:return None
        if event['strict']:self.first_hop_complete=True
        reached=event['foot_error_m']<=LANDING_RADIUS_M and event['tilt_deg']<=15.
        ready=self.ready(event)
        if not reached:
            if self._moving:
                self.state = 'RETRY'
                self._hide_next()
                return 'STEP_MISS_RETRY'
            if self.state in ('PREVIEW', 'PAUSED'):
                self._pause()
                return 'STEP_MISS_RETRY_HOLD'
            return None
        if self.state=='PAUSED':
            if ready and self.first_hop_complete:
                self.state='PREVIEW';self._publish_next(now)
                return 'STEP_PREVIEW_RESUMED'
            return 'STEP_PAUSED_HOLD'
        if self.state=='PREVIEW' and not ready:
            self._pause()
            return 'STEP_PREVIEW_PAUSED'
        if (self.state=='PREVIEW' and self.first_hop_complete
                and now-self.request_time>=PREVIEW_MIN_S):
            # Score the old point first, then publish the new current target
            # BEFORE inference on this contact tick. Do not reset memory.
            self._launch_next(now)
            return 'STEP_LAUNCH'
        if self._moving:
            if self.state == 'RETRY' and not ready:
                return 'STEP_RETRY_WAIT_READY'
            self.anchor_xy=self.targets_xy[0].copy()
            if not self.pending_targets:
                self.state='HOLD'
                self._publish_next(now)
                return 'STEP_ARRIVED' if event['strict'] else 'STEP_ARRIVED_UNDERHEIGHT'
            if not ready:
                self._pause()
                return 'STEP_ARRIVED_PAUSED'
            if self.request_time is None:
                self._publish_next(now)
            if now-self.request_time>=PREVIEW_MIN_S:
                self._launch_next(now)
                return ('STEP_ARRIVED_NEXT_LAUNCH' if event['strict']
                        else 'STEP_ARRIVED_UNDERHEIGHT_NEXT_LAUNCH')
            self.state='PREVIEW'
            return 'STEP_ARRIVED_WAIT_PREVIEW'
        return None

    def policy_targets(self,heading):
        c,s=math.cos(heading),math.sin(heading)
        world_to_policy=np.array([[c,s],[-s,c]])
        return (self.targets_xy-self.release_origin_xy)@world_to_policy.T

    def policy_edges(self, heading, targets):
        # A moving retry keeps the original commanded edge and mirror selection.
        current = np.zeros(2)
        if self._moving:
            c, s = math.cos(heading), math.sin(heading)
            current = np.array([[c, s], [-s, c]]) @ (self.targets_xy[0] - self.step_start_xy)
        return current, targets[1] - targets[0]

    def excursion_distance(self,xy):
        p=np.asarray(xy,dtype=np.float64)
        if not self._moving:return float(np.linalg.norm(p-self.targets_xy[0]))
        a=self.step_start_xy;b=self.targets_xy[0];edge=b-a
        fraction=np.clip(np.dot(p-a,edge)/np.dot(edge,edge),0.,1.)
        return float(np.linalg.norm(p-(a+fraction*edge)))

def world_yaw_rad(quat_xyzw):
    """Return standard body-X heading in the Vicon world frame."""
    matrix = R.from_quat(quat_xyzw).as_matrix()
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


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


def quaternion_matrix(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError('Invalid wxyz quaternion')
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    w, x, y, z = q
    matrix = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return q, matrix

class Observation73:
    """NumPy equivalent of crossing_obs73_quat, with inactive crossing14.

    Landing tracking is separate: update the queue before assembling the
    contact-tick observation. Geometry remains a proxy without a spring sensor.
    """
    def __init__(self,course):
        if course['observation_version']!=VERSION or course['task']!='obs73_omni_ground_targets_v1':
            raise ValueError('Expected the current ground-target quaternion73 course')
        self.target_height=course['stationary_pretraining']['apex_target_m']
        self.reset()

    def reset(self):
        self.states=np.zeros((2,13),dtype=np.float32)
        self.actions=np.zeros((2,4),dtype=np.float32)
        self.state_buffer=np.zeros((4,13),dtype=np.float32)
        self.action_buffer=np.zeros((3,4),dtype=np.float32)
        self.recent_action=np.zeros(4,dtype=np.float32)
        self.snapshot=None
        self.phase=0;self.air_steps=0;self.peak=0.;self.apex_seen=False
        self.completed_jumps=0;self.last_landing=None
        self.precontact_motion=None

    def track_landing(self,position,q_wxyz,velocity_world,target_xy,spring_q=None,omega_body=None):
        p=np.asarray(position,dtype=np.float64);v=np.asarray(velocity_world,dtype=np.float64)
        _,rotation=quaternion_matrix(q_wxyz)
        uz=rotation[2,2]
        if uz<=.2:raise ValueError('Invalid body up axis')
        spring=np.clip((.012+.273*uz-p[2])/uz,0.,.087) if spring_q is None else float(spring_q)
        if not math.isfinite(spring) or not 0<=spring<=.087001:raise ValueError('Invalid spring compression')
        foot=p+rotation@np.array([0.,0.,-.273+spring])
        before=self.phase;contact=spring>.002
        launch=before==1 and spring<=.001 and v[2]>0
        airborne=before in (2,3);touch=airborne and contact
        if airborne and not contact and foot[2]>.032 and v[2]<0 and omega_body is not None:
            # Foot radius12mm plus20mm clearance avoids using the first
            # impact vibration before the geometric contact threshold fires.
            omega=np.asarray(omega_body,dtype=np.float64)
            if omega.shape!=(3,) or not np.isfinite(omega).all():raise ValueError('Invalid approach gyro')
            forward=rotation[:,0];derivative=rotation@np.cross(omega,[1.,0.,0.])
            heading_rate=(forward[0]*derivative[1]-forward[1]*derivative[0])/max(np.dot(forward[:2],forward[:2]),.01)
            self.precontact_motion=(float(np.linalg.norm(omega[:2])),math.degrees(heading_rate))
        if airborne:
            self.peak=max(self.peak,p[2]);self.apex_seen|=v[2]<=0
        self.last_landing=None
        if touch and before==3 and self.air_steps>=5 and self.apex_seen:
            self.completed_jumps+=1
            error=float(np.linalg.norm(foot[:2]-target_xy))
            tilt=math.degrees(math.acos(np.clip(uz,-1.,1.)))
            self.last_landing=dict(jump=self.completed_jumps,apex_m=self.peak,
                foot_error_m=error,tilt_deg=tilt,
                strict=bool(self.peak>=1. and error<=LANDING_RADIUS_M and tilt<=15.))
            heading=math.degrees(math.atan2(rotation[1,0],rotation[0,0]))
            rate,heading_rate=self.precontact_motion or (math.inf,math.inf)
            self.last_landing.update(heading_error_deg=heading,
                predicted_heading_error_deg=heading+.14*heading_rate,
                roll_pitch_rate_rad_s=rate,foot_policy_xy_m=foot[:2].tolist())
        if launch:
            self.air_steps,self.peak,self.apex_seen=1,float(p[2]),False
            self.precontact_motion=None
        continuing=airborne and not contact
        self.air_steps+=int(continuing)
        if (before==0 and contact) or touch:self.phase=1
        if launch:self.phase=2
        if continuing and self.air_steps>=3:self.phase=3
        if touch:self.air_steps,self.apex_seen=0,False
        return self.last_landing

    def observe(self,position,q_wxyz,velocity_world,omega_body,targets_xy=None):
        p=np.asarray(position,dtype=np.float64);v=np.asarray(velocity_world,dtype=np.float64)
        omega=np.asarray(omega_body,dtype=np.float64)
        targets=np.zeros((2,2)) if targets_xy is None else np.asarray(targets_xy,dtype=np.float64)
        if any(a.shape!=(3,) or not np.isfinite(a).all() for a in (p,v,omega)):
            raise ValueError('Invalid A-point state')
        if targets.shape!=(2,2) or not np.isfinite(targets).all():raise ValueError('Invalid target pair')
        q,rotation=quaternion_matrix(q_wxyz);uz=rotation[2,2]
        vb=rotation.T@v
        three=targets[[0,1,1]]
        errors=(np.c_[three-p[:2],np.zeros(3)]@rotation)[:,:2].ravel()
        self.snapshot=np.r_[vb,omega,q,v].astype(np.float32)
        shadow=float(abs(uz)>=.2 and np.clip((.015-(p[2]-.285*uz))/uz,0.,.086)>.002)
        obs=np.r_[errors,vb,q,omega,v,self.states.ravel(),self.actions.ravel(),
            self.recent_action,self.target_height-p[2],shadow,np.zeros(14)].astype(np.float32)
        if obs.shape!=(73,) or not np.isfinite(obs).all():raise ValueError('Invalid quaternion73 vector')
        return obs

    def accept(self,action):
        action=np.asarray(action,dtype=np.float32)
        if self.snapshot is None or action.shape!=(4,) or not np.isfinite(action).all():
            raise ValueError('Accept requires an observation and finite4-motor action')
        self.state_buffer[:-1]=self.state_buffer[1:].copy()
        self.state_buffer[-1]=self.snapshot
        self.states[:]=self.state_buffer[[0,2]]
        self.action_buffer[:-1]=self.action_buffer[1:].copy()
        self.action_buffer[-1]=self.recent_action
        self.actions[:]=self.action_buffer[[0,2]]
        self.recent_action=np.clip(action,-1.,1.)
        self.snapshot=None


def obs73_state(pos, quat_xyzw, vel, anchor, heading):
    inverse_heading = R.from_euler('z', -heading)
    position = inverse_heading.apply(np.asarray(pos)-np.r_[anchor, VICON_GROUND_Z_M])
    velocity = inverse_heading.apply(vel)
    q = (inverse_heading*R.from_quat(quat_xyzw)).as_quat()
    return position, q[[3,0,1,2]], velocity


def obs73_contact(pos, quat_xyzw):
    uz = R.from_quat(quat_xyzw).as_matrix()[2,2]
    if uz <= .2:
        return 0.0
    spring = np.clip((.012+.273*uz-(pos[2]-VICON_GROUND_Z_M))/uz, 0., .087)
    return float(spring > .002)


def check_waypoint_retries():
    queue = StationaryWaypointQueue()
    hit = dict(strict=True, foot_error_m=LANDING_RADIUS_M, tilt_deg=2.,
               heading_error_deg=0., predicted_heading_error_deg=0., roll_pitch_rate_rad_s=.2)
    miss = dict(hit, strict=False, foot_error_m=.069)
    out_of_envelope = (
        dict(hit, heading_error_deg=-10.001),
        dict(hit, predicted_heading_error_deg=10.001),
        dict(hit, roll_pitch_rate_rad_s=1.201),
    )

    assert queue.request_step(0., b'a') and queue.request_step(.2, b'd')
    assert queue.on_landing(hit, .281) is None
    assert queue.on_landing(miss, 1.3) == 'STEP_MISS_RETRY_HOLD'
    assert queue.state == 'PAUSED' and queue.remaining_steps == 2
    np.testing.assert_array_equal(queue.targets_xy, [[0., 0.], [0., 0.]])
    np.testing.assert_array_equal(queue.pending_targets, [[0., STEP_M], [0., 0.]])
    assert queue.on_landing(out_of_envelope[0], 2.) == 'STEP_PAUSED_HOLD'
    assert queue.on_landing(hit, 3.) == 'STEP_PREVIEW_RESUMED'
    assert queue.on_landing(hit, 3.44) is None
    assert queue.on_landing(hit, 3.46) == 'STEP_LAUNCH'
    np.testing.assert_array_equal(queue.targets_xy, [[0., STEP_M], [0., 0.]])

    assert queue.on_landing(miss, 4.) == 'STEP_MISS_RETRY'
    assert queue.state == 'RETRY' and queue.remaining_steps == 2
    assert queue.request_step(4.1, b'a')
    assert queue.on_landing(miss, 5.) == 'STEP_MISS_RETRY'
    assert queue.remaining_steps == 3 and queue.request_time is None
    np.testing.assert_array_equal(queue.targets_xy, [[0., STEP_M], [0., STEP_M]])
    np.testing.assert_array_equal(queue.pending_targets, [[0., 0.], [0., STEP_M]])
    current, preview = queue.policy_edges(0., queue.policy_targets(0.))
    np.testing.assert_array_equal(current, [0., STEP_M])
    np.testing.assert_array_equal(preview, [0., 0.])
    assert queue.excursion_distance([0., .1]) == 0.0
    assert queue.excursion_distance([.31, .1]) > XY_ABORT_M

    for now, unsafe in enumerate(out_of_envelope, 6):
        assert queue.on_landing(unsafe, now) == 'STEP_RETRY_WAIT_READY'
        assert queue.state == 'RETRY' and queue.remaining_steps == 3
    assert queue.on_landing(hit, 10.) == 'STEP_ARRIVED_WAIT_PREVIEW'
    np.testing.assert_array_equal(queue.targets_xy, [[0., STEP_M], [0., 0.]])
    assert queue.on_landing(hit, 10.44) is None
    assert queue.on_landing(hit, 10.46) == 'STEP_LAUNCH'
    np.testing.assert_array_equal(queue.targets_xy, [[0., 0.], [0., STEP_M]])
    assert queue.on_landing(hit, 11.5) == 'STEP_ARRIVED_NEXT_LAUNCH'
    assert queue.on_landing(miss, 12.5) == 'STEP_MISS_RETRY'
    assert queue.remaining_steps == 1
    assert queue.on_landing(hit, 13.5) == 'STEP_ARRIVED'
    assert queue.remaining_steps == 0 and queue.state == 'HOLD'

    queue.request_step(14., b'd')
    assert queue.on_landing(miss, 15.) == 'STEP_MISS_RETRY_HOLD'
    queue.cancel()
    assert queue.remaining_steps == 0 and queue.state == 'HOLD'


def check_imu_framing():
    def packet(sequence):
        return IMU_SYNC[:2] + PAYLOAD_STRUCT.pack(
            0xAA, 0, sequence, 0, 0, 16384, 0, 0, 0, 12., 0., sequence, 0, 0
        )

    first = packet(10)
    second = packet(14)
    decoder = ImuFrameDecoder()
    assert not decoder.feed(b'\xaa\xbb\x00' + first[:1])
    assert not decoder.feed(first[1:10])
    assert [frame[11] for frame in decoder.feed(first[10:] + second)] == [10, 14]
    assert decoder.discarded_bytes == 3 and not decoder.buffer
    sequences = range(16, 140, 4)
    burst = b''.join(packet(sequence) for sequence in sequences)
    assert len(burst) > 1024
    assert [frame[11] for frame in decoder.feed(burst)] == list(sequences)


def obs73_self_check():
    check_imu_framing()
    observer = Observation73(OBS73_COURSE)
    policy = V73Controller()
    zero_edge = np.zeros(2)
    actions = []
    frames = []
    for _ in range(2):
        policy.reset()
        observer.reset()
        run = []
        for i in range(8):
            obs = observer.observe([0., 0., 1.], [1., 0., 0., 0.], [.1, 0., -.6],
                                   [0., 0., 0.], [[0., 0.], [-STEP_M, 0.]])
            edge = [-STEP_M, 0.] if i == 0 else zero_edge
            action = policy.step(obs, zero_edge, edge)
            assert policy.last_mirror and action.shape == (4,) and not obs[59:].any()
            for delay, offset in ((4, 45), (2, 49), (1, 53)):
                expected = run[i - delay] if i >= delay else np.zeros(4)
                np.testing.assert_array_equal(obs[offset:offset + 4], expected)
            observer.accept(action)
            run.append(action)
        actions.append(run)
    np.testing.assert_array_equal(actions[0], actions[1])
    policy.reset()
    assert not policy.last_mirror

    class FakeSerial:
        def write(self, data):
            frames.append(data)
            return len(data)

    port = FakeSerial()
    send_pwm(port, DISARMED_PWM)
    send_pwm(port, action_to_pwm(np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32)))
    assert [PWM_FRAME.unpack(frame) for frame in frames] == [
        (0x55, 0xAA, 0, 0, 0, 0), (0x55, 0xAA, 0, 500, 999, 750),
    ]

    arm_events = iter([[(True, 72, 104), (True, 72, 104)], [(True, 72, 104)],
                       [(False, 72, 104), (True, 72, 104)]])
    arm = ConsoleKeyPresses(lambda: next(arm_events, []))
    assert arm.kbhit() and arm.getch() == b'h' and not arm.kbhit()
    assert arm.kbhit() and arm.getch() == b'h'
    for key, direction in StationaryWaypointQueue.DIRECTIONS.items():
        queue = StationaryWaypointQueue()
        queue.lock_here([1.,2.])
        assert queue.request_step(0.,key)
        target = np.array([1.,2.])+STEP_M*np.array(direction)
        np.testing.assert_allclose(queue.targets_xy[1],target)
        heading = math.pi / 2.
        rotated_edge = STEP_M * np.array([direction[1], -direction[0]])
        current, preview = queue.policy_edges(heading, queue.policy_targets(heading))
        np.testing.assert_array_equal(current, zero_edge)
        np.testing.assert_allclose(preview, rotated_edge, atol=1e-12)
        event = dict(strict=True,foot_error_m=.02,tilt_deg=2.,heading_error_deg=0.,
                     predicted_heading_error_deg=0.,roll_pitch_rate_rad_s=.2,foot_world_xy_m=[1.,2.])
        assert queue.on_landing(event,.2) is None
        assert queue.on_landing(event,.5)=='STEP_LAUNCH'
        np.testing.assert_allclose(queue.targets_xy[0],target)
        current, preview = queue.policy_edges(heading, queue.policy_targets(heading))
        np.testing.assert_allclose(current, rotated_edge, atol=1e-12)
        np.testing.assert_array_equal(preview, zero_edge)
        assert queue.on_landing(event,1.5)=='STEP_ARRIVED' and queue.state=='HOLD'
    # Rapid distinct W presses survive typematic filtering at the real input API.
    batches = iter([[(True,87,119),(True,87,119)],
                    [(False,87,119),(True,87,119),(False,87,119),(True,87,119)],
                    [(True,87,119)], [(False,87,119)], []])
    keyboard = ConsoleKeyPresses(lambda: next(batches, []))
    queue = StationaryWaypointQueue()
    for now in (0.,.1,.2,.3):
        if keyboard.kbhit():
            while keyboard._buffer:
                queue.request_step(now,keyboard.getch())
    assert queue.remaining_steps == 3
    np.testing.assert_allclose(queue.pending_targets, [[STEP_M,0.],[2*STEP_M,0.],[3*STEP_M,0.]])
    assert queue.on_landing(event,.2) is None
    assert queue.on_landing(event,.5)=='STEP_LAUNCH'
    np.testing.assert_allclose(queue.targets_xy,[[STEP_M,0.],[2*STEP_M,0.]])
    # Verify actual policy observation exposes the next point, not just the queue.
    observer.reset()
    obs = observer.observe([STEP_M,0.,1.], [1.,0.,0.,0.], [0.,0.,0.], [0.,0.,0.],queue.policy_targets(0.))
    np.testing.assert_allclose(obs[:6],[0.,0.,STEP_M,0.,STEP_M,0.])
    observer.accept(np.zeros(4,dtype=np.float32))
    saved_states = observer.state_buffer.copy()
    # Real crash precursor at 62.09s: a reached point with unsafe world yaw.
    unsafe = dict(event,foot_error_m=min(.0873, LANDING_RADIUS_M * .9),tilt_deg=3.7326,
                  heading_error_deg=-15.5678,predicted_heading_error_deg=-15.)
    assert queue.on_landing(unsafe,1.5)=='STEP_ARRIVED_PAUSED'
    assert queue.state=='PAUSED' and queue.remaining_steps==2
    np.testing.assert_allclose(queue.targets_xy,[[STEP_M,0.]]*2)
    assert queue.on_landing(event,2.)=='STEP_PREVIEW_RESUMED'
    assert queue.on_landing(event,2.2) is None
    for now, current, future, remaining, transition in ((2.5,2,3,2,'STEP_LAUNCH'),(3.5,3,3,1,'STEP_ARRIVED_NEXT_LAUNCH')):
        assert queue.on_landing(event,now)==transition
        np.testing.assert_allclose(queue.targets_xy,[[current*STEP_M,0.],[future*STEP_M,0.]])
        assert queue.remaining_steps==remaining
    assert queue.on_landing(event,4.5)=='STEP_ARRIVED'
    assert queue.state=='HOLD' and queue.remaining_steps==0
    np.testing.assert_allclose(queue.targets_xy,[[3*STEP_M,0.]]*2)
    np.testing.assert_array_equal(observer.state_buffer,saved_states)
    # A command in the last instant of flight waits for preview at the reached point.
    queue.request_step(5.,b'w'); queue.on_landing(event,5.5)
    queue.request_step(6.4,b'a')
    assert queue.on_landing(event,6.5)=='STEP_ARRIVED_WAIT_PREVIEW'
    assert queue.on_landing(event,7.5)=='STEP_LAUNCH'
    np.testing.assert_allclose(queue.targets_xy[0],[4*STEP_M,STEP_M])
    queue.request_step(7.6,b's'); queue.cancel()
    assert queue.remaining_steps==0 and queue.state=='HOLD' and not queue.first_hop_complete
    np.testing.assert_allclose(queue.targets_xy[0],queue.targets_xy[1])
    check_waypoint_retries()
    # The real landing detector must supply the same guard fields. A large
    # gyro impulse at contact must not replace the last free-air body rate.
    tracker = Observation73(OBS73_COURSE)
    yaw = math.radians(-15.)
    q = [math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    states = [(.281,-2.),(.30,2.),(.5,1.5),(.8,1.),(1.05,.1),
              (1.,-.3),(.7,-1.),(.4,-2.),(.281,-2.5)]
    for i,(height,vz) in enumerate(states):
        landing = tracker.track_landing([0.,0.,height],q,[0.,0.,vz],[0.,0.],
            omega_body=[20.,20.,20.] if i==len(states)-1 else [0.,.2,.3])
    assert landing is not None and not StationaryWaypointQueue.ready(landing)
    assert abs(landing['heading_error_deg']+15.)<1e-6
    assert abs(landing['roll_pitch_rate_rad_s']-.2)<1e-6
    print('V88_TRIAL_SELF_CHECK=passed (LSTM, OBS73 history, retries, IMU framing and PWM protocol)')

def run_trial(*, powered):
    import serial

    teacher_policy = V73Controller()
    keyboard = ConsoleKeyPresses()
    mode = 'live' if powered else 'shadow'
    deployment_version = f'{DEPLOYMENT_VERSION}_{mode}'
    print('V88 LIVE: motors can spin after arming.' if powered else 'V88 SHADOW: serial PWM stays zero.',
          flush=True)
    print('After stationary IMU calibration, press H twice. '
          'W:+X S:-X A:+Y D:-Y in Vicon world, 0.5 m per press. SPACE stops, E exits.', flush=True)
    vicon = ViconReceiver(VICON_HOST)
    try:
        if not vicon.connected:
            raise RuntimeError(f'Vicon connection failed: {vicon.last_error}')
        frame_deadline = time.perf_counter() + 10.0
        initial_frame = vicon.latest_frame()
        while initial_frame is None and time.perf_counter() < frame_deadline:
            time.sleep(0.05)
            initial_frame = vicon.latest_frame()
        if initial_frame is None:
            raise RuntimeError(f'No tracked Vicon frame: {vicon.last_error}')
    except BaseException:
        vicon.close()
        raise

    eskf = ErrorStateKalmanFilter(initial_frame["pos"], initial_frame["quat"])

    log_data = []
    log_headers = trial_log_headers()
    log_columns = {name: i for i, name in enumerate(log_headers)}
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
    policy_heading_reference_rad = 0.0
    waypoint_queue = StationaryWaypointQueue()
    spatial_observer = Observation73(OBS73_COURSE)
    current_target_x, current_target_y = 0.0, 0.0
    previous_contact = True
    last_policy_clock = None
    candidate_pwm = DISARMED_PWM
    startup_ready_ticks = 0
    freefall_ticks = 0
    flight_start_time = None
    session_notice_printed = False
    last_stop_reason = ''
    last_inference_ms = last_motor_write_ms = float('nan')
    key_events = []
    land_log = ('',)*8
    def timing_fields(now,rx=None):
        # Raw GetFrame arrivals are a PC receive clock, not camera timestamps.
        frame_clock, recv_gap_ms = rx if rx is not None else getattr(vicon,'receive_clock',(None,float('nan')))
        return [recv_gap_ms, (now-frame_clock)*1000 if frame_clock is not None else float('nan'),
                (now-last_policy_clock)*1000 if last_policy_clock is not None else float('nan'),
                last_inference_ms, last_motor_write_ms, ';'.join(key_events), *land_log]
    soft_tilt_bad_ticks = 0
    hard_tilt_bad_ticks = 0
    touchdown_count = 0

    # Log every Vicon frame and mark the ones with a fresh graph inference.
    last_teacher_obs_log = np.zeros(73, dtype=np.float32)
    last_teacher_action_log = np.zeros(4, dtype=np.float32)
    last_allocated_action_log = np.zeros(4, dtype=np.float32)
    last_edges = np.zeros(4)
    last_pwm_raw_log = np.zeros(4, dtype=np.float32)
    last_pwm_final_log = np.zeros(4, dtype=np.float32)
    last_policy_vel_w_log = np.zeros(3, dtype=np.float32)
    policy_obs_valid = 0
    policy_update_this_frame = 0

    def reset_policy_state():
        nonlocal previous_contact, startup_ready_ticks, freefall_ticks
        nonlocal flight_start_time, soft_tilt_bad_ticks, hard_tilt_bad_ticks
        nonlocal session_notice_printed
        nonlocal touchdown_count, last_policy_clock, candidate_pwm
        nonlocal policy_obs_valid, policy_update_this_frame
        teacher_policy.reset()
        spatial_observer.reset()
        waypoint_queue.cancel()
        last_teacher_obs_log.fill(0.0)
        last_teacher_action_log.fill(0.0)
        last_allocated_action_log.fill(0.0)
        last_edges.fill(0.0)
        last_pwm_raw_log.fill(0.0)
        last_pwm_final_log.fill(0.0)
        last_policy_vel_w_log.fill(0.0)
        previous_contact = True
        startup_ready_ticks = 0
        freefall_ticks = 0
        flight_start_time = None
        session_notice_printed = False
        soft_tilt_bad_ticks = 0
        hard_tilt_bad_ticks = 0
        touchdown_count = 0
        last_policy_clock = None
        candidate_pwm = DISARMED_PWM
        policy_obs_valid = 0
        policy_update_this_frame = 0

    def imu_link_fault(now):
        critical = (fc_status_flags | fc_fault_in_batch) & FC_STATUS_CRITICAL_MASK
        if critical:
            return f'FC status fault 0x{critical:04x}'
        if serial_budget_exceeded:
            return 'IMU receive backlog exceeds read budget'
        if not 0 <= imu_buffer.age(now) <= IMU_STALE_TIMEOUT_S:
            return 'IMU stale'
        return None

    def sensor_fault(now, frame):
        if frame is None or not 0 <= now-frame['time_s'] <= VICON_STALE_TIMEOUT_S:
            return 'Vicon stale'
        return imu_link_fault(now)

    def sensors_ready():
        frame = vicon.latest_frame()
        return sensor_fault(time.perf_counter(), frame) is None

    def serial_diagnostics():
        return [fc_timestamp_ms, fc_fault_in_batch, imu_seq_delta, imu_rx_interval_ms,
                imu_rejected_frames, serial_rx_bytes, serial_valid_frames, decoder.discarded_bytes,
                serial_rx_buffer_peak, serial_budget_exceeded_polls]

    def stop_policy(reason):
        nonlocal flight_mode, last_stop_reason, land_log
        stop_rx = getattr(vicon,'receive_clock',(None,float('nan')))
        stop_clock = time.perf_counter()
        stop_timing = timing_fields(stop_clock, stop_rx)
        write_error = None
        try:
            send_pwm(ser, DISARMED_PWM)
        except Exception as exc:
            write_error = exc
            reason = f'{reason} | zero PWM write failed: {exc}'
        reset_policy_state()
        flight_mode = 'IDLE' if not is_calibrating else 'CALIBRATE'
        last_stop_reason = reason
        # Persist the actual stop cause even when the branch skips the next
        # Vicon row or exits. Pose is the most recent logged frame, not a new
        # measurement; Policy_Obs_Valid/Update are zero for this event row.
        if log_data and start_time is not None:
            event_row = log_data[-1].copy()
            event_row[0] = time.perf_counter()-start_time
            event_row[16:20] = DISARMED_PWM if write_error is None else ('',) * 4
            event_row[20:22] = [0, 0]
            snapshot = {
                'IMU_Age_s': imu_buffer.age(stop_clock),
                'IMU_Seq': imu_sequence,
                'IMU_Seq_Gap': imu_seq_gap,
                'FC_Status_Flags': fc_status_flags,
                'FC_Cmd_Age_ms': fc_cmd_age_ms,
                'Link_Health_OK': int(imu_link_fault(stop_clock) is None),
                'IMU_Degraded': int(imu_degraded),
                'Serial_Parse_Error_Count': decoder.invalid_frames,
            }
            snapshot.update(zip(SERIAL_DIAGNOSTIC_HEADERS, serial_diagnostics()))
            for column, value in snapshot.items():
                event_row[log_columns[column]] = value
            event_row[-17:-3] = stop_timing
            key_events.clear()
            land_log = ('',)*8
            event_row[-3:] = [flight_mode, deployment_version, reason]
            log_data.append(event_row)
        if write_error is not None:
            raise write_error
        print('ZERO PWM: '+reason, flush=True)

    latest_filt_acc = np.zeros(3)
    latest_filt_gyro = np.zeros(3)
    latest_mapped_accel = np.zeros(3)
    latest_mapped_gyro = np.zeros(3)
    latest_raw_g = [0.0, 0.0, 0.0]
    kf_vx, kf_vy, kf_vz = 0.0, 0.0, 0.0
    last_imu_sequence = None
    last_fc_timestamp_ms = None
    imu_sequence = 0
    fc_timestamp_ms = 0
    imu_seq_gap = imu_seq_delta = 0
    imu_rx_interval_ms = float('nan')
    last_valid_imu_rx = None
    imu_rejected_frames = 0
    fc_status_flags = fc_fault_in_batch = 0
    fc_cmd_age_ms = 0
    link_health_ok = False
    decoder = ImuFrameDecoder()
    serial_rx_bytes = serial_valid_frames = 0
    serial_rx_buffer_peak = serial_budget_exceeded_polls = 0
    serial_budget_exceeded = False
    imu_degraded = False
    output_tracking_error = np.zeros(3)
    delayed_state_miss_count = 0
    estimator_predict_count = 0
    deploy_tilt_deg = 0.0
    deploy_xy_error_m = 0.0
    deploy_pwm_spread = 0.0
    deploy_safety_code = 0

    last_calc_time = initial_frame["time_s"]
    last_imu_predict_time = None
    vicon_tick = 0


    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0, write_timeout=0.05)
        send_pwm(ser, DISARMED_PWM)
        ser.reset_input_buffer()

        start_time = None
        last_zero_send = -1.0
        next_status = 0.0

        while True:
            guard_now = time.perf_counter()
            if is_calibrating and guard_now-last_zero_send >= .020:
                send_pwm(ser, DISARMED_PWM)
                last_zero_send = guard_now
            if guard_now >= next_status and flight_mode != 'JUMP':
                print(f'STATUS {flight_mode}: calibration={calib_count}/{CALIB_SAMPLES}; '
                      f'Vicon/IMU ready={sensors_ready()}', flush=True)
                next_status = guard_now+1.0
            if flight_mode == 'JUMP' and flight_start_time is not None:
                if guard_now-flight_start_time >= SESSION_NOTICE_S and not session_notice_printed:
                    print('15s reached; control continues. SPACE: motors off; E: exit.', flush=True)
                    session_notice_printed = True
                if last_policy_clock is not None and guard_now-last_policy_clock > POLICY_GAP_ABORT_S:
                    stop_policy('policy control gap')
            # ==== 1. 键盘监听 ====
            if keyboard.kbhit():
                key = keyboard.getch()
                key_lower = key.lower()
                if key_lower == b'e':
                    stop_policy('operator exit')
                    break
                elif key_lower == b' ':
                    stop_policy('operator SPACE')
                elif key_lower in (b'w', b'a', b's', b'd'):
                    key_events.append(key_lower.decode()+':'+(flight_mode if flight_mode!='JUMP' else waypoint_queue.state))
                    if flight_mode != 'JUMP':
                        print('Move key ignored: waiting for JUMP state.', flush=True)
                    else:
                        waypoint_queue.request_step(time.perf_counter(), key_lower)
                        print(f'STEP_COMMAND: key={key_lower.decode().upper()}; world delta='
                              f'{(STEP_M*np.array(waypoint_queue.DIRECTIONS[key_lower])).tolist()}; '
                                  f'state={waypoint_queue.state}; '
                              f'remaining={waypoint_queue.remaining_steps}; '
                              f'next={waypoint_queue.targets_xy[1].tolist()}; '
                              f'last_queued={waypoint_queue.pending_targets[-1].tolist()}', flush=True)
                elif key_lower == b'h':
                    if is_calibrating or not sensors_ready():
                        print(f'H BLOCKED: calibration={calib_count}/{CALIB_SAMPLES}; '
                              f'Vicon/IMU ready={sensors_ready()}', flush=True)
                    elif flight_mode == 'IDLE':
                        flight_mode = 'PREHOVER'
                        policy_heading_reference_rad = (
                            world_yaw_rad(vicon.quat)
                            if ZERO_POLICY_YAW_AT_RELEASE else 0.0
                        )
                        waypoint_queue.lock_here(vicon.pos[:2])
                        current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                        reset_policy_state()
                        print("==== 预稳定模式：已锁定位置，电机零油门待命；再按 H 进入释放模式 ====")
                    elif flight_mode == 'PREHOVER':
                        flight_mode = 'JUMP_STARTUP'
                        policy_heading_reference_rad = (
                            world_yaw_rad(vicon.quat)
                            if ZERO_POLICY_YAW_AT_RELEASE else 0.0
                        )
                        waypoint_queue.lock_here(vicon.pos[:2])
                        current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                        reset_policy_state()
                        print("==== 等待释放或稳定放置，进入 JUMP ====")
                    else:
                        stop_policy('operator H')


            # Drain a bounded chunk and keep incomplete frames for the next read.
            waiting = ser.in_waiting
            serial_rx_buffer_peak = max(serial_rx_buffer_peak, waiting)
            serial_budget_exceeded = waiting > SERIAL_READ_LIMIT
            serial_budget_exceeded_polls += int(serial_budget_exceeded)
            fc_fault_in_batch = 0
            if waiting:
                chunk = ser.read(min(waiting, SERIAL_READ_LIMIT))
                serial_rx_bytes += len(chunk)
                for data in decoder.feed(chunk):
                    serial_valid_frames += 1
                    fc_status_flags = int(data[12])
                    fc_cmd_age_ms = int(data[13])
                    fc_fault_in_batch |= fc_status_flags & FC_STATUS_CRITICAL_MASK
                    imu_sample_valid = (fc_status_flags & FC_STATUS_IMU_READ_FAIL) == 0
                    imu_degraded = (fc_status_flags & FC_STATUS_DEGRADED_MASK) != 0

                    imu_sequence = int(data[11])
                    fc_timestamp_ms = int(data[2])
                    imu_seq_delta = ((imu_sequence - last_imu_sequence) & 0xFFFF
                                     if last_imu_sequence is not None else 0)
                    imu_seq_gap = max(0, imu_seq_delta - LEGACY_IMU_SEQ_STEP)
                    if last_imu_sequence is not None:
                        fc_delta = (fc_timestamp_ms - last_fc_timestamp_ms) & 0xFFFFFFFF
                        if imu_seq_delta == 0 or imu_seq_delta >= 0x8000 or fc_delta >= 0x80000000:
                            imu_rejected_frames += 1
                            # Rebase after an inactive receiver has timed out.
                            # The suspect packet itself never refreshes IMU age.
                            if (flight_mode in ('IDLE', 'CALIBRATE') and
                                    imu_buffer.age(time.perf_counter()) > IMU_STALE_TIMEOUT_S):
                                last_imu_sequence = last_fc_timestamp_ms = None
                            continue
                    last_imu_sequence = imu_sequence
                    last_fc_timestamp_ms = fc_timestamp_ms

                    if not imu_sample_valid:
                        continue
                    raw_ax_o, raw_ay_o, raw_az_o = (value / 16384 * 9.81 for value in data[3:6])
                    raw_gx_o, raw_gy_o, raw_gz_o = (math.radians(value / 16.4) for value in data[6:9])
                    latest_mapped_accel = mapped_accel = np.array([-raw_ay_o, raw_ax_o, raw_az_o])
                    latest_mapped_gyro = mapped_gyro = np.array([-raw_gy_o, raw_gx_o, raw_gz_o])
                    latest_vbat, latest_current = data[9:11]

                    if is_calibrating:
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
                            print("传感器校准成功。按 H 进入准备状态。")
                        continue

                    clean_acc = bias_estimator.correct_accel(mapped_accel)
                    clean_gyro = bias_estimator.correct_gyro(mapped_gyro)
                    latest_filt_acc = clean_acc if not USE_PC_IMU_LPF else accel_lpf.apply(clean_acc)
                    latest_filt_gyro = clean_gyro if not USE_PC_IMU_LPF else gyro_lpf.apply(clean_gyro)
                    imu_rx_time = time.perf_counter()
                    if last_valid_imu_rx is not None:
                        imu_rx_interval_ms = (imu_rx_time - last_valid_imu_rx) * 1000
                    last_valid_imu_rx = imu_rx_time
                    imu_buffer.add(fc_timestamp_ms, latest_filt_acc, latest_filt_gyro, imu_rx_time,
                                   degraded=imu_degraded)
                    if last_imu_predict_time is not None:
                        imu_dt = imu_rx_time - last_imu_predict_time
                        if 0.001 <= imu_dt <= IMU_STALE_TIMEOUT_S:
                            eskf.predict(imu_dt, latest_filt_acc, latest_filt_gyro,
                                         imu_degraded=imu_degraded)
                            estimator_predict_count += 1
                    last_imu_predict_time = imu_rx_time
                    estimator_replay_buffer.save_state(imu_rx_time, eskf)
                    latest_raw_g = [raw_gx_o, raw_gy_o, raw_gz_o]

            if flight_mode in ('PREHOVER', 'JUMP_STARTUP', 'JUMP'):
                frame = vicon.latest_frame()
                fault = sensor_fault(time.perf_counter(), frame)
                if fault:
                    stop_policy(fault)

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
                link_health_ok = imu_link_fault(curr_calc_time) is None
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
                # Define this every Vicon tick because the CSV is logged at
                # 200 Hz while the policy updates at 100 Hz.
                policy_target_xy = waypoint_queue.targets_xy[0].copy()

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
                        deploy_safety_code = 2
                        stop_policy('hard tilt')
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

                policy_update_this_frame = 0
                if flight_mode not in ('CALIBRATE', 'IDLE') and link_health_ok:
                    # 100Hz 策略降频执行
                    if vicon_tick % 2 == 0:
                        if flight_mode == 'JUMP' and waypoint_queue.excursion_distance(pos_w[:2]) > XY_ABORT_M:
                            stop_policy('XY excursion')
                            continue
                        current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                        vel_w = policy_vel_w
                        ang_vel_b = policy_gyro_b
                        is_contact = obs73_contact(pos_w, latest_pure_q)
                        bias_estimator.observe_static(
                            latest_mapped_accel,
                            latest_mapped_gyro,
                            r_mat.inv().apply([0, 0, 9.81]),
                            is_contact > 0.5
                        )

                        policy_obs_valid = 0
                        if flight_mode == 'JUMP':
                            p73, q73, v73 = obs73_state(policy_pos_w, latest_pure_q, vel_w,
                                                      waypoint_queue.release_origin_xy, policy_heading_reference_rad)
                            old_targets = waypoint_queue.policy_targets(policy_heading_reference_rad)
                            scored_target_world_xy = waypoint_queue.targets_xy[0].copy()
                            landing = spatial_observer.track_landing(p73, q73, v73, old_targets[0], omega_body=ang_vel_b)
                            if landing is not None:
                                c, s = math.cos(policy_heading_reference_rad), math.sin(policy_heading_reference_rad)
                                landing['foot_world_xy_m'] = (waypoint_queue.release_origin_xy +
                                    np.array([[c,-s],[s,c]]) @ np.asarray(landing['foot_policy_xy_m'])).tolist()
                            transition = waypoint_queue.on_landing(landing, curr_calc_time)
                            if landing is not None:
                                land_log = (transition or 'HOLD', int(waypoint_queue.ready(landing)),
                                    landing['heading_error_deg'],landing['predicted_heading_error_deg'],
                                    landing['roll_pitch_rate_rad_s'],landing['foot_error_m'],*scored_target_world_xy)
                            if landing is not None:
                                print('STEP_LANDING: '+str(dict(landing, state=waypoint_queue.state,
                                    transition=transition, remaining=waypoint_queue.remaining_steps,
                                    scored_target_world_xy=scored_target_world_xy.tolist(),
                                    target_world_xy=waypoint_queue.targets_xy[0].tolist())), flush=True)
                            if is_contact > 0.5 and not previous_contact:
                                touchdown_count += 1
                                # The release drop is excluded from the touchdown tilt guard.
                                if touchdown_count >= 2 and deploy_tilt_deg > TOUCHDOWN_TILT_ABORT_DEG:
                                    stop_policy(f'touchdown tilt {deploy_tilt_deg:.1f} deg')
                                    continue

                            current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                            policy_target_xy = waypoint_queue.targets_xy[0].copy()
                            targets = waypoint_queue.policy_targets(policy_heading_reference_rad)
                            current_edge, preview_edge = waypoint_queue.policy_edges(
                                policy_heading_reference_rad, targets)
                            teacher_obs = spatial_observer.observe(p73, q73, v73, ang_vel_b, targets)
                            inference_start = time.perf_counter()
                            try:
                                allocated_action = teacher_policy.step(teacher_obs, current_edge, preview_edge)
                            except (ValueError, RuntimeError) as exc:
                                stop_policy(f'V88 inference failed: {exc}')
                                continue
                            last_inference_ms = (time.perf_counter() - inference_start) * 1000
                            last_teacher_obs_log[:] = teacher_obs
                            last_teacher_action_log[:] = teacher_policy.last_raw
                            last_allocated_action_log[:] = allocated_action
                            last_edges[:] = np.r_[current_edge, preview_edge]
                            spatial_observer.accept(allocated_action)
                            policy_obs_valid = 1
                            policy_update_this_frame = 1

                            last_policy_clock = curr_calc_time
                            last_pwm_raw_log[:] = (allocated_action + 1.0) * (MAX_PWM * 0.5)
                            candidate_pwm = action_to_pwm(allocated_action)
                            last_pwm_final_log[:] = candidate_pwm
                            deploy_pwm_spread = max(candidate_pwm) - min(candidate_pwm)

                        if flight_mode == 'JUMP_STARTUP':
                            # 松手下坠检测：手拿时 Vz≈0；真正松手后 Vz 快速跌破阈值。
                            # 命中后本 tick 保持零油门，下一 tick 从清零的 LSTM 开始起跳。
                            if vel_w[2] < FREE_FALL_TRIGGER_VZ:
                                freefall_ticks += 1
                            else:
                                freefall_ticks = 0
                            if freefall_ticks >= FREE_FALL_TRIGGER_TICKS:
                                flight_mode = 'JUMP'
                                # The robot may have been moved or turned while
                                # armed.  Use its heading at the actual release,
                                # before resetting the policy's recurrent state.
                                if ZERO_POLICY_YAW_AT_RELEASE:
                                    policy_heading_reference_rad = world_yaw_rad(latest_pure_q)
                                waypoint_queue.lock_here(pos_w[:2])
                                current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                                reset_policy_state()
                                previous_contact = is_contact > 0.5
                                flight_start_time = time.perf_counter()
                                last_policy_clock = flight_start_time
                                print(f"==== 检测到释放，开始 V88 {mode.upper()} ====")

                        if flight_mode == 'JUMP_STARTUP':
                            # 原有地上放置路径：接触 + 静止 0.2 s 自动起跳
                            if handoff_ready(pos_w, vel_w, latest_pure_q, is_contact):
                                startup_ready_ticks += 1
                            else:
                                startup_ready_ticks = 0
                            required_ticks = max(1, int(HANDOFF_CONTACT_HOLD_S * 100.0))
                            if startup_ready_ticks >= required_ticks:
                                flight_mode = 'JUMP'
                                if ZERO_POLICY_YAW_AT_RELEASE:
                                    policy_heading_reference_rad = world_yaw_rad(latest_pure_q)
                                waypoint_queue.lock_here(pos_w[:2])
                                current_target_x, current_target_y = waypoint_queue.targets_xy[0]
                                reset_policy_state()
                                previous_contact = is_contact > 0.5
                                flight_start_time = time.perf_counter()
                                last_policy_clock = flight_start_time
                                touchdown_count = 1  # The first touchdown after a ground start counts.
                                print(f"==== 检测到稳定放置，开始 V88 {mode.upper()} ====")

                        previous_contact = is_contact > 0.5

                if flight_mode in ('CALIBRATE', 'IDLE', 'PREHOVER', 'JUMP_STARTUP') or not link_health_ok:
                    deploy_pwm_spread = 0.0

                motor_write_start = time.perf_counter()
                if flight_mode == 'JUMP':
                    fault = sensor_fault(motor_write_start, vicon_frame)
                    if fault:
                        stop_policy(fault)
                        continue
                    if last_policy_clock is None or motor_write_start - last_policy_clock > POLICY_GAP_ABORT_S:
                        stop_policy('policy control gap')
                        continue

                wire_pwm = candidate_pwm if powered and flight_mode == 'JUMP' and policy_obs_valid else DISARMED_PWM
                send_pwm(ser, wire_pwm)
                last_motor_write_ms = (time.perf_counter()-motor_write_start)*1000

                if start_time is None: start_time = time.perf_counter()

                log_row = [
                              time.perf_counter() - start_time, dt,
                              pos_w[0], pos_w[1], pos_w[2],
                              kf_vx, kf_vy, kf_vz,
                              *last_policy_vel_w_log.tolist(),
                              current_target_x, current_target_y, TARGET_Z,
                              policy_target_xy[0], policy_target_xy[1],
                              *wire_pwm,
                              policy_obs_valid, policy_update_this_frame,
                          ] + last_teacher_obs_log.tolist() + last_teacher_action_log.tolist() + \
                              last_allocated_action_log.tolist() + last_pwm_raw_log.tolist() + \
                              last_pwm_final_log.tolist() + [
                              int(teacher_policy.last_mirror), *last_edges,
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
                              decoder.invalid_frames,
                              vicon.error_count,
                              deploy_tilt_deg,
                              deploy_xy_error_m,
                              deploy_pwm_spread,
                              deploy_safety_code,
                              math.degrees(world_yaw_rad(latest_pure_q)),
                              math.degrees(policy_heading_reference_rad),
                              waypoint_queue.state,
                              *waypoint_queue.targets_xy[1].tolist(),
                              waypoint_queue.excursion_distance(pos_w[:2]),
                              waypoint_queue.remaining_steps,
                              *serial_diagnostics(),
                              *timing_fields(time.perf_counter()),
                              flight_mode,
                              deployment_version,
                              last_stop_reason if flight_mode == 'IDLE' else '',
                          ]

                log_data.append(log_row)
                key_events.clear()
                land_log = ('',)*8

    except KeyboardInterrupt:
        if 'ser' in locals() and ser.is_open:
            stop_policy('operator Ctrl+C')
    except Exception as exc:
        if 'ser' in locals() and ser.is_open:
            try:
                stop_policy(f'{type(exc).__name__}: {exc}')
            except Exception as stop_error:
                print(f'Serial stop failed: {stop_error}', flush=True)
        raise
    finally:
        try:
            if 'ser' in locals() and ser.is_open:
                try:
                    for _ in range(5):
                        try:
                            send_pwm(ser, DISARMED_PWM)
                        except Exception as exc:
                            print(f'Serial stop failed: {exc}', flush=True)
                            break
                        time.sleep(0.02)
                finally:
                    ser.close()
        finally:
            vicon.close()
            if log_data:
                write_trial_log(log_data, mode, log_headers)


def trial_log_headers():
    obs_headers = [f'obs_{i}' for i in range(73)]
    action_headers = [f'Policy_Action_Raw_m{i}' for i in range(1, 5)]
    pwm_raw_headers = [f'PWM_Raw_m{i}' for i in range(1, 5)]
    allocated_headers = [f'Policy_Action_Allocated_m{i}' for i in range(1, 5)]
    pwm_final_headers = [f'PWM_FinalFloat_m{i}' for i in range(1, 5)]

    return [
                  'Time_s', 'loop_dt', 'X', 'Y', 'Z', 'EKF_Vel_X', 'EKF_Vel_Y', 'EKF_Vel_Z',
                  'Policy_Vel_X', 'Policy_Vel_Y', 'Policy_Vel_Z',
                  'Target_X', 'Target_Y', 'Target_Z', 'Policy_Target_X', 'Policy_Target_Y',
                  'M1', 'M2', 'M3', 'M4',
                  'Policy_Obs_Valid', 'Policy_Update_This_Frame',
              ] + obs_headers + action_headers + allocated_headers + pwm_raw_headers + pwm_final_headers + [
                  'V73_Use_Mirror', 'Current_Edge_Policy_X', 'Current_Edge_Policy_Y',
                  'Preview_Edge_Policy_X', 'Preview_Edge_Policy_Y',
                  'Raw_IMU_Gx_deg', 'Raw_IMU_Gy_deg', 'Raw_IMU_Gz_deg', 'VBat_V', 'Current_A',
                  'Innov_Accepted', 'Innov_Rejected', 'IMU_Age_s', 'Accel_Bias_Norm', 'Gyro_Bias_Norm',
                  'IMU_Seq', 'IMU_Seq_Gap', 'FC_Status_Flags', 'FC_Cmd_Age_ms', 'Link_Health_OK', 'IMU_Degraded',
                  'Est_Pos_Test_Ratio', 'Est_Vel_Test_Ratio', 'Est_Att_Test_Ratio', 'Est_Fault_Flags', 'Est_Reset_Count',
                  'Output_Track_Ang', 'Output_Track_Vel', 'Output_Track_Pos',
                  'Estimator_Predict_Count', 'Delayed_State_Miss_Count', 'Serial_Parse_Error_Count', 'Vicon_Error_Count',
                  'Deploy_Tilt_Deg', 'Deploy_XY_Error_M', 'Deploy_PWM_Spread', 'Deploy_Safety_Code',
                  'Vicon_Yaw_Deg', 'Policy_Heading_Reference_Deg', 'Step_State', 'Next_Target_X', 'Next_Target_Y', 'Path_Excursion_M', 'Queued_Steps_Remaining', *SERIAL_DIAGNOSTIC_HEADERS, 'Vicon_Rx_Interval_ms', 'Vicon_Rx_Age_ms', 'Policy_Tick_Age_ms', 'Last_Actor_ms', 'Last_Motor_Write_ms', 'Keys_Before_Handler', 'Landing_Transition', 'Landing_Ready', 'Landing_Heading_deg', 'Landing_Pred_Heading_deg', 'Landing_Precontact_RP_rad_s', 'Landing_Foot_Error_m', 'Scored_Target_X', 'Scored_Target_Y', 'Flight_Mode',
                  'Deployment_Version', 'Disarm_Reason'
              ]



def write_trial_log(log_data, mode, headers):
    csv_filename = f"real_obs73_step_x05_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(log_data)
    print(f"V88 {mode.upper()} 日志已保存到 {Path(csv_filename).resolve()}", flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--self-check', action='store_true', help='CPU test without Vicon or serial')
    modes.add_argument('--shadow', action='store_true', help='log Vicon/IMU and transmit only zero PWM')
    args = parser.parse_args()
    if args.self_check:
        obs73_self_check()
    else:
        run_trial(powered=not args.shadow)
