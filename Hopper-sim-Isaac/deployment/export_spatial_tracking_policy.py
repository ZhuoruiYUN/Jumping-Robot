#!/usr/bin/env python3
"""Export the accepted 52-D spatial-tracking recurrent actor to ONNX."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from rsl_rl.modules import ActorCriticRecurrent

try:
    import onnxruntime as ort
except ImportError:
    ort = None


OBS_DIM = 52
HIDDEN_DIM = 256
V12_ACTION_HISTORY = "five_executed_post_limit_motor_actions_oldest_to_newest"
V12_ACTION_SHAPING = {
    "version": "height_spread_yaw_v1",
    "max_pwm_spread": 600.0,
    "airborne_yaw_diagonal_pwm_diff": 140.0,
    "grounded_yaw_diagonal_pwm_diff": 220.0,
    "constraint_projection_iterations": 2,
    "pwm_quantization": "floor_to_integer",
}
OBSERVATIONS = [
    {"slice": [0, 3], "name": "linear_velocity_body", "frame": "body", "units": "m/s"},
    {"slice": [3, 6], "name": "angular_velocity_body", "frame": "body", "units": "rad/s"},
    {"slice": [6, 10], "name": "root_quaternion_world", "order": "wxyz"},
    {"slice": [10, 13], "name": "active_landing_and_apex_error_body", "frame": "body", "units": "m"},
    {"slice": [13, 14], "name": "root_height_world", "units": "m"},
    {"slice": [14, 15], "name": "spring_contact", "definition": "spring_position > 0.002 m"},
    {"slice": [15, 16], "name": "spring_position", "units": "m"},
    {"slice": [16, 17], "name": "spring_velocity", "units": "m/s"},
    {"slice": [17, 37], "name": "last_five_actions", "layout": "oldest_to_newest, F1..F4"},
    {"slice": [37, 39], "name": "current_landing_xy_error_body", "scale": "divide by 0.20 m"},
    {"slice": [39, 41], "name": "next_hop_displacement_body", "scale": "divide by 0.20 m"},
    {"slice": [41, 42], "name": "current_absolute_apex_command", "scale": "divide by 2.0 m"},
    {"slice": [42, 43], "name": "next_absolute_apex_command", "scale": "divide by 2.0 m"},
    {"slice": [43, 46], "name": "path_carrot_error_body", "units": "m", "clip": [-1.0, 1.0]},
    {"slice": [46, 49], "name": "path_unit_tangent_body"},
    {"slice": [49, 52], "name": "next_hop_landing_up_hint_body"},
]


class RecurrentActorExporter(torch.nn.Module):
    def __init__(self, policy: ActorCriticRecurrent):
        super().__init__()
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
        self.actor = copy.deepcopy(policy.actor)

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalizer(obs)
        latent, (h_out, c_out) = self.rnn(normalized.unsqueeze(0), (h_in, c_in))
        return self.actor(latent.squeeze(0)), h_out, c_out


def build_policy(state_dict: dict[str, torch.Tensor]) -> ActorCriticRecurrent:
    has_actor_normalizer = any(key.startswith("actor_obs_normalizer.") for key in state_dict)
    has_critic_normalizer = any(key.startswith("critic_obs_normalizer.") for key in state_dict)
    policy = ActorCriticRecurrent(
        obs={"policy": torch.zeros(1, OBS_DIM)},
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=4,
        actor_obs_normalization=has_actor_normalizer,
        critic_obs_normalization=has_critic_normalizer,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=HIDDEN_DIM,
        rnn_num_layers=1,
        init_noise_std=0.04,
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def export_onnx(policy: ActorCriticRecurrent, output_path: Path) -> None:
    exporter = RecurrentActorExporter(policy).cpu().eval()
    torch.onnx.export(
        exporter,
        (
            torch.zeros(1, OBS_DIM),
            torch.zeros(1, 1, HIDDEN_DIM),
            torch.zeros(1, 1, HIDDEN_DIM),
        ),
        str(output_path),
        export_params=True,
        opset_version=18,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        dynamic_axes={},
        dynamo=False,
    )


def materialize_lstm_sequence_lengths(model_path: Path) -> None:
    """Fill the optional LSTM sequence-length input for strict runtimes."""
    model = onnx.load(model_path)
    changed = False
    for index, node in enumerate(model.graph.node):
        if node.op_type == "LSTM" and len(node.input) > 4 and not node.input[4]:
            name = f"lstm_sequence_lengths_{index}"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray([1], dtype=np.int32), name=name)
            )
            node.input[4] = name
            changed = True
    if changed:
        onnx.save(model, model_path)


def verify_onnx(policy: ActorCriticRecurrent, onnx_path: Path) -> float:
    if ort is None:
        raise RuntimeError("onnxruntime is required for mandatory export parity verification")
    generator = torch.Generator().manual_seed(20260909)
    obs = torch.randn(1, OBS_DIM, generator=generator) * 0.1
    h_in = torch.randn(1, 1, HIDDEN_DIM, generator=generator) * 0.01
    c_in = torch.randn(1, 1, HIDDEN_DIM, generator=generator) * 0.01
    with torch.inference_mode():
        normalized = policy.actor_obs_normalizer(obs)
        latent, (h_out, c_out) = policy.memory_a.rnn(
            normalized.unsqueeze(0), (h_in, c_in)
        )
        actions = policy.actor(latent.squeeze(0))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["actions", "h_out", "c_out"],
        {"obs": obs.numpy(), "h_in": h_in.numpy(), "c_in": c_in.numpy()},
    )
    expected = actions.numpy(), h_out.numpy(), c_out.numpy()
    return max(
        float(np.max(np.abs(got - want)))
        for got, want in zip(actual, expected, strict=True)
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = data["model_state_dict"]
    input_width = int(state_dict["memory_a.rnn.weight_ih_l0"].shape[1])
    if input_width != OBS_DIM:
        raise ValueError(f"Expected a {OBS_DIM}-D spatial policy, got {input_width}")
    contract = (data.get("infos") or {}).get("tracking_recovery_contract") or {}
    if contract.get("obs_dim") != OBS_DIM or contract.get("actions") != 4:
        raise ValueError("Checkpoint does not carry the expected 52-D/four-action contract")
    if (contract.get("simulation_only_phase_collective_baseline", False)
            or contract.get("simulation_only_phase_collective_controller", False)
            or contract.get("simulation_only_apex_bias_controller", False)):
        raise ValueError(
            "Refusing to export a simulation-only phase-collective-baseline checkpoint. "
            "It has not been validated as a deployment controller."
        )
    if contract.get("version") == 12:
        if contract.get("action_history") != V12_ACTION_HISTORY:
            raise ValueError("v12 checkpoint is missing the executed-action history contract")
        if contract.get("deployment_action_shaping") != V12_ACTION_SHAPING:
            raise ValueError("v12 checkpoint actuator shaping does not match the deployment contract")
        if not np.isclose(contract.get("policy_target_root_z_m", np.nan), 1.095):
            raise ValueError("v12 checkpoint does not carry the 1.095 m policy/sim target")
        if not np.isclose(contract.get("physical_ground_root_z_m", np.nan), 0.285):
            raise ValueError("v12 checkpoint does not carry the measured 0.285 m ground root height")
        if not np.isclose(contract.get("physical_target_root_z_m", np.nan), 1.0):
            raise ValueError("v12 checkpoint does not target a physical 1.000 m apex")

    policy = build_policy(state_dict)
    onnx_path = output_dir / "quadhopper_spatial_tracking_policy.onnx"
    export_onnx(policy, onnx_path)
    materialize_lstm_sequence_lengths(onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    max_error = verify_onnx(policy, onnx_path)
    if max_error > 1.0e-5:
        raise RuntimeError(f"ONNX parity failed: max absolute error {max_error}")

    checkpoint_copy = output_dir / checkpoint.name
    shutil.copy2(checkpoint, checkpoint_copy)
    shutil.copy2(Path(__file__).with_name("spatial_tracking_onnx_inference.py"), output_dir)
    shutil.copy2(Path(__file__).with_name("requirements-runtime.txt"), output_dir)
    shutil.copy2(Path(__file__).with_name("README_SPATIAL_TRACKING_EXPORT.md"), output_dir / "README.md")
    observations = copy.deepcopy(OBSERVATIONS)
    action_parameterization = contract.get("action_parameterization", "direct_motor")
    if action_parameterization == "collective_residual_v1":
        observations[7]["name"] = "last_five_raw_collective_residual_actions"
        observations[7]["layout"] = "oldest_to_newest, collective, roll, pitch, yaw"
        action_postprocess = (
            "clip [collective,roll,pitch,yaw] to [-1,1]; "
            "collective_u=0.5*(collective+1); residual=0.5*[roll,pitch,yaw] "
            "@ [[1,1,-1],[-1,1,1],[-1,-1,-1],[1,-1,1]]^T; "
            "uniformly scale the zero-mean residual to keep every motor in [0,1]; "
            "u=collective_u+scaled_residual"
        )
    else:
        observations[7]["name"] = (
            "last_five_executed_post_limit_motor_actions"
            if contract.get("version") == 12
            else "last_five_raw_policy_actions"
        )
        action_postprocess = "clip actions to [-1,1], then u=clip(0.5*action+0.5,0,1)"
    metadata = {
        "source_checkpoint": str(checkpoint),
        "checkpoint_iteration": int(data.get("iter", -1)),
        "checkpoint_sha256": sha256(checkpoint_copy),
        "onnx_sha256": sha256(onnx_path),
        "checkpoint_contract": contract,
        "policy": "ActorCriticRecurrent/LSTM",
        "control_frequency_hz": 100,
        "inputs": {
            "obs": [1, OBS_DIM],
            "h_in": [1, 1, HIDDEN_DIM],
            "c_in": [1, 1, HIDDEN_DIM],
        },
        "outputs": {
            "actions": [1, 4],
            "h_out": [1, 1, HIDDEN_DIM],
            "c_out": [1, 1, HIDDEN_DIM],
        },
        "normalization": "identity",
        "motor_order": ["F1", "F2", "F3", "F4"],
        "action_parameterization": action_parameterization,
        "action_postprocess": action_postprocess,
        "simulation_motor_time_constant_s": 0.0674,
        "simulation_action_delay": contract.get("action_delay", "legacy checkpoint contract"),
        "simulation_action_history": contract.get("action_history", "raw policy actions"),
        "simulation_deployment_action_shaping": contract.get(
            "deployment_action_shaping", "not recorded"
        ),
        "policy_excludes": [
            "state estimation",
            "waypoint queue",
            "untimed Hermite path and carrot construction",
            "contact/apex event detection",
            "motor delay and first-order lag",
            "safety supervisor",
        ],
        "onnx_opset": 18,
        "onnx_parity_max_abs_error": max_error,
        "observations": observations,
    }
    (output_dir / "policy_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {onnx_path}")
    print(f"ONNX parity max abs error: {max_error:.3e}")


if __name__ == "__main__":
    main()
