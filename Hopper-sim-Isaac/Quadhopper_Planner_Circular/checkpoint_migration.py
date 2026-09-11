from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch


OLD_OBS_DIM = 37
LEGACY_PLANNER_OBS_DIM = 42
NEW_OBS_DIM = 43
PLANNER_VELOCITY_OBS_DIM = 47
RECURRENT_INPUT_KEYS = ("memory_a.rnn.weight_ih_l0", "memory_c.rnn.weight_ih_l0")
LEGACY_FIXED_HEIGHT_COMMAND = 1.30 / 2.0


def absolute_next_to_relative_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Preserve LSTM outputs when Pt1_abs is replaced by Pt1-Pt.

    The planner command columns are [Pt_x, Pt_y, Pt1_x, Pt1_y] at indices
    37:41.  Since Pt1_abs = Pt + next_delta, folding the old Pt1 weights
    into the Pt weights makes the observation change algebraically exact.
    """
    converted = deepcopy(state_dict)
    for key in RECURRENT_INPUT_KEYS:
        weight = converted[key]
        if weight.shape[1] not in (NEW_OBS_DIM, PLANNER_VELOCITY_OBS_DIM):
            raise ValueError(
                f"{key} has observation width {weight.shape[1]}, expected {NEW_OBS_DIM} or {PLANNER_VELOCITY_OBS_DIM}"
            )
        weight[:, 37:39] = weight[:, 37:39] + weight[:, 39:41]
    return converted


def migrate_stable_checkpoint(
    source: str | Path,
    destination: str | Path,
    target_obs_dim: int = NEW_OBS_DIM,
) -> Path:
    """Expand recurrent inputs to the requested planner observation contract."""
    if target_obs_dim not in (NEW_OBS_DIM, PLANNER_VELOCITY_OBS_DIM):
        raise ValueError(
            f"target_obs_dim must be {NEW_OBS_DIM} or {PLANNER_VELOCITY_OBS_DIM}"
        )
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = deepcopy(checkpoint["model_state_dict"])
    migrated = False
    for key in RECURRENT_INPUT_KEYS:
        weight = state_dict[key]
        if weight.shape[1] == target_obs_dim:
            continue
        if weight.shape[1] not in (
            OLD_OBS_DIM,
            LEGACY_PLANNER_OBS_DIM,
            NEW_OBS_DIM,
        ):
            raise ValueError(
                f"{key} has observation width {weight.shape[1]}, expected 37, 42, 43 or {target_obs_dim}"
            )
        if weight.shape[1] > target_obs_dim:
            raise ValueError(
                f"{key} has observation width {weight.shape[1]}, cannot shrink to {target_obs_dim}"
            )
        expanded = torch.zeros(weight.shape[0], target_obs_dim, dtype=weight.dtype)
        expanded[:, : weight.shape[1]] = weight
        if weight.shape[1] == LEGACY_PLANNER_OBS_DIM:
            # V10 saw only the constant 1.30/2 height value, so its height
            # column cannot encode command conditioning. Fold that constant
            # contribution into the LSTM input bias, then initialize both new
            # height channels at zero. Their weights can now learn H_t/H_t+1
            # without an arbitrary fixed-height correlation.
            bias_key = key.replace("weight_ih_l0", "bias_ih_l0")
            state_dict[bias_key] = (
                state_dict[bias_key] + weight[:, 41] * LEGACY_FIXED_HEIGHT_COMMAND
            )
            expanded[:, 41] = 0.0
            expanded[:, 42] = 0.0
        state_dict[key] = expanded
        migrated = True
    output = {
        "model_state_dict": state_dict,
        "iter": 0 if migrated else checkpoint.get("iter", 0),
        "infos": {
            "source_checkpoint": str(source),
            "observation_migration": (
                f"{target_obs_dim}-D planner observation; "
                "base contract is stable37 + Pt_xy + Pt1_xy + H_t + H_t1; "
                "optional planner velocity channels initialized to zero; "
                "legacy fixed 1.30 m H_t contribution folded into LSTM bias; "
                "new H_t/H_t1 weights initialized to zero"
            ),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)
    return destination
