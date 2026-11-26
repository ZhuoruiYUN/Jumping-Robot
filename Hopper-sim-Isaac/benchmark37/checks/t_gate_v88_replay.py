"""Screen fixed +X gate placements on the five archived 100 Hz V88 first hops.

Reports sampled intersections of the provisional two-box T envelope. Zero
sampled intersections alone cannot distinguish clearance from an outside bypass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..t_frame import DEFAULT_ENVELOPE, boxes
from ..t_gate_asset import gate_layout

CASES = (("cardinal", 0), ("opposite", 1), ("opposite", 5), ("opposite", 9), ("opposite", 13))


def first_hop(directory, case_id):
    with np.load(directory / "straight_visual_trace.npz") as archive:
        column = list(archive["columns"]).index
        trace = archive["trace"][:, list(archive["case_ids"]).index(case_id)]
    events = json.loads((directory / "straight_visual_events.json").read_text())
    landing = next(event for event in events[str(case_id)] if event["moving"])
    if landing["direction"] != "E" or not landing["first_attempt_for_target"]:
        raise ValueError(f"Unexpected archived first hop for {directory.name}/{case_id}")
    flight = trace[(trace[:, column("target_index")] == landing["target_index"])
                   & (trace[:, column("phase")] == 3)
                   & (trace[:, column("step")] < landing["touchdown_step"])]
    positions = flight[:, [column(key) for key in ("root_x_m", "root_y_m", "height_m")]]
    quats = flight[:, [column(key) for key in ("q_x", "q_y", "q_z", "q_w")]]
    return positions, Rotation.from_quat(quats).as_matrix()


def intersection_count(positions, rotations, origin, strips, thickness):
    # Separating-axis test for a robot-oriented box and a gate-aligned box,
    # evaluated for all recorded frames. Both volumes include their interiors.
    body_axes = rotations.transpose(0, 2, 1)
    gate_axes = np.broadcast_to(np.eye(3), rotations.shape)
    cross = np.cross(gate_axes[:, :, None, :], body_axes[:, None, :, :])
    axes = np.concatenate((gate_axes, body_axes, cross.reshape(len(positions), 9, 3)), axis=1)
    hits = np.zeros(len(positions), dtype=bool)
    for lower, upper in map(np.asarray, boxes()):
        body_center = positions + rotations @ ((lower + upper) / 2)
        body_extent = np.abs(axes @ rotations) @ ((upper - lower) / 2)
        for left, right, low, high in strips.values():
            strip_center = origin + np.array([0, (left + right) / 2, (low + high) / 2])
            strip_extent = np.abs(axes) @ np.array([thickness, right - left, high - low]) / 2
            separation = np.abs(np.einsum("nai,ni->na", axes, body_center - strip_center))
            hits |= np.all(separation <= body_extent + strip_extent + 1e-10, axis=1)
    return int(hits.sum())


def forward_crossing_pose(positions, rotations, origin):
    """Interpolate the first forward A-point crossing of the fixed +X plane."""
    distance = positions[:, 0] - origin[0]
    crossings = np.flatnonzero((distance[:-1] < 0) & (distance[1:] >= 0))
    if not len(crossings):
        return None
    index = crossings[0]
    alpha = -distance[index] / (distance[index + 1] - distance[index])
    position = positions[index] + alpha * (positions[index + 1] - positions[index])
    key_rotations = Rotation.from_matrix(rotations[index:index + 2])
    attitude = Slerp([0, 1], key_rotations)(alpha)
    return position, attitude


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acceptance", type=Path, help="V88 acceptance directory containing cardinal/ and opposite/")
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--origin", nargs=3, type=float, metavar=("X", "Y", "Z"),
                          help="Gate origin at the crossbar opening centre, in metres")
    location.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                          help="Legacy aperture bounding-rectangle centre, in metres")
    parser.add_argument("--clearance-m", type=float, default=0.02)
    parser.add_argument("--crossing-pose", action="store_true",
                        help="Also report A-point error at the +X plane relative to the upright gate pose")
    args = parser.parse_args()
    strips = gate_layout(DEFAULT_ENVELOPE, args.clearance_m, 0.01, 0.02)
    origin = np.asarray(args.origin or args.center or (0.265, 0, 1.075), dtype=float)
    if not np.isfinite(origin).all():
        parser.error("Gate position must be finite")
    if args.center is not None:
        # Convert the archived bounding rectangle's midpoint to the crossbar centre.
        origin[2] -= (strips["Top"][2] + strips["Bottom"][3]) / 2
    for group, case_id in CASES:
        positions, rotations = first_hop(args.acceptance / group, case_id)
        hits = intersection_count(positions, rotations, origin, strips, 0.01)
        print(f"{group}/{case_id}: {hits}/{len(positions)} sampled poses intersect the frame")
        if args.crossing_pose:
            crossing = forward_crossing_pose(positions, rotations, origin)
            if crossing is None:
                print("  No forward A-point crossing")
                continue
            position, attitude = crossing
            dy, dz = (position - origin)[1:] * 100
            angle = np.degrees(attitude.magnitude())
            print(f"  A-point at plane: dY={dy:+.2f} cm, dZ={dz:+.2f} cm, "
                  f"in-plane error={np.hypot(dy, dz):.2f} cm, attitude error={angle:.2f} deg")
