"""Exact 52-D spatial-policy additions for a stationary physical waypoint.

The flight computer already builds the first 37 stable observations.  This
module builds the remaining 15 fields: current/next waypoint terms and the
untimed Hermite-path carrot, tangent and landing-up hint used by
``SpatialTrackingEnv``.  It deliberately contains no control or safety logic.
"""

from __future__ import annotations

import math

import numpy as np


class Spatial52StationaryObserver:
    """Stateful no-clock path generator for P_t = P_(t+1) = launch XY."""

    def __init__(
        self,
        landing_root_height: float = 0.38,
        apex_height: float = 1.0,
        waypoint_scale: float = 0.20,
        lookahead_m: float = 0.10,
        nodes: int = 33,
        landing_hint_tilt_deg: float = 4.0,
    ):
        if nodes < 9 or nodes % 2 == 0:
            raise ValueError("nodes must be odd and at least 9")
        self.landing_root_height = float(landing_root_height)
        self.apex_height = float(apex_height)
        self.waypoint_scale = float(waypoint_scale)
        self.lookahead_m = float(lookahead_m)
        self.nodes = int(nodes)
        self.axis_w = np.array(
            [0.0, 0.0, math.cos(math.radians(landing_hint_tilt_deg))],
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self.curve: np.ndarray | None = None
        self.descending = False
        self.minimum_arc_fraction = 0.0
        self.previous_contact = True

    def _hop_curve(self, start: np.ndarray, landing: np.ndarray) -> np.ndarray:
        # This is the stationary specialization of spatial_path.hop_curve:
        # incoming, outgoing and the landing tangent have zero XY components.
        peak = 0.5 * (start + landing)
        peak[2] = self.apex_height
        t0 = np.array([0.0, 0.0, 2.0 * max(self.apex_height - start[2], 0.0)])
        tm = np.zeros(3)
        t1 = np.array([0.0, 0.0, -2.0 * max(self.apex_height - landing[2], 0.0)])
        u = np.linspace(0.0, 1.0, self.nodes // 2 + 1)[:, None]

        def segment(a: np.ndarray, b: np.ndarray, ta: np.ndarray, tb: np.ndarray) -> np.ndarray:
            return (
                (2 * u**3 - 3 * u**2 + 1) * a
                + (u**3 - 2 * u**2 + u) * ta
                + (-2 * u**3 + 3 * u**2) * b
                + (u**3 - u**2) * tb
            )

        return np.concatenate((segment(start, peak, t0, tm), segment(peak, landing, tm, t1)[1:]))

    def _replan(self, start_w: np.ndarray, target_xy: np.ndarray) -> None:
        landing = np.array(
            [target_xy[0], target_xy[1], self.landing_root_height], dtype=np.float32
        )
        self.curve = self._hop_curve(np.asarray(start_w, dtype=np.float32), landing)
        self.descending = False
        self.minimum_arc_fraction = 0.0

    def update(
        self,
        position_w: np.ndarray,
        target_xy: np.ndarray,
        contact: bool,
        vertical_velocity_w: float,
    ) -> None:
        """Update physical phase; replan only on contact/liftoff transitions."""
        if self.curve is None or contact != self.previous_contact:
            self._replan(position_w, target_xy)
        if not contact and vertical_velocity_w <= 0.0:
            self.descending = True
        self.previous_contact = bool(contact)

    def _project(self, position_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        if self.curve is None:
            raise RuntimeError("update() must be called before observation()")
        start = self.curve[:-1]
        segment = self.curve[1:] - start
        length = np.linalg.norm(segment, axis=1).clip(1.0e-8)
        fraction = np.clip(
            np.sum((position_w[None, :] - start) * segment, axis=1) / np.square(length),
            0.0,
            1.0,
        )
        projected = start + fraction[:, None] * segment
        distance_sq = np.sum(np.square(position_w[None, :] - projected), axis=1)
        indices = np.arange(len(segment))
        allowed = indices >= len(segment) // 2 if self.descending else indices < len(segment) // 2
        chosen = np.where(allowed, distance_sq, np.inf).argmin()
        arc_starts = np.concatenate(([0.0], np.cumsum(length)[:-1]))
        total = float(np.sum(length))
        raw_arc = float(arc_starts[chosen] + fraction[chosen] * length[chosen])
        self.minimum_arc_fraction = max(self.minimum_arc_fraction, raw_arc / max(total, 1.0e-8))
        arc = max(raw_arc, self.minimum_arc_fraction * total)
        forward_arc = min(arc + self.lookahead_m, total)
        forward = min(int(np.sum(np.cumsum(length) < forward_arc)), len(segment) - 1)
        forward_fraction = np.clip(
            (forward_arc - arc_starts[forward]) / length[forward], 0.0, 1.0
        )
        carrot = start[forward] + forward_fraction * segment[forward]
        tangent = segment[forward] / length[forward]
        return carrot.astype(np.float32), tangent.astype(np.float32), float(np.sqrt(distance_sq[chosen]))

    def observation(
        self,
        position_w: np.ndarray,
        world_to_body,
        target_xy: np.ndarray,
        next_target_xy: np.ndarray,
    ) -> np.ndarray:
        """Return slices 37:52 in the trained policy's exact order."""
        carrot, tangent, _ = self._project(position_w)
        current_error_w = np.array(
            [target_xy[0] - position_w[0], target_xy[1] - position_w[1], 0.0],
            dtype=np.float32,
        )
        next_displacement_w = np.array(
            [next_target_xy[0] - target_xy[0], next_target_xy[1] - target_xy[1], 0.0],
            dtype=np.float32,
        )
        current_error_b = world_to_body.apply(current_error_w)[:2] / self.waypoint_scale
        next_displacement_b = world_to_body.apply(next_displacement_w)[:2] / self.waypoint_scale
        carrot_error_b = np.clip(world_to_body.apply(carrot - position_w), -1.0, 1.0)
        tangent_b = world_to_body.apply(tangent)
        axis_b = world_to_body.apply(self.axis_w)
        return np.concatenate(
            (
                current_error_b,
                next_displacement_b,
                np.array([self.apex_height / 2.0, self.apex_height / 2.0], dtype=np.float32),
                carrot_error_b,
                tangent_b,
                axis_b,
            )
        ).astype(np.float32)
