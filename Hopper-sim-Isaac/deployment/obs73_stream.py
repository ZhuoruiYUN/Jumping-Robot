"""V51 quaternion OBS73 input adapter for a continuous stream of ground waypoints.

Supply state and targets in the same release frame. It has no hardware I/O.
The observation/history and touchdown estimator are from the archived OBS73 runtime.
"""
import math

import numpy as np

from benchmark37.waypoint_stream import checked_preview, next_preview


VERSION = 'direct_motor_crossing_quaternion73_history40_20_v1'
LANDING_RADIUS_M = .05


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


class StreamingObservation73:
    """Build one OBS73 input per control tick, scoring the old goal before handoff."""

    DROP, GROUND, LIFTOFF, FLIGHT = range(4)

    def __init__(self, course, source=None, case_id=0):
        if course['observation_version'] != VERSION or course['task'] != 'obs73_omni_ground_targets_v1':
            raise ValueError('Expected the ground-target quaternion73 course')
        if (course['omni']['strict_foot_m'] != LANDING_RADIUS_M or
                course['moving_navigation']['version'] != 'position_and_tilt_after_first_hop_v1'):
            raise ValueError('Expected the 5cm OBS73 moving navigation contract')
        self.source = source
        self.case_id = case_id
        self.target_height = course['stationary_pretraining']['apex_target_m']
        self.reset()

    def reset(self):
        self.index = 0
        self.current_xy = (0., 0.)
        self.preview_xy = self._request_next(self.index, self.current_xy)
        self.state_buffer = np.zeros((4, 13), dtype=np.float32)
        self.action_buffer = np.zeros((3, 4), dtype=np.float32)
        self.recent_action = np.zeros(4, dtype=np.float32)
        self.snapshot = None
        self.phase = self.DROP
        self.air_steps = 0
        self.peak = 0.
        self.apex_seen = False
        self.last_landing = None

    def _request_next(self, index, current):
        if self.source is None:
            return current
        return next_preview(self.source, self.case_id, index + 1, current)

    def set_next(self, xy):
        """Replace the preview before the next inference tick."""
        self.preview_xy = checked_preview(self.current_xy, xy)

    def _track_landing(self, position, rotation, vz, spring_q):
        up_z = rotation[2, 2]
        if up_z <= .2:
            raise ValueError('Invalid body up axis')
        if spring_q is None:
            spring_q = np.clip((.012 + .273 * up_z - position[2]) / up_z, 0., .087)
        if not math.isfinite(spring_q) or not 0 <= spring_q <= .087001:
            raise ValueError('Invalid spring compression')
        self.last_landing = None
        contact = spring_q > .002
        if self.phase == self.DROP:
            if contact:
                self.phase = self.GROUND
            return
        if self.phase == self.GROUND:
            if spring_q <= .001 and vz > 0:
                self.phase = self.LIFTOFF
                self.peak, self.air_steps, self.apex_seen = position[2], 1, False
            return

        self.peak = max(self.peak, position[2])
        self.apex_seen |= vz <= 0
        if not contact:
            self.air_steps += 1
            if self.air_steps >= 3:
                self.phase = self.FLIGHT
            return

        confirmed = self.phase == self.FLIGHT and self.air_steps >= 5 and self.apex_seen
        self.phase = self.GROUND
        if confirmed:
            foot = position + rotation[:, 2] * (-.273 + spring_q)
            error = float(np.linalg.norm(foot[:2] - self.current_xy))
            tilt = math.degrees(math.acos(np.clip(up_z, -1., 1.)))
            reached = error <= LANDING_RADIUS_M and tilt <= 15.
            strict = bool(reached and self.peak >= 1.)
            advance = reached and (self.index > 0 or strict)
            self.last_landing = dict(target_index=self.index, target_xy=self.current_xy, apex_m=self.peak,
                                     foot_error_m=error, tilt_deg=tilt, strict=strict, navigation_advanced=advance)
            if advance:
                following = self._request_next(self.index + 1, self.preview_xy)
                self.current_xy, self.preview_xy = self.preview_xy, following
                self.index += 1

    def observe(self, position, q_wxyz, velocity_world, omega_body, spring_q=None):
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity_world, dtype=np.float64)
        omega = np.asarray(omega_body, dtype=np.float64)
        if any(a.shape != (3,) or not np.isfinite(a).all() for a in (position, velocity, omega)):
            raise ValueError('Invalid root-link state')
        q, rotation = quaternion_matrix(q_wxyz)
        self._track_landing(position, rotation, velocity[2], spring_q)
        targets = np.asarray((self.current_xy, self.preview_xy, self.preview_xy))
        errors = (np.c_[targets - position[:2], np.zeros(3)] @ rotation)[:, :2].ravel()
        body_velocity = rotation.T @ velocity
        self.snapshot = np.r_[body_velocity, omega, q, velocity].astype(np.float32)
        up_z = rotation[2, 2]
        shadow = np.clip((.015 - (position[2] - .285 * up_z)) / up_z, 0., .086) > .002
        obs = np.r_[errors, body_velocity, q, omega, velocity, self.state_buffer[[0, 2]].ravel(),
                    self.action_buffer[[0, 2]].ravel(), self.recent_action, self.target_height - position[2],
                    float(shadow), np.zeros(14)].astype(np.float32)
        if obs.shape != (73,) or not np.isfinite(obs).all():
            raise ValueError('Invalid quaternion73 observation')
        return obs

    def accept(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.snapshot is None or action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Accept requires an observation and four finite motor commands')
        self.state_buffer[:-1] = self.state_buffer[1:].copy()
        self.state_buffer[-1] = self.snapshot
        self.action_buffer[:-1] = self.action_buffer[1:].copy()
        self.action_buffer[-1] = self.recent_action
        self.recent_action = np.clip(action, -1., 1.)
        self.snapshot = None
