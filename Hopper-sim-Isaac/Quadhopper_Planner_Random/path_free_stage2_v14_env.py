"""Opt-in v14 corrections; inherited reward scales and physical contract unchanged."""
from .path_free_stage2_v13_env import PathFreeStage2V13Env
from .attitude_recovery_v14 import AttitudeRecovery, predicted_touchdown_tilt


class PathFreeStage2V14Env(PathFreeStage2V13Env):
    def _create_recovery_state(self):
        return AttitudeRecovery(self.num_envs, self.device, self.step_dt)

    def _update_recovery_state(self, clearance, vz):
        state = self._near_ground_recovery
        active, released = state.update(
            self._touchdown_event, self._completed_stationary_command,
            self._initial_drop_pending, clearance, vz,
            self._robot.data.root_quat_w, self._robot.data.root_ang_vel_b)
        self.extras['log'].update({
            'V14/unsafe_airborne_recovery_active_rate': (
                active & (clearance > state.settings['recovery_clearance_m'])).float().mean(),
            'V14/recovery_attitude_safe_rate': state.attitude_safe.float().mean(),
            'V14/recovery_release_rate': released.float().mean(),
        })
        return active, released

    def _forecast_tilt(self, quat, rates, clearance, vz):
        return predicted_touchdown_tilt(quat, rates, clearance, vz)

    def _recovery_coverage(self, clearance):
        # v12 already charged these samples. Exclude them from v13's fallback
        # to avoid charging airborne recovery twice.
        return self._near_ground_recovery.active
