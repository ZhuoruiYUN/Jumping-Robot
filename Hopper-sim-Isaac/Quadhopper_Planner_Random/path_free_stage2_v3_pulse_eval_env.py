"""V3 accounting task with an evaluation-only short early-ascent pulse."""
from .ascent_pulse_v26 import apply_ascent_pulse
from .path_free_stage2_v3_accounting_env import PathFreeStage2V3AccountingEnv


class PathFreeStage2V3PulseEvalEnv(PathFreeStage2V3AccountingEnv):
    def _task_shape_direct_motor_target_u(self, target_u):
        target_u = super()._task_shape_direct_motor_target_u(target_u)
        return apply_ascent_pulse(
            target_u,
            spring_q=self._robot.data.joint_pos[:, self._spring_joint_id],
            cycle_active=self._cycle_active,
            initial_drop=self._initial_drop_pending,
            root_vz=self._robot.data.root_lin_vel_w[:, 2],
            cycle_time_s=self._cycle_time,
            bias_pwm=self.cfg.diagnostic_ascent_pulse_pwm,
        )
