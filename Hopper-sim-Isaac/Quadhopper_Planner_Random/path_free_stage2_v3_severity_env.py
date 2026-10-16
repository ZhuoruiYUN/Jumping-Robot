"""V3 accounting task with reset-time hardware-strength curriculum."""

from .hardware_severity_curriculum import scale_hard_reset, severity_at_step
from .path_free_stage2_v3_accounting_env import PathFreeStage2V3AccountingEnv


class PathFreeStage2V3SeverityEnv(PathFreeStage2V3AccountingEnv):
    def _severity(self):
        return severity_at_step(
            getattr(self, 'common_step_counter', 0),
            self.cfg.curriculum_steps_per_iteration,
            self.cfg.hardware_severity_stage_iterations,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # DirectRLEnv can reset while its superclass is constructing buffers.
        if not hasattr(self, '_hardware_global_thrust_scale'):
            return
        ids = self._robot._ALL_INDICES if env_ids is None else env_ids
        scale_hard_reset(self, ids, self._severity())

    def _get_rewards(self):
        reward = super()._get_rewards()
        self.extras.setdefault('log', {})['Stage2/hardware_severity_target'] = self._severity()
        return reward
