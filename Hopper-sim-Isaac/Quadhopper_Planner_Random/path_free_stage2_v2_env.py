"""Task-only recovery curriculum; preserves the v1 zero-curve observation layout."""
import torch
from .path_free_stage2 import curriculum_values
from .path_free_stage2_env import PathFreeStage2Env
from .tracking_progress import LandingXYProgress


class PathFreeStage2V2Env(PathFreeStage2Env):
    def __init__(self, cfg, **kwargs):
        # Keep the final coefficient separate from the live scheduled value.
        self._stage2_final_energy_scale = cfg.stage2_energy_penalty_scale
        super().__init__(cfg, **kwargs)
        self._foot_progress = LandingXYProgress(self.num_envs, self.device)
        self._foot_progress_totals = torch.zeros(3, device=self.device, dtype=torch.float64)

    def _curriculum(self):
        probability, energy = curriculum_values(
            self.cfg, getattr(self, 'common_step_counter', 0),
            final_energy_scale=self._stage2_final_energy_scale)
        # The v2 task schedule predates the explicit hardware ramp.  In v3,
        # stage2_curriculum_training is false, so its probability is always
        # the final value unless we delegate to the base reset-time schedule.
        if getattr(self.cfg, 'hardware_randomization_curriculum_iterations', 0.0) > 0.0:
            probability = super()._current_hardware_randomization_probability()
        return probability, energy

    def _current_hardware_randomization_probability(self):
        return self._curriculum()[0]

    def _replan(self, env_ids):
        super()._replan(env_ids)
        if hasattr(self, '_foot_progress') and len(env_ids):
            target = self.commands.lookahead(env_ids)[0]
            distance = (self.foot_position_w()[env_ids, :2] - target).norm(dim=1)
            self._foot_progress.reanchor(env_ids, distance)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, '_foot_progress'):
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            self._foot_progress.reanchor(ids, 0.0)

    def _get_rewards(self):
        probability, energy_scale = self._curriculum()
        self.cfg.stage2_energy_penalty_scale = energy_scale
        reward = super()._get_rewards()
        target = self.commands.lookahead()[0]
        distance = (self.foot_position_w()[:, :2] - target).norm(dim=1)
        active = (self._cycle_active & ~self._touchdown_event
                  & ~self._initial_drop_pending)
        # Signed, unclipped change: approaching then retreating cancels.
        # Replan/reset breaks continuity; touchdown retains its event reward.
        progress = self._foot_progress.update(distance, active) * self.cfg.stage2_foot_progress_scale
        self._foot_progress_totals += torch.stack((
            progress.sum(), progress.clamp_min(0).sum(), progress.clamp_max(0).sum()))
        self.extras['log'].update({
            'Stage2/foot_progress_reward': progress.mean(),
            'Stage2/hardware_probability_target': probability,
            'Stage2/hard_env_fraction': self._hardware_hard_mask.float().mean(),
            'Stage2/energy_penalty_scale': energy_scale,
        })
        return reward + progress

    def stage2_summary(self):
        result = super().stage2_summary()
        result.update(dict(zip(
            ('stage2_foot_progress_reward_per_env', 'stage2_foot_approach_reward_per_env',
             'stage2_foot_retreat_penalty_per_env'),
            (self._foot_progress_totals / self.num_envs).tolist())))
        return result
