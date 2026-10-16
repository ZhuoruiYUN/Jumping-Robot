"""Frozen-policy V3 accounting task with one opt-in post-miss route decision."""

from .path_free_stage2_v3_accounting_env import PathFreeStage2V3AccountingEnv
from .route_recovery_eval import RouteRecoveryMetrics


class PathFreeStage2V3RouteEvalEnv(PathFreeStage2V3AccountingEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._route_recovery_metrics = RouteRecoveryMetrics(
            self.num_envs, self.device, cfg.route_recovery_severe_miss_m)

    def _advance_route(self, env_ids):
        if (self.cfg.route_recovery_mode == 'retry_large'
                and len(env_ids) and hasattr(self, '_target_miss_event')):
            # The parent calls this once for hits and once for misses. Keep
            # large foot misses on the same target; all other calls match V21.
            miss = self._target_miss_event[env_ids]
            target = self.commands.lookahead(env_ids)[0]
            foot_xy = self.foot_position_w()[env_ids, :2]
            severe = miss & ((foot_xy - target).norm(dim=1)
                             > self.cfg.route_recovery_severe_miss_m)
            env_ids = env_ids[~severe]
        super()._advance_route(env_ids)

    def _update_cycle_events(self):
        route_before = self.commands.route_index.clone()
        super()._update_cycle_events()
        metrics = getattr(self, '_route_recovery_metrics', None)
        if metrics is None:
            return
        ids = self._touchdown_event.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        next_target = self.commands.lookahead(ids)[0]
        foot_xy = self.foot_position_w()[ids, :2]
        metrics.observe(
            ids,
            task_hit=self._target_hit_event[ids],
            foot_error_m=self._stage2_foot_error[ids],
            settled_apex_m=self._settled_apex_height[ids],
            target_apex_m=self._settled_apex_target[ids],
            route_before=route_before[ids],
            route_after=self.commands.route_index[ids],
            next_target_distance_m=(foot_xy - next_target).norm(dim=1),
            step=self.common_step_counter,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        metrics = getattr(self, '_route_recovery_metrics', None)
        if metrics is not None:
            ids = self._robot._ALL_INDICES if env_ids is None else env_ids
            metrics.reset_pending(ids)

    def stage2_summary(self):
        result = super().stage2_summary()
        result.update({f'route_recovery_{name}': value for name, value in
                       self._route_recovery_metrics.summary().items()})
        return result
