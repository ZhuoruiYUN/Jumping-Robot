"""Drop -> stationary -> ring -> two stationary landings; task-only extension."""
import torch

TASK = 'drop_stationary_ring_stationary2_v1'


class LoopState:
    # Initial drop is filtered out by the inherited confirmed-flight tracker.
    STATIONARY, RING, POST1, POST2, COMPLETE = range(5)

    def __init__(self, n, device):
        self.stage = torch.zeros(n, dtype=torch.long, device=device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.completions = torch.zeros(4, dtype=torch.long, device=device)
        self.failures = torch.zeros((), dtype=torch.long, device=device)

    def reset(self, ids):
        self.stage[ids] = self.STATIONARY
        self.failed[ids] = False

    def step(self, touch, strict, passed, died):
        before = self.stage.clone()
        active = touch & (before < self.COMPLETE) & ~self.failed
        good = active & strict & ~died & ((before != self.RING) | passed)
        bad = active & ~good
        self.failed |= bad
        self.failures += bad.sum()
        for i in range(4):
            self.completions[i] += (good & (before == i)).sum()
        self.stage += good.long()
        return good & (before == self.STATIONARY), good & (before == self.POST2), bad


def env_class():
    # Keep LoopState usable for the small CPU event check without starting Kit.
    from .ring_env import RingEnv
    from .two_ground_targets import _set_active_goal

    class RingMinLoopEnv(RingEnv):
        loop_continue = False  # GUI can continue hopping after loop completion.

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loop = LoopState(self.num_envs, self.device)
            self.post_stability=self.course.get('post_ring_stability')
            if self.post_stability:
                from .ring_post_stability import validate
                validate(self.post_stability)
                for name in ('post_ring_pitch','post_ring_pitch_rate'):
                    self._episode_sums[name]=torch.zeros(self.num_envs,device=self.device)

        def _get_rewards(self):
            if not self.post_stability:return super()._get_rewards()
            from .ring_post_stability import active_mask,terms
            d=self._robot.data
            active=active_mask(self.loop.stage,self.ring_joint_success,self.reset_buf)
            extra,pitch=terms(d.root_quat_w,d.root_ang_vel_b[:,1],active,self.step_dt,self.post_stability)
            reward=super()._get_rewards()
            for name,value in extra.items():
                self._episode_sums[name]+=value
                reward=reward+value
            count=active.sum().clamp(min=1)
            self.extras.setdefault('log',{}).update({
                'PostRing/pitch_abs_deg':torch.where(active,torch.rad2deg(pitch.abs()),0.).sum()/count,
                'PostRing/pitch_rate_abs_rad_s':torch.where(active,d.root_ang_vel_b[:,1].abs(),0.).sum()/count})
            return reward

        def _reset_task_targets(self, ids):
            origin = self._robot.data.root_link_pos_w[ids, :2]
            self._two_ground_pt_xy_w[ids] = origin
            self._two_ground_pt1_xy_w[ids] = origin + origin.new_tensor(self.course['landing_xy_m'])

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            if hasattr(self, 'loop'):
                self.loop.reset(self._robot._ALL_INDICES if ids is None else ids)

        def _ring_eligible(self):
            f = self._foot_event
            return ((f.phase == f.FLIGHT) & (f.attempt_index == 1)
                    & (self.loop.stage == self.loop.RING))

        def _get_dones(self):
            died, timeout = super()._get_dones()
            if hasattr(self, 'loop'):
                died |= (self.loop.stage == self.loop.RING) & self.ring.failed
                if self.loop_continue:
                    timeout &= self.loop.stage != self.loop.COMPLETE
            return died, timeout

        def _landing_event(self, touch, designated, died):
            super()._landing_event(touch, designated, died)
            advance, complete, failed = self.loop.step(
                touch, self._foot_event.last_qualified, self.ring.passed,
                died | self.reset_terminated)
            ids = advance.nonzero(as_tuple=False).flatten()
            if len(ids):
                self._two_ground_pt_xy_w[ids] = self._two_ground_pt1_xy_w[ids].clone()
                # All subsequent ground targets coincide at the ring landing.
                self._two_ground_pt1_xy_w[ids] = self._two_ground_pt_xy_w[ids]
                self._two_ground_route_index[ids] += 1
                self._two_ground_total_advances += len(ids)
                _set_active_goal(self, ids)
            terminal = failed | (complete & (not self.loop_continue))
            # DirectRLEnv evaluates dones before rewards, but consumes these
            # buffers for auto-reset and wrapper dones AFTER rewards. Set the
            # just-observed landing terminal here, before histories are reset.
            self.reset_terminated |= terminal
            self.reset_buf |= terminal
            names = ('initial_stationary', 'ring_landing', 'post_stationary1', 'loop_complete')
            self.extras.setdefault('log', {}).update({
                'Loop/' + name + '_total': self.loop.completions[i]
                for i, name in enumerate(names)})
            self.extras['log']['Loop/landing_failures_total'] = self.loop.failures

        def _advance_targets(self):
            pass  # Explicit sequence owns the ground queue; never sample an edge.

    return RingMinLoopEnv
