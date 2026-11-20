"""Fixed wide-ring task on the v23 stationary reward/physics interface.

Drop -> strict stationary hop -> full-body ring hop -> three strict stationary
hops. Task state changes ground goals, never observations, motors or physics.
"""
import json
from pathlib import Path

import torch

TASK = 'obs73_fixed_wide_ring_stage_b_v1'
CASE_VERSION = 'obs73_stage_b_fixed_wide_response64_64_first128_v1'


def output_series(course):
    if course['stage_b'].get('ascent_power_relief'):
        return 'benchmark73_ring_stage_b_v4'
    progress = course['stage_b'].get('forward_progress')
    if not progress:
        return 'benchmark73_ring_stage_b_v1'
    versions = {'confirmed_flight_new_frontier_v1':'benchmark73_ring_stage_b_v2',
                'ascending_approach_new_frontier_v2':'benchmark73_ring_stage_b_v3'}
    return versions[progress['version']]


def stage_a_evidence(root, course, source):
    """Read an existing formal result; never evaluate or hash the source again."""
    path = Path(root)/course['stage_b']['stage_a_summary']
    summary = json.loads(path.read_text())
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('mode') != 'eval'
            or Path(manifest['checkpoint']).resolve() != Path(source).resolve()
            or summary.get('episodes') != 128 or summary.get('seed') != 7001):
        raise ValueError('Stage B requires the existing formal128 evaluation of its source')
    checks = dict(first_four_strict=summary['counts']['first_four_strict'] >= 116,
                  zero_physical_deaths=summary['counts']['physical_death'] == 0,
                  mean_pitch_span=summary['pitch']['mean_span_deg'] <= 5.,
                  pitch_span_tail=summary['pitch']['span_gt10_rate'] <= .05)
    if not all(checks.values()):
        raise ValueError(f'Stage A plan gate failed: {checks}')
    return dict(summary=str(path.resolve()),source=str(Path(source).resolve()),checks=checks,
                first_four_strict=summary['counts']['first_four_strict'],
                physical_deaths=summary['counts']['physical_death'],pitch=summary['pitch'],
                passed=True,scope='Execution-plan Stage A; not the separate3deg settled convergence gate')


class StageBSequence:
    STATIONARY, RING, POST1, POST2, POST3, COMPLETE = range(6)

    def __init__(self, n, device):
        self.stage = torch.zeros(n,dtype=torch.long,device=device)
        self.failed = torch.zeros(n,dtype=torch.bool,device=device)
        self.completions = torch.zeros(5,dtype=torch.long,device=device)
        self.failures = torch.zeros((),dtype=torch.long,device=device)

    def reset(self, ids):
        self.stage[ids] = self.STATIONARY
        self.failed[ids] = False

    def step(self, valid, strict, full_pass, bad_geometry, physical):
        before = self.stage.clone()
        active = valid & ~self.failed & (before < self.COMPLETE)
        good = active & strict & ~bad_geometry & ~physical & ((before != self.RING) | full_pass)
        bad = active & ~good
        self.failed |= bad
        self.failures += bad.sum()
        for stage in range(5):
            self.completions[stage] += (good & (before == stage)).sum()
        self.stage += good.long()
        return good & (before == self.STATIONARY), good & (before == self.POST3), bad


def env_class(evaluation=False):
    from .obs73_vertical import env_class as vertical_class
    from .ring_env import RingEnv
    from .two_ground_targets import _set_active_goal

    class Obs73RingStageBEnv(vertical_class(evaluation=evaluation)):
        loop_continue = False

        def __init__(self, cfg, course, **kwargs):
            if (course['task'] != TASK or course.get('height_randomization')
                    or course.get('targeted_adjustments') or course['stage_b']['post_ring_strict_hops'] != 3):
                raise ValueError('Stage B v1 is fixed geometry with three post-ring strict hops')
            if course['vertical_pretraining']['levels'] != [dict(name='fixed1m',release_height_m=[.8,1.2],
                    target_apex_m=1.,tilt_limit_deg=5.,upright_fraction=.2)]:
                raise ValueError('Preserve v23 fixed1m release/target contract')
            super().__init__(cfg,course,**kwargs)
            self.loop = StageBSequence(self.num_envs,self.device)
            self._episode_sums['stage_b_terminal_potential'] = torch.zeros(self.num_envs,device=self.device)
            self._episode_sums['ring_complete'] = torch.zeros(self.num_envs,device=self.device)
            self.forward_progress = None
            if course['stage_b'].get('forward_progress'):
                from .obs73_ring_progress import ForwardProgress
                self.forward_progress = ForwardProgress(self.num_envs,self.device,course)
                self._episode_sums['stage_b_forward_progress'] = torch.zeros(self.num_envs,device=self.device)
            self.ascent_power = None
            if course['stage_b'].get('ascent_power_relief'):
                if self.forward_progress is None:
                    raise ValueError('Ascent power relief requires the existing Stage B route/progress contract')
                from .obs73_ring_power import AscentPowerRelief
                self.ascent_power = AscentPowerRelief(self.num_envs,self.device,course['stage_b']['ascent_power_relief'])
                self._episode_sums['stage_b_ascent_power_relief'] = torch.zeros(self.num_envs,device=self.device)
            # RingEnv already allocates ring success counters. Keep them across
            # resets, while per-episode sequence and event memory reset normally.

        def _reset_task_targets(self, ids):
            origin = self._terrain.env_origins[ids,:2]
            self._two_ground_pt_xy_w[ids] = origin
            self._two_ground_pt1_xy_w[ids] = origin + origin.new_tensor(self.course['landing_xy_m'])
            self._desired_pos_w[ids,2] = self._terrain.env_origins[ids,2] + 1.

        def _reset_idx(self, ids):
            super()._reset_idx(ids)
            ids = self._robot._ALL_INDICES if ids is None else ids
            if hasattr(self,'ring'):
                # Re-prime AFTER the vertical parent's final pose write, so
                # the reset teleport never becomes a swept collision segment.
                # This also replaces the parent's randomized nuisance ring Z.
                d = self._robot.data
                self.ring.reset(ids,d.root_link_pos_w,d.root_quat_w,
                    d.joint_pos[:,self._spring_joint_id],self._terrain.env_origins)
            if hasattr(self,'loop'):
                self.loop.reset(ids)
            if getattr(self,'forward_progress',None) is not None:
                self.forward_progress.reset(ids)
            if getattr(self,'ascent_power',None) is not None:
                self.ascent_power.reset(ids)

        def _ring_eligible(self):
            if not hasattr(self,'loop'):
                return torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            f = self._foot_event
            return (self.loop.stage == self.loop.RING) & (f.phase == f.FLIGHT)

        def _ring_sample(self):
            RingEnv._ring_sample(self)

        def _get_dones(self):
            died,timeout = super()._get_dones()
            if hasattr(self,'loop'):
                died |= self.ring.failed & (self.loop.stage == self.loop.RING)
            return died,timeout

        def _get_rewards(self):
            d = self._robot.data;f = self._foot_event;q = d.root_quat_w
            physical = (d.root_link_pos_w[:,2] < .05) | (q[:,1:3].square().sum(-1) > .5)
            tilt = torch.rad2deg(torch.acos((1-2*q[:,1:3].square().sum(-1)).clamp(-1,1)))
            valid = ((f.phase == f.FLIGHT) & (f.air_steps >= 5)
                     & (d.joint_pos[:,self._spring_joint_id] > .002)
                     & (f.apex_seen | (d.root_com_lin_vel_w[:,2] <= 0)) & ~physical)
            peak = torch.maximum(f.peak,d.root_link_pos_w[:,2])-self._terrain.env_origins[:,2]
            error = torch.linalg.vector_norm(d.body_link_pos_w[:,self._foot_body_id,:2]-self._two_ground_pt_xy_w,dim=-1)
            strict = valid & (peak >= 1.) & (error <= .1) & (tilt <= 15.)
            before = self.loop.stage.clone()
            event = self.ring.event.clone()
            power_inputs = None
            if self.ascent_power is not None:
                c = self.ascent_power.config
                local = d.root_link_pos_w-self._terrain.env_origins
                direction = self.forward_progress.direction
                coordinate = (local[:,:2]*direction).sum(-1)
                lateral = (local[:,:2]*self.forward_progress.side).sum(-1).abs()
                corridor = ((coordinate >= 0) & (coordinate <= self.forward_progress.length)
                    & (lateral <= c['lateral_limit_m'])
                    & (local[:,2] >= c['root_height_range_m'][0])
                    & (local[:,2] <= c['root_height_range_m'][1]))
                free = ((d.joint_pos[:,self._spring_joint_id] <= c['free_spring_q_max_m'])
                    & (d.joint_vel[:,self._spring_joint_id].abs() <= c['free_spring_qdot_max_m_s'])
                    & (d.body_link_pos_w[:,self._foot_body_id,2]-self._terrain.env_origins[:,2] > .012))
                power_before = self._episode_sums['flight_power_p'].clone()
                power_inputs = dict(active=before==self.loop.RING,
                    airborne=(f.phase==f.CANDIDATE)|(f.phase==f.FLIGHT),spring_free=free,
                    corridor=corridor,vz=d.root_link_lin_vel_w[:,2],
                    unsafe=physical|self.ring.collision|self.ring.failed|self.reset_buf.bool(),
                    passed=self.ring.passed,dt=self.step_dt)
            progress = None
            if self.forward_progress is not None:
                c = self.forward_progress.config
                free = ((f.phase == f.FLIGHT) & (f.air_steps >= 5)
                        & (d.joint_pos[:,self._spring_joint_id] <= c['free_spring_q_max_m'])
                        & (d.joint_vel[:,self._spring_joint_id].abs() <= c['free_spring_qdot_max_m_s']))
                progress = self.forward_progress.step(d.root_link_pos_w-self._terrain.env_origins,
                    before == self.loop.RING,free,
                    physical | self.ring.collision | self.ring.failed | self.reset_buf.bool(),
                    vertical_velocity=d.root_com_lin_vel_w[:,2])
            # All existing v23 height/landing/attitude costs run against the
            # pre-transition ground goal; the outer OBS73 adapter records events
            # against that same goal before entering here.
            reward = super()._get_rewards()
            if power_inputs is not None:
                native = self._episode_sums['flight_power_p']-power_before
                relief = self.ascent_power.step(native,**power_inputs)
                reward += relief
                self._episode_sums['stage_b_ascent_power_relief'] += relief
            advance,complete,failed = self.loop.step(valid,strict,self.ring.passed,
                self.ring.collision | self.ring.failed,physical)
            joint = valid & strict & (before == self.loop.RING) & self.ring.passed & ~self.ring.collision & ~physical
            self.ring_joint_success |= joint
            self.ring_joint_total += joint.sum()
            bonus = event.float()*self.course['completion_reward']
            self.ring_success_total += event.sum()
            self.ring.event.zero_()
            self._episode_sums['ring_complete'] += bonus
            reward += bonus
            if progress is not None:
                reward += progress
                self._episode_sums['stage_b_forward_progress'] += progress
            ids = advance.nonzero(as_tuple=False).flatten()
            if len(ids):
                self._two_ground_pt_xy_w[ids] = self._two_ground_pt1_xy_w[ids].clone()
                self._two_ground_pt1_xy_w[ids] = self._two_ground_pt_xy_w[ids]
                self._two_ground_route_index[ids] += 1
                self._two_ground_total_advances += len(ids)
                _set_active_goal(self,ids)  # Also resets XY progress reference.
                if self.forward_progress is not None:
                    self.forward_progress.prime(ids,d.root_link_pos_w[ids]-self._terrain.env_origins[ids])
            terminal = failed | complete
            # New task endpoints need zero terminal potential. Preserve v23's
            # energy shaping coefficient/gamma, correcting only nonphysical
            # ring/sequence terminations introduced by this task.
            logical = (terminal | self.reset_terminated) & ~physical & ~self.reset_time_outs
            correction = torch.where(logical,-self.jump_cycle.config['gamma']*self.jump_cycle.previous_potential,0.)
            reward += correction
            self._episode_sums['stage_b_terminal_potential'] += correction
            self.reset_terminated |= terminal
            self.reset_buf |= terminal
            logs = self.extras.setdefault('log',{})
            if self.ascent_power is not None:
                logs.update({'Ring/ascent_power_relief_total':self.ascent_power.total,
                    'Ring/ascent_power_relief_active_steps':self.ascent_power.active_steps})
            for index,name in enumerate(('initial_stationary','ring_landing','post1','post2','post3_loop_complete')):
                logs[f'Loop/{name}_total'] = self.loop.completions[index]
            logs.update({'Ring/full_crossings_total':self.ring_success_total,
                         'Ring/strict_landing_total':self.ring_joint_total,
                         'Loop/failed_landings_total':self.loop.failures})
            if progress is not None:
                logs.update({'Ring/forward_progress_reward_mean':progress.mean(),
                    'Ring/forward_progress_reward_total':self.forward_progress.total,
                    'Ring/forward_progress_positive_steps':self.forward_progress.positive_steps})
                if self.forward_progress.early is not None:
                    logs.update({'Ring/early_ascent_progress_reward_total':self.forward_progress.early_total,
                        'Ring/early_ascent_progress_positive_steps':self.forward_progress.early_positive_steps})
            return reward

    return Obs73RingStageBEnv
