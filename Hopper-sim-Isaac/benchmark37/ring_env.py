"""Isolated Stage4l task extension. Frozen source files are never edited."""
import torch
from .calibration_20261028_v2 import CalibratedAirMixin
from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnv
from .real_release import reset_release
from .goal_radius import reset_fixed_attitude, goal_bonus_delta
from .pre_touch_control import prime_off_history, update_release_phase, suppress_off_wrench
from .energy_margin import energy_reward_delta
from .point_energy_margin import extra_energy_reward_delta
from .point_target_short import reset_target
from .two_ground_targets import _ensure, _set_active_goal, extra_observation
from .range08_curriculum import advance_on_hit
from .landing_gain_task import LandingGainTracker
from .ring_geometry import rotate
from .ring_geometry_v2 import make_ring_events


class RingEnv(CalibratedAirMixin, QuadhopperEnv):
    def __init__(self,cfg,course,diagnostic=False,**kwargs):
        self.landing_radius_m = course.get("omni", {}).get("strict_foot_m", .1)
        self.course=course;self.diagnostic=diagnostic;self.trajectory=[]
        self._physical_robustness_config = course.get('physical_robustness')
        self._physical_robustness = None
        if self._physical_robustness_config is not None:
            from .physical_robustness import validate
            if course.get('task') != 'obs73_omni_ground_targets_v1':
                raise ValueError('Planar perturbations require the OBS73 ground-target task')
            validate(self._physical_robustness_config)
        super().__init__(cfg,**kwargs)
        self._foot_event=LandingGainTracker(self.num_envs,self.device)
        self._foot_body_id=self._robot.find_bodies('SpringLeg')[0][0]
        self.ring=make_ring_events(self.num_envs,self.device,course)
        self.ring_joint_success=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.ring_success_total=torch.zeros((),device=self.device)
        self.ring_joint_total=torch.zeros((),device=self.device)
        for name in ('foot_touchdown','apex_deficit','foot_excess','ring_complete'):
            self._episode_sums[name]=torch.zeros(self.num_envs,device=self.device)

    def _reset_idx(self,ids):
        if ids is None or len(ids)==self.num_envs:ids=self._robot._ALL_INDICES
        super()._reset_idx(ids)
        if self._physical_robustness is not None:
            self._physical_robustness.reset(ids)
        d=self._robot.data
        self._robot.write_joint_state_to_sim(d.default_joint_pos[ids].clone(),d.default_joint_vel[ids].clone(),env_ids=ids)
        reset_release(self,ids)
        reset_fixed_attitude(self,ids,self.cfg.seed)
        prime_off_history(self,ids)
        plant = getattr(self, '_imu_link_plant', None)
        if plant is not None:
            plant.reset(ids)
        # Allocate the inherited target bookkeeping, then set the fixed R1 pair.
        reset_target(self,ids,self.cfg.seed,evaluation=False)
        _ensure(self,self.cfg.seed);self._range08_training=False
        self._two_ground_route_index[ids]=0
        self._reset_task_targets(ids)
        self._two_ground_total_resets+=len(ids)
        _set_active_goal(self,ids)
        if hasattr(self,'_foot_event'):self._foot_event.reset(ids)
        if hasattr(self,'ring'):
            self.ring.reset(ids,d.root_link_pos_w,d.root_quat_w,d.joint_pos[:,self._spring_joint_id],self._terrain.env_origins)
            self.ring_joint_success[ids]=False

    def _reset_task_targets(self,ids):
        d=self._robot.data
        origin=d.root_link_pos_w[ids,:2]
        self._two_ground_pt_xy_w[ids]=origin+origin.new_tensor(self.course['landing_xy_m'])
        self._two_ground_pt1_xy_w[ids]=self._two_ground_pt_xy_w[ids]+origin.new_tensor(self.course['next_edge_xy_m'])

    def _ring_eligible(self):
        f=self._foot_event
        return (f.phase==f.FLIGHT)&(f.attempt_index==0)

    def _landing_event(self,touch,designated,died):
        joint=touch&designated&self.ring.passed&self._foot_event.last_qualified&~died
        self.ring_joint_success|=joint
        self.ring_joint_total+=joint.sum()
        self.ring.failed|=touch&designated&~self.ring.passed

    def _advance_targets(self):
        advance_on_hit(self,self._foot_event.last_qualified&~self.reset_buf)

    def install_imu_link_plant(self):
        if not hasattr(self, '_imu_link_plant'):
            from .imu_link_plant import ImuLinkPlant
            self._imu_link_plant = ImuLinkPlant(self, self.course['imu_link_plant'])
            self._imu_link_plant.audit = self._imu_link_plant.install_spring()
        return self._imu_link_plant.audit

    def _pre_physics_step(self,actions):
        update_release_phase(self)
        if self._physical_robustness_config is not None and not self.omni_route.evaluation:
            # Delay the first draw until after the runner's nominal asset audit.
            if self._physical_robustness is None:
                from .physical_robustness import PlanarPhysicalRobustness
                self._physical_robustness = PlanarPhysicalRobustness(self, self._physical_robustness_config)
                self._physical_robustness.reset(self._robot._ALL_INDICES)
            self._physical_robustness.maybe_push()
        super()._pre_physics_step(actions)
        plant = getattr(self, '_imu_link_plant', None)
        running = None
        if plant is not None:
            plant.step()
            running = (plant.collective_u.amax(dim=-1) > 1e-8) | (plant.attitude_u.amax(dim=-1) > 1e-8)
        suppress_off_wrench(self, running)

    def _ring_sample(self):
        if not hasattr(self,'ring') or self.ring.previous is None:return
        d=self._robot.data;f=self._foot_event
        eligible=self._ring_eligible()
        self.ring.sample(d.root_link_pos_w,d.root_quat_w,d.joint_pos[:,self._spring_joint_id],eligible)
        if self.diagnostic:
            self.trajectory.append(torch.cat((
                d.root_link_pos_w-self._terrain.env_origins,d.root_quat_w,
                d.joint_pos[:,self._spring_joint_id,None],eligible[:,None].float()),-1).detach().clone())

    def _apply_action(self):
        self._ring_sample()
        super()._apply_action()

    def _get_dones(self):
        self._ring_sample()
        died,timeout=super()._get_dones()
        if hasattr(self,'ring') and not self.diagnostic:died=died|self.ring.collision
        return died,timeout

    def _get_rewards(self):
        reward=super()._get_rewards();d=self._robot.data
        gravity=abs(self.sim.cfg.gravity[2])
        args=(d.root_pos_w[:,2],d.root_lin_vel_w[:,2],self._desired_pos_w[:,2],gravity,
              self.cfg.energy_tracking_reward_scale,self.cfg.vertical_energy_error_scale,self.step_dt)
        # Exact inherited Stage4l energy target1.04m and root goal10cm changes.
        energy=energy_reward_delta(*args)+extra_energy_reward_delta(*args)
        goal=goal_bonus_delta(d.root_pos_w[:,:2],self._desired_pos_w[:,:2],self.cfg.goal_bonus_scale,self.step_dt)
        self._episode_sums['energy_tracking']+=energy;self._episode_sums['goal_bonus']+=goal
        quat=d.root_quat_w
        tilt=torch.rad2deg(torch.acos((1-2*quat[:,1:3].square().sum(-1)).clamp(-1,1)))
        died=(d.root_link_pos_w[:,2]<.05)|(quat[:,1:3].square().sum(-1)>.5)
        if not self.diagnostic:died|=self.ring.collision
        touch=(self._foot_event.phase==self._foot_event.FLIGHT)&(d.joint_pos[:,self._spring_joint_id]>.002)
        designated=self._ring_eligible()
        positive=self._foot_event.step(d.joint_pos[:,self._spring_joint_id],d.root_link_pos_w[:,2],
            d.root_com_lin_vel_w[:,2],d.body_link_pos_w[:,self._foot_body_id,:2],
            self._desired_pos_w[:,:2],tilt,died)
        deficit=self._foot_event.last_deficit_reward;excess=self._foot_event.last_foot_excess_reward
        self._episode_sums['foot_touchdown']+=positive;self._episode_sums['apex_deficit']+=deficit
        self._episode_sums['foot_excess']+=excess
        self._landing_event(touch,designated,died)
        bonus=self.ring.event.float()*self.course['completion_reward'] if not self.diagnostic else torch.zeros_like(reward)
        self.ring_success_total+=self.ring.event.sum();self.ring.event.zero_()
        self._episode_sums['ring_complete']+=bonus
        self._advance_targets()
        self.extras.setdefault('log',{}).update({
            'Ring/full_crossings_total':self.ring_success_total,
            'Ring/strict_landing_total':self.ring_joint_total})
        return reward+energy+goal+positive+deficit+excess+bonus

    def _get_observations(self):
        base=super()._get_observations()['policy']
        ground=extra_observation(self);ground+=torch.randn_like(ground)*.01
        old=torch.cat((base,ground),-1)
        if self.diagnostic:return {'policy':old}
        d=self._robot.data
        inverse=d.root_quat_w.clone();inverse[:,1:]*=-1
        center=rotate(inverse,self.ring.center-d.root_link_pos_w)
        normal=rotate(inverse,self.ring.normal)
        radius=torch.full((self.num_envs,1),self.course['inner_radius_m'],device=self.device)
        phase=torch.where(self.ring.passed,1.,.5)[:,None]
        return {'policy':torch.cat((old,center,normal,radius,phase),-1)}
