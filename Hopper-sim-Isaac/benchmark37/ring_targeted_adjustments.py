"""Opt-in task costs for approach clearance, landing braking and pitch work.

Full-body projection is a conservative approach feedback signal. The existing
swept collision/full-crossing detector remains the sole ring event authority.
No command filtering, reference trajectory, physics or observation changes.
"""
import math
import torch
from .ring_min_loop import LoopState
from .ring_post_stability import active_mask

TERM_NAMES=('ring_approach_feedback','ring_landing_brake','post_pitch_work')


def validate(config):
    for key in ('approach_weight','landing_brake_weight','pitch_work_weight'):
        if not math.isfinite(config[key]) or config[key]<0:raise ValueError(f'Invalid targeted cost: {key}')
    for key in ('lookahead_m','clearance_scale_m','excess_cap','brake_radius_m','speed_cap_m_s',
                'free_spring_q_max_m','free_spring_qdot_max_m_s'):
        if not math.isfinite(config[key]) or config[key]<=0:raise ValueError(f'Invalid targeted scale: {key}')


def approach_cost(vertices,center,normal,velocity,radius,margin,dt,config):
    relative=vertices-center[:,None,None,:]
    axial=(relative*normal[:,None,None,:]).sum(-1)
    radial=torch.linalg.vector_norm(relative-axial[...,None]*normal[:,None,None,:],dim=-1)
    excess=(radial.amax((1,2))-(radius-margin)).clamp(min=0.)
    near=(1+axial.amax((1,2))/config['lookahead_m']).clamp(0.,1.)
    forward=(velocity*normal).sum(-1).clamp(0.,config['speed_cap_m_s'])
    normalized=(excess/config['clearance_scale_m']).clamp(max=config['excess_cap'])
    cost=-config['approach_weight']*near*normalized.square()*forward.square()*dt
    return cost,excess


def landing_cost(foot_xy,velocity_xy,target_xy,dt,config):
    distance=torch.linalg.vector_norm(foot_xy-target_xy,dim=-1)
    near=(1-distance/config['brake_radius_m']).clamp(0.,1.)
    speed=torch.linalg.vector_norm(velocity_xy,dim=-1).clamp(max=config['speed_cap_m_s'])
    return -config['landing_brake_weight']*near*speed.square()*dt


def pitch_work_cost(moment_y,omega_y,dt,config):
    # Both are in the current Body frame. Positive power injects pitch energy;
    # negative power brakes it and receives no additional work cost.
    return -config['pitch_work_weight']*(moment_y*omega_y).clamp(min=0.)*dt


def phase_masks(stage,phase,passed,joint_success,spring,spring_rate,alive,reset_buf,config,*,flight_phase,candidate_phase):
    ring_flight=(stage==LoopState.RING)&(phase==flight_phase)&alive
    approach=ring_flight&~passed
    braking=ring_flight&passed
    post=active_mask(stage,joint_success,reset_buf)&alive
    post_air=post&((phase==candidate_phase)|(phase==flight_phase))
    free=post_air&(spring<=config['free_spring_q_max_m'])&(spring_rate.abs()<=config['free_spring_qdot_max_m_s'])
    return approach,braking,post_air,free


class FlightPitchStats:
    """Per-flight extrema/RMS; excludes initial drop and clears only local state."""
    def __init__(self,n,device):
        self.tracking=torch.zeros(n,dtype=torch.bool,device=device)
        self.low=torch.zeros(n,device=device);self.high=self.low.clone()
        self.rate_sq=self.low.clone();self.samples=self.low.clone()
        self.count=torch.zeros((),device=device)
        self.span_sum=self.count.clone();self.span_gt10=self.count.clone();self.rms_sum=self.count.clone()

    def reset(self,ids):
        self.tracking[ids]=False
        for value in (self.low,self.high,self.rate_sq,self.samples):value[ids]=0.

    def step(self,pitch,rate,airborne,contact,settle):
        start=airborne&~self.tracking
        self.low=torch.where(start,pitch,self.low);self.high=torch.where(start,pitch,self.high)
        self.rate_sq=torch.where(start,0.,self.rate_sq);self.samples=torch.where(start,0.,self.samples)
        self.tracking|=start
        self.low=torch.where(airborne,torch.minimum(self.low,pitch),self.low)
        self.high=torch.where(airborne,torch.maximum(self.high,pitch),self.high)
        self.rate_sq+=torch.where(airborne,rate.square(),0.);self.samples+=airborne.float()
        complete=settle&self.tracking
        span=torch.rad2deg(self.high-self.low)
        rms=torch.sqrt(self.rate_sq/self.samples.clamp(min=1.))
        self.count+=complete.sum();self.span_sum+=torch.where(complete,span,0.).sum()
        self.span_gt10+=(complete&(span>10.)).sum();self.rms_sum+=torch.where(complete,rms,0.).sum()
        self.tracking&=~contact

    def log(self):
        denominator=self.count.clamp(min=1.)
        return {'PostRing/completed_flights_total':self.count,
                'PostRing/cumulative_pitch_span_mean_deg':self.span_sum/denominator,
                'PostRing/cumulative_pitch_span_gt10_rate':self.span_gt10/denominator,
                'PostRing/cumulative_pitch_rate_rms_rad_s':self.rms_sum/denominator}


def env_class(evaluation=False):
    from .ring_height_loop import env_class as height_class

    class RingTargetedEnv(height_class(evaluation=evaluation)):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.targeted_config=self.course['targeted_adjustments'];validate(self.targeted_config)
            self.targeted_totals={name:torch.zeros((),device=self.device) for name in TERM_NAMES}
            self.targeted_active_samples={name:torch.zeros((),device=self.device) for name in TERM_NAMES}
            for name in TERM_NAMES:self._episode_sums[name]=torch.zeros(self.num_envs,device=self.device)
            self.flight_pitch_stats=FlightPitchStats(self.num_envs,self.device)

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if hasattr(self,'flight_pitch_stats'):
                self.flight_pitch_stats.reset(self._robot._ALL_INDICES if ids is None else ids)

        def _get_rewards(self):
            d=self._robot.data;f=self._foot_event;c=self.targeted_config;q=d.root_quat_w
            spring=d.joint_pos[:,self._spring_joint_id];spring_rate=d.joint_vel[:,self._spring_joint_id]
            alive=(d.root_link_pos_w[:,2]>=.05)&(q[:,1:3].square().sum(-1)<=.5)&~self.reset_time_outs
            approach,braking,post_air,free=phase_masks(self.loop.stage,f.phase,self.ring.passed,
                self.ring_joint_success,spring,spring_rate,alive,self.reset_buf,c,
                flight_phase=f.FLIGHT,candidate_phase=f.CANDIDATE)
            extra={name:torch.zeros(self.num_envs,device=self.device) for name in TERM_NAMES}
            ids=approach.nonzero(as_tuple=False).flatten()
            if len(ids):
                vertices=self.ring.shape.vertices(d.root_link_pos_w[ids],q[ids],spring[ids])
                extra[TERM_NAMES[0]][ids],_=approach_cost(vertices,self.ring.center[ids],self.ring.normal[ids],
                    d.root_link_lin_vel_w[ids],self.course['inner_radius_m'],self.course['delta_m'],self.step_dt,c)
            extra[TERM_NAMES[1]]=torch.where(braking,landing_cost(d.body_link_pos_w[:,self._foot_body_id,:2],
                d.root_link_lin_vel_w[:,:2],self._two_ground_pt_xy_w,self.step_dt,c),0.)
            extra[TERM_NAMES[2]]=torch.where(free,pitch_work_cost(self._moment[:,0,1],d.root_ang_vel_b[:,1],self.step_dt,c),0.)
            pitch=torch.asin((2*(q[:,0]*q[:,2]-q[:,3]*q[:,1])).clamp(-1.,1.))
            contact=(spring>.002)
            settle=post_air&contact&(f.phase==f.FLIGHT)&(f.air_steps>=5)&(f.apex_seen|(d.root_com_lin_vel_w[:,2]<=0))
            self.flight_pitch_stats.step(pitch,d.root_ang_vel_b[:,1],post_air,contact,settle)
            reward=super()._get_rewards()
            for name,mask in zip(TERM_NAMES,(approach,braking,free)):
                value=extra[name];reward=reward+value;self._episode_sums[name]+=value
                self.targeted_totals[name]+=value.sum();self.targeted_active_samples[name]+=mask.sum()
            self.extras.setdefault('log',{}).update(self.flight_pitch_stats.log())
            return reward

    return RingTargetedEnv
