"""59 proprioception values plus T-frame crossing command14 with unit quaternion.

This73D interface is distinct from historical OBS73 ring8. T-frame +X is the
directed normal, +Y follows the crossbar, -Z follows the stem. The target body
orientation is derived from the same T-frame pose used to place the aperture.
"""
import math

import torch

from . import crossing_obs72 as previous
from .ring_obs73 import canonical_quaternion
from .ring_geometry import rotate
from .t_frame import quaternion_product,quaternion_from_rpy

VERSION='direct_motor_crossing_quaternion73_history40_20_v1'
DIM=73
SLICES={k:v[:] for k,v in previous.SLICES.items() if v[1]<=59}
SLICES.update(crossing_position_body_m=[59,62],crossing_relative_quaternion_wxyz=[62,66],
              crossing_velocity_body_m_s=[66,69],crossing_angular_velocity_body_rad_s=[69,72],
              crossing_state=[72,73])


def contract():
    result=previous.contract(previous.HISTORY_VERSION)
    result.update(version=VERSION,dimension=DIM,slices=SLICES,
        replacement='ring/event histories removed; desired crossing position3+quaternion4+velocity3+angular velocity3+state1',
        attitude='q_relative=conjugate(q_WB)*q_WB_des, normalized wxyz with canonical hemisphere w>=0',
        t_frame='q_WB_des=q_WT*q_TB_des; q_WT is T-frame->world, q_TB_des is desired body->T-frame; identity means body axes align with the T template',
        t_frame_observation='T orientation is encoded in the relative desired quaternion; no redundant absolute-world quaternion input',
        t_frame_timing='Task samples/fixes T pose before inference and uses that same pose for geometry and target/reward; never generate unrelated random observation targets',
        orientation_error='1-dot(normalized(q_WB),normalized(q_WB_des))^2; sign invariant, zero only for the desired rotation',
        state='0 inactive(all13 command values zero),0.5 pending,1 passed; reset histories only on true reset',
        limitation='Quaternion input avoids Euler angle singularities; representation alone does not establish improved PPO convergence or reachability',
        checkpoint='Exact version required; old73 ring8 rejected despite equal dimension; old72 requires explicit quaternion migration')
    return result


def orientation_error(q_world_body,q_world_desired):
    """Sign-invariant task loss for the exact quaternion used by the observation."""
    q=canonical_quaternion(q_world_body);target=canonical_quaternion(q_world_desired)
    return (1-(q*target).sum(-1).square()).clamp(0,1)


def command_from_t_frame(center_world,q_world_t,offset_A_t,q_t_body,velocity_t,omega_t):
    """Derive the desired body pose and physical velocities from one T-frame pose.

    offset_A_t is the planner-selected A-point offset from the aperture centre,
    expressed in T coordinates. Velocities are desired robot velocities in the
    fixed T frame; they are not the frame's own velocity or Euler derivatives.
    """
    if any(not torch.isfinite(v).all() for v in
           (center_world,q_world_t,offset_A_t,q_t_body,velocity_t,omega_t)):
        raise ValueError('T-frame commands must be finite')
    if (q_world_t.norm(dim=-1)<1e-6).any() or (q_t_body.norm(dim=-1)<1e-6).any():
        raise ValueError('T-frame and target-body quaternions must be nonzero')
    frame=canonical_quaternion(q_world_t);mount=canonical_quaternion(q_t_body)
    target_q=canonical_quaternion(quaternion_product(frame,mount))
    return (center_world+rotate(frame,offset_A_t),target_q,
            rotate(frame,velocity_t),rotate(frame,omega_t))


def assemble(position,q,velocity_world,omega_body,targets_xy,ground_z,apex_target_ground,
             memory,target_position,target_q,target_velocity_world,target_omega_world,state):
    # Reuse the established measured state/history assembly; replace its whole
    # command block. No Euler angles enter this interface.
    old,snapshot=previous.assemble(position,q,velocity_world,omega_body,targets_xy,ground_z,
        apex_target_ground,memory,target_position,target_q,target_velocity_world,target_omega_world,state)
    measured_q=canonical_quaternion(q)
    inverse=measured_q.clone();inverse[:,1:]*=-1
    relative=canonical_quaternion(quaternion_product(inverse,canonical_quaternion(target_q)))
    command=torch.cat((old[:,59:62],relative,old[:,65:71]),-1)
    command=torch.where((state!=0)[:,None],command,torch.zeros_like(command))
    obs=torch.cat((old[:,:59],command,state[:,None]),-1)
    assert obs.shape==(len(position),DIM)
    return obs,snapshot


def wrap_env(base):
    adapted=previous.wrap_env(base,supported_versions=(VERSION,),observation_size=DIM,
                              assemble_fn=assemble,memory_version=previous.HISTORY_VERSION)

    class CrossingQuaternionEnv(adapted):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.t_frame_position_w=torch.zeros(self.num_envs,3,device=self.device)
            self.t_frame_quaternion_w=torch.zeros(self.num_envs,4,device=self.device)
            self.t_frame_quaternion_w[:,0]=1
            self._t_frame_rng=torch.Generator(device=self.device).manual_seed(int(self.cfg.seed)+421073)
            self._reset_t_frame_command(self._robot._ALL_INDICES)

        def set_t_frame_command(self,ids,center_world,quaternion_world_t,offset_A_t,
                               quaternion_t_body,velocity_t,omega_t,state=.5):
            p,q,v,w=command_from_t_frame(center_world,quaternion_world_t,offset_A_t,
                                         quaternion_t_body,velocity_t,omega_t)
            self.set_crossing_command(ids,p,q,v,w,state)
            self.t_frame_position_w[ids]=center_world
            self.t_frame_quaternion_w[ids]=canonical_quaternion(quaternion_world_t)

        def _reset_t_frame_command(self,ids):
            c=self.course.get('t_frame_command',{})
            if not c.get('enabled',False):return
            # Opt-in by an actual crossing task, which must use this pose for
            # both its aperture geometry and its quaternion tracking reward.
            if self.course.get('task')=='obs73_omni_ground_targets_v1':
                raise ValueError('Ground-only omni reward does not train T-frame passage; enable commands in a crossing task with matching geometry/reward')
            n=len(ids);origin=self._terrain.env_origins[ids]
            def values(key,size):
                value=origin.new_tensor(c[key])
                if value.shape!=(size,):raise ValueError(f'Invalid T-frame field: {key}')
                return value[None].expand(n,-1)
            frame=values('quaternion_world_t_wxyz',4)
            bounds=origin.new_tensor(c.get('local_rpy_jitter_deg',[0.,0.,0.]))
            if bounds.shape!=(3,) or not torch.isfinite(bounds).all() or (bounds<0).any():
                raise ValueError('Invalid T-frame attitude sampling bounds')
            rpy=(2*torch.rand((n,3),device=self.device,generator=self._t_frame_rng)-1)*bounds*(math.pi/180)
            frame=quaternion_product(canonical_quaternion(frame),quaternion_from_rpy(rpy))
            self.set_t_frame_command(ids,origin+values('center_local_m',3),frame,
                values('offset_A_in_t_m',3),values('quaternion_t_body_wxyz',4),
                values('velocity_in_t_m_s',3),values('angular_velocity_in_t_rad_s',3))

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if hasattr(self,'t_frame_quaternion_w'):
                ids=self._robot._ALL_INDICES if ids is None else ids
                self.t_frame_position_w[ids]=0
                self.t_frame_quaternion_w[ids]=self.t_frame_quaternion_w.new_tensor([1.,0.,0.,0.])
                self._reset_t_frame_command(ids)

        def crossing_orientation_error(self):
            error=orientation_error(self._robot.data.root_quat_w,self.crossing_quaternion_w)
            return torch.where(self.crossing_state!=0,error,0.)

    return CrossingQuaternionEnv
