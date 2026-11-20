"""59 retained proprioception/ground-target values + crossing state command13."""
import torch

from .ring_obs73 import canonical_quaternion, inverse_rotate, shadow_contact, SLICES as OLD_SLICES
from .t_frame import quaternion_product, rpy_from_quaternion

VERSION = 'direct_motor_no_event_history_crossing_state72_v1'
HISTORY_VERSION = 'direct_motor_no_event_history_crossing_state72_history40_20_v2'
SUPPORTED_VERSIONS = (VERSION,HISTORY_VERSION)
DIM = 72
LEGACY_SLICES = {k: v[:] for k, v in OLD_SLICES.items() if v[1] <= 58}
LEGACY_SLICES.update(shadow_contact=[58,59], crossing_position_body_m=[59,62],
              crossing_relative_rpy_rad=[62,65], crossing_velocity_body_m_s=[65,68],
              crossing_angular_velocity_body_rad_s=[68,71], crossing_state=[71,72])
SLICES = {k:v[:] for k,v in LEGACY_SLICES.items() if not (19<=v[0]<57)}
SLICES.update(state_history_40ms=[19,32],state_history_20ms=[32,45],
              motor_action_40ms=[45,49],motor_action_20ms=[49,53],motor_action_10ms=[53,57])
SLICES = dict(sorted(SLICES.items(),key=lambda item:item[1][0]))


def contract(version=HISTORY_VERSION):
    if version not in SUPPORTED_VERSIONS:raise ValueError('Unknown OBS72 history version')
    newer=version==HISTORY_VERSION
    return dict(version=version, dimension=DIM, actor_critic_same=True,
        slices=SLICES if newer else LEGACY_SLICES,
        reference_point='A=root link, not aperture centre or COM',
        removed='old73[58:64]: three apices and three landing errors; no event history in new memory',
        replacement='old ring8 replaced by desired crossing state12 plus state1',
        position='R_WB^T (p_A_des_world-p_A_world), metres',
        attitude='RPY of R_WB^T R_WB_des, Rz(yaw)Ry(pitch)Rx(roll), radians; order roll,pitch,yaw',
        velocity='R_WB^T v_A_des_world, m/s; target velocity, not frame velocity or velocity error',
        angular_velocity='R_WB^T omega_des_world, rad/s; physical angular velocity, not Euler derivatives',
        state='0 inactive (all command12 zero), 0.5 pending, 1 passed; no LSTM reset on passage',
        frame='body->world quaternion wxyz; aperture +X forward,+Y lateral,+Z up',
        history=('state[t-40ms],state[t-20ms]; accepted motor commands[t-40ms,t-20ms,t-10ms]'
                 if newer else 'state[t-20ms],state[t-10ms]; accepted motor commands[t-10ms,t-30ms,t-20ms]'),
        state_history_offsets_ms=[40,20] if newer else [20,10],
        action_history_offsets_ms=[40,20,10] if newer else [10,30,20],
        history_timing='100Hz observations; commit the inference snapshot and previous accepted action before accepting new action; no repeated noise, interpolation or averaging; unavailable pre-reset samples zero',
        action_interface='accepted policy commands F1,F2,F3,F4, not delayed commands or actual rotor states',
        noise='same coherent std0.01 state sample as OBS73; command transforms use that same sample',
        aperture='shape/size handled by upstream T-envelope fit; fixed robot size is not a policy input',
        limitation='RPY singular at relative pitch +/-pi/2; initial curriculum near upright, not arbitrary flips',
        checkpoint='strict same observation version only; history timing change requires explicit history migration, fresh optimizer and adaptation')


class ObservationMemory:
    def __init__(self, n, device, version=HISTORY_VERSION):
        if version not in SUPPORTED_VERSIONS:raise ValueError('Unknown OBS72 history version')
        self.version=version
        self.states = torch.zeros(n,2,13,device=device)
        self.actions = torch.zeros(n,2,4,device=device)
        self.recent_action = torch.zeros(n,4,device=device)
        # At inference t: state buffer ends at s[t-1], action buffer ends at
        # a[t-2]; recent_action is a[t-1]. One policy tick is10ms.
        self.state_buffer = torch.zeros(n,4,13,device=device)
        self.action_buffer = torch.zeros(n,3,4,device=device)

    def reset(self, ids):
        for value in (self.states,self.actions,self.recent_action,self.state_buffer,self.action_buffer): value[ids] = 0

    def commit(self, snapshot):
        if self.version==HISTORY_VERSION:
            self.state_buffer[:,:-1] = self.state_buffer[:,1:].clone()
            self.state_buffer[:,-1] = snapshot
            self.states.copy_(self.state_buffer[:,[0,2]])
            self.action_buffer[:,:-1] = self.action_buffer[:,1:].clone()
            self.action_buffer[:,-1] = self.recent_action
            self.actions.copy_(self.action_buffer[:,[0,2]])
            return
        self.states[:,0] = self.states[:,1].clone()
        self.states[:,1] = snapshot
        self.actions[:,0] = self.actions[:,1].clone()
        self.actions[:,1] = self.recent_action

    def motor_observation(self):
        if self.version==HISTORY_VERSION:
            return torch.cat((self.actions.flatten(1),self.recent_action),-1)
        return torch.cat((self.recent_action,self.actions.flatten(1)),-1)


def assemble(position,q,velocity_world,omega_body,targets_xy,ground_z,apex_target_ground,
             memory,target_position,target_q,target_velocity_world,target_omega_world,state):
    q = canonical_quaternion(q)
    vb = inverse_rotate(q,velocity_world)
    errors = torch.nn.functional.pad(targets_xy-position[:,None,:2],(0,1))
    errors = inverse_rotate(q[:,None].expand(-1,3,-1),errors)[...,:2].flatten(1)
    snapshot = torch.cat((vb,omega_body,q,velocity_world),-1)
    qi = q.clone(); qi[:,1:] *= -1
    relative_q = canonical_quaternion(quaternion_product(qi,canonical_quaternion(target_q)))
    command = torch.cat((inverse_rotate(q,target_position-position),rpy_from_quaternion(relative_q),
                         inverse_rotate(q,target_velocity_world),inverse_rotate(q,target_omega_world)),-1)
    command = torch.where((state != 0)[:,None],command,torch.zeros_like(command))
    obs = torch.cat((errors,vb,q,omega_body,velocity_world,memory.states.flatten(1),
                     memory.motor_observation(),
                     (apex_target_ground-(position[:,2]-ground_z))[:,None],
                     shadow_contact(position,q,ground_z),command,state[:,None]),-1)
    assert obs.shape == (len(position),DIM)
    return obs,snapshot


def wrap_env(base, *, supported_versions=SUPPORTED_VERSIONS, observation_size=DIM,
             assemble_fn=assemble, memory_version=None):
    class CrossingObs72Env(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.course['observation_version'] not in supported_versions or self.cfg.observation_space != observation_size:
                raise ValueError('Crossing observation course/config mismatch')
            if abs(self.step_dt-.01)>1e-9:
                raise ValueError('OBS72 history ages require a10ms policy step')
            self.crossing_memory = ObservationMemory(self.num_envs,self.device,
                memory_version or self.course['observation_version'])
            self._crossing_snapshot = None
            self.crossing_position_w = torch.zeros(self.num_envs,3,device=self.device)
            self.crossing_quaternion_w = torch.zeros(self.num_envs,4,device=self.device)
            self.crossing_quaternion_w[:,0] = 1
            self.crossing_velocity_w = torch.zeros_like(self.crossing_position_w)
            self.crossing_omega_w = torch.zeros_like(self.crossing_position_w)
            self.crossing_state = torch.zeros(self.num_envs,device=self.device)

        def set_crossing_command(self,ids,position_w,quaternion_w,velocity_w,omega_w,state=.5):
            # Explicit task/planner API. Positions already include environment origins.
            values = (position_w,quaternion_w,velocity_w,omega_w)
            if any(not bool(torch.isfinite(v).all()) for v in values):
                raise ValueError('Crossing command must be finite')
            if bool((quaternion_w.norm(dim=-1) < 1e-6).any()) or state not in (0.,.5,1.):
                raise ValueError('Invalid crossing quaternion/state')
            self.crossing_position_w[ids] = position_w
            self.crossing_quaternion_w[ids] = canonical_quaternion(quaternion_w)
            self.crossing_velocity_w[ids] = velocity_w
            self.crossing_omega_w[ids] = omega_w
            self.crossing_state[ids] = state

        def _reset_idx(self,ids):
            super()._reset_idx(ids)
            if hasattr(self,'crossing_memory'):
                ids = self._robot._ALL_INDICES if ids is None else ids
                self.crossing_memory.reset(ids)
                self.crossing_state[ids] = 0
                self._crossing_snapshot = None

        def _get_observations(self):
            d = self._robot.data
            p = d.root_link_pos_w + .01*torch.randn_like(d.root_link_pos_w)
            q = canonical_quaternion(d.root_quat_w + .01*torch.randn_like(d.root_quat_w))
            v = d.root_link_lin_vel_w + .01*torch.randn_like(d.root_link_lin_vel_w)
            w = d.root_ang_vel_b + .01*torch.randn_like(d.root_ang_vel_b)
            targets = torch.stack((self._two_ground_pt_xy_w,self._two_ground_pt1_xy_w,self._two_ground_pt1_xy_w),1)
            ground = self._terrain.env_origins[:,2]
            obs,self._crossing_snapshot = assemble_fn(p,q,v,w,targets,ground,self._desired_pos_w[:,2]-ground,
                self.crossing_memory,self.crossing_position_w,self.crossing_quaternion_w,
                self.crossing_velocity_w,self.crossing_omega_w,self.crossing_state)
            return {'policy':obs}

        def _pre_physics_step(self,actions):
            if self._crossing_snapshot is None: self._get_observations()
            self.crossing_memory.commit(self._crossing_snapshot)
            self._crossing_snapshot = None
            super()._pre_physics_step(actions)
            self.crossing_memory.recent_action.copy_(self._actions)

    return CrossingObs72Env
