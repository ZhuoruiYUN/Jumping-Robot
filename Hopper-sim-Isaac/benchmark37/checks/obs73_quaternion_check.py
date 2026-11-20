"""CPU T-pose->goal->relative-quaternion tests; no PPO/Isaac/hardware."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .. import crossing_obs72 as old
from .. import crossing_obs73_quat as new
from ..crossing_transfer import (migrate_quaternion,migrate_history,validate_quaternion_source)
from .obs72_history_check import CPUBase
from ..ring_obs73 import observation_dim,validate_checkpoint_course,inverse_rotate
from ..t_frame import quaternion_from_rpy,quaternion_product

ROOT=Path(__file__).resolve().parents[2]


def run():
    config=dict(enabled=True,center_local_m=[.27,0.,.85],quaternion_world_t_wxyz=[1.,0.,0.,0.],
                local_rpy_jitter_deg=[0.,0.,0.],offset_A_in_t_m=[0.,0.,.0763333333],
                quaternion_t_body_wxyz=[1.,0.,0.,0.],velocity_in_t_m_s=[.8,0.,0.],
                angular_velocity_in_t_rad_s=[0.,0.,.1])
    cfg=SimpleNamespace(observation_space=73,seed=42)
    course=dict(observation_version=new.VERSION,task='CPU_T_frame_pose_fixture',t_frame_command=config)
    env=new.wrap_env(CPUBase)(cfg,course)
    ids=torch.arange(2)
    # T frame yaw90deg, robot yaw30deg -> relative desired yaw60deg.
    frame=quaternion_from_rpy(torch.tensor([[0.,0.,math.pi/2]])).repeat(2,1)
    body=quaternion_from_rpy(torch.tensor([[0.,0.,math.pi/6]])).repeat(2,1)
    env._robot.data.root_quat_w.copy_(body)
    center=torch.tensor([[.27,0.,.85]]).repeat(2,1)
    offset=torch.tensor([[0.,0.,.0763333333]]).repeat(2,1)
    mount=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
    velocity=torch.tensor([[.8,0.,0.]]).repeat(2,1);omega=torch.tensor([[0.,0.,.1]]).repeat(2,1)
    env.set_t_frame_command(ids,center,frame,offset,mount,velocity,omega)
    with patch('torch.randn_like',side_effect=lambda value:torch.zeros_like(value)):
        obs=env._get_observations()['policy']
        expected=quaternion_from_rpy(torch.tensor([[0.,0.,math.pi/3]])).repeat(2,1)
        assert obs.shape==(2,73)
        torch.testing.assert_close(obs[:,62:66],expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(obs[:,62:66].norm(dim=-1),torch.ones(2))
        torch.testing.assert_close(obs[:,59:62],inverse_rotate(body,env.crossing_position_w))
        torch.testing.assert_close(env.crossing_velocity_w,torch.tensor([[0.,.8,0.]]).repeat(2,1),atol=1e-6,rtol=0)
        torch.testing.assert_close(obs[:,66:69],inverse_rotate(body,env.crossing_velocity_w))
        torch.testing.assert_close(obs[:,69:72],inverse_rotate(body,env.crossing_omega_w))
        torch.testing.assert_close(env.crossing_orientation_error(),1-obs[:,62].square())
        # q and -q must publish identical inputs and identical orientation loss.
        env.set_t_frame_command(ids,center,-frame,offset,-mount,velocity,omega)
        torch.testing.assert_close(env._get_observations()['policy'],obs)
        torch.testing.assert_close(new.orientation_error(body,frame),new.orientation_error(-body,-frame))
        # General noncommuting frame/body rotations verify multiplication order.
        tilt=quaternion_from_rpy(torch.tensor([[.3,-.2,.4]])).repeat(2,1)
        mounting=quaternion_from_rpy(torch.tensor([[-.1,.25,.2]])).repeat(2,1)
        env.set_t_frame_command(ids,center,tilt,offset,mounting,velocity,omega)
        torch.testing.assert_close(env.crossing_quaternion_w,quaternion_product(tilt,mounting))
        # Frame orientation is generated before the first inference and reset.
        env._reset_idx(torch.tensor([0]))
        reset_obs=env._get_observations()['policy']
        assert reset_obs[0,72]==.5 and not reset_obs[0,19:57].any()
        torch.testing.assert_close(env.t_frame_quaternion_w[0],torch.tensor([1.,0.,0.,0.]))
        env._robot.data.root_quat_w[0]=env.crossing_quaternion_w[0]
        assert env.crossing_orientation_error()[0]<1e-6
        # True quaternion targets remain valid at Euler pitch90deg.
        pitch90=quaternion_from_rpy(torch.tensor([[0.,math.pi/2,0.]])).repeat(2,1)
        env.set_t_frame_command(ids,center,pitch90,offset,mount,velocity,omega)
        assert torch.isfinite(env._get_observations()['policy']).all()
        # Histories retain40/20ms state ages and40/20/10ms command ages.
        for tick in range(5):env._get_observations();env._pre_physics_step(torch.full((2,4),tick*.1))
        obs=env._get_observations()['policy']
        torch.testing.assert_close(obs[:,45:57],torch.tensor([[.1]*4+[.3]*4+[.4]*4]).repeat(2,1))
        env.crossing_state.zero_()
        assert not env._get_observations()['policy'][:,59:].any()
        assert not env.crossing_orientation_error().any()
    # A ground-only course cannot masquerade as conditional T-orientation learning.
    bad=copy.deepcopy(course);bad['task']='obs73_omni_ground_targets_v1'
    try:new.wrap_env(CPUBase)(cfg,bad)
    except ValueError as exc:assert 'matching geometry/reward' in str(exc)
    else:raise AssertionError('Enabled T orientation with unrelated ground reward')
    # A crossing task may sample frame attitude once per reset, then hold it.
    sampled=copy.deepcopy(course);sampled['t_frame_command']['local_rpy_jitter_deg']=[5.,5.,45.]
    randomized=new.wrap_env(CPUBase)(cfg,sampled)
    assert not torch.equal(randomized.t_frame_quaternion_w[0],randomized.t_frame_quaternion_w[1])
    torch.testing.assert_close(randomized.t_frame_quaternion_w.norm(dim=-1),torch.ones(2))
    # Real72D checkpoint migration; same-number historical73D stays rejected.
    c=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_omni_course_v5.json').read_text())
    source=ROOT/c['omni']['initial_checkpoint'];previous=validate_quaternion_source(source,c)
    state=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    migrated=migrate_quaternion(state,previous['observation_version'])
    foundation=migrate_history(state) if previous['observation_version']==old.VERSION else state
    for side in ('a','c'):
        key=f'memory_{side}.rnn.weight_ih_l0'
        assert migrated[key].shape==(1024,73) and not migrated[key][:,59:].any()
        torch.testing.assert_close(migrated[key][:,:59],foundation[key][:,:59])
    for key,value in migrated.items():
        if 'obs_normalizer.' in key and value.shape==(1,73):
            torch.testing.assert_close(value[:,:59],foundation[key][:,:59])
            target=torch.ones(1,14) if key.endswith(('._var','._std')) else torch.zeros(1,14)
            torch.testing.assert_close(value[:,59:],target)
        elif key not in ('memory_a.rnn.weight_ih_l0','memory_c.rnn.weight_ih_l0'):
            assert torch.equal(value,state[key]),key
    assert observation_dim(c)==73
    try:validate_checkpoint_course(source,c)
    except ValueError:pass
    else:raise AssertionError('Old checkpoint silently accepted')
    assert new.contract()['slices']['crossing_relative_quaternion_wxyz']==[62,66]
    return dict(status='passed',dimension=73,attitude='unit quaternion wxyz',
                t_frame_pose_before_inference_and_reset=True,relative_quaternion_order_checked=True,
                q_sign_invariance=True,noncommuting_frames_checked=True,pitch90_finite=True,
                position_velocity_and_angular_velocity_frame_checks=True,
                state_history_ms=[40,20],action_history_ms=[40,20,10],
                task_target_sampling_checked=True,unrelated_ground_reward_guard=True,
                real72_to73_initialization_checked=True,optimizer_updates=0,simulation_started=False,
                limit='Command assembly/CPU task seam only; full T crossing reward/swept geometry and policy training remain a separate stage')


if __name__=='__main__':print(json.dumps(run(),indent=2))
