"""Timestamped CPU environment checks for OBS72 history; no Isaac or learning."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..crossing_obs72 import VERSION,HISTORY_VERSION,wrap_env,contract
from ..crossing_transfer import HISTORY_COLUMNS,migrate_history,validate_history_source
from ..ring_obs73 import validate_checkpoint_course

ROOT=Path(__file__).resolve().parents[2]


class CPUBase:
    """Expose the environment lifecycle called by the actual observation adapter."""
    def __init__(self,cfg,course):
        self.cfg=cfg;self.course=course;self.num_envs=2;self.device='cpu';self.step_dt=.01
        p=torch.zeros(2,3);q=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1)
        data=SimpleNamespace(root_link_pos_w=p,root_quat_w=q,
                             root_link_lin_vel_w=p.clone(),root_ang_vel_b=p.clone())
        self._robot=SimpleNamespace(data=data,_ALL_INDICES=torch.arange(2))
        self._terrain=SimpleNamespace(env_origins=p.clone())
        self._two_ground_pt_xy_w=torch.zeros(2,2)
        self._two_ground_pt1_xy_w=torch.zeros(2,2)
        self._desired_pos_w=p.clone();self._desired_pos_w[:,2]=1.
        self._actions=torch.zeros(2,4)

    def _pre_physics_step(self,actions):self._actions=actions.clamp(-1,1).clone()
    def _reset_idx(self,ids):self._actions[ids]=0.


def lifecycle(version,state_ages,action_ages):
    env=wrap_env(CPUBase)(SimpleNamespace(observation_space=72),dict(observation_version=version))
    snapshots=[];actions=[]
    # Noisy measurement timing is unchanged; disable noise only for timestamps.
    with patch('torch.randn_like',side_effect=lambda value:torch.zeros_like(value)):
        for tick in range(9):
            env._robot.data.root_link_lin_vel_w[:]=torch.tensor([tick+.1,tick+.2,tick+.3])
            env._robot.data.root_ang_vel_b[:]=torch.tensor([tick+.4,tick+.5,tick+.6])
            obs=env._get_observations()['policy']
            assert obs.shape==(2,72)
            for slot,age in enumerate(state_ages):
                expected=snapshots[tick-age] if tick>=age else torch.zeros(2,13)
                torch.testing.assert_close(obs[:,19+13*slot:32+13*slot],expected)
            for slot,age in enumerate(action_ages):
                expected=actions[tick-age] if tick>=age else torch.zeros(2,4)
                torch.testing.assert_close(obs[:,45+4*slot:49+4*slot],expected)
            snapshot=env._crossing_snapshot.clone()
            # Repeated observation reads must not shift either history.
            again=env._get_observations()['policy']
            torch.testing.assert_close(again,obs)
            command=torch.tensor([tick*.1+.01,tick*.1+.02,tick*.1+.03,tick*.1+.04]).repeat(2,1)
            snapshots.append(snapshot);actions.append(command)
            env._pre_physics_step(command)
        untouched={key:getattr(env.crossing_memory,key)[1].clone()
                   for key in ('states','actions','recent_action','state_buffer','action_buffer')}
        env._reset_idx(torch.tensor([0]))
        for key,value in untouched.items():
            assert not getattr(env.crossing_memory,key)[0].any(),key
            torch.testing.assert_close(getattr(env.crossing_memory,key)[1],value)
        reset_obs=env._get_observations()['policy']
        assert not reset_obs[0,19:57].any()
        # Reset env0 must fill20ms samples only after two new policy ticks;
        # pre-reset data must never return when hidden buffers shift.
        new_snapshots=[];new_actions=[]
        for tick in range(5):
            env._robot.data.root_link_lin_vel_w[:]=10.+tick
            obs=env._get_observations()['policy']
            for slot,age in enumerate(state_ages):
                expected=new_snapshots[tick-age][0] if tick>=age else torch.zeros(13)
                torch.testing.assert_close(obs[0,19+13*slot:32+13*slot],expected)
            for slot,age in enumerate(action_ages):
                expected=new_actions[tick-age][0] if tick>=age else torch.zeros(4)
                torch.testing.assert_close(obs[0,45+4*slot:49+4*slot],expected)
            new_snapshots.append(env._crossing_snapshot.clone())
            command=torch.full((2,4),-.1*(tick+1));new_actions.append(command)
            env._pre_physics_step(command)


def run():
    lifecycle(HISTORY_VERSION,[4,2],[4,2,1])
    lifecycle(VERSION,[2,1],[1,3,2])
    c=contract();assert c['state_history_offsets_ms']==[40,20] and c['action_history_offsets_ms']==[40,20,10]
    course=json.loads((ROOT/'benchmark37/configs/obs72_omni_course_v4.json').read_text())
    source=ROOT/course['omni']['initial_checkpoint']
    validate_history_source(source,course)
    try:validate_checkpoint_course(source,course)
    except ValueError:pass
    else:raise AssertionError('Same72D input count must not silently bypass the changed history contract')
    old_course=json.loads((source.parent/'ring_config.json').read_text())
    validate_checkpoint_course(source,old_course)
    old=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    migrated=migrate_history(old)
    assert sorted(HISTORY_COLUMNS)==list(range(72))
    for key,value in old.items():
        if key in ('memory_a.rnn.weight_ih_l0','memory_c.rnn.weight_ih_l0') or ('obs_normalizer.' in key and value.shape==(1,72)):
            torch.testing.assert_close(migrated[key],value[:,HISTORY_COLUMNS])
        else:assert torch.equal(migrated[key],value),key
    for side in ('a','c'):
        key=f'memory_{side}.rnn.weight_ih_l0'
        x=torch.linspace(-1.,1.,72).repeat(2,1)
        torch.testing.assert_close(x@old[key].T,x[:,HISTORY_COLUMNS]@migrated[key].T,rtol=1e-5,atol=3e-6)
    old_range=json.loads((ROOT/'benchmark37/configs/obs72_omni_course_v3.json').read_text())
    for key in ('omni','vertical_pretraining','stationary_pretraining','runner_config_source','training_iterations'):
        assert course[key]==old_range[key],key
    return dict(status='passed',dimension=72,state_history_ms=[40,20],motor_action_history_ms=[40,20,10],
                actual_adapter_lifecycle_checked=True,numbered_frames_checked=9,
                repeated_reads_do_not_commit=True,partial_reset_and_zero_warmup_checked=True,
                legacy72_timing_preserved=True,checkpoint_contract_guard_checked=True,
                real_checkpoint_action_columns_and_normalizers_reordered=True,
                rewards_distance_physics_ppo_unchanged_from_v3=True,
                behavior_equivalence_claimed=False,optimizer_updates=0,simulation_started=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
