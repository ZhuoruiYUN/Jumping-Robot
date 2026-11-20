"""Explicit deployed-v23 ring73 -> crossing-quaternion73 initialization.

The dimension happens to match; input semantics do not. Deleted event/ring
signals and changed sample ages make this a partial transfer, not an export.
"""
import json
from pathlib import Path

import torch

from .crossing_obs73_quat import VERSION
from .ring_obs73 import VERSION as LEGACY_VERSION

ROOT=Path(__file__).resolve().parents[1]
COLUMNS=list(range(45))+list(range(49,57))+list(range(45,49))+[57,64]


def validate_source(path,course):
    path=Path(path).resolve(strict=True)
    previous=json.loads((path.parent/'ring_config.json').read_text())
    recovery=course.get('stationary_quaternion_recovery',{})
    if (course.get('observation_version')!=VERSION
            or previous.get('observation_version')!=LEGACY_VERSION
            or course.get('task')!='obs73_stationary_foundation_v1'
            or previous.get('task')!=course.get('task')):
        raise ValueError('Explicit stationary migration requires legacy stationary ring73 -> new quaternion73')
    if path!=(ROOT/recovery['initial_checkpoint']).resolve():
        raise ValueError('Recovery must start from the configured deployed-v23 checkpoint')
    baseline=course['stationary_retention']
    if path!=(ROOT/baseline['baseline_checkpoint']).resolve():
        raise ValueError('Migration source and stationary regression baseline must match')
    deployment=json.loads((path.parent/'deployment_export.json').read_text())
    if Path(deployment['source_checkpoint']).resolve()!=path:
        raise ValueError('Source differs from the recorded deployed checkpoint')
    for key in ('stationary_pretraining','vertical_pretraining'):
        if course[key]!=previous[key]:
            raise ValueError('Stationary recovery keeps deployed-v23 rewards, release and response randomization unchanged')
    if course.get('omni') or course.get('t_frame_command',{}).get('enabled'):
        raise ValueError('Recovery is stationary only, with inactive crossing commands')
    return previous


def migrate(state):
    result={key:value.clone() for key,value in state.items()}
    for side in ('a','c'):
        key=f'memory_{side}.rnn.weight_ih_l0'
        source=state[key]
        if source.shape!=(1024,73):raise ValueError('Expected deployed LSTM256/73D')
        result[key]=torch.zeros_like(source)
        result[key][:,:59]=source[:,COLUMNS]
    for key,value in state.items():
        if 'obs_normalizer.' in key and value.shape==(1,73):
            result[key]=torch.ones_like(value) if key.endswith(('._var','._std')) else torch.zeros_like(value)
            result[key][:,:59]=value[:,COLUMNS]
    return result


def initialize(runner,path,course):
    validate_source(path,course)
    if runner.alg.optimizer.state:raise ValueError('Explicit migration requires a fresh optimizer')
    original=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    state=migrate(original)
    runner.alg.policy.load_state_dict(state,strict=True)
    runner.current_learning_iteration=0
    return dict(source=str(path),old_dim=73,new_dim=73,
        transfer='explicit deployed stationary ring73 -> crossing quaternion73',
        retained_source_columns=COLUMNS,cold_command_columns=14,
        source_state_ages_ms=[20,10],target_state_ages_ms=[40,20],
        source_action_ages_ms=[10,30,20],target_action_ages_ms=[40,20,10],
        action_std=state['std'].tolist(),optimizer='fresh',iteration=0,
        behavior_equivalent=False,
        limitation='Event/ring inputs removed and sample ages changed; manual stationary adaptation and saved-v23 regression evaluation required')
