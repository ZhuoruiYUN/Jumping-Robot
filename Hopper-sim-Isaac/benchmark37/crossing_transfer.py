"""Explicit OBS73 -> OBS72 partial initialization, fresh optimizer only."""
import json

import torch

from .crossing_obs72 import VERSION, HISTORY_VERSION, DIM
from .ring_obs73 import VERSION as OLD_VERSION

KEEP = list(range(58)) + [64]
HISTORY_COLUMNS = list(range(45))+list(range(49,57))+list(range(45,49))+list(range(57,72))


def validate_source(path, course):
    previous = json.loads((path.parent/'ring_config.json').read_text())
    if course['observation_version'] != VERSION or previous['observation_version'] != OLD_VERSION:
        raise ValueError('Explicit migration requires OBS73 source and OBS72 destination')
    if previous['task'] != 'obs73_stationary_foundation_v1' or course['task'] != 'obs73_omni_ground_targets_v1':
        raise ValueError('First migration is stationary v23 -> short omni bridge only')
    return previous


def migrate(state, previous, course):
    # Fold the formerly identical Pt/Pnext inputs before selecting columns.
    # This part preserves their old contribution. Removing event/ring columns
    # does NOT preserve the whole policy: it evaluates them at normalized zero.
    from .obs73_omni import prepare_source
    state = {k: v.clone() for k,v in state.items()}
    folded = prepare_source(state,previous,course)
    for side in ('a','c'):
        key = f'memory_{side}.rnn.weight_ih_l0'
        old = state[key]
        if old.shape != (1024,73): raise ValueError('Expected LSTM256 with73 inputs')
        new = old.new_zeros((old.shape[0],DIM))
        new[:,:59] = old[:,KEEP]
        state[key] = new
    for key,old in list(state.items()):
        if 'obs_normalizer.' in key and old.shape == (1,73):
            new = old.new_ones((1,DIM)) if key.endswith(('._std','._var')) else old.new_zeros((1,DIM))
            new[:,:59] = old[:,KEEP]
            state[key] = new
    return state,folded


def initialize(runner,path,course):
    previous = validate_source(path,course)
    if runner.alg.optimizer.state: raise ValueError('Migration requires fresh optimizer')
    old = torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    state,folded = migrate(old,previous,course)
    runner.alg.policy.load_state_dict(state,strict=True)
    runner.current_learning_iteration = 0
    return dict(source=str(path),old_dim=73,new_dim=72,retained_columns=KEEP,
                new_command_columns_zero=True,folded_ground_inputs=folded,
                action_std=state['std'].tolist(),optimizer='fresh',iteration=0,
                behavior_equivalent=False,
                limitation='Deleted history and ring inputs lose their learned contribution; needs adaptation and evaluation')


def validate_history_source(path,course):
    previous=json.loads((path.parent/'ring_config.json').read_text())
    if (previous.get('observation_version')!=VERSION or course.get('observation_version')!=HISTORY_VERSION
            or previous.get('task')!=course.get('task')):
        raise ValueError('History migration requires old72 timing ->40/20ms72 timing with the same task')
    return previous


def migrate_history(state):
    """Move newest-action weights with the newest-action slot; age changes remain.

    New40ms action uses old30ms input weights, new20ms uses old20ms, and
    new10ms uses old10ms. State slots retain older/newer roles, with new ages.
    This is partial initialization, never a behavior-equivalence claim.
    """
    result={k:v.clone() for k,v in state.items()}
    for side in ('a','c'):
        key=f'memory_{side}.rnn.weight_ih_l0'
        if state[key].shape!=(1024,72):raise ValueError('History migration expects single-layer LSTM256/72D')
        result[key]=state[key][:,HISTORY_COLUMNS].clone()
    for key,value in state.items():
        if 'obs_normalizer.' in key and value.shape==(1,72):
            result[key]=value[:,HISTORY_COLUMNS].clone()
    return result


def initialize_history(runner,path,course):
    validate_history_source(path,course)
    if runner.alg.optimizer.state:raise ValueError('History migration requires a fresh optimizer')
    state=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    migrated=migrate_history(state)
    runner.alg.policy.load_state_dict(migrated,strict=True)
    runner.current_learning_iteration=0
    return dict(source=str(path),old_dim=72,new_dim=72,transfer='explicit history timing/slot migration',
                source_state_ages_ms=[20,10],target_state_ages_ms=[40,20],
                source_action_ages_ms=[10,30,20],target_action_ages_ms=[40,20,10],
                source_columns=HISTORY_COLUMNS,action_std=migrated['std'].tolist(),
                optimizer='fresh',iteration=0,behavior_equivalent=False,
                limitation='State sample ages and oldest action age changed; requires manual adaptation training/evaluation')


def validate_quaternion_source(path,course):
    from .crossing_obs73_quat import VERSION as QUAT_VERSION
    previous=json.loads((path.parent/'ring_config.json').read_text())
    if (previous.get('observation_version') not in (VERSION,HISTORY_VERSION)
            or course.get('observation_version')!=QUAT_VERSION
            or previous.get('task')!=course.get('task')):
        raise ValueError('Quaternion migration requires72D crossing RPY ->73D crossing quaternion with the same task; historical ring8 OBS73 is incompatible')
    return previous


def migrate_quaternion(state,source_version):
    if source_version not in (VERSION,HISTORY_VERSION):raise ValueError('Expected a72D crossing source')
    state=migrate_history(state) if source_version==VERSION else {k:v.clone() for k,v in state.items()}
    # The previous command block was inactive in the omni source. Cold-start
    # all14 command columns/statistics to avoid its collapsed zero variance.
    for side in ('a','c'):
        key=f'memory_{side}.rnn.weight_ih_l0'
        old=state[key]
        if old.shape!=(1024,72):raise ValueError('Expected LSTM256 with72 source inputs')
        new=old.new_zeros((1024,73));new[:,:59]=old[:,:59];state[key]=new
    for key,old in list(state.items()):
        if 'obs_normalizer.' in key and old.shape==(1,72):
            new=old.new_ones((1,73)) if key.endswith(('._var','._std')) else old.new_zeros((1,73))
            new[:,:59]=old[:,:59];state[key]=new
    return state


def initialize_quaternion(runner,path,course):
    previous=validate_quaternion_source(path,course)
    if runner.alg.optimizer.state:raise ValueError('Quaternion migration requires a fresh optimizer')
    old=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    migrated=migrate_quaternion(old,previous['observation_version'])
    runner.alg.policy.load_state_dict(migrated,strict=True);runner.current_learning_iteration=0
    return dict(source=str(path),old_dim=72,new_dim=73,
                transfer='explicit72D RPY crossing ->73D quaternion crossing initialization',
                history_timing_migrated=previous['observation_version']==VERSION,
                retained_foundation=59,cold_command_columns=14,action_std=migrated['std'].tolist(),
                optimizer='fresh',iteration=0,behavior_equivalent=False)
