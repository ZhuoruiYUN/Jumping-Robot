"""41->49 input transfer only; optimizer and iteration are never resumed."""
import torch


def prepare_stationary_to_ring(state,source_course,course):
    """Keep the learned65D foundation; enable the8 previously irrelevant inputs
    from zero weights and unit normalization, avoiding constant-phase variance.
    No other encoder, recurrent, head, normalizer or std tensor is changed.
    """
    if not (source_course.get('task')=='obs73_stationary_foundation_v1'
            and course.get('task') in ('drop_stationary_ring_stationary2_v1','obs73_fixed_wide_ring_stage_b_v1')):
        return []
    fold = course.get('stage_b',{}).get('fold_duplicate_ground_inputs',False)
    if fold:
        # Stationary Pt/Pnext/Pnext were identical measured vectors. Preserve
        # their aggregate contribution while cold-starting the newly distinct
        # lookahead command, which otherwise enters at ~24 source stds.
        for side in ('actor','critic'):
            for suffix in ('_mean','_var','_std'):
                stats=state[f'{side}_obs_normalizer.{suffix}']
                if not (torch.equal(stats[:,:2],stats[:,2:4]) and torch.equal(stats[:,:2],stats[:,4:6])):
                    raise ValueError('Duplicate ground-target transfer requires identical saved normalization statistics')
    changed=[]
    for key,value in state.items():
        if key in ('memory_a.rnn.weight_ih_l0','memory_c.rnn.weight_ih_l0'):
            if value.shape[1]!=73:raise ValueError('Stationary transfer requires73D input')
            if fold:
                value[:,:2]+=value[:,2:4]+value[:,4:6]
                value[:,2:6]=0.
            value[:,65:73]=0.;changed.append(key)
        elif key.startswith(('actor_obs_normalizer.','critic_obs_normalizer.')) and value.shape==(1,73):
            if key.endswith(('._mean','._var','._std')):
                value[:,65:73]=0. if key.endswith('._mean') else 1.
                changed.append(key)
    return changed


def initialize_same_contract(runner,path,course):
    import json
    from .ring_obs73 import observation_dim,validate_checkpoint_course
    validate_checkpoint_course(path,course)
    dim=observation_dim(course)
    if runner.alg.optimizer.state:raise ValueError('Expected a fresh optimizer')
    state=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    if any(state[f'memory_{side}.rnn.weight_ih_l0'].shape[1]!=dim for side in ('a','c')):
        raise ValueError(f'Expected matching{dim}D actor and critic')
    source_course=json.loads((path.parent/'ring_config.json').read_text())
    if dim == 72 or course.get('observation_version')=='direct_motor_crossing_quaternion73_history40_20_v1':
        if source_course['task'] != course['task']:
            raise ValueError('Crossing same-contract initialization currently supports the same task only')
        runner.alg.policy.load_state_dict(state,strict=True)
        runner.current_learning_iteration=0
        return dict(source=str(path),old_dim=dim,new_dim=dim,transfer=f'strict crossing OBS{dim} copy',
                    optimizer='fresh',iteration=0,action_std=state['std'].tolist())
    if course.get('task')=='obs73_omni_ground_targets_v1':
        from .obs73_omni import prepare_source
        folded=prepare_source(state,source_course,course)
        runner.alg.policy.load_state_dict(state,strict=True)
        runner.current_learning_iteration=0
        return dict(source=str(path),old_dim=dim,new_dim=dim,
            transfer='fold duplicate ground-target columns; preserve all other weights including ring8' if folded else 'strict same-contract copy',
            folded_ground_inputs=folded,reset_ring_inputs=[],optimizer='fresh',iteration=0,
            action_std=state['std'].tolist())
    changed=prepare_stationary_to_ring(state,source_course,course)
    runner.alg.policy.load_state_dict(state,strict=True)
    runner.current_learning_iteration=0
    ground_fold=bool(changed and course.get('stage_b',{}).get('fold_duplicate_ground_inputs'))
    return dict(source=str(path),old_dim=dim,new_dim=dim,
                transfer=('fold duplicate targets; retain other65D foundation weights; zero ring8 inputs and reset ring8 normalization' if ground_fold
                          else 'copy65D foundation; zero ring8 input weights and reset ring8 normalization' if changed else 'strict same-contract copy'),
                ground_target_transfer=('fold weights of three formerly identical target vectors into Pt; zero Pnext input weights; saved target normalization retained' if ground_fold else 'unchanged'),
                reset_ring_inputs=changed,
                optimizer='fresh',iteration=0,action_std=state['std'].tolist())


def initialize_49(runner,path):
    """Copy a trained49D ring policy, retaining std and normalizer statistics."""
    if runner.alg.optimizer.state:raise ValueError('Expected a fresh optimizer')
    state=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    if any(state[f'memory_{side}.rnn.weight_ih_l0'].shape[1]!=49 for side in ('a','c')):
        raise ValueError('Expected a49D actor and critic checkpoint')
    runner.alg.policy.load_state_dict(state,strict=True)
    runner.current_learning_iteration=0
    return dict(source=str(path),old_dim=49,new_dim=49,expanded=[],
                transfer='strict actor/critic/LSTM/normalizers/std copy',action_std=state['std'].tolist(),
                optimizer='fresh',optimizer_state_entries=len(runner.alg.optimizer.state),iteration=0)


def transfer(policy,path):
    old=torch.load(path,map_location='cpu',weights_only=False)['model_state_dict']
    if old['memory_a.rnn.weight_ih_l0'].shape[1]!=41:raise ValueError('Expected Stage4l41D')
    new=policy.state_dict();expanded=[]
    for key,value in old.items():
        if value.shape==new[key].shape:new[key]=value
        elif key in ('memory_a.rnn.weight_ih_l0','memory_c.rnn.weight_ih_l0'):
            target=torch.zeros_like(new[key]);target[:,:41]=value
            new[key]=target;expanded.append(key)
        elif 'obs_normalizer.' in key and value.shape==(1,41) and new[key].shape==(1,49):
            target=torch.ones_like(new[key]) if key.endswith(('._var','._std')) else torch.zeros_like(new[key])
            target[:,:41]=value;new[key]=target;expanded.append(key)
        else:raise ValueError(f'Unmapped tensor {key}: {value.shape}->{new[key].shape}')
    policy.load_state_dict(new,strict=True)
    # Direct LSTM/MLP equivalence with nonzero hidden/cell states, normalized
    # old inputs and arbitrary new inputs. One compatibility check, no seeds.
    device=next(policy.parameters()).device
    with torch.no_grad():
        x=torch.linspace(-1,1,41,device=device)[None].repeat(2,1)
        for side,prefix,head in (('a','actor',policy.actor),('c','critic',policy.critic)):
            norm=getattr(policy,prefix+'_obs_normalizer')
            full=torch.cat((x,torch.full((2,8),.37,device=device)),1)
            xn=norm(full)
            # Compare old and expanded recurrent cells using the saved weights.
            rnn=getattr(policy,'memory_'+side).rnn
            old_rnn=torch.nn.LSTM(41,256).to(device)
            old_rnn.load_state_dict({k.removeprefix('memory_'+side+'.rnn.'):v for k,v in old.items() if k.startswith('memory_'+side+'.rnn.')})
            h=torch.full((1,2,256),.03,device=device);c=torch.full_like(h,-.02)
            a,_=old_rnn(xn[:,:41][None],(h,c));b,_=rnn(xn[None],(h,c))
            torch.testing.assert_close(head(a),head(b),rtol=1e-5,atol=1e-6)
    return dict(source=str(path),old_dim=41,new_dim=49,new_columns_zero=True,
                expanded=expanded,action_value_equivalence=True,action_std=old['std'].tolist(),
                optimizer='fresh',iteration=0)
