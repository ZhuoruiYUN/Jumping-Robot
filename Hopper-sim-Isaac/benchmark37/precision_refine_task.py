"""Frozen Stage4k: additional optimization of the selected low-entropy policy."""
import copy
import json
from types import SimpleNamespace

from .artifacts import ROOT,sha
from . import entropy_refine_ab as source_task

SOURCE=ROOT/'outputs/benchmark37_stage4j_entropy_low/train_seed42_20261101_081253_659776Z/model_49.pt'
SOURCE_SHA256='8785170ace732a386f1c3c3ecf968198f637a98836de14d0459fdcdd4b7a4dcb'
VARIANT='real_release_two_ground_range08_precision_refine'
SERIES='benchmark37_stage4k_precision_refine'


def read(path):return json.loads(path.read_text())


def contract():
    return dict(name='precision_refine_v1',variant=VARIANT,source_checkpoint=str(SOURCE),
        source_sha256=SOURCE_SHA256,intervention='One50-update768x256 seed42 refinement; no task or PPO hyperparameter change',
        initialization='Exact41D actor/critic/LSTM/normalizers and learned std; fresh optimizer and iteration0, not exact resume',
        unchanged='Stage4j low task/physics/DR/rewards/sampler/41D observations/five4-motor action histories, entropy0.002,15s training horizon, full-circle0..0.8m,strict10cm/1m/15deg,retry/history',
        energy='Diagnostic only, not a performance-selection gate; existing reward weights retained',code_sha256=sha(__file__))


def reward_contract():return source_task.reward_contract(.002)


def configure_ppo(cfg):
    source=read(SOURCE.parent/'runner_config.json');normalized=copy.deepcopy(cfg)
    for key in ('seed','max_iterations'):normalized[key]=source[key]
    if normalized!=source or source['algorithm']['entropy_coef']!=.002:
        raise ValueError('Precision refinement must retain every source PPO setting')
    return dict(source_sha256=SOURCE_SHA256,entropy_coef=.002,obs_dim=41,
        learning_rate=source['algorithm']['learning_rate'],schedule=source['algorithm']['schedule'],
        desired_kl=source['algorithm']['desired_kl'],initial_std='full source value, not reset')


def audit_runner(runner,training):
    audit=source_task.audit_runner(runner,.002)
    audit.update(source_sha256=SOURCE_SHA256,optimizer_state_entries=len(runner.alg.optimizer.state),
        iteration=runner.current_learning_iteration)
    if training and (audit['optimizer_state_entries'] or audit['iteration']):
        raise ValueError('Expected fresh optimizer and iteration0, not an optimizer resume')
    return audit


def validate_source(path,source_lock,physics_lock):
    path=path.resolve(strict=True)
    if path!=SOURCE.resolve() or sha(path)!=SOURCE_SHA256 or source_task.validate_checkpoint(path)!=.002:
        raise ValueError('Precision refinement requires the pinned formal Stage4j low model49')
    manifest=read(path.parent/'manifest.json')
    if manifest['source_lock']!=source_lock or manifest['physics_lock']!=physics_lock:
        raise ValueError('Source reference/physics locks differ')
    return manifest


def api(training=False,paired=False):
    base=source_task.api(.002,training,paired)
    return SimpleNamespace(**{**vars(base),'VARIANT':VARIANT,'SERIES':SERIES,
        'contract':contract,'validate_source':validate_source})


def validate_checkpoint(path,allow_smoke=False):
    import torch
    path=path.resolve(strict=True);m=read(path.parent/'manifest.json')
    smoke=bool(m.get('smoke'));updates=2 if smoke else 50;count=16 if smoke else 768
    config=read(path.parent/'runner_config.json');runtime=configure_ppo(config)
    cfg=read(path.parent/'env_config.json');expected=read(SOURCE.parent/'env_config.json')
    expected['scene']['num_envs']=count;expected['csv_log_path']=str(path.parent/'single_env.csv')
    std=torch.load(SOURCE,map_location='cpu',weights_only=False)['model_state_dict']['std'].tolist()
    audit=m.get('precision_refine_runtime',{});warm=m.get('warm_start_audit',{})
    sampler=m.get('two_ground_runtime_end',{});rewards=m.get('first_hop_margin_runtime',{})
    if (smoke and not allow_smoke or m.get('status')!='complete' or m.get('mode')!='train'
            or path.parent.parent!=ROOT/'outputs'/SERIES or path.name!=f'model_{updates-1}.pt'
            or m.get('variant')!=VARIANT or m.get('requested_iterations')!=updates or m.get('num_envs')!=count
            or m.get('seed')!=42 or config['seed']!=42 or config['max_iterations']!=updates
            or m.get('task_intervention')!=contract() or m.get('reward_intervention')!=reward_contract()
            or m.get('initial_checkpoint_sha256')!=SOURCE_SHA256
            or m.get('checkpoint_hashes',{}).get(path.name)!=sha(path)
            or m.get('precision_refine_runtime_config')!=runtime or cfg!=expected
            or m.get('physics_revision')!=source_task.inherited.REVISION
            or m.get('calibration_lock')!=source_task.inherited.physics_contract()
            or audit.get('entropy_coef')!=.002 or audit.get('initial_action_std')!=std
            or audit.get('optimizer_state_entries')!=0 or audit.get('iteration')!=0
            or warm.get('source_sha256')!=SOURCE_SHA256 or warm.get('new_iteration')!=0
            or warm.get('source_checkpoint')!=str(SOURCE)
            or warm.get('optimizer_state_entries')!=0 or not m.get('arguments',{}).get('precision_refine')
            or sampler.get('training_mode') is not True or sampler.get('training_mixture_enabled')
            or sampler.get('active_hard_rows')!=0 or sampler.get('current_sampling_max_m')!=.8
            or rewards.get('foot_excess_scale')!=0. or rewards.get('scale')!=50.
            or rewards.get('reference_world_m_by_slot')!=[1.,1.,1.]
            or m['arguments'].get('foot_excess_scale')!=0.):
        raise ValueError('Incomplete/wrong precision-refinement checkpoint or changed training config')
    return path
