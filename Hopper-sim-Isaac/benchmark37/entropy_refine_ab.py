"""Frozen Stage4j: only PPO entropy regularization differs; Stage4i K0 task."""
import copy
import json
from types import SimpleNamespace

from .artifacts import ROOT,sha
from . import foot_excess_ab as source_task

SOURCE=ROOT/'outputs/benchmark37_stage4i_foot_excess_control/train_seed42_20261029_142556_238686Z/model_49.pt'
SOURCE_SHA256='1459f712168be5bd05db4d55c8a9516866016dafd51d7c146035ad445a8640c5'
COEFS=(.01,.002)
VARIANTS={k:'real_release_two_ground_range08_entropy_'+('control' if k==.01 else 'low') for k in COEFS}
SERIES={k:'benchmark37_stage4j_entropy_'+('control' if k==.01 else 'low') for k in COEFS}
inherited=source_task.inherited


def contract(coef):
    if coef not in COEFS:raise ValueError('Expected entropy0.01 or0.002')
    return dict(name='entropy_refine_v1',variant=VARIANTS[coef],entropy_coef=coef,
        source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA256,
        initialization='Exact41D actor/critic/LSTM/normalizers including learned std; fresh optimizer and iteration0; no std reset',
        intervention='Only existing PPO entropy_coef0.01 vs0.002;768x50x256 per arm,seed42',
        unchanged='Stage4i K0 task/rewards/sampler,41D next targets/five action histories,v2 physics/DR,release0.8..1.2m immediate control,15s,strict10cm/1m/15deg,retry/history; learning rate3e-4 adaptive/KL0.015 and all other PPO settings',
        code_sha256=sha(__file__))


def reward_contract(coef):
    if coef not in COEFS:raise ValueError('Invalid entropy')
    return source_task.reward_contract(0.)


def configure_ppo(cfg,coef,evaluation=False):
    """Validate the complete source PPO config; change exactly one existing field."""
    contract(coef)
    source=json.loads((SOURCE.parent/'runner_config.json').read_text())
    if source['algorithm']['entropy_coef']!=.01:raise ValueError('Pinned source entropy changed')
    expected=coef if evaluation else .01
    if cfg['algorithm']['entropy_coef']!=expected:raise ValueError('Unexpected input PPO entropy')
    normalized=copy.deepcopy(cfg)
    normalized['algorithm']['entropy_coef']=.01
    for key in ('seed','max_iterations'):normalized[key]=source[key]
    if normalized!=source:raise ValueError('Another PPO setting differs from pinned source')
    cfg['algorithm']['entropy_coef']=coef
    return dict(entropy_coef=coef,source_sha256=SOURCE_SHA256,obs_dim=41,
                learning_rate=source['algorithm']['learning_rate'],schedule=source['algorithm']['schedule'],
                desired_kl=source['algorithm']['desired_kl'],initial_std='copied exactly from source; not reset')


def audit_runner(runner,coef):
    if runner.alg.entropy_coef!=coef:raise ValueError('Actual PPO did not apply entropy setting')
    policy=runner.alg.policy
    if policy.memory_a.rnn.input_size!=41 or policy.memory_c.rnn.input_size!=41:
        raise ValueError('Observation architecture changed')
    return dict(entropy_coef=runner.alg.entropy_coef,initial_action_std=policy.std.detach().cpu().tolist(),obs_dim=41)


def validate_source(path,source_lock,physics_lock):
    if path.resolve()!=SOURCE.resolve() or sha(path)!=SOURCE_SHA256 or source_task.validate_checkpoint(path)!=0.:
        raise ValueError('Entropy refinement requires pinned formal Stage4i K0 model49')
    m=json.loads((path.parent/'manifest.json').read_text())
    if m['source_lock']!=source_lock or m['physics_lock']!=physics_lock:raise ValueError('Source locks differ')
    return m


def api(coef,training=False,paired=False):
    base=source_task.api(0.,training,paired)
    return SimpleNamespace(**{**vars(base),'VARIANT':VARIANTS[coef],'SERIES':SERIES[coef],
        'contract':lambda:contract(coef),'validate_source':validate_source})


def validate_checkpoint(path,allow_smoke=False):
    path=path.resolve(strict=True);m=json.loads((path.parent/'manifest.json').read_text())
    coef=next((k for k,v in VARIANTS.items() if m.get('variant')==v),None)
    if coef is None:raise ValueError('Not an entropy refinement checkpoint')
    smoke=bool(m.get('smoke'));updates=2 if smoke else 50
    cfg=json.loads((path.parent/'runner_config.json').read_text())
    audit=configure_ppo(cfg,coef,evaluation=True)
    if (smoke and not allow_smoke or m.get('status')!='complete' or m.get('mode')!='train'
            or path.parent.parent!=ROOT/'outputs'/SERIES[coef] or path.name!=f'model_{updates-1}.pt'
            or m.get('requested_iterations')!=updates or m.get('num_envs')!=(16 if smoke else 768)
            or m.get('task_intervention')!=contract(coef) or m.get('reward_intervention')!=reward_contract(coef)
            or m.get('initial_checkpoint_sha256')!=SOURCE_SHA256 or m.get('checkpoint_hashes',{}).get(path.name)!=sha(path)
            or m.get('physics_revision')!=inherited.REVISION or m.get('calibration_lock')!=inherited.physics_contract()
            or m.get('entropy_refine_runtime_config')!=audit
            or m.get('arguments',{}).get('foot_excess_scale')!=0.
            or m.get('entropy_refine_runtime',{}).get('entropy_coef')!=coef):
        raise ValueError('Incomplete/wrong entropy checkpoint provenance')
    return coef
