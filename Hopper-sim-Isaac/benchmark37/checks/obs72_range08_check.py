"""CPU checks for the explicit OBS72 distance/precision curriculum; no PPO."""
import copy
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ..obs73_omni import GroundRoute, case_version, distance_bins, TASK, CASE_VERSION
from ..metrics import distribution, summarize
from ..ring_obs73 import validate_checkpoint_course

ROOT=Path(__file__).resolve().parents[2]


def run():
    # Execute the actual collector class against a CPU-only base. Its normal
    # import pulls in RingEnv/Isaac; class methods under test need neither.
    class CPUBase:
        def restore_cases(self,path):raise AssertionError('Should reject before simulator restore')
    module=ast.parse((ROOT/'benchmark37/obs73_omni_evaluation.py').read_text())
    node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='OmniEvaluation')
    namespace=dict(RingEvaluation=CPUBase,TASK=TASK,CASE_VERSION=CASE_VERSION,case_version=case_version,distance_bins=distance_bins,
                   torch=torch,json=json,math=math,distribution=distribution,summarize=summarize,
                   DIRECTIONS=('E','NE','N','NW','W','SW','S','SE'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual OmniEvaluation CPU seam>','exec'),namespace)
    OmniEvaluation=namespace['OmniEvaluation']
    course=json.loads((ROOT/'benchmark37/configs/obs72_omni_course_v3.json').read_text())
    source=ROOT/course['omni']['initial_checkpoint']
    old=json.loads((source.parent/'ring_config.json').read_text())
    assert json.loads((source.parent/'manifest.json').read_text())['status']=='complete'
    validate_checkpoint_course(source,course)
    assert old['vertical_pretraining']==course['vertical_pretraining']
    assert old['stationary_pretraining']==course['stationary_pretraining']
    previous=json.loads((source.parent/'runner_config.json').read_text())
    runner=json.loads((ROOT/course['runner_config_source']).read_text())
    expected=copy.deepcopy(previous);expected['algorithm']['entropy_coef']=.002
    assert expected==runner
    ids=torch.arange(128)
    route=GroundRoute(128,'cpu',course['omni'],42)
    assert abs(route.current_max(0)-.2)<1e-7
    assert abs(route.current_max(60*256)-.5)<1e-7
    assert abs(route.current_max(120*256)-.8)<1e-7
    assert abs(route.current_max(200*256)-.8)<1e-7
    route.reset(ids)
    initial=route.points.clone()
    assert route.points.diff(dim=1).norm(dim=-1).max()<=.20001
    route.current_max(120*256)
    assert torch.equal(initial,route.points),'Curriculum must not move queued targets'
    route.reset(ids[:64],60*256)
    assert torch.equal(initial[64:],route.points[64:]),'Partial reset changed other routes'
    assert route.points[:64].diff(dim=1).norm(dim=-1).max()<=.50001
    route.reset(ids,120*256)
    assert (route.points.norm(dim=-1)<=2.00001).all()
    lengths=route.points.diff(dim=1).norm(dim=-1)
    assert ((lengths<1e-6)|((lengths>=.12-1e-5)&(lengths<=.8+1e-5))).all()
    assert (lengths>.7).any(),'Final course failed to sample long jumps'
    before=route.points.clone();valid=torch.zeros(128,dtype=torch.bool);strict=valid.clone()
    valid[:2]=True;strict[0]=True;route.advance(valid,strict)
    assert route.index[0]==1 and not route.index[1:].any() and route.misses==1
    assert torch.equal(before,route.points),'Advancing/retrying must retain queued points'
    evaluation=GroundRoute(128,'cpu',course['omni'],7001,evaluation=True)
    evaluation.reset(ids)
    assert abs(evaluation.current_max(0)-.8)<1e-7
    edge=evaluation.points[:,1]
    directions=((torch.atan2(edge[:,1],edge[:,0])/(torch.pi/4)+.5).floor().long()%8)
    assert torch.equal(torch.bincount(directions,minlength=8),torch.full((8,),16))
    assert (edge.norm(dim=-1)>.6).any() and (edge.norm(dim=-1)>.1).all()
    legacy=GroundRoute(128,'cpu',old['omni'],7001,evaluation=True)
    legacy.reset(ids)
    assert legacy.points.diff(dim=1).norm(dim=-1).max()<=.20001
    assert case_version(old)!=case_version(course)
    # A short saved route must be rejected before mutating any simulator state.
    old_eval=ROOT/'outputs/benchmark72_omni_v2/eval_seed7001_20261113_081712_184479Z'
    fake=SimpleNamespace(env=SimpleNamespace(course=course,_robot=SimpleNamespace(_ALL_INDICES=ids)))
    try:OmniEvaluation.restore_cases(fake,old_eval/'cases.json')
    except ValueError as exc:assert 'full final range' in str(exc)
    else:raise AssertionError('Short cases accepted for long-range evaluation')
    # Summary plumbing uses a saved ledger as a fixture, not a new evaluation.
    rows=json.loads((old_eval/'episodes.json').read_text())
    for row,boundary in zip(rows,(.2,.4,.6,.8)):
        next(ev for ev in row['events'] if ev['moving'] and ev['first_attempt_for_target'])['commanded_edge_m']=boundary
    env=SimpleNamespace(course=course)
    summary,_=OmniEvaluation.result(SimpleNamespace(rows=rows,env=env,case_source='CPU fixture only'))
    assert sum(v['attempts'] for v in summary['per_distance_first_attempt'].values())==summary['moving_first_attempt']['attempts']
    assert all(v['attempts']>0 for v in summary['per_distance_first_attempt'].values())
    assert 'within_5cm' in summary['moving_first_position_accuracy']
    assert not summary['promotion_gate']['passed'],'Old short-range ledger must not pass the new precision gate'
    return dict(status='passed',dimension=72,distance_max_schedule_m=[.2,.5,.8],
        last80_updates_hold_max_m=.8,queued_targets_fixed=True,partial_reset_checked=True,
        final_evaluation_long_edges=True,balanced_first_directions=[16]*8,
        old_short_cases_rejected=True,distance_bins_nonoverlapping=True,
        old_short_course_preserved=True,entropy=.002,all_reward_and_physics_config_preserved=True,
        precision_gate_p95_m=.065,training_updates=0,simulation_started=False,
        limit='CPU wiring checks only; policy reachability/accuracy need manual training and a new final-range evaluation')


if __name__=='__main__':print(json.dumps(run(),indent=2))
