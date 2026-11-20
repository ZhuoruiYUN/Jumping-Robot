"""CPU checks for user-requested horizontal0.5m cap; apex remains1m."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ..obs73_omni import TASK,CASE_VERSION,case_version,distance_bins
from ..stationary_retention import RetentionRoute,compare_stationary
from ..crossing_transfer import validate_quaternion_source,migrate_quaternion
from ..crossing_obs73_quat import VERSION
from ..metrics import distribution,summarize

ROOT=Path(__file__).resolve().parents[2]


def run():
    c=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_range05_course_v8.json').read_text())
    source=ROOT/c['omni']['initial_checkpoint'];previous=validate_quaternion_source(source,c)
    m=json.loads((source.parent/'manifest.json').read_text())
    assert m['status']=='complete' and m['iteration']==99
    source_eval=ROOT/'outputs/benchmark72_omni_v2/eval_seed7001_20261113_081712_184479Z'
    evidence=json.loads((source_eval/'manifest.json').read_text())
    summary=json.loads((source_eval/'summary.json').read_text())
    assert Path(evidence['checkpoint']).resolve()==source.resolve()
    assert summary['promotion_gate']['passed'] and summary['counts']['physical_death']==0
    assert c['observation_version']==VERSION and c['training_iterations']==200
    assert c['omni']['moving_distance_range_m']==[.12,.5]
    assert c['omni']['strict_height_m']==1.
    assert all(level['target_apex_m']==1. for level in c['vertical_pretraining']['levels'])
    assert c['vertical_pretraining']==previous['vertical_pretraining']
    assert c['stationary_pretraining']==previous['stationary_pretraining']
    assert not c['t_frame_command']['enabled'] and not c['crossing_command']['enabled']
    rc=json.loads((ROOT/c['runner_config_source']).read_text())
    old_rc=json.loads((source.parent/'runner_config.json').read_text())
    expected=copy.deepcopy(old_rc['algorithm']);expected['entropy_coef']=.002
    assert rc['algorithm']==expected and rc['policy']==old_rc['policy']
    assert rc['num_steps_per_env']==old_rc['num_steps_per_env']==256
    state=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    migrated=migrate_quaternion(state,previous['observation_version'])
    assert migrated['memory_a.rnn.weight_ih_l0'].shape==(1024,73)
    assert torch.equal(migrated['std'],state['std'])
    assert not migrated['memory_a.rnn.weight_ih_l0'][:,59:].any()
    ids=torch.arange(128);route=RetentionRoute(128,'cpu',c['omni'],42)
    for updates,expected_max in ((0,.2),(60,.35),(120,.5),(200,.5)):
        assert abs(route.current_max(updates*256)-expected_max)<1e-6
        route.reset(ids,updates*256)
        assert int(route.stationary_mask.sum())==64
        assert not route.points[route.stationary_mask].any()
        lengths=route.points.diff(dim=1).norm(dim=-1)
        assert ((lengths<1e-6)|((lengths>=.12-1e-5)&(lengths<=expected_max+1e-5))).all()
        assert (lengths>expected_max-.01).any()
        queued=route.points.clone();route.current_max(200*256)
        assert torch.equal(queued,route.points),'Curriculum moved existing queued targets'
    moving=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True)
    moving.reset(ids)
    assert moving.current_max()==.5 and not moving.stationary_mask.any()
    directions=(torch.atan2(moving.points[:,1,1],moving.points[:,1,0])/(math.pi/4)+.5).floor().long()%8
    assert torch.equal(torch.bincount(directions,minlength=8),torch.full((8,),16))
    stationary=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True,stationary_only=True)
    stationary.reset(ids);assert not stationary.points.any()
    assert distance_bins(c)==[(.1,.2),(.2,.3),(.3,.4),(.4,.5)]
    # Execute the actual saved-case restoration and report logic on a CPU
    # seam; importing its simulator base is unnecessary for these methods.
    class CPUBase:
        def restore_cases(self,path):raise AssertionError('Reject incompatible range before restoring simulator state')
    tree=ast.parse((ROOT/'benchmark37/obs73_omni_evaluation.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='OmniEvaluation')
    ns=dict(RingEvaluation=CPUBase,torch=torch,json=json,math=math,TASK=TASK,
        CASE_VERSION=CASE_VERSION,case_version=case_version,distance_bins=distance_bins,
        distribution=distribution,summarize=summarize,DIRECTIONS=('E','NE','N','NW','W','SW','S','SE'))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<actual omni range05 report>','exec'),ns)
    collector=ns['OmniEvaluation']
    fake=SimpleNamespace(env=SimpleNamespace(course=c,_robot=SimpleNamespace(_ALL_INDICES=ids)))
    for wrong in (source_eval/'cases.json',ROOT/'outputs/benchmark73_quaternion_preserve_v6/eval_seed7001_20261113_124552_267210Z/cases.json'):
        try:collector.restore_cases(fake,wrong)
        except ValueError as exc:assert 'full final range' in str(exc)
        else:raise AssertionError('Incompatible short/0.8m cases accepted for0.5m evaluation')
    rows=json.loads((source_eval/'episodes.json').read_text())
    first=[e for row in rows for e in row['events'] if e['moving'] and e['first_attempt_for_target']]
    for i,event in enumerate(first):event['commanded_edge_m']=(.2,.3,.4,.5)[i%4]
    report,_=collector.result(SimpleNamespace(rows=rows,env=SimpleNamespace(course=c),case_source='CPU report fixture, not policy evaluation'))
    assert report['case_version']==c['omni']['case_version']
    assert list(report['per_distance_first_attempt'])==['0.10..0.20m','0.20..0.30m','0.30..0.40m','0.40..0.50m']
    assert sum(v['attempts'] for v in report['per_distance_first_attempt'].values())==len(first)
    assert report['promotion_gate']['foot_error_p95_m_max']==.065
    baseline=json.loads((ROOT/c['stationary_retention']['baseline_summary']).read_text())
    assert compare_stationary(baseline,baseline)['passed']
    return dict(status='passed',dimension=73,source=str(source),source_short_range_gate_passed=True,
        horizontal_max_schedule_m=[.2,.35,.5],apex_target_m=1.,
        stationary_rehearsal_fraction=.5,history_and_quaternion_contract_unchanged=True,
        long_and_short_saved_cases_rejected=True,distance_report_ends_at_50cm=True,
        stationary_baseline_gate_unchanged=True,entropy=.002,
        optimizer_updates=0,simulation_started=False,
        limitation='New0.5m candidate still needs manual training and separate moving/stationary evaluations; no claim of non-regression')


if __name__=='__main__':print(json.dumps(run(),indent=2))
