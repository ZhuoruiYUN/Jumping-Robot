"""CPU retention checks using the deployed-v23 artifacts; no training/evaluation."""
import copy
import json
from pathlib import Path

import torch

from ..stationary_retention import RetentionRoute,stationary_reward_delta,compare_stationary
from ..obs73_reference_reward import flight_terms
from ..crossing_transfer import validate_quaternion_source,migrate_quaternion
from ..crossing_obs73_quat import VERSION

ROOT=Path(__file__).resolve().parents[2]


def run():
    c=json.loads((ROOT/'benchmark37/configs/obs73_quaternion_preserve_course_v6.json').read_text())
    source=ROOT/c['omni']['initial_checkpoint']
    previous=validate_quaternion_source(source,c)
    manifest=json.loads((source.parent/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['iteration']==199
    baseline=ROOT/c['stationary_retention']['baseline_checkpoint']
    baseline_course=json.loads((baseline.parent/'ring_config.json').read_text())
    deployed=json.loads((baseline.parent/'deployment_export.json').read_text())
    assert Path(deployed['source_checkpoint']).resolve()==baseline.resolve()
    reference=json.loads((ROOT/c['stationary_retention']['baseline_summary']).read_text())
    ref_manifest=json.loads((ROOT/c['stationary_retention']['baseline_summary']).with_name('manifest.json').read_text())
    assert Path(ref_manifest['checkpoint']).resolve()==baseline.resolve()
    cases=json.loads((ROOT/c['stationary_retention']['baseline_cases']).read_text())['cases']
    assert len(cases)==128 and [v['case_id'] for v in cases]==list(range(128))
    assert [v['response_group'] for v in cases]==['nominal']*64+['weaker']*64
    assert c['observation_version']==VERSION and c['training_iterations']==100
    assert not c['t_frame_command']['enabled']
    ids=torch.arange(128)
    route=RetentionRoute(128,'cpu',c['omni'],42)
    route.reset(ids)
    mask=route.stationary_mask
    assert int(mask.sum())==64 and not route.points[mask].any()
    assert route.points[~mask].abs().sum()>0
    assert route.current_max(0)==.8 and route.current_max(100*256)==.8
    lengths=route.points[~mask].diff(dim=1).norm(dim=-1)
    assert (lengths>.6).any() and lengths.max()<=.80001
    before=route.points.clone();valid=torch.zeros(128,dtype=torch.bool);strict=valid.clone()
    valid[:3]=True;strict[:2]=True;route.advance(valid,strict)
    assert route.index[0]==route.index[1]==1 and route.index[2]==0
    assert not route.points[mask].any() and torch.equal(before,route.points)
    route.reset(torch.tensor([0]));assert not route.points[0].any()
    torch.testing.assert_close(route.points[1:],before[1:])
    # Moving evaluation must contain no rehearsal cases; stationary evaluation
    # must contain no hidden moving goals, including Pnext.
    moving=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True);moving.reset(ids)
    stationary=RetentionRoute(128,'cpu',c['omni'],7001,evaluation=True,stationary_only=True);stationary.reset(ids)
    assert not moving.stationary_mask.any() and stationary.stationary_mask.all()
    assert not stationary.points.any() and (moving.points[:,1].norm(dim=-1)>.1).all()
    assert torch.equal(route.stationary_mask,ids%2==0),'Task identity must persist across reset'
    # Dense-reward replacement is exact for stationary samples and zero for moving ones.
    z=torch.tensor([.3,.7,1.,1.04,1.06,1.1]);v=torch.tensor([3.,-1.,0.,0.,.3,-.2]);target=torch.ones(6)
    mask=torch.tensor([True,False,True,True,False,True])
    scales=dict(energy_tracking=12.,distance_to_z=2.,apex=8.)
    current=c['vertical_pretraining']['reference_flight_reward']
    ref=c['stationary_retention']['baseline_reference_flight_reward']
    assert ref==baseline_course['vertical_pretraining']['reference_flight_reward']
    assert ref['energy_margin_m']==ref['height_reward_margin_m']==.04
    assert current['energy_margin_m']==current['height_reward_margin_m']==.06
    deltas=stationary_reward_delta(z,v,target,9.81,2.,scales,.01,current,ref,mask)
    old=flight_terms(z,v,target,9.81,2.,scales,.01,current)
    expected=flight_terms(z,v,target,9.81,2.,scales,.01,ref)
    for key,value in deltas.items():
        assert value.shape==(6,) and torch.isfinite(value).all()
        assert not value[~mask].any()
        torch.testing.assert_close((old[key]+value)[mask],expected[key][mask])
    # Strict gates fail closed on specific regressions, even when all other
    # metrics exactly equal the reference. This fixture is not a new policy result.
    assert compare_stationary(reference,reference)['passed']
    variants=[]
    x=copy.deepcopy(reference);x['counts']['physical_death']=1;variants.append(x)
    x=copy.deepcopy(reference);x['counts']['first_strict']-=1;variants.append(x)
    x=copy.deepcopy(reference);x['response_strata']['nominal']['foot_error_m']['p95']+=.0001;variants.append(x)
    x=copy.deepcopy(reference);x['response_strata']['weaker']['pitch']['mean_span_deg']+=.001;variants.append(x)
    x=copy.deepcopy(reference);x['response_strata']['weaker']['strict_landing_rate']-=.0001;variants.append(x)
    x=copy.deepcopy(reference);x['response_strata']['nominal']['completed_landings']-=1;variants.append(x)
    x=copy.deepcopy(reference);x['response_strata']['nominal']['episodes_with_enough_later_landings']=63;variants.append(x)
    x=copy.deepcopy(reference);x.pop('response_strata');variants.append(x)
    assert all(not compare_stationary(x,reference)['passed'] for x in variants)
    state=torch.load(source,map_location='cpu',weights_only=False)['model_state_dict']
    initialized=migrate_quaternion(state,previous['observation_version'])
    assert initialized['memory_a.rnn.weight_ih_l0'].shape==(1024,73)
    assert torch.equal(initialized['std'],state['std'])
    rc=json.loads((ROOT/c['runner_config_source']).read_text())
    source_rc=json.loads((source.parent/'runner_config.json').read_text())
    assert rc['algorithm']==source_rc['algorithm'] and rc['policy']==source_rc['policy']
    assert rc['algorithm']['entropy_coef']==.002
    return dict(status='passed',source=str(source),deployed_reference=str(baseline),
        dimension=73,training_updates=0,formal_evaluations=0,
        training_rehearsal_fraction=.5,full_distance_max_m=.8,
        task_allocation_and_partial_reset_checked=True,stationary_reward_exactly_restores_v23=True,
        moving_and_stationary_evaluations_separate=True,saved128_reference_cases_checked=True,
        regression_gate_adversarial_fixtures=len(variants),checkpoint_std_preserved=True,
        deployment_written=False,limit='Checks do not establish candidate non-regression; requires user training and both formal evaluations')


if __name__=='__main__':print(json.dumps(run(),indent=2))
