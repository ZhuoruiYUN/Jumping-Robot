"""Balanced keyboard-style ground routes and explicit 60-second acceptance.

Task commands only: no observation, motor or robot-physics changes.
"""
import math
from .step_hold import DIRECTIONS


def queued_points(count,spacing,direction,warmup=3,moves=3,holds=3):
    if (direction not in range(4) or not isinstance(count,int) or count<16
            or not math.isfinite(spacing) or spacing<=0
            or any(not isinstance(x,int) or x<1 for x in (warmup,moves,holds))):
        raise ValueError('Invalid cardinal queued route')
    vector=DIRECTIONS[direction]
    # Warm up, three outward steps, hold, three return steps, hold; repeat.
    cycle=list(range(1,moves+1))+[moves]*holds+list(range(moves-1,-1,-1))+[0]*holds
    levels=[0]*warmup+[cycle[i%len(cycle)] for i in range(count-warmup)]
    return [[spacing*level*v for v in vector] for level in levels]


def evaluation_spec(course):
    count=course['omni']['route_points']
    if count<64 or course['omni']['moving_distance_range_m'][1]<.5:
        raise ValueError('60s queued evaluation needs at least64 points and0.5m support')
    return dict(version='obs73_cardinal_queued05_60s_v1',
        evaluation_scope='cardinal_queued05_warmup3_move3_hold3_reverse3_hold3_60s',
        spacing_m=.5,warmup_hops=3,moving_steps=3,hold_hops=3,episode_s=60.,
        minimum_landings=40,route_points=count,
        route_scope='128 first60s episodes; case_id%4 gives +X,-X,+Y,-Y. '
        'Three stationary warmup hops; three0.5m moves,three holds,three reverse moves,three holds,repeat. '
        'All misses,retries,deaths and yaw remain failures; no hardware acceptance implied.')


def restore_route(env,payload,apply=False):
    spec=getattr(env,'cardinal_queue_evaluation_spec',None)
    saved=payload.get('cardinal_queue_evaluation')
    if saved is not None and saved!=spec:raise ValueError('Queued cases require matching explicit evaluation scope')
    if spec and (getattr(env,'straight_evaluation_spec',None) or getattr(env,'step_hold_evaluation_spec',None)):
        raise ValueError('Queued evaluation is exclusive of straight and step/hold')
    if spec and apply:
        if len(payload['cases'])!=128:raise ValueError('Queued evaluation requires128 release/DR cases')
        env.omni_route.points.copy_(env.omni_route.points.new_tensor([
            queued_points(spec['route_points'],spec['spacing_m'],i%4,
                spec['warmup_hops'],spec['moving_steps'],spec['hold_hops']) for i in range(128)]))
        env.omni_route.index.zero_()
    return spec is not None


def batch_metrics(rows,spec,yaw,foot_limit_m=.1):
    from .metrics import landing_precision_metrics,rebound_motion_metrics
    from .step_hold import body_rate_metrics
    if len(rows)!=128 or sorted(r['case_id'] for r in rows)!=list(range(128)):
        raise ValueError('Queued evaluation requires128 unique cases')
    cases=[]
    for row in sorted(rows,key=lambda r:r['case_id']):
        points=queued_points(spec['route_points'],spec['spacing_m'],row['case_id']%4)
        events=row['events']
        for ev in events:
            i=ev['target_index']
            if (i not in range(len(points)) or len(ev['target_local_xy_m'])!=2
                    or any(not math.isfinite(x) for x in ev['target_local_xy_m'])
                    or any(abs(a-b)>1e-4 for a,b in zip(ev['target_local_xy_m'],points[i]))
                    or bool(ev['moving'])!=(i>0 and points[i]!=points[i-1])):
                raise ValueError('Queued event target differs from declared route')
        coverage=len(events)>=spec['minimum_landings']
        seconds=row.get('seconds',float('nan'))
        full_horizon=math.isfinite(seconds) and seconds>=spec['episode_s']-.02 and bool(row.get('timed_out',False))
        sequence=coverage and full_horizon and all(ev['target_index']==i and ev['first_attempt_for_target']
            and ev['strict'] for i,ev in enumerate(events))
        cases.append(dict(case_id=row['case_id'],direction=('+X','-X','+Y','-Y')[row['case_id']%4],
            enough_landings=coverage,strict_sequence_and_survived=bool(sequence) and not row['physical_death'],
            full_horizon=full_horizon,
            retries=sum(not e['first_attempt_for_target'] for e in events),physical_death=bool(row['physical_death'])))
    precision=landing_precision_metrics(rows,foot_limit_m);motion=rebound_motion_metrics(rows,yaw)
    precision['scope']=spec['route_scope'];motion['scope']=spec['route_scope']
    rates=body_rate_metrics(rows)
    def group(selected):
        return dict(episodes=len(selected),enough_landings=sum(c['enough_landings'] for c in selected),
            full_horizon=sum(c['full_horizon'] for c in selected),
            strict_sequence_and_survived=sum(c['strict_sequence_and_survived'] for c in selected),
            retries=sum(c['retries'] for c in selected),physical_deaths=sum(c['physical_death'] for c in selected))
    checks=dict(all128_have40_landings=all(c['enough_landings'] for c in cases),
        all128_reach60s=all(c['full_horizon'] for c in cases),
        all_targets_first_attempt_strict=all(c['strict_sequence_and_survived'] for c in cases),
        **{f'all60s_errors_within{foot_limit_m*100:g}cm':precision['passed']},all60s_yaw=motion['yaw_gate']['passed'],
        body_rate_coverage=rates['metric_coverage_complete'])
    return dict(**spec,counts=group(cases),cases=cases,checks=checks,passed=all(checks.values()),
        per_direction={name:group([c for c in cases if c['direction']==name]) for name in ('+X','-X','+Y','-Y')},
        precision=precision,motion=motion,body_rates=rates)
