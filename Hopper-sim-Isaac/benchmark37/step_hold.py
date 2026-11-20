"""Explicit cardinal one-step/hold exposure and evaluation; no motor control."""
import math

DIRECTIONS=((1.,0.),(-1.,0.),(0.,1.),(0.,-1.))


def step_hold_points(count,spacing,warmup_hops,direction):
    if (not isinstance(count,int) or not isinstance(warmup_hops,int)
            or not 1<=warmup_hops<count-1 or direction not in range(4)
            or not math.isfinite(spacing) or spacing<=0):
        raise ValueError('Invalid cardinal step/hold route')
    target=[spacing*v for v in DIRECTIONS[direction]]
    return [[0.,0.] if i<warmup_hops else target.copy() for i in range(count)]


def step_hold_eval_spec(course):
    low,high=course['omni']['moving_distance_range_m']
    count=course['omni']['route_points']
    if not low<=.4<=high or count<12:
        raise ValueError('Step/hold evaluation requires0.4m commands and12 route points')
    return dict(version='obs73_cardinal_step_hold_response64_64_v1',
        evaluation_scope='step_hold_cardinal_04_warmup3_hold6',spacing_m=.4,
        warmup_hops=3,minimum_post_move_landings=6,route_points=count,
        route_scope='Three stationary hops; one0.4m step along+X,-X,+Y,-Y(case_id%4); hold that target for remaining15s. NEXT point visible during the final warmup hop; no recurrent reset at handoff.')


def restore_step_hold_route(env,payload,apply=False):
    """Same route/scope seam for CPU checks and the physical case restorer."""
    spec=getattr(env,'step_hold_evaluation_spec',None)
    saved=payload.get('step_hold_evaluation')
    if saved is not None and saved!=spec:
        raise ValueError('Step/hold cases require their explicit matching evaluation scope')
    if spec and getattr(env,'straight_evaluation_spec',None):
        raise ValueError('Evaluation route scopes are exclusive')
    if spec and apply:
        count=len(payload['cases'])
        if count!=128:raise ValueError('Step/hold route restoration requires128 cases')
        points=env.omni_route.points.new_tensor([step_hold_points(spec['route_points'],
            spec['spacing_m'],spec['warmup_hops'],i%4) for i in range(count)])
        env.omni_route.points.copy_(points);env.omni_route.index.zero_()
    return spec is not None


def body_rate_metrics(rows):
    """Time-weighted body rates, separate from Euler-angle derivatives."""
    values=[row.get('body_rate_totals') for row in rows]
    coverage=bool(values) and all(v is not None and len(v)==9
        and all(math.isfinite(x) and x>=0 for x in v) and v[0]>0 and v[6]>0 for v in values)
    if not coverage:return dict(metric_coverage_complete=False)
    total=[sum(v[i] for v in values) for i in range(9)]
    return dict(metric_coverage_complete=True,free_air_seconds=total[0],
        roll_pitch_rate_rms_rad_s=math.sqrt(total[1]/total[0]),
        roll_pitch_rate_peak_rad_s=max(v[2] for v in values),
        body_yaw_rate_rms_rad_s=math.sqrt(total[3]/total[0]),
        body_yaw_rate_peak_rad_s=max(v[4] for v in values),
        roll_pitch_rate_gt06_time_fraction=total[5]/total[0],launch_events=int(total[6]),
        launch_roll_pitch_rate_rms_rad_s=math.sqrt(total[7]/total[6]),
        launch_roll_pitch_rate_peak_rad_s=max(v[8] for v in values),
        scope='Moving-route free flight and once-only free liftoff after first hop,including holds; spring motion,drop,physical terminals and pure stationary members excluded. Diagnostic;0.6rad/s is a training allowance,not a hard actuator limit.')


def step_hold_batch_metrics(rows,spec,yaw,foot_limit_m=.1):
    from .metrics import landing_precision_metrics,rebound_motion_metrics
    if len(rows)!=128 or sorted(r['case_id'] for r in rows)!=list(range(128)):
        raise ValueError('Step/hold evaluation requires128 unique case IDs')
    warm=spec['warmup_hops'];holds=spec['minimum_post_move_landings'];required=warm+1+holds
    cases=[]
    for row in sorted(rows,key=lambda r:r['case_id']):
        expected=step_hold_points(spec['route_points'],spec['spacing_m'],warm,row['case_id']%4)
        events=row['events'];first={}
        for ev in events:
            i=ev['target_index'];point=ev['target_local_xy_m']
            if (i not in range(spec['route_points']) or len(point)!=2
                    or any(not math.isfinite(x) for x in point)
                    or any(abs(a-b)>1e-4 for a,b in zip(point,expected[i]))
                    or bool(ev['moving'])!=(i==warm)):
                raise ValueError('Observed command differs from the declared step/hold route')
            first.setdefault(i,ev)
        sequence=(len(events)>=required and all(ev['target_index']==i and ev['strict']
            and ev['first_attempt_for_target'] for i,ev in enumerate(events[:required])))
        move=first.get(warm)
        later=[first[i] for i in range(warm+1,required) if i in first]
        cases.append(dict(case_id=row['case_id'],direction=('+X','-X','+Y','-Y')[row['case_id']%4],
            response_group='nominal' if row['case_id']<64 else 'weaker',
            moving_exposed=move is not None,moving_first_strict=bool(move and move['strict'] and move['first_attempt_for_target']),
            required_later_holds_observed=len(later),retries=sum(not ev['first_attempt_for_target'] for ev in events),
            strict_sequence_and_survived=bool(sequence) and not row['physical_death'],
            physical_death=bool(row['physical_death'])))
    def group(selected):
        return dict(episodes=len(selected),moving_first_strict=sum(c['moving_first_strict'] for c in selected),
            enough_later_holds=sum(c['required_later_holds_observed']==holds for c in selected),
            strict_sequence_and_survived=sum(c['strict_sequence_and_survived'] for c in selected),
            retries=sum(c['retries'] for c in selected),physical_deaths=sum(c['physical_death'] for c in selected))
    precision=landing_precision_metrics(rows,foot_limit_m);motion=rebound_motion_metrics(rows,yaw)
    rotation=body_rate_metrics(rows)
    checks=dict(all128_warmup_step_and_holds_first_attempt_strict_and_survived=all(c['strict_sequence_and_survived'] for c in cases),
        **{f'all15s_errors_within{foot_limit_m*100:g}cm':precision['passed']},all15s_yaw=motion['yaw_gate']['passed'],
        body_rate_coverage=rotation['metric_coverage_complete'])
    return dict(**spec,counts=group(cases),cases=cases,checks=checks,passed=all(checks.values()),
        per_direction={name:group([c for c in cases if c['direction']==name]) for name in ('+X','-X','+Y','-Y')},
        response_groups={name:group([c for c in cases if c['response_group']==name]) for name in ('nominal','weaker')},
        body_rates=rotation,acceptance='Additional simulation scope only; unchanged original straight,omni and stationary-retention gates still required. Rate improvement needs comparison to source; no hardware acceptance implied.')
