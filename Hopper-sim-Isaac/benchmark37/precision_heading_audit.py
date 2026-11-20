"""Matched opposite-heading diagnostic using identical initial states and a frozen policy."""
import math

from .precision_trials import read


def strict_prefix(row):
    prefix=0
    for index,event in enumerate(row['events'][:11]):
        if (event['target_index']!=index or not event['first_attempt_for_target'] or not event['strict']
                or not 0<=event['foot_error_m']<=.05):
            break
        prefix=index
    return prefix,int(prefix==10 and row['timed_out'] and not row['physical_death'])


def paired_report(reference,reverse,checkpoint):
    data=[]
    for folder,suite in ((reference,'development'),(reverse,'opposite_straight')):
        m=read(folder/'manifest.json')
        if (m['status']!='complete' or m['checkpoint']!=str(checkpoint) or m['optimizer_updates']!=0
                or m['seed']!=7001 or m['evaluation_scope']!='precision10_v1_'+suite):
            raise ValueError('Invalid paired evaluation provenance')
        cases={c['case_id']:c for c in read(folder/'cases.json')['cases']}
        rows=read(folder/'episodes.json')
        if len(rows)!=128 or set(cases)!=set(range(128)) or {r['case_id'] for r in rows}!=set(cases):
            raise ValueError('Incomplete paired evaluation')
        for row in rows:
            points=cases[row['case_id']]['omni_route_local_m']
            for event in row['events']:
                idx=event['target_index']
                if not 0<=idx<len(points) or math.dist(event['target_local_xy_m'],points[idx])>1.e-5:
                    raise ValueError('Touchdown was scored at a different waypoint')
        data.append((cases,{r['case_id']:r for r in rows}))
    (before,normal),(after,opposite)=data
    sectors=('E','NE','N','NW','W','SW','S','SE')
    pairs={}
    control_delta=[]
    for i,c in before.items():
        other=after[i]
        ignored={'omni_route_local_m','next_ground_target_xy'}
        if {k:v for k,v in c.items() if k not in ignored}!={k:v for k,v in other.items() if k not in ignored}:
            raise ValueError(f'Release or physical conditions differ in case {i}')
        sign=-1 if i%2 else 1
        points=c['omni_route_local_m']
        desired=[[sign*x,sign*y] for x,y in points]
        preview=[x+y for x,y in zip(c['ground_target_xy'],desired[1])]
        if other['omni_route_local_m']!=desired or math.dist(other['next_ground_target_xy'],preview)>1.e-5:
            raise ValueError(f'Incorrect paired command in case {i}')
        if not i%2:
            control_delta.append(strict_prefix(opposite[i])[0]-strict_prefix(normal[i])[0])
        else:
            heading=(i//2)%8
            left,right=strict_prefix(normal[i]),strict_prefix(opposite[i])
            if heading>=4:
                left,right=right,left
            axis=heading%4
            entry=pairs.setdefault(sectors[axis]+'/'+sectors[axis+4],
                dict(cases=0,first_prefix=0,opposite_prefix=0,first_clean=0,opposite_clean=0,
                     first_longer=0,opposite_longer=0))
            entry['cases']+=1
            entry['first_prefix']+=left[0]
            entry['opposite_prefix']+=right[0]
            entry['first_clean']+=left[1]
            entry['opposite_clean']+=right[1]
            entry['first_longer']+=left[0]>right[0]
            entry['opposite_longer']+=right[0]>left[0]
    return dict(checkpoint=str(checkpoint),seed=7001,paired_physical_cases=64,axes=pairs,
                unchanged_omni=dict(same_prefix=control_delta.count(0),cases=len(control_delta),
                                    mean_absolute_prefix_change=sum(map(abs,control_delta))/len(control_delta)))
