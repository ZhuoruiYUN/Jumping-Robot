"""Descriptive, matched touchdown-error accounting; never a promotion gate."""
from collections import defaultdict
import math
import statistics


PARTS = ('release_plan','flight_duration','release_speed','post_release_motion','foot_offset_change')


def reversal_apex_evidence(rows, seed, release_allowance):
    """Calibrate on strict first landings, compare two speed gates on disjoint cases."""
    from .precision_routes import routes
    bank=routes(seed,32)
    calibration=[]
    validation=[]
    for row in rows:
        case=row['case_id']
        for event in row['events']:
            hop=event['target_index']
            if not (1<=hop<=10 and event['first_attempt_for_target'] and 'post_release_motion_along_m' in event):
                continue
            a,b,c=bank[case][hop-1:hop+2]
            edge=(b[0]-a[0],b[1]-a[1])
            preview=(c[0]-b[0],c[1]-b[1])
            length,future=math.hypot(*edge),math.hypot(*preview)
            if length<.40 or future<.20 or sum(x*y for x,y in zip(edge,preview))>=-.866*length*future:
                continue
            descent=event['release_to_touch_s']-event['release_to_apex_sample_s']
            at_release=-event['launch_error_along_m_s']
            at_apex=at_release-event['post_release_motion_along_m'][1]/descent
            hit=event['strict'] and event['foot_error_m']<=.05
            short=event['foot_error_m']>.05 and event['foot_error_along_m'] < -abs(event['foot_error_cross_m'])
            if (case//16)%2==0:
                if hit:
                    calibration.append(at_apex)
            elif hit or short:
                validation.append((hit,at_release,at_apex))
    hits=[v for v in validation if v[0]]
    misses=[v for v in validation if not v[0]]
    if min(len(calibration),len(hits),len(misses))<8:
        return None
    allowance=statistics.quantiles(calibration,n=4,method='inclusive')[2]
    if allowance<=0:
        return None
    def flagged(pool,index,limit):
        return sum(row[index]>limit for row in pool)/len(pool)
    old_gap=flagged(misses,1,release_allowance)-flagged(hits,1,release_allowance)
    sensitivity=flagged(misses,2,allowance)
    false_alarm=flagged(hits,2,allowance)
    gap=sensitivity-false_alarm
    return dict(calibration_hits=len(calibration),validation_hits=len(hits),validation_short=len(misses),
                apex_allowance_m_s=allowance,apex_short_flagged=sensitivity,apex_hit_flagged=false_alarm,
                release_gap=old_gap,supported=gap>=.25 and gap-old_gap>=.15)


def matched_components(rows, failures, seed=None):
    from .precision_routes import routes
    bank=routes(seed,32) if seed is not None else None
    indexed={row['case_id']:row for row in rows}
    controls=defaultdict(list)
    for row in rows:
        for hop,event in enumerate(row['events'][:11]):
            if (event['target_index']!=hop or not event['strict'] or not event['first_attempt_for_target']
                    or not 0<=event['foot_error_m']<=.05):
                break
            key=(row['case_id']%2,row['response_group'],hop,event['direction'])
            controls[key].append(event)
    groups={}
    for failure in failures:
        reason=failure['reason']
        if reason not in ('short','long','cross'):
            continue
        case_id,hop=failure['case_id'],failure['hop']
        row=indexed[case_id]
        event=row['events'][hop]
        axis='cross' if reason=='cross' else 'along'
        field=f'foot_error_components_{axis}_m'
        key=(case_id%2,row['response_group'],hop,event['direction'])
        name=('straight' if case_id%2 else 'omni')+'_'+reason
        pool=groups.setdefault(name,dict(eligible=0,deltas=[]))
        pool['eligible']+=1
        subset=None
        if bank is not None and name=='omni_short' and 0<hop<10:
            previous,current,following=bank[case_id][hop-1:hop+2]
            edge=[end-start for start,end in zip(previous,current)]
            preview=[end-start for start,end in zip(current,following)]
            if (math.hypot(*edge)>=.40 and math.hypot(*preview)>=.20 and
                    sum(a*b for a,b in zip(edge,preview))<-.866*math.hypot(*edge)*math.hypot(*preview)):
                subset=groups.setdefault('matched_pre_reversal_short',dict(eligible=0,deltas=[]))
                subset['eligible']+=1
        peers=[peer for peer in controls[key] if field in peer and abs(peer['commanded_edge_m']-event['commanded_edge_m'])<.06]
        if field not in event or not peers:
            continue
        sign=-1 if reason=='short' or (reason=='cross' and event['foot_error_cross_m']<0) else 1
        for item in (event,*peers):
            if len(item[field])!=5 or not math.isclose(sum(item[field]),item[f'foot_error_{axis}_m'],abs_tol=1.e-4):
                raise ValueError('Release-to-touch components do not reconstruct the measured foot error')
        delta=[100*sign*(value-statistics.mean(peer[field][index] for peer in peers))
               for index,value in enumerate(event[field])]
        pool['deltas'].append(delta)
        if subset is not None:
            subset['deltas'].append(delta)
    return {name:dict(eligible=pool['eligible'],matched=len(pool['deltas']),
                      excess_cm=dict(zip(PARTS,map(statistics.mean,zip(*pool['deltas'])))))
            for name,pool in groups.items()}
