"""Check the pre-impact leg reference against measured, successful consecutive long hops."""
import math
from statistics import median


def unit_axis(xy):
    return (*xy, math.sqrt(max(0., 1. - sum(v*v for v in xy))))


def angle_deg(axis, other):
    return math.degrees(math.acos(max(-1., min(1., sum(x*y for x,y in zip(axis,other))))))


def contact_reference_evidence(episodes, limit_deg):
    calibration=[]
    validation=[]
    for episode in episodes:
        case=episode['case_id']
        events=episode['events']
        for before,after in zip(events, events[1:]):
            if not (1<=before['target_index']<10 and after['target_index']==before['target_index']+1
                    and all(e['strict'] and e['first_attempt_for_target'] and e['commanded_edge_m']>=.4
                            for e in (before,after))):
                continue
            requested=before.get('continuation_reference_xy_m_s_before_touchdown')
            if requested is None or 'release_velocity_xy_m_s' not in after or 'release_vertical_speed_m_s' not in after:
                continue
            incoming=before['root_velocity_10ms_before_touchdown_m_s']
            change=after['release_vertical_speed_m_s']-incoming[2]
            if not math.isfinite(change) or change<=0:
                continue
            if (case//16)%2==0:
                calibration.append((case,change))
            else:
                validation.append((case,before,after))
    if (min(len(calibration),len(validation))<16 or
            min(len({row[0] for row in group}) for group in (calibration,validation))<6):
        return None
    vertical=median(x[1] for x in calibration)
    old_errors=[]
    errors=[]
    residual=[]
    measured=[]
    for _,before,after in validation:
        actual=unit_axis(before['approach_normal_xy_10ms_before_touchdown'])
        desired=unit_axis(before['desired_normal_xy_10ms_before_touchdown'])
        vx,vy,_=before['root_velocity_10ms_before_touchdown_m_s']
        tx,ty=before['continuation_reference_xy_m_s_before_touchdown']
        dx,dy=tx-vx,ty-vy
        scale=min(1.,vertical*math.tan(math.radians(limit_deg))/max(math.hypot(dx,dy),1.e-8))
        vector=(scale*dx,scale*dy,vertical)
        candidate=tuple(x/math.hypot(*vector) for x in vector)
        old_errors.append(angle_deg(actual,desired))
        errors.append(angle_deg(actual,candidate))
        rx,ry=after['release_velocity_xy_m_s']
        measured.append(math.hypot(rx-vx,ry-vy))
        residual.append(math.hypot(rx-vx-vertical*actual[0]/max(actual[2],1.e-8),
                                   ry-vy-vertical*actual[1]/max(actual[2],1.e-8)))
    closer=sum(y<x for x,y in zip(old_errors,errors))/len(errors)
    return dict(vertical_change_m_s=vertical,calibration_contacts=len(calibration),
                validation_contacts=len(validation),validation_cases=len({x[0] for x in validation}),
                old_median_deg=median(old_errors),candidate_median_deg=median(errors),closer_fraction=closer,
                contact_residual_m_s=median(residual),measured_horizontal_change_m_s=median(measured),
                supported=(median(old_errors)-median(errors)>=2 and closer>=2/3 and median(residual)<median(measured)))
