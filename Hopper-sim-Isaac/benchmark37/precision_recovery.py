"""Identify terminal flights covered by the existing near-retry launch-tail objective."""
import math
from statistics import quantiles


def long_release_tail(control_rows, trial_rows):
    def attempts(rows):
        return {(r['case_id'],e['target_index']):e for r in rows for e in r['events']
                if e['first_attempt_for_target'] and 1<=e['target_index']<=10 and e['commanded_edge_m']>=.4
                and e.get('release_window_samples',0)>0}
    control,trial=attempts(control_rows),attempts(trial_rows)
    paired=[key for key in control.keys() & trial.keys() if key[0]%2]
    hits=[sum(events[k]['strict'] for k in paired) for events in (control,trial)]
    good=[-e['launch_error_along_m_s'] for e in control.values() if e['strict']]
    if len(good)<128 or len(paired)<128 or hits[1]<=hits[0]:
        return None
    cuts=quantiles(good,n=100,method='inclusive')
    fade=[round(cuts[i],4) for i in (89,94)]
    if not 0<fade[0]<fade[1]:
        return None
    severe=[]
    for row in trial_rows:
        terminal=row.get('terminal_flight',{})
        i=row['case_id']
        if not i%2 or not row['physical_death'] or not terminal.get('confirmed_release'):
            continue
        target=terminal['target_index']
        if not 1<=target<=10 or any(e['target_index']==target for e in row['events']):
            continue
        heading=i//2%8*math.pi/4
        delta=[a-b for a,b in zip(terminal['desired_release_velocity_xy_m_s'],terminal['release_velocity_xy_m_s'])]
        deficit=delta[0]*math.cos(heading)+delta[1]*math.sin(heading)
        if deficit>=fade[1]:
            severe.append(dict(case_id=i,target=target,deficit_m_s=deficit))
    if severe:
        return dict(fade_m_s=fade,matched_straight=len(paired),strict_hits=hits,control_successes=len(good),
                    terminal_first_attempts=severe)


def recovery_tail_deaths(rows, near_speed):
    found=[]
    for row in rows:
        terminal=row.get('terminal_flight',{})
        if not row['physical_death'] or not terminal.get('confirmed_release') or not row['events']:
            continue
        contact=row['events'][-1]
        if (contact['target_index']!=terminal['target_index'] or not contact['moving']
                or contact['strict'] or contact['navigation_advanced']):
            continue
        actual=terminal['release_velocity_xy_m_s']
        desired=terminal['desired_release_velocity_xy_m_s']
        error=math.dist(actual,desired)
        if math.hypot(*desired)<=near_speed and error>.35:
            found.append(dict(case_id=row['case_id'],target=terminal['target_index'],
                              previous_attempt=contact['attempt_for_target'],prior_error_m=contact['foot_error_m'],
                              actual_speed_m_s=math.hypot(*actual),desired_speed_m_s=math.hypot(*desired),
                              release_error_m_s=error,seconds_since_release=terminal['seconds_since_release']))
    return found
