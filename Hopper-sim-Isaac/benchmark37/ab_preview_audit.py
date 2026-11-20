"""Same-release, same first target comparison of straight and reverse previews."""
from math import cos, dist, pi, sin
from statistics import mean,median

from .precision_routes import routes
from .precision_trials import read


def apex_forecast_bias(folder):
    """Median observed minus forecast along error on first long pre-reversal landings."""
    points={c['case_id']:c['omni_route_local_m'] for c in read(folder/'cases.json')['cases']}
    residual=[]
    for row in read(folder/'episodes.json'):
        for e in row['events']:
            i=e['target_index']
            if i<1 or not e['first_attempt_for_target'] or 'post_release_motion_along_m' not in e:
                continue
            a,b,c=points[row['case_id']][i-1:i+2]
            u,v=dist(a,b),dist(b,c)
            dot=sum((y-x)*(z-y) for x,y,z in zip(a,b,c))
            if u<.40-1.e-6 or v<.20-1.e-6 or dot>=-.866*u*v:
                continue
            ascent=e['release_to_apex_sample_s']
            planned=e['planned_flight_time_at_release_s']
            motion=e['post_release_motion_along_m']
            predicted=(e['foot_error_components_along_m'][0]+e['launch_error_along_m_s']*planned+
                       motion[0]+motion[1]*(planned-ascent)/(e['release_to_touch_s']-ascent))
            residual.append(e['foot_error_along_m']-predicted)
    return median(residual)


def paired_preview(folder,checkpoint):
    suites=('ab_continue','ab_reverse')
    cases=[]
    events=[]
    for suite in suites:
        p=folder/suite
        manifest=read(p/'manifest.json')
        if (manifest['evaluation_scope']!=suite or manifest['optimizer_updates']!=0 or manifest['seed']!=8531
                or manifest['checkpoint']!=str(checkpoint)):
            raise ValueError(f'Unexpected paired evaluation in {p}')
        bank=read(p/'cases.json')['cases']
        expected=routes(8531,32,ab_preview=suite)
        rows={r['case_id']:r for r in read(p/'episodes.json')}
        if sorted(rows)!=list(range(128)) or sorted(c['case_id'] for c in bank)!=list(range(128)):
            raise ValueError('Incomplete paired evaluation')
        physical={}
        for c in bank:
            idx=c['case_id']
            points=c['omni_route_local_m']
            if len(points)!=32 or max(dist(a,b) for a,b in zip(points,expected[idx]))>1.e-5:
                raise ValueError('Unexpected AB commands')
            physical[idx]={k:v for k,v in c.items() if k!='omni_route_local_m'}
        cases.append(physical)
        events.append(rows)
    if cases[0]!=cases[1]:
        raise ValueError('Release or dynamics differ between preview arms')
    report={}
    for parity,group in enumerate(('direct','moving_warmup')):
        a_idx=1+parity
        delta=[]
        counts={s:dict(a_hit=0,b_hit=0,b_reached=0,both=0) for s in suites}
        for i in range(parity,128,2):
            speed=[]
            for s,rows in zip(suites,events):
                first={e['target_index']:e for e in rows[i]['events'] if e['first_attempt_for_target']}
                a,b=first.get(a_idx),first.get(a_idx+1)
                qa=bool(a and a['strict'])
                qb=bool(b and b['strict'])
                counts[s]['a_hit']+=qa
                counts[s]['b_hit']+=qb
                counts[s]['b_reached']+=b is not None
                counts[s]['both']+=qa and qb
                if a and 'post_release_motion_along_m' in a:
                    theta=i//2%8*pi/4
                    vx,vy=a['release_velocity_xy_m_s']
                    release=vx*cos(theta)+vy*sin(theta)
                    descent=a['release_to_touch_s']-a['release_to_apex_sample_s']
                    speed.append((release,release+a['post_release_motion_along_m'][1]/descent,a['foot_error_along_m']))
            if len(speed)==2:
                delta.append(tuple(y-x for x,y in zip(*speed)))
        report[group]=dict(cases=64,counts=counts,speed_pairs=len(delta),
            reverse_minus_continue=dict(zip(('release_m_s','apex_m_s','landing_along_m'),map(mean,zip(*delta)))))
    return report
