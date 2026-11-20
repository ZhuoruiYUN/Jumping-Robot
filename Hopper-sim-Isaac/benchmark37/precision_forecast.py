"""Matched pre-touchdown forecast errors on measured long steps before a turn."""
import math
import statistics

from .precision_routes import routes


def weak_straight_prefix(rows):
    """Score the actual in-circle/outside forecast up to the first broken jump."""
    report={}
    for ms in (150,250,350):
        hit=miss=false_alarm=clear=0
        clock_errors=[]
        for row in rows:
            sector=row['case_id']//2%8
            if not row['case_id']%2 or sector not in (4,5,6):
                continue
            theta=sector*math.pi/4
            u,v=math.cos(theta),math.sin(theta)
            for index,event in enumerate(row['events'][:11]):
                if event['target_index']!=index or not event['first_attempt_for_target']:
                    break
                residual=event.get(f'dense_forecast_minus_landing_xy_{ms}ms_m')
                clock=event.get(f'dense_remaining_s_{ms}ms')
                if index and residual is not None and clock is not None:
                    dx,dy=residual
                    predicted=math.hypot(event['foot_error_along_m']+u*dx+v*dy,
                                         event['foot_error_cross_m']-v*dx+u*dy)>.05
                    outside=event['foot_error_m']>.05
                    hit+=predicted and outside
                    miss+=not predicted and outside
                    false_alarm+=predicted and not outside
                    clear+=not predicted and not outside
                    clock_errors.append(abs(clock-ms*.001))
                if not event['strict']:
                    break
        report[str(ms)]=dict(hit=hit,miss=miss,false_alarm=false_alarm,clear=clear,
                             clock_error_s=statistics.median(clock_errors) if clock_errors else None)
    return report


def compare(rows,seed,turn_deg):
    bank=routes(seed,32)
    pairs=[]
    for row in rows:
        points=bank[row['case_id']]
        for event in row['events']:
            index=event['target_index']
            if not 1<=index<=10 or not event['first_attempt_for_target']:
                continue
            previous,current,following=points[index-1:index+2]
            edge=[b-a for a,b in zip(previous,current)]
            preview=[b-a for a,b in zip(current,following)]
            norm=math.hypot(*edge)
            future=math.hypot(*preview)
            if norm<.30 or future<=.1 or sum(a*b for a,b in zip(edge,preview))>norm*future*math.cos(math.radians(turn_deg)):
                continue
            suffix='_forecast_minus_landing_xy_250ms_m'
            if 'dense'+suffix in event and 'preview'+suffix in event:
                pairs.append(tuple(math.hypot(*event[key+suffix]) for key in ('dense','preview')))
    return dict(pairs=len(pairs),better=sum(foot<dense for dense,foot in pairs),
                dense_m=statistics.median(pair[0] for pair in pairs) if pairs else None,
                foot_m=statistics.median(pair[1] for pair in pairs) if pairs else None)
