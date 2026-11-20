"""Frozen ten-command qualification routes; coordinates are metres in world XY."""
import math
import random
from collections import Counter
from functools import lru_cache


SEEDS = {'development': 7001, 'holdout': 9109, 'opposite_straight': 7001, 'continuous': 8171}
SEEDS.update(ab_continue=8531,ab_reverse=8531)


@lru_cache(maxsize=8)
def routes(seed, count, opposite_straight=False, continuous=False, ab_preview=None):
    bank = []
    for case_id in range(128):
        rng = random.Random(seed * 128 + case_id)
        heading = (case_id // 2 % 8) * math.pi / 4
        points = [(0., 0.)]
        if ab_preview:
            if case_id%2:
                warm_heading=rng.uniform(-math.pi,math.pi)
                points.append((.3*math.cos(warm_heading),.3*math.sin(warm_heading)))
            x,y=points[-1]
            dx,dy=.5*math.cos(heading),.5*math.sin(heading)
            a=(x+dx,y+dy)
            b=(x,y) if ab_preview=='ab_reverse' else (a[0]+dx,a[1]+dy)
            points.extend((a,b,(b[0]+.2,b[1]+.1)))
            bank.append(tuple(points+[points[-1]]*(count-len(points))))
            continue
        for hop in range(1, count if continuous else 11):
            radius = .5
            if case_id % 2 == 0:
                if continuous:
                    turn=rng.random()
                    if turn<.2:
                        heading+=math.pi
                    elif turn>=.4:
                        heading=rng.uniform(-math.pi,math.pi)
                    length=rng.random()
                    radius=0. if length<.1 else (.5 if length<.4 else .5*rng.random())
                elif hop == 4:
                    radius = 0.
                elif hop == 7:
                    heading += math.pi
                else:
                    heading = rng.uniform(-math.pi, math.pi)
                    radius = rng.uniform(0., .1) if hop == 2 else (.5 if hop in (6, 9) else rng.uniform(0., .5))
            previous = points[-1]
            points.append((previous[0]+radius*math.cos(heading), previous[1]+radius*math.sin(heading)))
        if opposite_straight and case_id%2:
            points=[(-x,-y) for x,y in points]
        bank.append(tuple(points + [points[-1]] * (count-len(points))))
    return tuple(bank)


def continuous_score(rows):
    if sorted(r['case_id'] for r in rows)!=list(range(128)):
        raise ValueError('Continuous evaluation needs all original episodes')
    groups={}
    for parity,name in enumerate(('omni','straight')):
        group=[r for r in rows if r['case_id']%2==parity]
        hits=attempts=pairs=confirmed=late_hits=late_attempts=0
        longest=[]
        for row in group:
            streak=peak=0
            previous=None
            for event in row['events']:
                idx=event['target_index']
                first=event['first_attempt_for_target']
                ok=first and event['strict'] and event['foot_error_m']<=.05
                if idx>0:
                    attempts+=first
                    hits+=ok
                    late_attempts+=first and idx>10
                    late_hits+=ok and idx>10
                    if previous==idx-1 and first:
                        pairs+=1
                        confirmed+=ok
                    streak=(streak+1 if previous==idx-1 else 1) if ok else 0
                    peak=max(peak,streak)
                previous=idx if ok else None
            longest.append(peak)
        groups[name]=dict(first_hits=hits,first_attempts=attempts,following_hits=confirmed,
            observed_following_attempts=pairs,after_ten_hits=late_hits,after_ten_attempts=late_attempts,
            longest_streak_per_case=longest,deaths=sum(r['physical_death'] for r in group))
    return groups


def straight_openings(failures):
    first={item['case_id']:item['hop'] for item in failures}
    sectors=[dict(origin_hits=0,strict_prefix4=0) for _ in range(8)]
    for case_id in range(1,128,2):
        missed=first.get(case_id,11)
        if missed>0:
            sector=sectors[case_id//2%8]
            sector['origin_hits']+=1
            sector['strict_prefix4']+=min(4,missed-1)
    return sectors


def score(rows, summary, seed):
    if sorted(row['case_id'] for row in rows) != list(range(128)):
        raise ValueError('Expected all 128 original episodes')
    bank = routes(seed, 32)
    totals = {name:dict(clean=0, prefix=0) for name in ('omni','straight')}
    failures = Counter()
    details = []
    deaths = 0
    for row in rows:
        case_id = row['case_id']
        group = totals['straight' if case_id % 2 else 'omni']
        deaths += bool(row['physical_death'])
        clean = True
        for index in range(11):
            if index >= len(row['events']):
                failures['missing'] += 1
                details.append(dict(case_id=case_id,hop=index,reason='missing'))
                clean = False
                break
            event = row['events'][index]
            target = event['target_index']
            if target not in range(32) or not math.dist(event['target_local_xy_m'], bank[case_id][target]) < 1e-5:
                raise ValueError('Recorded target differs from frozen route')
            if target != index or not event['first_attempt_for_target'] or not event['strict'] or not 0 <= event['foot_error_m'] <= .05:
                reason = 'height_or_tilt'
                if target != index or not event['first_attempt_for_target']:
                    reason = 'retry_or_skip'
                elif event['foot_error_m'] > .05:
                    along = event.get('foot_error_along_m')
                    cross = event.get('foot_error_cross_m')
                    reason = 'unprojected_xy' if along is None or cross is None else (
                        'cross' if abs(cross) >= abs(along) else ('short' if along < 0 else 'long'))
                failures[reason] += 1
                details.append(dict(case_id=case_id,hop=index,reason=reason,direction=event['direction'],
                                    edge_m=event['commanded_edge_m'],error_m=event['foot_error_m']))
                clean = False
                break
            group['prefix'] += int(index > 0)
        group['clean'] += int(clean and row['timed_out'] and not row['physical_death'])
    plane = summary['flight_plane']
    attitude = all(plane[key]['passed'] for key in ('hold','travel')) and summary['rebound_diagnostics']['yaw_gate']['passed']
    passed = not deaths and attitude and all(group['clean']==64 for group in totals.values())
    return dict(groups=totals, straight_openings=straight_openings(details),
                first_failures=dict(failures), first_failed_events=details, deaths=deaths, attitude=attitude,
                flight_plane=plane, yaw=summary['rebound_diagnostics']['yaw_gate'], passed=passed)
