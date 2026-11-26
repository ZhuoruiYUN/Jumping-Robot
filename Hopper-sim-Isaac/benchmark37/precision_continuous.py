"""Unattended simulation-only precision training with immutable round artifacts."""
import fcntl
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
from statistics import median
import sys
import time

from .precision_routes import score, straight_openings
from .precision_impulse import contact_reference_evidence
from .precision_trials import ROOT, SERIES, measured, rank, read, run, save


HOME = SERIES/'ab_only_v52'
AB_COURSE = ROOT/'training_config_v51/ab_only/course.json'
BASE = SERIES/'round_03'
START = BASE/'train/model_49.pt'
def worker_revision():
    return tuple(path.stat().st_mtime_ns for path in (Path(__file__),Path(__file__).with_name('train_ab_pairs.py')))


WORKER_REVISION=worker_revision()


def proposal(parent, feedback, number, auxiliary):
    course = read(parent.parent/'ring_config.json') if parent != START else read(AB_COURSE)
    if not course['omni'].get('ab_pair_only'):
        raise ValueError('AB worker needs an AB-only parent course after the initial transfer')
    course['omni']['initial_checkpoint']=str(parent)
    rng=random.Random(1051+number)
    terms={}
    for parity,name in enumerate(('omni','straight')):
        counts={key:1 for key in ('short','long','cross')}
        for event in feedback['first_failed_events']:
            if event['case_id']%2==parity and event['reason'] in counts:
                counts[event['reason']]+=1
        gain=rng.uniform(.65,1.8)
        terms[name]={key:round(gain*value/max(counts.values()),4) for key,value in counts.items()}
    course['directional_landing']=terms
    short=feedback.get('matched_failure_components',{}).get('matched_pre_reversal_short',{})
    if short.get('matched',0)>=8 and short['excess_cm'].get('release_speed',0.)>=2.:
        course['precision_pre_reversal_release'].update(allowance_m_s=.17,weight=rng.uniform(.8,1.8))
    if (not feedback['attitude'] and number%3==0) or auxiliary=='stationary':
        for key in ('plane_peak','plane_lean','plane_leg_speed'):
            course['flight_plane_control'][key]=min(480.,course['flight_plane_control'][key]*rng.uniform(1.05,1.25))
    return course


def accepted(result,best):
    if result['passed']:
        return True
    return (rank(result)>rank(best) and result['deaths']<=best['deaths']
            and all(result['groups'][name]['clean']>=entry['clean'] for name,entry in best['groups'].items())
            and (result['attitude'] or not best['attitude'])
            and all(result['flight_plane'][name]['lean_peak_deg']<=max(10.,best['flight_plane'][name]['lean_peak_deg']+1.)
                    for name in ('hold','travel')))


def heading_weights(feedback):
    sectors=straight_openings(feedback['first_failed_events'])
    retention=[(entry['strict_prefix4']+4)/(4*entry['origin_hits']+8) for entry in sectors]
    spread=max(retention)-min(retention)
    if min(entry['origin_hits'] for entry in sectors)<4 or spread<.20:
        return None
    return [round(1.+2.*(max(retention)-value)/spread,4) for value in retention]


def keyboard_qualified(data,checkpoint):
    rows=data['cases']
    return (data['checkpoint']==str(checkpoint) and data['optimizer_updates']==0 and
            data['direction']=='four-axis-rebound' and data['landing_radius_m']==.05 and
            data['both_arrived']==data['all_ten']==128 and data['requested']==data['strict_arrivals']==256 and
            data['deaths']==data['endpoint_over_30cm']==0 and sorted(row['case_id'] for row in rows)==list(range(128))
            and all(row['requested']==row['strict_arrivals']==2 and row['warm'] and len(row['endpoint'])==10
                    and row['rebound_count']==6 and all(0<=hop['landing_foot_error_m']<=.05
                    for hop in row['warm']+row['endpoint']) for row in rows))


def preserve(path):
    if path.exists():
        path.rename(path.with_name(path.name+f'.interrupted-{time.time_ns()}'))


def stage(folder, name, *args):
    target=folder/name
    manifest=target/'manifest.json'
    if manifest.is_file() and read(manifest)['status']=='complete':
        return target
    preserve(target)
    log=folder/(name+'.log')
    preserve(log)
    run(log,*args,'--output',target)
    if read(manifest)['status']!='complete':
        raise RuntimeError(f'Incomplete {manifest}')
    return target


def persist(state,phase):
    state['phase']=phase
    save(HOME/'status.json',state)


def plateau_probe(home,number,parent,state):
    if number<6 or state.get('early_probe_parent')==str(parent) or state['extra']:
        return None
    for index in range(number-5,number):
        path=home/f'round_{index:05d}'/'decision.json'
        if not path.exists():
            return None
        decision=read(path)
        if decision['accepted_as_parent'] or decision['parent']!=str(parent) or decision.get('mode','train')!='train':
            return None
    last=home/f'round_{number-1:05d}'
    train=last/'train'
    if (read(train/'manifest.json')['status']!='complete' or not (train/'model_10.pt').is_file()
            or read(last/'decision.json')['checkpoint']!=str(train/'model_49.pt')):
        return None
    return train/'model_10.pt'


def extend_plateau(home,number,parent,state):
    if number<9 or state.get('early_probe_parent')!=str(parent) or state.get('extended_parent')==str(parent) or state['extra']:
        return False
    for index in range(number-3,number):
        path=home/f'round_{index:05d}'/'decision.json'
        if not path.exists():
            return False
        decision=read(path)
        if decision['accepted_as_parent'] or decision['parent']!=str(parent):
            return False
    return True


def evaluate_window(folder,name,checkpoint):
    development=stage(folder,name,'eval','--headless','--checkpoint',checkpoint,'--precision-suite','development')
    if read(development/'cases.json')['cases']!=read(SERIES/'reference/cases.json')['cases']:
        raise ValueError('Development cases differ from the frozen reference')
    return development,measured(development,checkpoint,'development')


def rejected_controls(home,number,parent,lookback):
    for index in range(number-1,max(0,number-lookback-1),-1):
        folder=home/f'round_{index:05d}'
        if not (folder/'decision.json').is_file():
            continue
        decision=read(folder/'decision.json')
        if (decision.get('mode')!='train' or decision['parent']!=str(parent) or decision['deaths']
                or decision['accepted_as_parent'] or decision['checkpoint']!=str(folder/'train/model_49.pt')):
            continue
        course=read(folder/'course.json')
        if course['training_iterations']==50:
            yield folder,course


def distance_control(home,number,parent,state):
    if number<13 or state['extra'] or state.get('distance_probe_parent')==str(parent):
        return None
    for folder,control in rejected_controls(home,number,parent,5):
        if control['omni'].get('precision_distance_practice'):
            continue
        events=[event for row in read(folder/'development/episodes.json') if row['case_id']%2==0
                for event in row['events'] if 1<=event['target_index']<=10 and event['first_attempt_for_target']]
        bands=[[event['strict'] for event in events if low<event['commanded_edge_m']<=high]
               for low,high in ((.25,.4),(.4,.50001))]
        medium,long=bands
        if min(map(len,bands))>=40 and sum(medium)/len(medium)-sum(long)/len(long)>=.15:
            control['omni']['precision_distance_practice']=True
            control['precision_distance_control']=dict(source=str(folder),mid=[sum(medium),len(medium)],
                                                       long=[sum(long),len(long)])
            return control
    return None


def hold_peak_control(home,number,parent,state):
    if number<14 or state['extra'] or state.get('hold_peak_probe_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,5):
        plane=course['flight_plane_control']
        if not plane.get('plane_peak') or plane.get('include_hold_peak'):
            continue
        limit=math.radians(course['flight_plane_limits']['lean_peak_deg'])
        rows=read(folder/'development/episodes.json')
        violations=sum(row['flight_plane_totals'][0][4]>limit for row in rows)
        if violations>=16:
            plane['include_hold_peak']=True
            course['precision_hold_peak_control']=dict(source=str(folder),hold_violations=violations,cases=len(rows))
            return course
    return None


def foot_bonus_control(home,number,parent,state):
    if number<15 or state['extra'] or state.get('foot_bonus_probe_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,3):
        if (course.get('moving_foot_goal_bonus')
                or not course['vertical_pretraining']['jump_cycle_reward'].get('strict_goal_bonus')):
            continue
        radius=course['omni']['strict_foot_m']
        events=[e for row in read(folder/'development/episodes.json') for e in row['events']
                if e['first_attempt_for_target'] and e['strict'] and 1<=e['target_index']<=10
                and e['commanded_edge_m']>=.30 and e['foot_error_m']<=radius]
        excluded=sum(e['root_link_error_m']>=radius for e in events)
        if excluded>=16 and excluded>=.1*len(events):
            course['moving_foot_goal_bonus']=True
            course['precision_foot_bonus_control']=dict(source=str(folder),strict_landings=len(events),
                                                        excluded_by_root=excluded)
            return course
    return None


def startup_control(home,number,parent,state):
    if number<16 or state['extra'] or state.get('startup_probe_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,3):
        if course['omni'].get('startup_rehearsal') or not course['flight_plane_control'].get('include_hold_peak'):
            continue
        rows=read(folder/'development/episodes.json')
        limit=math.radians(course['flight_plane_limits']['lean_peak_deg'])
        violations=[r for r in rows if r['flight_plane_totals'][0][4]>limit]
        first_flights=[]
        for row in violations:
            peak=row.get('flight_plane_peak_at',{}).get('hold')
            if (peak and peak['target_index']==0 and row['events'] and row['first_touch_step'] is not None
                    and row['first_touch_step']<peak['control_step']<=row['events'][0]['touchdown_step']):
                first_flights.append(row)
        if len(first_flights)>=16 and len(first_flights)*2>=len(violations):
            course['omni']['startup_rehearsal']=True
            course['precision_startup_control']=dict(source=str(folder),first_flight_violations=len(first_flights),
                                                     hold_violations=len(violations))
            return course
    return None


def release_window_control(home,number,parent,state):
    if number<17 or state['extra'] or state.get('window_probe_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,3):
        window=course.get('launch_velocity_window')
        if not window or window.get('first_long_along_discount_m_s'):
            continue
        groups=[]
        rows=read(folder/'development/episodes.json')
        for parity in (0,1):
            values=[e['window_along_error_halves_m_s'] for row in rows if row['case_id']%2==parity
                    for e in row['events'] if 1<=e['target_index']<=10 and e['strict'] and e['first_attempt_for_target']
                    and e['commanded_edge_m']>=.40 and e['foot_error_m']<=.04
                    and None not in e.get('window_along_error_halves_m_s',[None])]
            if len(values)<48:
                break
            halves=list(zip(*values))
            if any(sum(x<=-.04 for x in half)<.7*len(half) for half in halves):
                break
            groups.append(dict(events=len(values),median_m_s=list(map(median,halves))))
        if len(groups)==2:
            discount=min(.15,-max(x for group in groups for x in group['median_m_s']))
            window['first_long_along_discount_m_s']=round(discount,3)
            course['precision_window_control']=dict(source=str(folder),omni=groups[0],straight=groups[1])
            return course
    return None


def straight_descent_control(home,number,parent,state):
    if number<19 or state['extra'] or state.get('straight_descent_parent')==str(parent):
        return None
    paired=home/'paired_heading_audit/paired.json'
    if not paired.is_file() or read(paired)['checkpoint']!=str(parent):
        return None
    from .precision_forecast import weak_straight_prefix
    for folder,course in rejected_controls(home,number,parent,3):
        guidance=course['moving_landing_guidance']
        if (guidance.get('straight_descent_precision') or guidance.get('objective')!='projected_touchdown_error_v1'
                or guidance['projected_error_width_m']!=.1 or guidance['landing_width_m']!=.05):
            continue
        proof=weak_straight_prefix(read(folder/'development/episodes.json'))
        if all(p['hit']>=12 and p['hit']>=4*max(p['miss'],p['false_alarm']) and p['clear']>=6
               and p['clock_error_s'] is not None and p['clock_error_s']<=.05 for p in proof.values()):
            guidance['straight_descent_precision']=True
            course['precision_straight_descent_control']=dict(source=str(folder),weak_first_failures=proof)
            return course
    return None


def rehearsal_balance_control(home,number,parent,state):
    if number<22 or state['extra'] or state.get('rehearsal_balance_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,4):
        retention=course['stationary_retention']
        if retention['training_stationary_fraction']!=.5:
            continue
        cases=read(folder/'development/cases.json')['cases']
        holds=sum(a==b for case in cases for a,b in
                  zip(case['omni_route_local_m'][:10],case['omni_route_local_m'][1:11]))
        rows=read(folder/'development/summary.json')['precision10']['groups']
        if holds<=.2*(10*len(cases)) and max(r['clean'] for r in rows.values())<len(cases)//4:
            retention['training_stationary_fraction']=.25
            retention['allocation']='Training environment ids divisible by four rehearse stationary hops. Other ids train moving routes. Membership persists across resets.'
            course['precision_rehearsal_balance_control']=dict(source=str(folder),development_hold_edges=holds,
                                                               development_edges=10*len(cases))
            return course
    return None


def impulse_control(home,number,parent,state):
    if number<23 or state['extra'] or state.get('impulse_probe_parent')==str(parent):
        return None
    for folder,course in rejected_controls(home,number,parent,4):
        handoff=course['omni_rebound_handoff']
        if (handoff.get('long_continuation_impulse_m_s') or not handoff.get('continuation_foot_offset_fraction')
                or handoff.get('continuation_lead_scale')!=0):
            continue
        proof=contact_reference_evidence(read(folder/'development/episodes.json'),
                                         course['rebound_transition']['touchdown_tilt_limit_deg'])
        if proof and proof['supported']:
            handoff['long_continuation_impulse_m_s']=round(proof['vertical_change_m_s'],3)
            course['precision_impulse_control']=dict(source=str(folder),evidence=proof)
            return course
    return None


def reversal_apex_control(home,number,parent,state):
    if number<24 or state['extra'] or state.get('reversal_apex_parent')==str(parent):
        return None
    from .precision_phase_diagnosis import reversal_apex_evidence
    options=[]
    for folder,course in rejected_controls(home,number,parent,5):
        cfg=course.get('precision_pre_reversal_release')
        if not cfg or 'first_apex_allowance_m_s' in cfg:
            continue
        proof=reversal_apex_evidence(read(folder/'development/episodes.json'),7001,cfg['allowance_m_s'])
        if proof and proof['supported']:
            groups=read(folder/'decision.json')['groups']
            balance=min(group['clean'] for group in groups.values())
            options.append((balance,proof['apex_short_flagged']-proof['apex_hit_flagged'],folder,course,proof))
    if options:
        _,_,folder,course,proof=max(options,key=lambda row:row[:2])
        course['precision_pre_reversal_release']['first_apex_allowance_m_s']=round(proof['apex_allowance_m_s'],4)
        course['precision_reversal_apex_control']=dict(source=str(folder),evidence=proof)
        return course
    return None


def alternate_control(home,number,champion,best,state):
    if number<12 or state['extra'] or state.get('alternate_tried_for')==str(champion):
        return None
    weaker=min(best['groups'],key=lambda name:best['groups'][name]['clean'])
    options=[]
    control_folder=home/f'round_{number-1:05d}'
    for index in range(number-8,number):
        folder=home/f'round_{index:05d}'
        if not (folder/'decision.json').exists():
            return None
        decision=read(folder/'decision.json')
        if decision['accepted_as_parent'] or decision['parent']!=str(champion):
            return None
        if decision.get('mode')!='train' or decision['deaths']:
            continue
        candidate=decision['groups'][weaker]
        reference=best['groups'][weaker]
        if candidate['clean']<=reference['clean'] or candidate['prefix']<=reference['prefix']:
            continue
        checkpoint=Path(decision['checkpoint'])
        for evaluation in sorted(folder.glob('development*')):
            manifest=evaluation/'manifest.json'
            if not manifest.is_file():
                continue
            meta=read(manifest)
            if meta.get('status')!='complete' or meta.get('checkpoint')!=str(checkpoint):
                continue
            report=measured(evaluation,checkpoint,'development')
            if all(report['flight_plane'][kind]['lean_peak_deg']<=max(10.,best['flight_plane'][kind]['lean_peak_deg']+1.)
                   for kind in ('hold','travel')):
                options.append((candidate['clean'],candidate['prefix'],str(checkpoint)))
            break
    if not options:
        return None
    course=read(control_folder/'course.json')
    if course['training_iterations']!=50 or read(control_folder/'decision.json').get('mode')!='train':
        return None
    course['omni']['initial_checkpoint']=max(options)[-1]
    course['precision_alternate_control']=dict(champion=str(champion),control=str(control_folder),focus=weaker)
    return course


def recovery_control(home,number,parent):
    from .precision_recovery import recovery_tail_deaths
    for index in range(number-1,max(0,number-4),-1):
        folder=home/f'round_{index:05d}'
        if not (folder/'decision.json').exists():
            continue
        decision=read(folder/'decision.json')
        course=read(folder/'course.json')
        if decision['parent']!=str(parent):
            continue
        if course.get('precision_recovery_control'):
            return None
        if (not decision['deaths'] or decision.get('mode')!='train' or decision['accepted_as_parent']
                or course['omni_rebound_handoff'].get('retry_launch_tail_weight')
                or decision['checkpoint']!=str(folder/'train/model_49.pt') or course['training_iterations']!=50):
            continue
        rows=read(folder/'development/episodes.json')
        deaths=recovery_tail_deaths(rows,course['omni_rebound_handoff']['near_retry_speed_max_m_s'])
        if deaths:
            course['precision_recovery_control']=dict(source=str(folder),confirmed_retry_deaths=deaths)
            course['omni_rebound_handoff']['retry_launch_tail_weight']=2.
            return course
    return None


def release_tail_control(home,number,parent,state):
    if number<20 or state['extra'] or state.get('release_tail_parent')==str(parent):
        return None
    from .precision_recovery import long_release_tail
    for index in range(number-1,max(0,number-6),-1):
        folder=home/f'round_{index:05d}'
        if not (folder/'decision.json').is_file():
            continue
        decision=read(folder/'decision.json')
        course=read(folder/'course.json')
        window=course.get('launch_velocity_window',{})
        if (decision.get('mode')!='train' or not decision['deaths'] or decision['parent']!=str(parent)
                or decision['accepted_as_parent'] or decision['checkpoint']!=str(folder/'train/model_49.pt')
                or course['training_iterations']!=50 or not course.get('precision_window_control')
                or not window.get('first_long_along_discount_m_s') or window.get('release_deficit_fade_m_s')):
            continue
        reference=Path(course['precision_window_control']['source'])
        prior=read(reference/'decision.json')
        if prior['deaths'] or prior['parent']!=str(parent):
            continue
        proof=long_release_tail(read(reference/'development/episodes.json'),read(folder/'development/episodes.json'))
        if proof:
            window['release_deficit_fade_m_s']=proof['fade_m_s']
            course['precision_release_tail_control']=dict(source=str(folder),comparison=str(reference),**proof)
            return course
    return None


def forecast_control(previous,parent,state):
    from .precision_forecast import compare
    if state['extra'] or state.get('forecast_probe_parent')==str(parent) or not (previous/'decision.json').exists():
        return None
    decision=read(previous/'decision.json')
    control=read(previous/'course.json')
    if (decision.get('mode')!='train' or decision['accepted_as_parent'] or decision['parent']!=str(parent)
            or control['training_iterations']!=50 or control['moving_landing_guidance'].get('measured_foot_blend_s')):
        return None
    proof=compare(read(previous/'development/episodes.json'),7001,control['omni_rebound_handoff']['turn_min_deg'])
    if proof['pairs']<64 or proof['better']/proof['pairs']<.75 or proof['dense_m']-proof['foot_m']<.015:
        return None
    control['moving_landing_guidance'].update(measured_foot_blend_s=.4,long_turn_blend_only=True)
    control['precision_forecast_control']=dict(source=str(previous),paired=proof)
    return control


def main():
    HOME.mkdir(exist_ok=True)
    with (HOME/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        scalar_command=['bash',str(ROOT/'run_benchmark37_ring.sh'),'log-precision-results']
        subprocess.run(scalar_command,check=False)
        status=HOME/'status.json'
        state=read(status) if status.exists() else dict(round=1,best=str(START),best_eval=str(BASE/'development'),
            feedback=str(BASE/'development'),extra=None,phase='new')
        while state['phase']!='simulation_candidate_only':
            if shutil.disk_usage(HOME).free<8*1024**3:
                persist(state,'waiting_for_disk')
                time.sleep(300)
                continue
            for request in sorted((HOME/'ab_pairs').glob('*/course.json')):
                if not any((request.parent/name).exists() for name in ('result.json','error.json')):
                    if not read(request)['omni'].get('ab_pair_only'):
                        save(request.parent/'error.json',dict(error='Expected an AB-only training course'))
                        continue
                    from .train_ab_pairs import service_trial
                    service_trial(request.parent)
            number=state['round']
            folder=HOME/f'round_{number:05d}'
            folder.mkdir(exist_ok=True)
            parent=Path(state['best'])
            best=measured(Path(state['best_eval']),parent,'development')
            if number==18:
                from .precision_heading_audit import paired_report
                audit=HOME/'paired_heading_audit'
                audit.mkdir(exist_ok=True)
                if not (audit/'paired.json').exists() and not (audit/'error.json').exists():
                    persist(state,'paired_heading_diagnostic')
                    try:
                        reverse=stage(audit,'opposite','eval','--headless','--checkpoint',parent,
                                      '--precision-suite','opposite_straight')
                        save(audit/'paired.json',paired_report(Path(state['best_eval']),reverse,parent))
                    except (RuntimeError,ValueError,OSError,subprocess.CalledProcessError) as exc:
                        save(audit/'error.json',dict(error=str(exc)))
            feedback=read(Path(state['feedback'])/'summary.json')['precision10']
            config=folder/'course.json'
            probe_plan=folder/'probe.json'
            early=Path(read(probe_plan)['checkpoint']) if probe_plan.exists() else None
            if not config.exists() and early is None:
                early=plateau_probe(HOME,number,parent,state)
                if early:
                    save(probe_plan,dict(checkpoint=str(early),reason='five rejected finals from same parent'))
            if early:
                checkpoint=early
                if not config.exists():
                    save(config,read(early.parent/'ring_config.json'))
            elif not config.exists():
                course=proposal(parent,feedback,number,state['extra'])
                previous=HOME/f'round_{number-1:05d}'
                repair=None if state['extra'] else recovery_control(HOME,number,parent)
                if repair is None:
                    repair=hold_peak_control(HOME,number,parent,state)
                if repair is None:
                    repair=foot_bonus_control(HOME,number,parent,state)
                if repair is None:
                    repair=release_window_control(HOME,number,parent,state)
                if repair is None:
                    repair=release_tail_control(HOME,number,parent,state)
                if repair is None:
                    repair=straight_descent_control(HOME,number,parent,state)
                if repair is None:
                    repair=reversal_apex_control(HOME,number,parent,state)
                if repair is None:
                    repair=impulse_control(HOME,number,parent,state)
                if repair is None:
                    repair=alternate_control(HOME,number,parent,best,state)
                if repair is None and number>=11:
                    repair=forecast_control(previous,parent,state)
                if repair:
                    course=repair
                course.pop('precision_compare_budget',None)
                extended=extend_plateau(HOME,number,parent,state)
                course['training_iterations']=100 if extended else 50
                if extended:
                    course['precision_compare_budget']=[49,99]
                save(config,course)
            windows=read(config).get('precision_compare_budget') if not early else None
            if not early:
                if not read(config)['omni'].get('ab_pair_only'):
                    raise ValueError(f'Expected an AB-only training course in {config}')
                persist(state,'training')
                train=stage(folder,'train','train','--headless','--ring-config',config)
                checkpoint=train/('model_99.pt' if windows else 'model_49.pt')
            persist(state,'development_early_snapshot' if early else 'development')
            development,result=evaluate_window(folder,'development',checkpoint)
            if windows:
                alternate=train/'model_49.pt'
                if not alternate.exists():
                    alternate=train/'model_50.pt'
                alternate_updates=int(alternate.stem.split('_')[-1])+1
                earlier,short=evaluate_window(folder,f'development_{alternate_updates}',alternate)
                save(folder/'budget_comparison.json',[
                    dict(checkpoint=str(path),updates=int(path.stem.split('_')[-1])+1,groups=report['groups'],deaths=report['deaths'],
                         attitude=report['attitude'],passed=report['passed'],accepted=accepted(report,best),
                         travel_peak_deg=report['flight_plane']['travel']['lean_peak_deg'])
                    for path,report in ((alternate,short),(checkpoint,result))])
                if (short['passed'],accepted(short,best),rank(short))>(result['passed'],accepted(result,best),rank(result)):
                    checkpoint,development,result=alternate,earlier,short
            next_feedback=development
            keep=accepted(result,best)
            extra=state['extra']
            final=False
            if result['passed']:
                persist(state,'holdout')
                seed=20000+number
                holdout=stage(folder,'holdout','eval','--headless','--checkpoint',checkpoint,'--precision-suite','holdout',
                              '--fresh-holdout-seed',seed)
                meta=read(holdout/'manifest.json')
                if meta['optimizer_updates']!=0 or meta['checkpoint']!=str(checkpoint) or meta['seed']!=seed:
                    raise ValueError('Holdout provenance mismatch')
                final=score(read(holdout/'episodes.json'),read(holdout/'summary.json'),seed)['passed']
                if not final:
                    next_feedback=holdout
                if final:
                    persist(state,'stationary')
                    stationary=stage(folder,'stationary','eval-stationary','--headless','--checkpoint',checkpoint)
                    final=read(stationary/'summary.json')['promotion_gate']['passed']
                    extra=None if final else 'stationary'
                if final:
                    persist(state,'keyboard')
                    keys=folder/'keyboard.json'
                    if not keys.exists():
                        preserve(folder/'keyboard.log')
                        run(folder/'keyboard.log','eval-guarded-keys','--checkpoint',checkpoint,'--cases-file',
                            ROOT/'outputs/benchmark73_long_pairs_v48/moving/cases.json','--terminal-hold','four-axis-rebound',
                            '--hold-report',keys)
                    data=read(keys)
                    final=keyboard_qualified(data,checkpoint)
                    extra=None if final else 'keyboard'
            source_course=read(config)
            save(folder/'decision.json',dict(mode='early_snapshot' if early else 'train',
                 parent=source_course['omni']['initial_checkpoint'],comparison_checkpoint=str(parent),
                 checkpoint=str(checkpoint),accepted_as_parent=keep,
                 simulation_candidate=final,groups=result['groups'],first_failures=result['first_failures'],
                 deaths=result['deaths'],attitude=result['attitude']))
            if keep:
                state.update(best=str(checkpoint),best_eval=str(development))
            if early:
                state['early_probe_parent']=str(parent)
            if windows:
                state['extended_parent']=str(parent)
            if not early and source_course.get('precision_alternate_control',{}).get('champion')==str(parent):
                state['alternate_tried_for']=str(parent)
            if not early and read(config).get('precision_forecast_control'):
                state['forecast_probe_parent']=str(parent)
            if not early and source_course.get('precision_distance_control'):
                state['distance_probe_parent']=str(parent)
            if not early and source_course.get('precision_hold_peak_control'):
                state['hold_peak_probe_parent']=str(parent)
            if not early and source_course.get('precision_foot_bonus_control'):
                state['foot_bonus_probe_parent']=str(parent)
            if not early and source_course.get('precision_startup_control'):
                state['startup_probe_parent']=str(parent)
            if not early and source_course.get('precision_window_control'):
                state['window_probe_parent']=str(parent)
            if not early and source_course.get('precision_straight_descent_control'):
                state['straight_descent_parent']=str(parent)
            if not early and source_course.get('precision_release_tail_control'):
                state['release_tail_parent']=str(parent)
            if not early and source_course.get('precision_rehearsal_balance_control'):
                state['rehearsal_balance_parent']=str(parent)
            if not early and source_course.get('precision_impulse_control'):
                state['impulse_probe_parent']=str(parent)
            if not early and source_course.get('precision_reversal_apex_control'):
                state['reversal_apex_parent']=str(parent)
            state.update(feedback=str(next_feedback),extra=extra,last_round=number,round=number+1)
            persist(state,'simulation_candidate_only' if final else 'next_round')
            subprocess.run(scalar_command,check=False)
            print(f"第 {number} 轮：全向 {result['groups']['omni']['clean']}/64，直线 {result['groups']['straight']['clean']}/64，保留 {keep}",flush=True)
            if not final and worker_revision()!=WORKER_REVISION:
                os.execv(sys.executable,[sys.executable,'-u','-m','benchmark37.precision_continuous'])


if __name__=='__main__':
    main()
