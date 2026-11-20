"""Bounded precision PPO search. No exports, hardware actions or deployment changes."""
import json
from pathlib import Path
import subprocess
import sys

from .precision_routes import SEEDS, score


ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT/'outputs/benchmark73_precision10_v50'
SOURCE = ROOT/'outputs/benchmark73_second_release_v49/train/model_99.pt'
WRAPPER = ROOT/'run_benchmark73_phase_motion.sh'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def run(log, *args):
    print('运行：', *args, flush=True)
    with log.open('x') as stream:
        subprocess.run(['bash',str(WRAPPER),*map(str,args)], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def measured(folder, checkpoint, split):
    manifest = read(folder/'manifest.json')
    if (manifest['status']!='complete' or manifest['checkpoint']!=str(checkpoint) or
            manifest['optimizer_updates']!=0 or manifest['seed']!=SEEDS[split] or
            manifest['evaluation_scope']!='precision10_v1_'+split):
        raise ValueError(f'Incomplete or mismatched qualification in {folder}')
    return score(read(folder/'episodes.json'),read(folder/'summary.json'), SEEDS[split])


def evaluate(folder, checkpoint, split):
    run(folder.with_suffix('.log'),'eval','--headless','--precision-suite',split,'--checkpoint',checkpoint,'--output',folder)
    if split=='development' and read(folder/'cases.json')['cases'] != read(SERIES/'reference/cases.json')['cases']:
        raise ValueError('Development release, response, physical DR or route changed')
    return measured(folder,checkpoint,split)


def rank(report):
    groups = list(report['groups'].values())
    return (-report['deaths'],min(item['clean'] for item in groups),sum(item['clean'] for item in groups),
            min(item['prefix'] for item in groups),report['attitude'], -report['flight_plane']['travel']['lean_peak_deg'])


def recipe(report, tried):
    failures = report['first_failures']
    choices = [('plane',failures.get('cross',0) + 12*int(not report['attitude'])),
               ('release',failures.get('short',0)+failures.get('long',0)),
               ('routes',abs(report['groups']['straight']['prefix']-report['groups']['omni']['prefix'] )/8),
               ('approach',failures.get('height_or_tilt',0)+failures.get('missing',0))]
    return max((priority,name) for name,priority in choices if name not in tried)[1]


def configure(parent, change):
    course = read(parent.parent/'ring_config.json')
    course.update(version='obs73_precision10_v50',output_series=SERIES.name,training_iterations=50)
    course['omni'].update(initial_checkpoint=str(parent),ten_step_fraction=course['omni'].get('ten_step_fraction',.20))
    if change=='plane':
        for field in ('plane_peak','plane_lean','plane_leg_speed'):
            course['flight_plane_control'][field] *= 1.35
    elif change=='release':
        course['rebound_transition']['launch_weight'] *= 1.3
    elif change=='routes':
        course['omni']['ten_step_fraction'] = .35
    elif change=='approach':
        course['rebound_transition']['approach_weight'] *= 1.25
    return course


def main():
    parent = SOURCE
    best = measured(SERIES/'reference',parent,'development')
    tried = set()
    state = dict(status='running',reference=best, best_checkpoint=str(parent),rounds=[])
    status = SERIES/'status.json'
    if status.exists():
        raise FileExistsError('Refusing to overwrite an existing search')
    save(status,state)
    try:
        for number in range(1,5):
            folder = SERIES/f'round_{number:02d}'
            folder.mkdir()
            change = recipe(best,tried)
            tried.add(change)
            save(folder/'course.json', configure(parent,change))
            state.update(stage='train',round=number,repair=change)
            save(status,state)
            run(folder/'train.log','train','--headless','--ring-config',folder/'course.json','--output',folder/'train')
            checkpoint = folder/'train/model_49.pt'
            state['stage']='development'
            save(status,state)
            result = evaluate(folder/'development',checkpoint,'development')
            improved = rank(result)>rank(best) or result['passed']
            state['rounds'].append(dict(round=number,repair=change,parent=str(parent),result=result,accepted=improved))
            if improved:
                best, parent = result, checkpoint
                state['best_checkpoint']=str(parent)
            save(status,state)
            print(f"第 {number} 轮 | {change} | 全向 {result['groups']['omni']['clean']}/64 | 直线 {result['groups']['straight']['clean']}/64 | 失稳 {result['deaths']} | 保留 {improved}",flush=True)
            if not result['passed']:
                continue
            state['stage']='final_validation'
            save(status,state)
            blind = evaluate(folder/'holdout',checkpoint,'holdout')
            state['holdout']=blind
            state['status']='holdout_failed'
            if blind['passed']:
                run(folder/'stationary.log','eval-stationary','--headless','--checkpoint',checkpoint,'--output',folder/'stationary')
                run(folder/'keyboard.log','eval-guarded-keys','--checkpoint',checkpoint,'--cases-file',
                    ROOT/'outputs/benchmark73_long_pairs_v48/moving/cases.json','--terminal-hold','four-axis-rebound',
                    '--hold-report',folder/'keyboard.json')
                keys=read(folder/'keyboard.json')
                state['stationary_ok']=read(folder/'stationary/summary.json')['promotion_gate']['passed']
                state['keyboard_ok']=(keys['both_arrived']==keys['all_ten']==128 and keys['requested']==keys['strict_arrivals']==256 and keys['deaths']==keys['endpoint_over_30cm']==0)
                state['status']='simulation_candidate_only' if state['keyboard_ok'] and state['stationary_ok'] else 'interface_or_hold_failed'
            break
        else:
            state['status']='budget_exhausted_not_qualified'
    except Exception as error:
        state.update(status='error',error=str(error))
        raise
    finally:
        save(status,state)


if __name__=='__main__':
    main()
