"""Independent Benchmark37 R0 / R1 launcher; no legacy hash-validation path."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/benchmark37_stage4l_landing_gain/train_seed42_20261102_034630_836993Z/model_49.pt'
COURSE=ROOT/'benchmark37/configs/ring_r1_course.json'
SERIES=ROOT/'outputs/benchmark37_ring_r1'
TRAINING_PROFILES={'vram_768':768,'vram_1024':1024}


def write(path,data):path.write_text(json.dumps(data,indent=2,default=str)+'\n')


def initial_course():
    from .ring_geometry import envelope_report
    e=envelope_report()
    return dict(version='fixed_wide_first_rebound_v1',status='candidate_not_frozen',envelope=e,
                center_local_m=[.30,0.,.80],normal_world=[1.,0.,0.],inner_radius_m=e['D0_m']/2,
                axial_thickness_m=.020,tube_radial_width_m=.025,delta_m=e['delta_m'],
                landing_xy_m=[.60,0.],next_edge_xy_m=[.40,0.],designated_flight=1,
                completion_reward=12.,reward_source='one full-body event, comparable to max inherited first-hop touchdown12',
                geometry_source='virtual engineering ring; no physical ring measurement',
                observation_version='stage4l41_plus_ring8_v1',episode_s=15.)


def offline():
    import torch
    from .ring_geometry import RingEvents,envelope_report
    course=initial_course();course['center_local_m']=[0.,0.,.80]
    p=torch.tensor([[-.5,0.,.94]]);q=torch.tensor([[1.,0.,0.,0.]]);s=torch.zeros(1)
    def run(xs,y=0.,z=.94,enabled=True):
        e=RingEvents(1,'cpu',course);start=p.clone();start[0]=start.new_tensor([xs[0],y,z]);e.reset(torch.tensor([0]),start,q,s,torch.zeros_like(p))
        for x in xs:
            pos=start.clone();pos[:,0]=x;e.sample(pos,q,s,torch.tensor([enabled]))
        return e
    xs=torch.linspace(-.5,.5,101).tolist()
    assert bool(run(xs).passed[0]),'full assembly forward'
    assert not bool(run(list(reversed(xs))).passed[0]),'reverse crossing'
    assert not bool(run(xs,y=.30).passed[0]),'rotor/leg clearance'
    bypass=run(xs,y=1.0);assert not bool(bypass.passed[0]) and not bool(bypass.collision[0]),'external bypass'
    half=run(torch.linspace(-.5,.01,52).tolist());assert bool(half.root_crossed[0]) and not bool(half.passed[0]),'root-only crossing'
    assert not bool(run([-.5,.5],y=.30).passed[0]),'swept high speed hit'
    assert not bool(run(xs,enabled=False).passed[0]),'initial free drop disabled'
    e=run(xs);e.event.zero_();e.sample(torch.tensor([[.6,0.,.94]]),q,s,torch.tensor([True]));assert not bool(e.event[0]),'once only'
    assert not bool(run([-.5,-.1,.001,-.001,.001,-.1,-.5]).passed[0]),'jitter/partial return'
    e.reset(torch.tensor([0]),p,q,s,torch.zeros_like(p));assert not bool(e.passed[0]),'reset'
    # Body passing but low foot striking the lower rim.
    assert not bool(run(xs,z=.65).passed[0]),'foot clearance'
    write(ROOT/'benchmark37/ring_envelope.json',envelope_report())
    print('RING_OFFLINE=passed (full body, rotor/foot margin, reverse, bypass, root-only, swept hit, drop, once, jitter, reset)',flush=True)


def select_course(trajectory,folder):
    import torch
    from .ring_geometry import RingEvents
    # Nine fixed placements for the SAME fixed landing point and same16 source
    # trajectories. Exploratory placement selection, never a success-rate eval.
    candidates=[(x,z) for x in (.27,.30,.33) for z in (.75,.80,.85)]
    t=trajectory.cpu();n=t.shape[1];m=len(candidates)
    c=initial_course();event=RingEvents(n*m,'cpu',c)
    first=t[0].repeat(m,1);ids=torch.arange(n*m)
    event.reset(ids,first[:,:3],first[:,3:7],first[:,7],torch.zeros(n*m,3))
    event.center=torch.tensor([[x,0.,z] for x,z in candidates]).repeat_interleave(n,0)
    for row in t:
        a=row.repeat(m,1);event.sample(a[:,:3],a[:,3:7],a[:,7],a[:,8].bool())
    results=[]
    for i,(x,z) in enumerate(candidates):
        sl=slice(i*n,(i+1)*n);passed=event.passed[sl]
        clearance=event.min_clearance[sl][passed]
        results.append(dict(center_local_m=[x,0.,z],source_complete_samples=int(passed.sum()),
                            minimum_pass_clearance_m=float(clearance.min()) if len(clearance) else None))
    viable=[r for r in results if r['source_complete_samples']>0]
    write(folder/'placement_diagnostic.json',dict(exploratory=True,seed=42,source=str(SOURCE),cases=n,candidates=results))
    if not viable:raise RuntimeError('R0 no feasible full-body source crossing; do not train')
    chosen=max(viable,key=lambda r:(r['source_complete_samples'],r['minimum_pass_clearance_m']))
    c.update(status='frozen_for_r1',center_local_m=chosen['center_local_m'],
             r0_evidence=str(folder/'placement_diagnostic.json'),r0_trajectory=str(folder/'source_trajectory.pt'),
             r0_source_complete_samples=chosen['source_complete_samples'],r0_minimum_clearance_m=chosen['minimum_pass_clearance_m'])
    write(COURSE,c);print('R0_FROZEN='+json.dumps(c),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('offline','diagnose','smoke','train','eval','obs-check'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--initialize-from',type=Path,help='Initialize from a completed ring checkpoint with the same observation contract; fresh optimizer and iteration0')
    parser.add_argument('--initialize-obs73-from',type=Path,help='Explicit partial OBS73->OBS72 initialization; deleted inputs are not behavior preserving')
    parser.add_argument('--initialize-history-from',type=Path,help='Explicit old72->new72 history timing/slot migration; fresh optimizer and iteration0')
    parser.add_argument('--initialize-quaternion-from',type=Path,help='Explicit72D RPY crossing ->73D quaternion crossing initialization; fresh optimizer')
    parser.add_argument('--initialize-stationary-quaternion-from',type=Path,help='Explicit deployed-v23 legacy stationary ring73 -> new quaternion73; fresh optimizer')
    parser.add_argument('--initialize-distilled-from',type=Path,help='Initialize73D stationary PPO from an approved completed teacher distillation; fresh optimizer and iteration0')
    parser.add_argument('--from-scratch',action='store_true',help='Initialize a new OBS73 policy and normalizers; no49D/CTBR weight transfer')
    parser.add_argument('--profile',choices=tuple(TRAINING_PROFILES),help='Training parallelism; OBS73 omni defaults to 1024 environments')
    parser.add_argument('--iterations',type=int,help='Overrides the course training budget (default50 for ring fine-tuning)')
    parser.add_argument('--ring-config',type=Path,help='Explicit independent geometry configuration; defaults to the checkpoint course for eval')
    parser.add_argument('--cases-file',type=Path,help='Restore the saved128-case release and DR states for evaluation')
    parser.add_argument('--training-difficulty',action='store_true',help='Vertical eval only: freeze the checkpoint current curriculum level; retain separate strict1m scores')
    parser.add_argument('--stationary-retention-eval',action='store_true',help='Evaluate the new observation policy on the saved deployed-v23 stationary cases; no moving targets')
    parser.add_argument('--straight-x-step-m',type=float,help='Evaluation-only fixed+X route, with all128 saved release/DR cases')
    parser.add_argument('--straight-jumps',type=int,default=10,help='First moving targets required for straight evaluation (default10); retain full15s exposure')
    parser.add_argument('--precision-suite',choices=('development','holdout','opposite_straight','continuous','ab_continue','ab_reverse'))
    parser.add_argument('--fresh-holdout-seed',type=int)
    parser.add_argument('--step-hold-eval',action='store_true',help='Separate128-case cardinal0.4m step then hold evaluation')
    parser.add_argument('--cardinal-queue-eval',action='store_true',help='Separate128-case four-direction0.5m move3/hold/reverse60s evaluation')
    parser.add_argument('--output',type=Path)
    # Offline geometry has no Kit/GPU dependency.
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='offline':offline();return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    # Kit107.3 telemetry collects display info on a worker during startup.
    # Observed headless SIGSEGV: XOpenDisplay -> getenv, before model loading.
    # The installed omni.kit.telemetry docs support serial startup via this
    # setting. Preserve explicit user overrides; simulation settings are intact.
    if args.headless and '/exts/omni.kit.telemetry/skipDeferredStartup' not in args.kit_args:
        args.kit_args=(args.kit_args+' --/exts/omni.kit.telemetry/skipDeferredStartup=true').strip()
    migrate72=args.initialize_obs73_from is not None
    migrate_history=args.initialize_history_from is not None
    migrate_quaternion=args.initialize_quaternion_from is not None
    migrate_stationary_quaternion=args.initialize_stationary_quaternion_from is not None
    if migrate_stationary_quaternion:
        if (args.initialize_from or migrate72 or migrate_history or migrate_quaternion
                or args.initialize_distilled_from or args.from_scratch or args.mode!='train'):
            parser.error('--initialize-stationary-quaternion-from is exclusive and only supports train')
        args.initialize_from=args.initialize_stationary_quaternion_from
    if migrate_quaternion:
        if args.initialize_from or migrate72 or migrate_history or args.initialize_distilled_from or args.from_scratch or args.mode!='train':
            parser.error('--initialize-quaternion-from is exclusive and only supports train')
        args.initialize_from=args.initialize_quaternion_from
    if migrate_history:
        if args.initialize_from or args.initialize_obs73_from or args.initialize_distilled_from or args.from_scratch or args.mode!='train':
            parser.error('--initialize-history-from is exclusive and only supports train')
        args.initialize_from=args.initialize_history_from
    if migrate72:
        if args.initialize_from or args.initialize_distilled_from or args.from_scratch or args.mode != 'train':
            parser.error('--initialize-obs73-from is exclusive and only supports train')
        args.initialize_from=args.initialize_obs73_from
    evaluating=args.mode=='eval'
    retention_eval=args.stationary_retention_eval
    straight_eval=args.straight_x_step_m is not None
    if args.cardinal_queue_eval and (not evaluating or retention_eval or straight_eval or args.step_hold_eval or not args.cases_file):
        parser.error('Cardinal queue requires eval and --cases-file; incompatible with other evaluation scopes')
    if args.step_hold_eval and (not evaluating or retention_eval or straight_eval or not args.cases_file):
        parser.error('Step/hold requires eval and --cases-file; incompatible with straight/stationary scopes')
    if straight_eval and (not evaluating or retention_eval or not args.cases_file):
        parser.error('Straight evaluation requires eval and --cases-file; incompatible with stationary retention')
    if not straight_eval and args.straight_jumps!=10:
        parser.error('--straight-jumps requires --straight-x-step-m')
    if retention_eval and not evaluating:parser.error('--stationary-retention-eval is evaluation-only')
    from .precision_routes import SEEDS
    if args.precision_suite and (not evaluating or args.cases_file or retention_eval or straight_eval or args.step_hold_eval or args.cardinal_queue_eval):
        parser.error('Precision qualification uses its own routes, releases and DR')
    expected_seed=SEEDS[args.precision_suite] if args.precision_suite else (7001 if evaluating else 42)
    if args.fresh_holdout_seed is not None:
        if args.precision_suite!='holdout' or not 10000 <= args.fresh_holdout_seed < 2**31:
            parser.error('Fresh seeds apply only to the final holdout')
        expected_seed=args.fresh_holdout_seed
    args.seed=args.seed if args.seed is not None else expected_seed
    if args.seed!=expected_seed:raise ValueError('Seed does not match the selected protocol')
    diagnostic=args.mode=='diagnose';smoke=args.mode=='smoke';obs_check=args.mode=='obs-check'
    if args.cases_file and not evaluating:parser.error('--cases-file is evaluation-only')
    if args.initialize_from and args.mode not in ('train','smoke'):parser.error('--initialize-from is training-only')
    if args.initialize_distilled_from and (args.mode!='train' or args.initialize_from or args.from_scratch):
        parser.error('--initialize-distilled-from is exclusive with other initialization options and only supports train')
    if args.from_scratch and (args.initialize_from or args.mode not in ('train','smoke','obs-check')):
        parser.error('--from-scratch is exclusive with --initialize-from and only supports train/smoke/obs-check')
    if args.iterations is not None and args.iterations<1:parser.error('--iterations must be positive')
    initial_source=SOURCE
    if args.initialize_from:
        initial_source=args.initialize_from.resolve(strict=True)
        parent=json.loads((initial_source.parent/'manifest.json').read_text())
        if parent.get('mode')!='train' or parent.get('status')!='complete' or parent.get('num_envs') not in TRAINING_PROFILES.values():
            raise ValueError('Initialize from a completed formal ring training run')
    if args.initialize_distilled_from:
        initial_source=args.initialize_distilled_from.resolve(strict=True)
        parent=json.loads((initial_source.parent/'manifest.json').read_text())
        if (parent.get('mode')!='distill' or parent.get('status')!='complete'
                or parent.get('seed')!=42 or not parent.get('teacher_gate',{}).get('passed')
                or not parent.get('student_gate',{}).get('passed')):
            raise ValueError('Use a completed73D distillation that passed both teacher and student gates')
    if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
    if evaluating:
        if args.checkpoint is None:parser.error('eval requires --checkpoint')
        args.checkpoint=args.checkpoint.resolve(strict=True)
        trained=json.loads((args.checkpoint.parent/'manifest.json').read_text())
        if trained.get('mode')!='train' or trained.get('status')!='complete':raise ValueError('Use a completed formal ring training checkpoint')
        course=json.loads((args.ring_config or args.checkpoint.parent/'ring_config.json').read_text())
    elif diagnostic:course=initial_course()
    else:
        course=json.loads((args.ring_config or COURSE).read_text())
        if course.get('task') not in ('obs73_stationary_foundation_v1','obs73_fixed_wide_ring_stage_b_v1','obs73_omni_ground_targets_v1'):
            if course['status']!='frozen_for_r1' or course['r0_source_complete_samples']<1:raise ValueError('R0 required')
    min_loop=course.get('task')=='drop_stationary_ring_stationary2_v1'
    stage_b=course.get('task')=='obs73_fixed_wide_ring_stage_b_v1'
    stationary=course.get('task')=='obs73_stationary_foundation_v1'
    omni=course.get('task')=='obs73_omni_ground_targets_v1'
    retention=bool(course.get('stationary_retention'))
    evaluation_source_course=course
    if evaluating and args.cardinal_queue_eval and omni:
        from copy import deepcopy
        course=deepcopy(course);course['omni']['route_points']=max(96,course['omni']['route_points'])
    if evaluating and course.get('long_moving_horizon') and not args.cardinal_queue_eval:
        # Preserve the original15s acceptance route format/case files. The
        # dedicated60s scope and training alone use the longer96-point queue.
        from copy import deepcopy
        course=deepcopy(course);course['omni']['route_points']=32
    if retention_eval:
        if not (retention and (omni or stationary)):parser.error('Stationary retention eval requires an explicit baseline comparison course')
        if args.training_difficulty:parser.error('Stationary retention always uses the saved full-difficulty cases')
        expected_cases=(ROOT/course['stationary_retention']['baseline_cases']).resolve(strict=True)
        if args.cases_file and args.cases_file.resolve()!=expected_cases:
            parser.error('Stationary retention uses exactly the saved deployed-v23 cases')
        args.cases_file=expected_cases
    evaluation_scope='stationary_retention' if retention_eval else 'moving'
    if args.precision_suite:
        if not omni or course['omni']['strict_foot_m']!=.05 or not course.get('long_moving_horizon'):
            parser.error('Precision qualification requires the 5cm long-horizon omni course')
        course['omni']['precision_eval_seed']=args.seed
        if args.precision_suite=='opposite_straight':
            course['omni']['opposite_straight_audit']=True
        evaluation_scope='precision10_v1_'+args.precision_suite
        if args.precision_suite in ('ab_continue','ab_reverse'):
            course['omni']['ab_preview_eval']=args.precision_suite
            evaluation_scope=args.precision_suite
        if args.precision_suite=='continuous':
            course['omni'].update(continuous_route_eval=True,route_points=96)
            evaluation_scope='continuous60_v1'
    straight_spec=None
    if straight_eval:
        if not omni or not course.get('rebound_transition'):
            parser.error('Straight evaluation requires the saved OBS73 moving/rebound course')
        from .obs73_omni import straight_eval_spec
        try:straight_spec=straight_eval_spec(course,args.straight_x_step_m,args.straight_jumps)
        except ValueError as exc:parser.error(str(exc))
        evaluation_scope=straight_spec['evaluation_scope']
    step_hold_spec=None
    if args.step_hold_eval:
        if not omni or not course.get('rebound_transition'):
            parser.error('Step/hold evaluation requires a saved OBS73 moving/rebound course')
        from .step_hold import step_hold_eval_spec
        step_hold_spec=step_hold_eval_spec(course)
        evaluation_scope=step_hold_spec['evaluation_scope']
    queue_spec=None
    if args.cardinal_queue_eval:
        if not omni or not course.get('rebound_transition'):
            parser.error('Cardinal queue requires the OBS73 moving/rebound course')
        from .cardinal_queue import evaluation_spec
        queue_spec=evaluation_spec(course)
        evaluation_scope=queue_spec['evaluation_scope']
    vertical=(stationary or omni) and bool(course.get('vertical_pretraining'))
    if args.training_difficulty and not (vertical and evaluating and not omni):
        parser.error('--training-difficulty requires a vertical eval checkpoint')
    if vertical and args.initialize_distilled_from:parser.error('Independent vertical training does not use a teacher or distilled weights')
    if vertical and args.initialize_from:
        if not parent.get('course',{}).get('vertical_pretraining') or not parent.get('vertical_curriculum'):
            parser.error('Continue this independent vertical benchmark only from its own final checkpoint, or use --from-scratch')
        if initial_source.name!=f'model_{parent["iteration"]}.pt':
            parser.error('Use the final checkpoint matching the saved vertical curriculum state')
    from .ring_obs73 import observation_dim, contract as obs73_contract, validate_checkpoint_course
    obs_dim=41 if diagnostic else observation_dim(course)
    quaternion_crossing=course.get('observation_version')=='direct_motor_crossing_quaternion73_history40_20_v1'
    if quaternion_crossing:
        from .crossing_obs73_quat import contract as obs73_contract
    elif obs_dim==72:
        if not omni:parser.error('OBS72 currently trains the short omni bridge; crossing reward curriculum is a subsequent stage')
        from .crossing_obs72 import contract as crossing_contract
        obs73_contract=lambda:crossing_contract(course['observation_version'])
    if migrate72 and obs_dim!=72:parser.error('Migration requires an OBS72 course')
    if migrate_history and obs_dim!=72:parser.error('History migration requires an OBS72 course')
    if migrate_quaternion and not quaternion_crossing:parser.error('Quaternion migration requires the crossing-quaternion73 contract')
    if migrate_stationary_quaternion and not (stationary and quaternion_crossing):
        parser.error('Deployed stationary migration requires the stationary quaternion73 recovery task')
    if course.get('stationary_quaternion_recovery'):
        if args.mode not in ('train','eval'):parser.error('Recovery supports manual train or saved-case eval; use its dedicated zero-update check')
        if args.mode=='train' and not (migrate_stationary_quaternion or args.initialize_from):
            parser.error('Recovery requires explicit deployed-v23 migration or its own matching checkpoint')
        if evaluating:
            if not retention_eval:parser.error('Recovery evaluation requires --stationary-retention-eval')
            if course!=json.loads((args.checkpoint.parent/'ring_config.json').read_text()):
                parser.error('Recovery evaluation must use its checkpoint saved course')
            for old in (ROOT/'outputs'/course['output_series']).glob('*/manifest.json'):
                prior=json.loads(old.read_text())
                if (prior.get('mode')=='eval' and prior.get('checkpoint')==str(args.checkpoint)
                        and (prior.get('status') in ('running','complete') or (old.parent/'summary.json').exists())):
                    raise ValueError(f'Recovery checkpoint already evaluated: {old.parent}; read that result')
    if stationary and obs_dim!=73:parser.error('Stationary foundation requires the73D observation contract')
    if args.initialize_distilled_from and not stationary:parser.error('Distilled initialization is for the stationary foundation PPO stage')
    if obs_dim==73 and not (min_loop or stationary or stage_b or omni):parser.error('OBS73 requires an explicit73D task course')
    if omni:
        if args.mode not in ('train','eval') or args.from_scratch or args.initialize_distilled_from:
            parser.error('Omni bridge requires matching initialization (or explicit73->72 migration), or its saved model for eval; use dedicated zero-update check')
        if not evaluating:
            if not args.initialize_from:parser.error('Omni bridge requires --initialize-from')
            qualified=(ROOT/course['omni']['initial_checkpoint']).resolve()
            if initial_source!=qualified and parent.get('course',{}).get('task')!=course['task']:
                parser.error('Omni source must be its v23 stationary foundation or own completed omni model')
        else:
            if evaluation_source_course!=json.loads((args.checkpoint.parent/'ring_config.json').read_text()):
                parser.error('Omni eval must use its checkpoint saved course')
            for old in (ROOT/'outputs'/course.get('output_series','benchmark73_omni_v1')).glob('*/manifest.json'):
                prior=json.loads(old.read_text())
                if (prior.get('mode')=='eval' and prior.get('checkpoint')==str(args.checkpoint)
                        and prior.get('evaluation_scope','moving')==evaluation_scope
                        and (prior.get('status') in ('running','complete') or (old.parent/'summary.json').exists())):
                    raise ValueError(f'Omni checkpoint already evaluated: {old.parent}; read that result')
    stage_a_gate=None
    if stage_b:
        if args.mode not in ('train','eval') or args.from_scratch or args.initialize_distilled_from:
            parser.error('Stage B uses a qualified73D source for train, or its saved checkpoint for eval; no from-scratch/distillation/smoke')
        if not evaluating and not args.initialize_from:
            parser.error('Stage B requires --initialize-from')
        if not evaluating:
            from .obs73_ring_stage_b import stage_a_evidence
            qualified=ROOT/course['stage_b']['initial_checkpoint']
            stage_a_gate=stage_a_evidence(ROOT,course,qualified)
            source_course=json.loads((initial_source.parent/'ring_config.json').read_text())
            if initial_source!=qualified.resolve() and source_course.get('task')!=course['task']:
                parser.error('Stage B source must be its qualified foundation or an own Stage B checkpoint')
        else:
            saved=json.loads((args.checkpoint.parent/'ring_config.json').read_text())
            if saved!=course or saved.get('task')!='obs73_fixed_wide_ring_stage_b_v1':
                parser.error('Stage B eval must use a trained Stage B checkpoint and its saved course')
            for old in (ROOT/'outputs').glob('benchmark73_ring_stage_b_v*/*/manifest.json'):
                prior=json.loads(old.read_text())
                if (prior.get('mode')=='eval' and prior.get('checkpoint')==str(args.checkpoint)
                        and (prior.get('status') in ('running','complete') or (old.parent/'summary.json').exists())):
                    raise ValueError(f'Stage B checkpoint already has an evaluation: {old.parent}; read it instead')
    if (args.from_scratch or obs_check) and obs_dim!=73:parser.error('--from-scratch/obs-check requires an explicit OBS73 course')
    if obs_check and not args.from_scratch:parser.error('obs-check requires --from-scratch; it performs no training')
    if obs_dim==73 and not (evaluating or args.initialize_from or args.initialize_distilled_from or args.from_scratch):
        parser.error('OBS73 needs --from-scratch or a matching73D --initialize-from checkpoint')
    if min_loop and not (evaluating or args.initialize_from or args.from_scratch):parser.error('Minimal loop requires --initialize-from')
    if evaluating:validate_checkpoint_course(args.checkpoint,course)
    if args.initialize_from:
        if migrate_stationary_quaternion:
            from .stationary_quaternion_transfer import validate_source
            validate_source(initial_source,course)
        elif migrate72:
            from .crossing_transfer import validate_source
            validate_source(initial_source,course)
        elif migrate_history:
            from .crossing_transfer import validate_history_source
            validate_history_source(initial_source,course)
        elif migrate_quaternion:
            from .crossing_transfer import validate_quaternion_source
            validate_quaternion_source(initial_source,course)
        else:validate_checkpoint_course(initial_source,course)
    if args.initialize_distilled_from:validate_checkpoint_course(initial_source,course)
    if args.profile is None:
        args.profile='vram_1024' if omni and obs_dim==73 else 'vram_768'
    n=128 if evaluating else (16 if diagnostic or smoke or obs_check else TRAINING_PROFILES[args.profile])
    iterations=args.iterations if args.iterations is not None else course.get('training_iterations',50)
    updates=0 if evaluating or obs_check else (2 if smoke else iterations)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    series=ROOT/'outputs/benchmark37_ring_r1_v2' if course['envelope']['version']=='full_assembly_disk_prisms_swept_v2' else SERIES
    if min_loop:series=ROOT/'outputs/benchmark37_ring_min_loop'
    if min_loop and course.get('post_ring_stability'):series=ROOT/'outputs/benchmark37_ring_min_loop_pitch'
    if min_loop and course.get('height_randomization'):series=ROOT/'outputs/benchmark37_ring_min_loop_height'
    if min_loop and course.get('targeted_adjustments'):series=ROOT/'outputs/benchmark37_ring_targeted'
    if obs_dim==73:series=ROOT/'outputs/benchmark37_ring_obs73'
    if stationary:series=ROOT/'outputs/benchmark37_obs73_stationary'
    if vertical:series=ROOT/'outputs/benchmark73_vertical'
    if course.get('stationary_quaternion_recovery'):series=ROOT/'outputs'/course['output_series']
    if stage_b:
        from .obs73_ring_stage_b import output_series
        series=ROOT/'outputs'/output_series(course)
    if omni:series=ROOT/'outputs'/course.get('output_series','benchmark73_omni_v1')
    folder=(args.output or series/f'{args.mode}_seed{args.seed}_{stamp}').resolve()
    folder.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='starting',mode=args.mode,seed=args.seed,num_envs=n,requested_iterations=updates,
                  steps_per_update=256,source=str(initial_source),course=course,
                  physics_revision='sysid_20261028_spring_air_v2_local_wrench',
                  initialization='41->49 zero input expansion, copied std/normalizers, fresh optimizer, iteration0')
    if evaluating:manifest.update(checkpoint=str(args.checkpoint),initialization=f'strict{obs_dim}D checkpoint inference; no optimizer or normalizer updates',case_version='ring_r1_seed7001_initial128_v1')
    if evaluating and (omni or course.get('stationary_quaternion_recovery')):manifest['evaluation_scope']=evaluation_scope
    if straight_spec:manifest['straight_evaluation']=straight_spec
    if queue_spec:manifest['cardinal_queue_evaluation']=queue_spec
    if args.ring_config:manifest['explicit_ring_config']=str(args.ring_config.resolve())
    if args.cases_file:manifest['case_source']=str(args.cases_file.resolve())
    if stage_a_gate is not None:manifest['stage_a_gate']=stage_a_gate
    response_config = course.get('vertical_pretraining', {}).get('response_robustness')
    if response_config:
        manifest['additional_response_randomization'] = response_config
        if evaluating:
            from .obs73_response_robustness import CASE_VERSION as RESPONSE_CASE_VERSION
            manifest['case_version'] = RESPONSE_CASE_VERSION
    if args.initialize_from or args.initialize_distilled_from:
        manifest.update(initialization=f'strict{obs_dim}D policy/critic/LSTM/normalizers/std; fresh optimizer and iteration0',
                        initial_checkpoint=str(initial_source),profile=args.profile)
    if obs_dim in (72,73):
        manifest['observation_contract']=obs73_contract()
        write(folder/'observation_contract.json',obs73_contract())
    if args.from_scratch:
        manifest.update(source=None,runner_config_source=str(initial_source.parent/'runner_config.json'),
                        initialization='new73D actor/critic/LSTM, fresh normalizers/optimizer, configured initial std, iteration0; no source weights loaded')
    write(folder/'manifest.json',manifest);print('BENCHMARK37_RING_RUN='+str(folder),flush=True)
    app=None;raw=None
    try:
        app=AppLauncher(args).app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure,validate_asset
        from .collector import audit_asset
        from .artifacts import clean
        from .ring_env import RingEnv
        from .ring_transfer import transfer,initialize_49,initialize_same_contract
        torch.manual_seed(args.seed)
        cfg=QuadhopperEnvCfg();configure(cfg)
        cfg.seed=args.seed;cfg.scene.num_envs=n;cfg.sim.device=args.device
        cfg.observation_space=obs_dim;cfg.episode_length_s=15.
        cfg.power_model_path=str(ROOT/'Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt')
        cfg.csv_log_path=str(folder/'single_env.csv')
        # Verify every inherited setting directly against the source artifact,
        # allowing only the declared task dimensions and output/count changes.
        expected=json.loads((SOURCE.parent/'env_config.json').read_text())
        actual=clean(cfg.to_dict())
        for key in ('seed','observation_space','csv_log_path'):expected[key]=actual[key]
        expected['scene']['num_envs']=n
        if expected!=actual:
            differences=[k for k in actual if actual[k]!=expected.get(k)]
            raise RuntimeError('Source environment configuration differs: '+str(differences))
        # Physics audit above remains unchanged. Only the declared task timeout
        # extends; original evaluation scopes and stationary members stay15s.
        horizon=course.get('long_moving_horizon')
        if horizon:
            if not (omni and retention and horizon.get('moving_s')==60. and horizon.get('stationary_s')==15.):
                raise ValueError('Long moving horizon requires moving60s and retained stationary15s')
            cfg.episode_length_s=60. if queue_spec or not evaluating or args.precision_suite=='continuous' else 15.
            actual=clean(cfg.to_dict());manifest['task_episode_horizon_s']=cfg.episode_length_s
        elif queue_spec:
            cfg.episode_length_s=60.;actual=clean(cfg.to_dict());manifest['task_episode_horizon_s']=60.
        write(folder/'env_config.json',actual);write(folder/'ring_config.json',course)
        runner_config_path=(args.checkpoint.parent if evaluating else initial_source.parent)/'runner_config.json'
        if (vertical or stage_b) and not evaluating and course.get('runner_config_source'):
            runner_config_path=ROOT/course['runner_config_source']
        rc=json.loads(runner_config_path.read_text())
        if (vertical or stage_b) and course['vertical_pretraining'].get('jump_cycle_reward'):
            if course['vertical_pretraining']['jump_cycle_reward']['gamma'] != rc['algorithm']['gamma']:
                raise ValueError('Jump-cycle potential shaping must use the PPO discount')
        manifest['runner_config_source']=str(runner_config_path)
        rc.update(seed=args.seed,max_iterations=updates,experiment_name=series.name)
        # Save early for launch verification; logging cadence does not change PPO.
        rc['save_interval']=course.get('checkpoint_interval',1)
        write(folder/'runner_config.json',rc)
        env_type=RingEnv
        if stationary or omni:
            from .obs73_stationary import env_class
            env_type=env_class()
            if vertical:
                from .obs73_vertical import env_class as vertical_env_class
                level=trained['vertical_curriculum']['level'] if args.training_difficulty else None
                env_type=vertical_env_class(evaluation=evaluating,evaluation_level=level)
        if omni:
            if retention:
                from .stationary_retention import env_class as retention_env_class
                env_type=retention_env_class(evaluation=evaluating,stationary_only=retention_eval)
            else:
                from .obs73_omni import env_class as omni_env_class
                env_type=omni_env_class(evaluation=evaluating)
            if course.get('moving_phase_reward'):
                from .moving_phase_reward import wrap_env as motion_reward_wrapper
                env_type=motion_reward_wrapper(env_type)
        if min_loop:
            from .ring_min_loop import env_class
            env_type=env_class()
            if course.get('height_randomization'):
                from .ring_height_loop import env_class as height_env_class
                env_type=height_env_class(evaluation=evaluating)
            if course.get('targeted_adjustments'):
                from .ring_targeted_adjustments import env_class as targeted_env_class
                env_type=targeted_env_class(evaluation=evaluating)
        if stage_b:
            from .obs73_ring_stage_b import env_class as stage_b_env_class
            env_type=stage_b_env_class(evaluation=evaluating)
        if quaternion_crossing:
            from .crossing_obs73_quat import wrap_env
            env_type=wrap_env(env_type)
        elif obs_dim==73:
            from .ring_obs73 import wrap_env
            env_type=wrap_env(env_type)
        elif obs_dim==72:
            from .crossing_obs72 import wrap_env
            env_type=wrap_env(env_type)
        if retention_eval:
            from .stationary_retention import eval_env_class as retention_eval_class
            env_type=retention_eval_class(env_type,ROOT,course)
            manifest['case_version']='obs73_quaternion_stationary_retention_deployed_v23_seed7001_v1'
        elif omni and evaluating:
            from .obs73_omni_evaluation import eval_env_class
            from .obs73_omni import case_version
            env_type=eval_env_class(env_type)
            manifest['case_version']=case_version(course)
            if straight_spec:manifest['case_version']=straight_spec['version']
            if step_hold_spec:manifest['case_version']=step_hold_spec['version']
            if queue_spec:manifest['case_version']=queue_spec['version']
        elif stationary and evaluating:
            from .obs73_stationary import eval_env_class,CASE_VERSION
            env_type=eval_env_class(env_type)
            manifest['case_version']=CASE_VERSION
            if vertical:
                manifest['case_version']=('obs73_vertical_training_level_seed7001_initial128_v1' if args.training_difficulty
                                          else 'obs73_vertical_seed7001_initial128_v1')
                manifest['evaluation_difficulty']='checkpoint_training_level' if args.training_difficulty else 'full1m'
                if response_config:
                    from .obs73_response_robustness import CASE_VERSION as RESPONSE_CASE_VERSION
                    manifest['case_version'] = RESPONSE_CASE_VERSION
        elif stage_b and evaluating:
            from .obs73_ring_stage_b_evaluation import eval_env_class
            from .obs73_ring_stage_b import CASE_VERSION
            env_type=eval_env_class(env_type)
            manifest['case_version']=CASE_VERSION
        elif min_loop:
            if evaluating:
                from .ring_min_loop_evaluation import eval_env_class
                env_type=eval_env_class(env_type)
                manifest['case_version']='ring_min_loop_seed7001_initial128_v1'
                if course.get('height_randomization'):
                    from .ring_height_loop import CASE_VERSION
                    manifest['case_version']=CASE_VERSION
        elif evaluating:
            from .ring_evaluation import RingEvalEnv
            env_type=RingEvalEnv
        raw=env_type(cfg,course,diagnostic=diagnostic)
        if straight_spec:raw.straight_evaluation_spec=straight_spec
        if step_hold_spec:raw.step_hold_evaluation_spec=step_hold_spec
        if queue_spec:raw.cardinal_queue_evaluation_spec=queue_spec
        if vertical and args.initialize_from:
            if parent['course']['vertical_pretraining']['levels']==course['vertical_pretraining']['levels']:
                raw.vertical_curriculum.restore(parent['vertical_curriculum'])
                manifest['restored_vertical_curriculum']=raw.vertical_curriculum.state()
            else:
                manifest['reset_vertical_curriculum']=dict(reason='Explicit course target/release levels changed; retain policy weights, discard old level/window',
                                                          state=raw.vertical_curriculum.state())
        if vertical:
            print('VERTICAL_OBJECTIVE='+json.dumps(dict(
                target_apex_m=raw.vertical_curriculum.current['target_apex_m'],
                curriculum_levels=len(raw.vertical_config['levels']),
                height_tolerance_m=raw.vertical_config['training_height_tolerance_m'])),flush=True)
        if course.get('rigid_body_randomization'):
            manifest['nominal_asset_audit'] = validate_asset(audit_asset(raw,cfg.robot.spawn.usd_path))
        wrapped=RslRlVecEnvWrapper(raw)
        if evaluating and args.cases_file:raw.evaluation.restore_cases(args.cases_file)
        obs=wrapped.get_observations()
        assert obs['policy'].shape==(n,obs_dim) and wrapped.num_actions==4
        assert raw.physics_dt==.005 and raw.step_dt==.01
        if course.get('rigid_body_randomization'):
            from .flight_plane import audit_rigid_scales
            manifest['rigid_body_randomization'] = audit_rigid_scales(raw)
            manifest['asset_audit'] = manifest['nominal_asset_audit']
        else:
            manifest['asset_audit'] = validate_asset(audit_asset(raw,cfg.robot.spawn.usd_path))
        runner_type = OnPolicyRunner
        route_metrics = getattr(getattr(raw,'omni_route',None),'adjacent_metrics',None)
        if route_metrics is not None:
            class AdjacentHopRunner(OnPolicyRunner):
                def log(self, locs, *args, **kwargs):
                    super().log(locs,*args,**kwargs)
                    if not evaluating and course.get('precision_compare_budget')==[49,99] and locs['it']==49:
                        self.save(str(folder/'model_49.pt'))
                    route_metrics.log_rollout(self.writer,locs['it'])
                    if hasattr(raw,'release_discount_counts'):
                        launches,protected=raw.release_discount_counts.tolist()
                        if launches:
                            self.writer.add_scalar('Launch/severe_deficit_fraction',protected/launches,locs['it'])
                        raw.release_discount_counts.zero_()
                    if hasattr(raw.omni_route,'startup_starts'):
                        self.writer.add_scalar('Route/startup_rehearsal_starts',raw.omni_route.startup_starts.item(),locs['it'])
                    practice=getattr(raw.omni_route,'distance_practice',None)
                    if practice is not None:
                        graded=practice.level[practice.mask]
                        self.writer.add_scalar('Route/practice_cap_mean_m',practice.caps[graded].mean().item(),locs['it'])
                        self.writer.add_scalar('Route/practice_full_fraction',(graded==2).float().mean().item(),locs['it'])
                    guard = getattr(raw,'key_guard',None)
                    if guard is not None and guard.pair_gap:
                        guard.log_pairs(self.writer,locs['it'])
                    if hasattr(raw,'rollout_travel_peak'):
                        self.writer.add_scalar('Travel/forbidden_lean_peak_deg',raw.rollout_travel_peak.rad2deg().item(),locs['it'])
                        raw.rollout_travel_peak.zero_()
                    if hasattr(raw,'south_release_stats'):
                        launches,short,speed = raw.south_release_stats.tolist()
                        raw.south_release_stats.zero_()
                        self.writer.add_scalar('Travel/south_pair_releases',launches,locs['it'])
                        if launches:
                            self.writer.add_scalar('Travel/south_pair_below_floor_rate',short/launches,locs['it'])
                            self.writer.add_scalar('Travel/south_pair_release_speed_m_s',speed/launches,locs['it'])
            runner_type = AdjacentHopRunner
        runner=runner_type(wrapped,rc,log_dir=str(folder),device=args.device)
        if diagnostic or evaluating:
            checkpoint=args.checkpoint if evaluating else SOURCE
            runner.alg.policy.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=False)['model_state_dict'],strict=True)
        elif args.initialize_from or args.initialize_distilled_from:
            if migrate_stationary_quaternion:
                from .stationary_quaternion_transfer import initialize
                manifest['transfer']=initialize(runner,initial_source,course)
            elif migrate72:
                from .crossing_transfer import initialize
                manifest['transfer']=initialize(runner,initial_source,course)
            elif migrate_history:
                from .crossing_transfer import initialize_history
                manifest['transfer']=initialize_history(runner,initial_source,course)
            elif migrate_quaternion:
                from .crossing_transfer import initialize_quaternion
                manifest['transfer']=initialize_quaternion(runner,initial_source,course)
            else:
                manifest['transfer']=(initialize_same_contract(runner,initial_source,course) if obs_dim in (72,73)
                                      else initialize_49(runner,initial_source))
            if manifest['transfer'].get('folded_ground_inputs'):
                manifest['initialization']='fold formerly duplicate target inputs into Pt,cold-start Pnext columns only; keep ring8/other73D weights/std; fresh optimizer and iteration0'
            if manifest['transfer'].get('reset_ring_inputs'):
                manifest['initialization']='copy73D stationary foundation; zero ring8 input weights and reset ring8 normalization; retain std; fresh optimizer and iteration0'
                if stage_b and manifest['transfer']['reset_ring_inputs']:
                    manifest['initialization']='fold duplicate ground-target inputs; cold-start newly distinct Pnext/ring inputs; retain other73D weights and std; fresh optimizer and iteration0'
            if migrate72:
                manifest['initialization']='explicit73->72 column migration; removed event/ring contributions, new command weights zero, retained std; fresh optimizer/iteration0; behavior not equivalent'
            if migrate_history:
                manifest['initialization']='explicit72D history timing migration; state ages40/20ms and action ages40/20/10ms; reorder action weights/statistics, preserve std; fresh optimizer/iteration0, behavior not equivalent'
            if migrate_quaternion:
                manifest['initialization']='explicit72->73 quaternion crossing migration; preserve59 foundation inputs, cold-start14 command inputs; fresh optimizer/iteration0, behavior not equivalent'
            if migrate_stationary_quaternion:
                manifest['initialization']='explicit deployed-v23 stationary ring73->quaternion73 partial migration; retain59 mapped inputs/std, cold-start14 commands; fresh optimizer/iteration0; behavior not equivalent'
            assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
            print('RING_INITIALIZATION='+json.dumps(manifest['transfer']),flush=True)
        elif not args.from_scratch:
            manifest['transfer']=transfer(runner.alg.policy,SOURCE)
            assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
        manifest.update(status='running',gpu=torch.cuda.get_device_name(),simulation_started=True)
        write(folder/'manifest.json',manifest)
        if obs_check:
            from .checks.ring_obs73_check import runtime_check
            manifest['observation_check']=runtime_check(raw,wrapped,runner,folder)
            print('RING_OBS73_CHECK='+json.dumps(manifest['observation_check']),flush=True)
        elif evaluating:
            raw.evaluation.freeze_cases(folder)
            policy=runner.get_inference_policy(device=args.device)
            frozen_state={key:value.detach().clone() for key,value in runner.alg.policy.state_dict().items()}
            def forbid_update(*unused,**kwargs):raise RuntimeError('Evaluation forbids optimizer updates')
            runner.alg.optimizer.step=forbid_update
            with torch.inference_mode():
                for step in range(raw.max_episode_length+1):
                    obs,_,done,_=wrapped.step(policy(obs))
                    runner.alg.policy.reset(done)
                    if step%250==0:print(f'RING_EVAL_STEP={step} COMPLETE={len(raw.evaluation.rows)}/128',flush=True)
                    if all(raw.evaluation.finished):break
            assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
            assert all(torch.equal(value,runner.alg.policy.state_dict()[key]) for key,value in frozen_state.items()),'Evaluation changed policy/normalizer state'
            manifest['optimizer_updates']=0
            result,rows=raw.evaluation.result()
            if args.precision_suite in ('continuous','opposite_straight','ab_continue','ab_reverse'):
                if args.precision_suite=='continuous':
                    from .precision_routes import continuous_score
                    result['continuous']=continuous_score(rows)
                result.update(seed=args.seed,evaluation_scope=evaluation_scope,
                              promotion_gate=dict(scope=evaluation_scope,passed=False,diagnostic_only=True))
            elif args.precision_suite:
                from .precision_routes import score
                report=score(rows,result,args.seed)
                from .precision_phase_diagnosis import matched_components
                report['matched_failure_components']=matched_components(rows,report['first_failed_events'],args.seed)
                result.update(seed=args.seed,evaluation_scope=evaluation_scope,precision10=report,
                              promotion_gate=dict(scope=evaluation_scope,passed=report['passed']))
            write(folder/'summary.json',result);write(folder/'episodes.json',rows)
            manifest.update(evaluation_summary=str(folder/'summary.json'),promotion_gate=result['promotion_gate'])
            result_keys=('counts','rates','failure_counts_overlapping','promotion_gate')
            if 'current_target' in result:result_keys+=('current_target',)
            if 'settled' in result:result_keys+=('settled',)
            if 'landing_precision' in result:result_keys+=('landing_precision',)
            if 'rebound_diagnostics' in result:result_keys+=('rebound_diagnostics',)
            if 'body_rate_diagnostics' in result:result_keys+=('body_rate_diagnostics',)
            if 'navigation' in result:result_keys+=('navigation',)
            if straight_spec:result_keys+=('straight_evaluation',)
            if step_hold_spec:result_keys+=('step_hold_evaluation',)
            if queue_spec:result_keys+=('cardinal_queue_evaluation',)
            if 'moving_continuity' in result and not step_hold_spec and not queue_spec:
                # Keep the console compact; full conditional error/tilt
                # distributions remain in summary.json.
                result['moving_continuity_brief']={key:result['moving_continuity'][key]
                    for key in ('episodes','physical_deaths','prefix')}
                result_keys+=('moving_continuity_brief',)
            label='CARDINAL_QUEUE_EVAL_RESULT=' if queue_spec else ('STEP_HOLD_EVAL_RESULT=' if step_hold_spec else ('STRAIGHT_EVAL_RESULT=' if straight_spec else 'RING_EVAL_RESULT='))
            if args.precision_suite=='continuous':
                for name,data in result['continuous'].items():
                    print(f"60 秒连续跳 | {name} | 首次达标 {data['first_hits']}/{data['first_attempts']} | "
                          f"第 10 点后 {data['after_ten_hits']}/{data['after_ten_attempts']} | 倒地 {data['deaths']}",flush=True)
            elif args.precision_suite in ('ab_continue','ab_reverse'):
                print(f"AB 预览对照 {args.precision_suite} | 回合 {len(rows)} | {folder/'episodes.json'}",flush=True)
            elif args.precision_suite=='opposite_straight':
                print(f"反向直线配对诊断完成，{len(rows)} 条路线，数据文件 {folder/'episodes.json'}",flush=True)
            elif args.precision_suite:
                print(f"\n十跳验收 | {args.precision_suite} | 全向 {report['groups']['omni']['clean']}/64 | "
                      f"直线 {report['groups']['straight']['clean']}/64 | 失稳 {report['deaths']} | "
                      f"姿态通过 {report['attitude']}\n数据：{folder/'summary.json'}",flush=True)
            else:
                print(label+json.dumps({k:result[k] for k in result_keys}),flush=True)
        elif diagnostic:
            policy=runner.get_inference_policy(device=args.device)
            with torch.inference_mode():
                for step in range(600):
                    obs,_,done,_=wrapped.step(policy(obs))
                    runner.alg.policy.reset(done)
                    if step%100==0:print(f'R0_TRAJECTORY_STEP={step}/600',flush=True)
            trajectory=torch.stack(raw.trajectory).cpu()
            torch.save(trajectory,folder/'source_trajectory.pt')
            select_course(trajectory,folder)
        else:
            runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False)
            manifest.update(iteration=int(runner.current_learning_iteration),
                            full_crossings_total=float(raw.ring_success_total),strict_landing_total=float(raw.ring_joint_total))
            if min_loop or stage_b:manifest['loop_stage_completions']=raw.loop.completions.tolist()
            if stage_b and raw.forward_progress is not None:
                manifest['forward_progress_totals']=dict(reward=float(raw.forward_progress.total),
                    positive_steps=int(raw.forward_progress.positive_steps),
                    per_episode_budget=raw.forward_progress.config['budget'])
                if raw.forward_progress.early is not None:
                    manifest['forward_progress_totals'].update(
                        early_ascent_reward=float(raw.forward_progress.early_total),
                        early_ascent_positive_steps=int(raw.forward_progress.early_positive_steps))
            if stage_b and raw.ascent_power is not None:
                manifest['ascent_power_relief_totals']=dict(reward=float(raw.ascent_power.total),
                    active_steps=int(raw.ascent_power.active_steps))
            if course.get('targeted_adjustments'):
                manifest['targeted_cost_totals']={k:float(v) for k,v in raw.targeted_totals.items()}
                manifest['post_ring_flight_metrics']={k:float(v) for k,v in raw.flight_pitch_stats.log().items()}
            if stationary or stage_b or omni:
                manifest['stationary_strict_landings_total']=int(raw._foot_event.qualified)
                manifest['stationary_flight_metrics']={k.replace('PostRing/','Stationary/'):float(v) for k,v in raw.stationary_stats.log().items()}
            if omni:
                manifest['omni_totals']=dict(target_advances=int(raw.omni_route.advances),missed_landings=int(raw.omni_route.misses))
                if course.get('moving_navigation'):
                    manifest['omni_totals'].update(strict_target_advances=int(raw.omni_route.strict_advances),
                        navigation_advances_without_strict=int(raw.omni_route.advances_without_strict))
            if vertical or stage_b:
                manifest['vertical_curriculum']=raw.vertical_curriculum.state()
                manifest['vertical_native_landings']=float(raw.vertical_target_hits)
                manifest['vertical_valid_landings']=float(raw.vertical_valid_landings)
                if raw.jump_cycle is not None:
                    manifest['jump_cycle_apex_events']=int(raw.jump_cycle.apex_count)
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        write(folder/'manifest.json',manifest)
        if raw is not None:raw.close()
        if app is not None:app.close()


if __name__=='__main__':main()
