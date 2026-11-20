"""Manual teacher49 -> student73 initialization for the stationary foundation.

--check-only runs teacher inference and shape/gradient checks, with zero updates.
The normal command fits the student's recurrent actor/critic on teacher data,
checks the student in the simulator, then optionally launches unchanged PPO.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / 'outputs/benchmark37_ring_targeted/train_seed42_20261105_135410Z/model_49.pt'
COURSE = ROOT / 'benchmark37/configs/obs73_stationary_course.json'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')


def make_policy(dim, config, device, n=1):
    import torch
    from rsl_rl.modules import ActorCriticRecurrent
    kwargs = copy.deepcopy(config['policy'])
    if kwargs.pop('class_name') != 'ActorCriticRecurrent':
        raise ValueError('Expected the existing recurrent actor/critic')
    return ActorCriticRecurrent({'policy': torch.zeros(n, dim, device=device)},
                               config['obs_groups'], 4, **kwargs).to(device)


def initialize_student(student, teacher):
    """Reuse hidden dynamics/heads/std, learn the new observation encoder.

    Input kernels are zero, not index-padded. All73D normalizers start fresh.
    This is an initialization for supervised learning, not a runnable policy.
    """
    import torch
    old = teacher.state_dict()
    state = student.state_dict()
    for key, value in state.items():
        if 'obs_normalizer.' in key:
            continue
        if key in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
            if old[key].shape[1] != 49 or value.shape[1] != 73:
                raise ValueError('Expected teacher49/student73')
            value.zero_()
        elif key in old and old[key].shape == value.shape:
            value.copy_(old[key])
        else:
            raise ValueError(f'Unmapped shared parameter: {key}')
    student.load_state_dict(state, strict=True)
    student.std.requires_grad_(False)
    torch.testing.assert_close(student.std, teacher.std)


def sequence(rnn, obs, reset_before, hidden=None):
    """Time-major recurrent forward; reset each environment before its sample.

    Fast chunks without an internal episode boundary use a single LSTM call.
    Hidden state is carried between truncated chunks, never between episodes.
    """
    import torch
    layers = rnn.num_layers
    n = obs.shape[1]
    if hidden is None:
        hidden = tuple(obs.new_zeros(layers, n, rnn.hidden_size) for _ in range(2))
    if not bool(reset_before[1:].any()):
        keep = (~reset_before[0]).to(obs.dtype)[None, :, None]
        return rnn(obs, tuple(value * keep for value in hidden))
    out = []
    for row, reset in zip(obs, reset_before):
        keep = (~reset).to(obs.dtype)[None, :, None]
        value, hidden = rnn(row[None], tuple(state * keep for state in hidden))
        out.append(value)
    return torch.cat(out), hidden


def predictions(student, obs, reset_before, hidden_a=None, hidden_c=None):
    actor_input = student.actor_obs_normalizer(obs)
    critic_input = student.critic_obs_normalizer(obs)
    a, hidden_a = sequence(student.memory_a.rnn, actor_input, reset_before, hidden_a)
    c, hidden_c = sequence(student.memory_c.rnn, critic_input, reset_before, hidden_c)
    return student.actor(a), student.critic(c), a, c, hidden_a, hidden_c


def loss_terms(prediction, target):
    import torch.nn.functional as F
    actions, values = prediction[:2]
    # Critic regression is scaled by its target magnitude; it estimates the
    # teacher's value only for initialization, not the new task's true return.
    value_scale = target['value'].square().mean().sqrt().clamp(min=1.).detach()
    action_loss = F.mse_loss(actions, target['action'].clamp(-1., 1.))
    value_loss = F.mse_loss(values / value_scale, target['value'] / value_scale)
    # The teacher has spring/action-history inputs the student cannot directly
    # observe. Match its bounded motor command, not its private latent state.
    # Do NOT clamp student predictions here: a wrong saturated output still
    # needs a corrective gradient back toward the bounded target.
    return action_loss + value_loss, action_loss


def motor_reference(teacher_mean, student_mean=None, teacher_beta=1.):
    if not 0. <= teacher_beta <= 1.:
        raise ValueError('Teacher mixture must lie in[0,1]')
    target = teacher_mean.clamp(-1., 1.)
    if student_mean is None:
        if teacher_beta != 1.:
            raise ValueError('Student action is required for mixed rollouts')
        return target, target
    return target, teacher_beta * target + (1. - teacher_beta) * student_mean.clamp(-1., 1.)


def legacy_obs(raw):
    # Deliberately call the original49D assembler, not a slice of student73.
    from .ring_env import RingEnv
    return RingEnv._get_observations(raw)


def collect(raw, wrapped, teacher, steps, student=None, teacher_beta=1.):
    import torch
    teacher.eval()
    teacher.reset()
    if student is not None:
        student.eval()
        student.reset()
    before_valid = int(raw._foot_event.touchdowns)
    before_strict = int(raw._foot_event.qualified)
    obs = wrapped.reset()[0]
    reset_before = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
    rows = {key: [] for key in ('obs', 'action', 'value', 'reset_before')}
    deaths = 0
    saturated = 0
    components = 0
    with torch.no_grad():
        for step in range(steps):
            old = legacy_obs(raw)
            teacher_mean = teacher.act_inference(old)
            student_mean = student.act_inference(obs) if student is not None else None
            target, action = motor_reference(teacher_mean, student_mean, teacher_beta)
            saturated += int((teacher_mean.abs() > 1.).sum())
            components += teacher_mean.numel()
            value = teacher.evaluate(old)
            item = dict(obs=obs['policy'], action=target, value=value, reset_before=reset_before)
            for key, value in item.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(f'Nonfinite teacher data: {key}')
                rows[key].append(value.detach().cpu().clone())
            obs, _, done, _ = wrapped.step(action)
            deaths += int((raw.reset_terminated & done.bool()).sum())
            teacher.reset(done)
            if student is not None:
                student.reset(done)
            reset_before = done.bool()
            if step % 250 == 0:
                print(f'DISTILL_COLLECT step={step}/{steps} teacher_beta={teacher_beta:.2f} strict_landings={int(raw._foot_event.qualified)-before_strict} deaths={deaths}', flush=True)
    data = {key: torch.stack(value) for key, value in rows.items()}
    valid = int(raw._foot_event.touchdowns) - before_valid
    strict = int(raw._foot_event.qualified) - before_strict
    result = dict(steps=steps, environments=raw.num_envs, valid_landings=valid, strict_landings=strict,
                  strict_rate=strict / max(1, valid), physical_deaths=deaths,
                  teacher_beta=teacher_beta, teacher_saturation_fraction=saturated/max(1,components),
                  passed=valid >= raw.num_envs and strict / max(1, valid) >= .9 and deaths == 0)
    return data, result


def fit(student, data, epochs, device, log_path):
    import torch
    # Hold out complete environments, not shuffled frames from the same flight.
    n = data['obs'].shape[1]
    split = n * 3 // 4
    train_ids = torch.arange(split)
    valid_ids = torch.arange(split, n)
    student.train()
    # A resumed student keeps its trained normalization. Once fitted, freeze
    # it throughout DAgger instead of moving its coordinate system each round.
    if int(student.actor_obs_normalizer.count) == 0:
        samples = data['obs'][:, train_ids].flatten(0, 1).to(device)
        student.update_normalization({'policy': samples})
        del samples
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=3e-4)
    history = []
    for epoch in range(epochs):
        order = train_ids[torch.randperm(len(train_ids))]
        losses = []
        for ids in order.split(16):
            hidden_a = hidden_c = None
            for start in range(0, data['obs'].shape[0], 64):
                batch = {key: value[start:start + 64, ids].to(device) for key, value in data.items()}
                prediction = predictions(student, batch['obs'], batch['reset_before'], hidden_a, hidden_c)
                loss, action_loss = loss_terms(prediction, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite distillation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite distillation gradients')
                optimizer.step()
                hidden_a = tuple(state.detach() for state in prediction[4])
                hidden_c = tuple(state.detach() for state in prediction[5])
                losses.append(float(action_loss.detach()))
        student.eval()
        square_sum = 0.
        count = 0
        hidden_a = hidden_c = None
        with torch.no_grad():
            for start in range(0, data['obs'].shape[0], 64):
                batch = {key: value[start:start + 64, valid_ids].to(device) for key, value in data.items()}
                prediction = predictions(student, batch['obs'], batch['reset_before'], hidden_a, hidden_c)
                square_sum += float((prediction[0] - batch['action']).square().sum())
                count += batch['action'].numel()
                hidden_a, hidden_c = prediction[4:6]
        row = dict(epoch=epoch + 1, train_action_mse=sum(losses) / len(losses),
                   heldout_action_rmse=(square_sum / count) ** .5)
        history.append(row)
        with log_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print('DISTILL_EPOCH=' + json.dumps(row), flush=True)
        student.train()
    student.eval()
    student.reset()
    return history


def student_gate(raw, wrapped, student):
    """One bounded readiness check; not the fixed128 promotion benchmark."""
    import torch
    before_valid = int(raw._foot_event.touchdowns)
    before_strict = int(raw._foot_event.qualified)
    obs = wrapped.reset()[0]
    student.eval()
    student.reset()
    deaths = 0
    with torch.no_grad():
        for _ in range(1500):
            actions = student.act_inference(obs)
            if not torch.isfinite(actions).all():
                raise RuntimeError('Nonfinite distilled student action')
            obs, _, done, _ = wrapped.step(actions)
            deaths += int((raw.reset_terminated & done.bool()).sum())
            student.reset(done)
    valid = int(raw._foot_event.touchdowns) - before_valid
    strict = int(raw._foot_event.qualified) - before_strict
    return dict(scope='single bounded startup gate; not128-case promotion', valid_landings=valid,
                strict_landings=strict, strict_rate=strict / max(1, valid), physical_deaths=deaths,
                passed=valid >= raw.num_envs and strict / max(1, valid) >= .9 and deaths == 0)


def check_gradients(student, data, device):
    import torch
    batch = {key: value[:32, :4].to(device) for key, value in data.items()}
    student.train()
    prediction = predictions(student, batch['obs'], batch['reset_before'])
    loss, _ = loss_terms(prediction, batch)
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Distillation gradient check failed')
    student.zero_grad(set_to_none=True)
    return dict(sequence_shape=list(batch['obs'].shape), loss=float(loss.detach()), finite_gradients=True,
                optimizer_updates=0)


def resume_student(student, path, teacher_path, course):
    from .ring_obs73 import validate_checkpoint_course
    validate_checkpoint_course(path, course)
    previous = json.loads((path.parent / 'manifest.json').read_text())
    if (previous.get('mode') != 'distill' or Path(previous['teacher']).resolve() != teacher_path
            or not previous.get('teacher_gate', {}).get('passed')):
        raise ValueError('Resume a distillation candidate from this verified teacher')
    import torch
    state = torch.load(path, map_location=next(student.parameters()).device, weights_only=False)['model_state_dict']
    student.load_state_dict(state, strict=True)
    student.std.requires_grad_(False)


def fit_dagger(raw, wrapped, teacher, student, folder, manifest):
    import torch
    replay = None
    manifest['rounds'] = []
    for index, beta in enumerate((1., .75, .5, .25, .1, 0.)):
        data, result = collect(raw, wrapped, teacher, 1500, student=student, teacher_beta=beta)
        if index == 0:
            manifest['teacher_gate'] = result
            if not result['passed']:
                raise RuntimeError('Teacher failed the stationary collection gate; no supervised/PPO updates started')
        replay = data if replay is None else {key: torch.cat((replay[key], data[key]), 0) for key in data}
        # Save the labeled trajectories so a failure does not discard the
        # already collected correction data. Each round starts with reset=1.
        torch.save(replay, folder / 'replay.pt')
        manifest.update(status='distilling', current_round=index + 1)
        write(folder / 'manifest.json', manifest)
        history = fit(student, replay, 4, raw.device, folder / f'round_{index+1}_losses.jsonl')
        row = dict(round=index + 1, teacher_beta=beta, rollout=result,
                   heldout_executed_target_rmse=history[-1]['heldout_action_rmse'])
        manifest['rounds'].append(row)
        manifest['heldout_action_rmse'] = row['heldout_executed_target_rmse']
        torch.save(dict(model_state_dict=student.state_dict(), iter=0, infos={'distillation': True}), folder / 'student_73.pt')
        write(folder / 'manifest.json', manifest)
        print('DISTILL_DAGGER_ROUND=' + json.dumps(row), flush=True)


def bootstrap(args, root):
    """Keep the supervisor free of Kit/CUDA, so fast Kit shutdown cannot
    swallow the handoff and both stages never hold GPU memory concurrently.
    """
    command = [sys.executable, '-m', 'benchmark37.obs73_distillation',
               '--teacher', str(args.teacher.resolve()), '--ring-config', str(args.ring_config.resolve()),
               '--seed', '42', '--profile', args.profile, '--device', args.device, '--output', str(root)]
    if args.student_from:
        command.extend(('--student-from', str(args.student_from.resolve())))
    if args.headless:
        command.append('--headless')
    subprocess.run(command, check=True)
    folder = root / 'distillation'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('mode') != 'distill'
            or not manifest.get('teacher_gate', {}).get('passed')
            or not manifest.get('student_gate', {}).get('passed')):
        raise SystemExit('Distillation readiness gate did not pass; PPO was not started. See distillation/manifest.json.')
    command = [sys.executable, '-m', 'benchmark37.ring_runner', 'train',
               '--initialize-distilled-from', str(folder / 'student_73.pt'),
               '--ring-config', str(args.ring_config.resolve()), '--seed', '42',
               '--profile', args.profile, '--iterations', str(args.iterations),
               '--device', args.device, '--output', str(root / 'ppo')]
    if args.headless:
        command.append('--headless')
    print('DISTILL_HANDOFF=' + json.dumps(command), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=TEACHER)
    parser.add_argument('--student-from', type=Path, help='Continue a saved distillation candidate; this does not approve it for PPO')
    parser.add_argument('--ring-config', type=Path, default=COURSE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile', choices=('vram_768',), default='vram_768')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--then-ppo', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.seed != 42 or args.iterations < 1 or args.check_only and args.then_ppo:
        parser.error('Use seed42; positive PPO iterations; check-only never launches training')
    teacher_path = args.teacher.resolve(strict=True)
    source_manifest = json.loads((teacher_path.parent / 'manifest.json').read_text())
    if source_manifest.get('mode') != 'train' or source_manifest.get('status') != 'complete':
        raise ValueError('Teacher must be a completed formal49D ring policy')
    config = json.loads((teacher_path.parent / 'runner_config.json').read_text())
    course = json.loads(args.ring_config.read_text())
    from .ring_obs73 import VERSION, contract
    if course.get('task') != 'obs73_stationary_foundation_v1' or course['observation_version'] != VERSION:
        raise ValueError('Distillation requires the73D stationary foundation course')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    root = (args.output or ROOT / 'outputs/benchmark37_obs73_stationary_bootstrap' / f'run_seed42_{stamp}').resolve()
    if args.then_ppo:
        bootstrap(args, root)
        return
    folder = root / 'distillation'
    folder.mkdir(parents=True, exist_ok=False)
    mode = 'distill_check' if args.check_only else 'distill'
    manifest = dict(status='starting', mode=mode, seed=42, teacher=str(teacher_path), observation_contract=contract(),
                    method='bounded_motor_targets_dagger_v2',
                    collection_envs=16 if args.check_only else 64, collection_steps=1500,
                    supervised_epochs=0 if args.check_only else 24, teacher_frozen=True,
                    initialization='teacher shared recurrent/MLP/std copied;73D input kernels zero; new73D normalizers',
                    supervised_config=dict(learning_rate=3e-4, gradient_clip=1., sequence_length=64,
                        env_batch_size=16, heldout_environment_fraction=.25,
                        teacher_beta_schedule=[1.,.75,.5,.25,.1,0.], epochs_per_round=4,
                        normalization='fit once on train environments for fresh students; preserve for resumed students; freeze during DAgger',
                        action_target='clipped teacher four-motor command in[-1,1]; student loss remains unclamped for corrective gradients',
                        critic_target='teacher critic values for initialization, not true new-task returns'))
    write(folder / 'manifest.json', manifest)
    write(folder / 'ring_config.json', course)
    write(folder / 'observation_contract.json', contract())
    write(folder / 'runner_config.json', config)
    app = None
    raw = None
    try:
        app = AppLauncher(args).app
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from Quadhopper_PPO_Reference_SysID.quadhopper_env import QuadhopperEnvCfg
        from .calibration_20261028_v2 import configure
        from .obs73_stationary import env_class
        from .ring_obs73 import wrap_env
        torch.manual_seed(42)
        cfg = QuadhopperEnvCfg()
        configure(cfg)
        cfg.seed = 42
        cfg.scene.num_envs = manifest['collection_envs']
        cfg.sim.device = args.device
        cfg.observation_space = 73
        cfg.episode_length_s = 15.
        cfg.power_model_path = str(ROOT / 'Quadhopper_PPO_Reference_SysID/model/quadhopper_memory_power.pt')
        cfg.csv_log_path = str(folder / 'unused.csv')
        raw = wrap_env(env_class())(cfg, course)
        wrapped = RslRlVecEnvWrapper(raw)
        teacher = make_policy(49, config, args.device, raw.num_envs)
        state = torch.load(teacher_path, map_location=args.device, weights_only=False)['model_state_dict']
        if any(state[f'memory_{side}.rnn.weight_ih_l0'].shape[1] != 49 for side in ('a', 'c')):
            raise ValueError('Teacher must have exactly49D inputs')
        teacher.load_state_dict(state, strict=True)
        teacher.requires_grad_(False)
        student = make_policy(73, config, args.device, raw.num_envs)
        initialize_student(student, teacher)
        if args.student_from:
            resume_student(student,args.student_from.resolve(strict=True),teacher_path,course)
            torch.testing.assert_close(student.std, teacher.std)
            manifest['student_source']=str(args.student_from.resolve())
            manifest['initialization']='resume saved73D distillation candidate and its normalizers/std; new supervised optimizer; PPO still requires gate'
        manifest.update(status='collecting', gpu=torch.cuda.get_device_name())
        write(folder / 'manifest.json', manifest)
        if args.check_only:
            data, teacher_result = collect(raw, wrapped, teacher, manifest['collection_steps'])
            manifest['teacher_gate'] = teacher_result
            print('DISTILL_TEACHER_GATE=' + json.dumps(teacher_result), flush=True)
            if not teacher_result['passed']:
                raise RuntimeError('Teacher cannot reliably perform this stationary course; no updates started')
            manifest['gradient_check'] = check_gradients(student, data, args.device)
            print('DISTILL_INTERFACE_CHECK=' + json.dumps(manifest['gradient_check']), flush=True)
        else:
            fit_dagger(raw, wrapped, teacher, student, folder, manifest)
            manifest['student_gate'] = student_gate(raw, wrapped, student)
            print('DISTILL_STUDENT_GATE=' + json.dumps(manifest['student_gate']), flush=True)
            torch.save(dict(model_state_dict=student.state_dict(), iter=0, infos={'distillation': True}), folder / 'student_73.pt')
            if not manifest['student_gate']['passed']:
                raise RuntimeError('Student startup gate failed; saved candidate is not approved for PPO initialization')
            student.std.requires_grad_(True)
        manifest['status'] = 'complete'
        print('DISTILL_COMPLETE=' + str(folder), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write(folder / 'manifest.json', manifest)
        if raw is not None:
            raw.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
