"""Focused observation contract checks; no optimizer updates or success scoring."""
import io
import math

import torch

from ..ring_obs73 import ObservationMemory, assemble, canonical_quaternion, shadow_contact


def offline_check():
    memory = ObservationMemory(2, 'cpu')
    p = torch.tensor([[0., 0., 2.], [0., 0., 2.]])
    q = torch.tensor([[-math.cos(math.pi / 6), 0., -math.sin(math.pi / 6), 0.], [1., 0., 0., 0.]])
    v = torch.tensor([[2., 0., 0.], [0., 1., -2.]])
    w = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    targets = torch.tensor([[[1., 0.], [2., 0.], [3., 0.]]]).repeat(2, 1, 1)
    center = torch.tensor([[1., 0., 3.]]).repeat(2, 1)
    normal = torch.tensor([[1., 0., 0.]]).repeat(2, 1)
    ground = torch.zeros(2)
    height = torch.tensor([3., 3.])
    passed = torch.tensor([False, True])

    def build():
        return assemble(p, q, v, w, targets, ground, height, memory, center, normal, .34, passed)

    obs, snapshot = build()
    assert obs.shape == (2, 73) and torch.isfinite(obs).all()
    torch.testing.assert_close(obs[0, :6], torch.tensor([.5, 0., 1., 0., 1.5, 0.]))
    torch.testing.assert_close(obs[0, 6:9], torch.tensor([1., 0., math.sqrt(3)]))
    torch.testing.assert_close(obs[:, 9:13], canonical_quaternion(q))
    torch.testing.assert_close(obs[:, 13:16], w)
    torch.testing.assert_close(obs[:, 16:19], v)
    assert not obs[:, 19:57].any() and not obs[:, 58:64].any()
    torch.testing.assert_close(obs[:, 57], torch.ones(2))
    torch.testing.assert_close(obs[:, 72], torch.tensor([.5, 1.]))
    torch.testing.assert_close(torch.linalg.vector_norm(obs[:, 68:71], dim=-1), torch.ones(2))
    # Translating tiles must not alter an input, including A-point height.
    shift = torch.tensor([10., -20., 4.])
    shifted, _ = assemble(p + shift, q, v, w, targets + shift[:2], ground + shift[2], height,
                          memory, center + shift, normal, .34, passed)
    torch.testing.assert_close(obs, shifted)
    # Reads never advance histories. The old adopted action is queued after
    # inference, before the new action becomes visible in the next observation.
    build(); build()
    assert not memory.states.any()
    memory.commit(snapshot)
    first = snapshot.clone()
    memory.recent_action[:] = torch.tensor([.1, .2, .3, .4])
    v += 1
    obs, snapshot = build()
    torch.testing.assert_close(obs[:, 32:45], first)
    assert not obs[:, 19:32].any() and not obs[:, 49:57].any()
    memory.commit(snapshot)
    memory.recent_action[:] = .5
    obs, _ = build()
    torch.testing.assert_close(obs[:, 19:32], first)
    torch.testing.assert_close(obs[:, 32:45], snapshot)
    torch.testing.assert_close(obs[:, 53:57], torch.tensor([[.1, .2, .3, .4]]).repeat(2, 1))
    assert not obs[:, 49:53].any()
    # Initial drop does not produce a hop apex/landing; each confirmed apex
    # enters once, and failed-precision landings still provide measured error.
    def events(phase, spring=0., vz=-1., peak=1.2, error=.15):
        memory.events(torch.full((2,), phase), torch.full((2,), 8), torch.full((2,), spring),
                      torch.ones(2), torch.full((2,), vz), torch.full((2,), peak),
                      torch.full((2,), error), torch.ones(2, dtype=torch.bool))
    events(0, .003)
    assert not memory.apices.any() and not memory.landings.any()
    events(3); events(3)
    torch.testing.assert_close(memory.apices, torch.tensor([[0., 0., 1.2]]).repeat(2, 1))
    events(3, .003)
    torch.testing.assert_close(memory.landings, torch.tensor([[0., 0., .15]]).repeat(2, 1))
    events(1); events(3, peak=1.3)
    torch.testing.assert_close(memory.apices, torch.tensor([[0., 1.2, 1.3]]).repeat(2, 1))
    other = memory.states[1].clone()
    memory.reset(torch.tensor([0]))
    assert not memory.states[0].any() and not memory.apices[0].any() and not memory.recent_action[0].any()
    torch.testing.assert_close(memory.states[1], other)
    upright = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    contact = shadow_contact(torch.tensor([[0., 0., .285], [0., 0., .31]]), upright, torch.zeros(2))
    torch.testing.assert_close(contact[:, 0], torch.tensor([1., 0.]))
    invalid = shadow_contact(torch.tensor([[0., 0., float('nan')], [0., 0., .1]]),
                             torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., math.sqrt(.5), 0.]]), torch.zeros(2))
    assert not invalid.any()
    return dict(layout=True, A_point_frames=True, full_rotation=True, history_timing=True,
                event_once=True, drop_excluded=True, partial_reset=True, shadow_contact=True)


def runtime_check(raw, wrapped, runner, folder):
    offline = offline_check()
    policy = runner.get_inference_policy(device=raw.device)
    obs = wrapped.get_observations()
    assert obs['policy'].shape == (16, 73)
    assert not obs['policy'][:, 19:57].any() and not obs['policy'][:, 58:64].any()
    assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
    resets = 0
    stationary = raw.course.get('task') == 'obs73_stationary_foundation_v1'
    with torch.inference_mode():
        for _ in range(256):
            before = obs['policy'].clone()
            actions = policy(obs)
            assert torch.isfinite(actions).all()
            obs, rewards, done, _ = wrapped.step(actions)
            runner.alg.policy.reset(done)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(rewards).all()
            if stationary:
                torch.testing.assert_close(raw._two_ground_pt_xy_w, raw._two_ground_pt1_xy_w)
                expected_height=torch.ones(16,device=raw.device)
                if raw.course.get('vertical_pretraining'):
                    levels=raw.course['vertical_pretraining']['levels']
                    heights=torch.tensor([level['target_apex_m'] for level in levels],device=raw.device)
                    expected_height=heights[raw._vertical_episode_level]
                    assert not raw._episode_sums['stationary_pitch'].any()
                    assert not raw._episode_sums['stationary_pitch_work'].any()
                torch.testing.assert_close(raw._desired_pos_w[:, 2] - raw._terrain.env_origins[:, 2], expected_height)
                assert not (raw.ring.passed | raw.ring.collision | raw.ring.event | raw.ring.failed | raw._ring_eligible()).any()
            good = ~done.bool()
            expected_snapshot = torch.cat((before[:, 6:9], before[:, 13:16], before[:, 9:13], before[:, 16:19]), -1)
            torch.testing.assert_close(obs['policy'][good, 32:45], expected_snapshot[good])
            torch.testing.assert_close(obs['policy'][good, 19:32], before[good, 32:45])
            torch.testing.assert_close(obs['policy'][good, 53:57], before[good, 45:49])
            torch.testing.assert_close(obs['policy'][good, 45:49], actions.clamp(-1, 1)[good])
            if done.any():
                assert not obs['policy'][done.bool(), 19:57].any()
                assert not obs['policy'][done.bool(), 58:64].any()
            resets += int(done.sum())
        keep = raw.obs73_memory.states[1].clone()
        raw._reset_idx(torch.tensor([0, 3], device=raw.device))
        torch.testing.assert_close(raw.obs73_memory.states[1], keep)
        assert not raw.obs73_memory.states[[0, 3]].any()
        # Actual model and its empirical normalizers round-trip in memory only;
        # no smoke checkpoint can accidentally become a training source.
        buffer = io.BytesIO()
        torch.save(runner.alg.policy.state_dict(), buffer)
        buffer.seek(0)
        runner.alg.policy.load_state_dict(torch.load(buffer, map_location=raw.device, weights_only=False), strict=True)
        assert torch.isfinite(policy(wrapped.get_observations())).all()
        if stationary:
            # A precision miss must not terminate or advance the stationary
            # course; its continuous-attempt curriculum is intentional.
            old_done = raw.reset_buf.clone()
            old_goal = raw._two_ground_pt_xy_w.clone()
            raw._landing_event(torch.ones(16, dtype=torch.bool, device=raw.device),
                               torch.zeros(16, dtype=torch.bool, device=raw.device),
                               torch.zeros(16, dtype=torch.bool, device=raw.device))
            torch.testing.assert_close(raw.reset_buf, old_done)
            torch.testing.assert_close(raw._two_ground_pt_xy_w, old_goal)
            # Check the real actor/critic encoder and normalization shapes used
            # for the next stage, without loading or training another model.
            from ..ring_transfer import prepare_stationary_to_ring
            before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
            state = {key: value.clone() for key, value in before.items()}
            changed = prepare_stationary_to_ring(state, raw.course, {'task': 'drop_stationary_ring_stationary2_v1'})
            assert 'memory_a.rnn.weight_ih_l0' in changed and 'memory_c.rnn.weight_ih_l0' in changed
            for key, value in state.items():
                if key in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
                    torch.testing.assert_close(value[:, :65], before[key][:, :65])
                    assert not value[:, 65:].any()
                elif key in changed:
                    torch.testing.assert_close(value[:, :65], before[key][:, :65])
                    torch.testing.assert_close(value[:, 65:], torch.full_like(value[:, 65:], 0. if key.endswith('._mean') else 1.))
                else:
                    torch.testing.assert_close(value, before[key])
    assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
    return dict(offline=offline, shape=[16, 73], steps=256, resets=resets,
                finite_obs_actions_rewards=True, real_history_timing=True,
                partial_reset=True, model_normalizer_roundtrip=True, optimizer_updates=0,
                stationary_targets_and_ring_disabled=stationary,
                stationary_retry_and_ring_transfer=stationary)


if __name__ == '__main__':
    import json
    print(json.dumps(offline_check()))
