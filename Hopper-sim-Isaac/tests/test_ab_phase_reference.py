import unittest

import torch

from benchmark37.ab_phase_reference import motor_advantage


class TestMotorAdvantage(unittest.TestCase):
    def test_centered_direction_and_saturated_commands(self):
        # One positive/negative exploration sample per motor. At equal command
        # means the reference must contribute no advantage, even during noise.
        base = torch.tensor([0., -.2, 1.5, -.4]).expand(8, -1)
        sigma = torch.full_like(base, .2)
        noise = torch.cat([torch.eye(4), -torch.eye(4)])
        actions = base + noise * sigma
        ref = base + torch.tensor([.4, -.1, 2., 0.])
        guide = motor_advantage(actions, base, sigma, ref)
        self.assertEqual(guide.shape, (8,))
        self.assertAlmostEqual(guide.mean().item(), 0.)
        self.assertGreater(guide[0].item(), 0.)
        self.assertLess(guide[1].item(), 0.)
        self.assertEqual(guide[2].item(), 0.)
        self.assertEqual(guide[3].item(), 0.)
        self.assertLessEqual(guide.square().sum().item(),2.00001)
        torch.testing.assert_close(motor_advantage(actions,base,sigma,base),torch.zeros(8))

    def test_each_lane_uses_its_own_command(self):
        base = torch.zeros(2,4)
        noise = torch.ones_like(base)
        positive = torch.ones_like(base)
        target = torch.stack([positive[0],-positive[1]])
        a = motor_advantage(noise,base,positive,target)
        self.assertGreater(a[0].item(),0)
        self.assertEqual(a[0].item(),-a[1].item())


class TestPhaseReferenceRollout(unittest.TestCase):
    def test_phase_switch_and_actor_only_update(self):
        import tempfile
        from types import SimpleNamespace
        from pathlib import Path
        from benchmark37.ab_phase_reference import PhaseReferencePPO

        class FakePPO:
            def __init__(self):
                self.policy = SimpleNamespace(
                    actor_obs_normalizer=torch.nn.Identity(),
                    memory_a=SimpleNamespace(rnn=torch.nn.LSTM(2, 3)),
                    actor=torch.nn.Linear(3,4), get_actor_obs=lambda obs: obs)
                self.normalize_advantage_per_mini_batch = False
                self.storage = SimpleNamespace(step=0,advantages=torch.zeros(2,3,1),returns=torch.zeros(2,3,1))
                self.transition = SimpleNamespace(action_mean=torch.zeros(3,4),action_sigma=torch.ones(3,4))
                self.rewards = []

            def act(self, obs):
                return torch.ones(3,4)

            def process_env_step(self, obs, rewards, dones, extras):
                self.rewards.append(rewards.clone())
                self.storage.step += 1

            def compute_returns(self, obs):
                self.storage.returns.copy_(torch.stack(self.rewards)[:,:,None])
                self.storage.advantages.fill_(.5)

        ppo = FakePPO()
        seen = torch.tensor([False, False, True])
        points = torch.zeros(3,2,2)
        points[:,1,0] = torch.tensor([.5,.2,.5])
        raw = SimpleNamespace(episode_length_buf=torch.zeros(3,dtype=torch.long),device='cpu',num_envs=3,
              omni_route=SimpleNamespace(points=points,ab_pairs=SimpleNamespace(seen_a=seen,
              all_lanes=True,a_index=torch.ones(3,dtype=torch.long))))
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for value in (-1.,1.):
                weights={'memory_a.rnn.'+k:v for k,v in ppo.policy.memory_a.rnn.state_dict().items()}
                weights.update({'actor.weight':torch.zeros_like(ppo.policy.actor.weight),
                                'actor.bias':torch.full((4,),value)})
                path=Path(tmp)/str(value)
                torch.save(dict(model_state_dict=weights),path)
                paths.append(path)
            alg = PhaseReferencePPO(ppo,raw,dict(long_edge_m=.4,startup_steps=300),*paths)
        obs=torch.ones(3,2)
        reward=torch.arange(3,dtype=torch.float32)
        alg.act(obs)
        alg.process_env_step(obs,reward,torch.tensor([0,1,0]),{})
        for teacher in alg.teachers:
            self.assertTrue(all(h[:,1].count_nonzero()==0 for h in teacher.hidden))
            self.assertFalse(teacher.training)
        seen[0] = True
        points[1,1,0] = .5
        raw.episode_length_buf[1] = 300
        alg.act(obs)
        alg.process_env_step(obs,reward,torch.zeros(3),{})
        alg.compute_returns(obs)
        torch.testing.assert_close(ppo.storage.returns[:,:,0],reward.expand(2,-1))
        torch.testing.assert_close(ppo.storage.advantages[:,:,0],torch.tensor([[2.5,-1.5,-1.5],[-1.5,-1.5,-1.5]]))
        self.assertEqual(alg.last_metrics['start_frames'],1)
        self.assertEqual(alg.last_metrics['precision_frames'],5)
