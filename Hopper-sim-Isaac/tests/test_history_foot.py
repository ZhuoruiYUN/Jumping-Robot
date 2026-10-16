"""CPU checks for foot sign, temporal ordering and exact recurrent transfer."""
import math
import unittest
import torch
from Quadhopper_Planner_Random.history_foot import (
    FRAME_DIM, OBS_DIM, ObservationHistory, foot_offset_world, expand_from_spatial52,
)


class HistoryFootTests(unittest.TestCase):
    def test_foot_pitch_roll_yaw_and_compression(self):
        rest = (0., 0., -.26)
        s, c = math.sin(math.pi/4), math.cos(math.pi/4)
        quat = torch.tensor([[1.,0,0,0], [c,0,s,0], [c,s,0,0], [c,0,0,s]])
        offset = foot_offset_world(quat, torch.tensor([.02,0,0,0]), rest)
        torch.testing.assert_close(offset, torch.tensor([[0.,0,-.24],[-.26,0,0],
                                                        [0,.26,0],[0,0,-.26]]), atol=1e-6, rtol=0)

    def test_history_tick_cache_and_partial_reset(self):
        h = ObservationHistory(2, 'cpu')
        def frame(value):
            return torch.full((2, FRAME_DIM), float(value))
        for tick in range(7):
            h.update(frame(tick), tick)
        self.assertEqual(h.frames[0,:,0].tolist(), [6,5,4,3,2,1])
        before = h.frames.clone()
        h.update(frame(99), 6)
        torch.testing.assert_close(h.frames, before)
        h.reset(torch.tensor([0]))
        h.update(frame(10), 6)
        self.assertEqual(h.frames[0,:,0].tolist(), [10]*6)
        torch.testing.assert_close(h.frames[1], before[1])
        saved = h.update(frame(11), 7)
        h.update(frame(12), 8)
        self.assertEqual(saved[0,0].item(), 11)

    def test_actor_and_critic_transfer_over_multiple_timesteps(self):
        torch.manual_seed(3)
        old = torch.nn.LSTM(52, 16)
        state = {f'memory_{branch}.rnn.{k}': v for branch in ('a','c')
                 for k,v in old.state_dict().items()}
        expanded = expand_from_spatial52(state)
        x = torch.randn(8,3,OBS_DIM)
        expected, _ = old(x[:,:,:52])
        for branch in ('a','c'):
            new = torch.nn.LSTM(OBS_DIM,16)
            prefix = f'memory_{branch}.rnn.'
            new.load_state_dict({k[len(prefix):]:v for k,v in expanded.items() if k.startswith(prefix)})
            actual, _ = new(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
