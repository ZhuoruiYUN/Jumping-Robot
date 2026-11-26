import math
import unittest
from collections import deque

import torch

from benchmark37.ascent_coast_preview import _queued_thrust, queued_coast
from benchmark37.v81_ascent_budget import retained_ascent_pulse_cost, validate_retained_pulse
from benchmark37.yaw_thrust_feedback import _thrust_from_duty


class QueuedAscentPreviewTest(unittest.TestCase):
    def setUp(self):
        self.z = torch.tensor([.83, .99], dtype=torch.float64)
        self.v = torch.tensor([1.5, .65], dtype=self.z.dtype)
        self.up = torch.tensor([.98, .97], dtype=self.z.dtype)
        self.mass = torch.tensor([.1556, .146], dtype=self.z.dtype)
        self.drag = torch.tensor([.05, .075], dtype=self.z.dtype)
        self.scale = torch.tensor([.95, 1.02], dtype=self.z.dtype)
        self.gain = torch.tensor([.91, .97], dtype=self.z.dtype)
        self.tau = torch.tensor([.145, .065], dtype=self.z.dtype)
        self.delay = torch.tensor([0., 4.507124999], dtype=self.z.dtype)
        self.history = torch.linspace(0.1, .9, 48, dtype=self.z.dtype).reshape(2, 6, 4)
        self.motor = torch.tensor([[.33, .22, .15, .21], [.1, .4, .3, .5]], dtype=self.z.dtype)

    def test_motor_preview_matches_live_queue_order_and_fractional_delay(self):
        candidate = torch.tensor([[.7, .8, .5, .4], [.8, .4, .85, .45]], dtype=self.z.dtype)
        output = _queued_thrust(self.motor, self.history, self.tau, self.delay, .01, self.gain, self.scale, candidate)
        expected = []
        for row, (history, initial) in enumerate(zip(self.history, self.motor, strict=True)):
            fifo = deque(history.unbind(), maxlen=len(history))
            state = initial.clone()
            alpha = -math.expm1(-.01 / self.tau[row].item())
            delay = self.delay[row].item()
            whole = math.floor(delay)
            forces = []
            for tick in range(16):
                fifo.append(candidate[row] if tick < 8 else torch.zeros_like(state))
                command = fifo[-1 - whole].lerp(fifo[-2 - whole], delay - whole)
                state.lerp_((command * self.gain[row]).clamp(0., 1.), alpha)
                forces.append(self.scale[row] * _thrust_from_duty(state).sum())
            expected.append(torch.stack(forces))
        torch.testing.assert_close(output, torch.stack(expected), atol=1.e-13, rtol=1.e-13)
        shutdown = _queued_thrust(self.motor, self.history, self.tau, self.delay, .01, self.gain, self.scale, None)
        torch.testing.assert_close(output[1, :4], shutdown[1, :4])
        self.assertGreater(output[1, 4].item(), shutdown[1, 4].item())

    def preview(self, candidate=None):
        return queued_coast(self.z, self.v, self.up, self.mass, self.drag, self.scale, self.gain,
                            self.motor, self.history, self.tau, self.delay, .01, candidate)

    def test_pending_energy_increases_apex_above_immediate_coast(self):
        apex, time = self.preview()
        ratio = self.drag / self.mass
        immediate_apex = self.z + torch.log1p(ratio * self.v.square() / 9.81) / (2. * ratio)
        self.assertTrue((apex > immediate_apex).all())
        pulsed, pulsed_time = self.preview(torch.full_like(self.motor, .65))
        self.assertTrue((pulsed > apex).all())
        self.assertTrue((pulsed_time > time).all())

    def test_small_speed_zero_crossing_and_existing_descent(self):
        self.v = torch.tensor([.02, -.9], dtype=self.z.dtype)
        self.motor.zero_()
        self.history.zero_()
        apex, remaining = self.preview()
        self.assertGreater(apex[0].item(), self.z[0].item())
        self.assertLess(apex[0].item() - self.z[0].item(), 3.e-5)
        self.assertEqual(apex[1].item(), self.z[1].item())
        distance = self.z[1].item() - .283
        descent_time = ((.9 ** 2 + 2. * 8.24 * distance) ** .5 - .9) / 8.24
        self.assertAlmostEqual(remaining[1].item(), descent_time, places=12)
        after, future = self.preview(torch.ones_like(self.motor))
        torch.testing.assert_close(after[1], apex[1])
        torch.testing.assert_close(future[1], remaining[1])

    def test_retained_pulse_scores_avoidable_surplus_before_zero_command_is_high_enough(self):
        height = torch.tensor([.58, .72, .88, .91, .88])
        gate = torch.tensor([1., 1., 1., 1., 0.])
        vz = torch.tensor([1.8, 1.8, 1.8, -1.8, 1.8])
        state = (height, vz, torch.full_like(height, .98), torch.full_like(height, .1556),
                 torch.full_like(height, .06), torch.ones_like(height), torch.ones_like(height),
                 torch.full((5, 4), .30), torch.full((5, 6, 4), .30),
                 torch.full_like(height, .08), torch.full_like(height, 3.), .01)
        actions = torch.full((5, 4), .30)
        actions[2] = torch.tensor([-1., .9, .9, .9])  # No equal-thrust removal is available.
        original = actions.clone()
        settings = dict(minimum_apex_m=1.025, max_collective_trim_n=.12,
                        min_removable_apex_m=.003, control_ticks=24)
        validate_retained_pulse(settings)
        cost, request, reduced = retained_ascent_pulse_cost(actions, gate, *state, 1.10, 14., settings)
        coast, _ = queued_coast(*state)

        self.assertLess(coast[1].item(), 1.005)
        self.assertGreater(request[1].item(), 1.10)
        self.assertGreater(reduced[1].item(), 1.025)
        self.assertLess(request[0].item(), 1.0)
        self.assertGreater(request[2].item(), 1.10)
        self.assertEqual((cost < 0.).nonzero().flatten().tolist(), [1])
        torch.testing.assert_close(actions, original, atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
