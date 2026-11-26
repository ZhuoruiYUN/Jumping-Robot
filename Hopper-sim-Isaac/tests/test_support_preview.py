import unittest

import torch

from benchmark37.support_preview import SupportPreview, predict_support_velocity


class SupportPreviewTest(unittest.TestCase):
    def test_pivot_and_rotation_in_world_axes(self):
        axis = torch.tensor([[0., 0., .273]]).repeat(3, 1)
        vel = torch.tensor([[.4, 0., -3.], [0., .4, -3.], [0., 0., -3.]])
        # Same pitch rate, with a 90-degree heading change in lane 1.
        q = torch.tensor([[1., 0., 0., 0.], [2**-.5, 0., 0., 2**-.5], [1., 0., 0., 0.]])
        omega = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 0., 2.]])
        v, clock = predict_support_velocity(axis, q, omega, vel, torch.full((3,), .1), 9.81, 1.81, .03)
        self.assertTrue((clock > 0.).all())
        self.assertGreater(v[0, 0], .4)
        torch.testing.assert_close(v[0].flip(0), v[1])
        torch.testing.assert_close(v[2], torch.zeros(2))

    def test_case31_five_frames_before_target18_contact(self):
        # Frozen +X case 31, frames 2200 (V62) and 2352 (V57), world XY.
        leg = torch.tensor([[-.03685, .01373, .27016], [-.01778, -.00368, .27240]])
        q = torch.tensor([[.99508, -.02976, -.06580, .06784], [.99916, .00752, -.03241, -.02394]])
        rate = torch.tensor([[-1.27170, -.44955, -.27036], [.08098, -.01691, -.31737]])
        velocity = torch.tensor([[.44879, .06414, -2.73805], [.46390, .03589, -3.08506]])
        pred, clock = predict_support_velocity(leg, q, rate, velocity, torch.tensor([.13782, .16112]),
                                               9.81, 1.81, .03)
        self.assertTrue(((clock > .03) & (clock < .14)).all())
        self.assertLess(pred[0, 0].item(), 0.)
        self.assertGreater(pred[0, 1].item(), .3)
        self.assertGreater(pred[1, 0].item(), 0.)
        actual = torch.tensor([[-.466, .712], [.208, -.068]])
        self.assertTrue(((pred - actual).norm(dim=-1) < .30).all())

    def test_confirmed_release_healthy_branch_and_lane_reset(self):
        tracker = SupportPreview(6, 'cpu')
        lanes = torch.ones(6, dtype=torch.bool)
        predicted = torch.tensor([[-.4,.65], [.54,0.], [.55,0.], [.1,0.], [-.4,.6], [.7,0.]])
        tracker.sample(lanes, predicted)
        goal = torch.tensor([[.52,0.], [0.,0.], [0.,0.], [0.,0.], [.52,0.], [0.,0.]])
        incoming = torch.tensor([[.4,0.], [.32,0.], [.65,0.], [.4,0.], [.4,0.], [.2,0.]])
        retry = torch.tensor([False, True, True, True, False, True])
        tracker.contact(lanes, ~retry, retry, incoming)
        tracker.reset(torch.tensor([4]))
        actual = predicted.clone()
        actual[3] = torch.tensor([.58, 0.])  # unpredicted error
        actual[5] = torch.tensor([.26, 0.])  # recovered
        def settle():
            return tracker.release(lanes, actual, goal, torch.ones(6), .47, .1, 4.)
        cost = settle()
        self.assertEqual((cost < 0).tolist(), [True, True, False, False, False, False])
        physical = torch.tensor([True, False, False, False, False, False])
        landed = ~physical
        errors = torch.full((6,), .04)
        paid = tracker.settle(landed, physical, errors, .05)
        self.assertEqual((paid < 0).tolist(), [True, False, False, False, False, False])
        self.assertFalse(tracker.settle(landed, physical, errors, .05).any())
        self.assertFalse(settle().any())

if __name__ == '__main__':
    unittest.main()
