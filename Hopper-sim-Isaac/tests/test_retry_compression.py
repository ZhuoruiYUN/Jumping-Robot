import unittest

import torch

from benchmark37.ab_recovery import ABRecovery
from benchmark37.retry_compression import RetryCompression, projected_release_at_contact


class CompressionTest(unittest.TestCase):
    def test_case107_and_successful_or_truncated_recovery(self):
        state = RetryCompression(5, 'cpu', .47, .1, 4.)
        yes = torch.ones(5, dtype=torch.bool)
        no = ~yes
        selected = torch.tensor([True, True, True, False, True])
        state.contact(yes, selected, torch.tensor([.378, .367, .378, .378, .48]))
        penalties = []
        for j, (v63, v57) in enumerate(zip([.352,.405,.500,.593,.598,.598], [.328,.317,.334,.352,.340,.335])):
            err = torch.tensor([v63,v57,v63,v63,v63])
            penalties.append(state.step(err, torch.ones(5), yes if j==5 else no, no, no, no, no))
        ledger = torch.stack(penalties)
        self.assertTrue((ledger[:2] == 0).all())
        self.assertTrue(ledger[2,0] < 0)
        self.assertFalse(ledger[:, [1,3,4]].any())
        self.assertFalse(state.tracking.any())
        hit = torch.tensor([False,True,True,False,False])
        refund = state.step(err, torch.ones(5), no, yes, hit, no, no)
        self.assertLess(float(ledger[:,0].sum()+refund[0]),0)
        torch.testing.assert_close(ledger[:,2].sum()+refund[2], torch.tensor(0.), atol=1e-6, rtol=0)
        self.assertFalse(state.debt.any())
        state.contact(yes, yes, torch.zeros(5))
        debit = state.step(err, torch.ones(5), no, no, no, no, no)
        state.reset(torch.tensor([0]))
        cancelled = state.step(err, torch.ones(5), no, no, no, no, yes)
        torch.testing.assert_close(cancelled[1:], -debit[1:])
        self.assertFalse(cancelled[0] or state.debt.any())

    def test_recovery_during_support_and_fall_after_liftoff(self):
        state = RetryCompression(2, 'cpu', .47, .1, 4.)
        flight = torch.ones(2)
        yes = torch.ones(2, dtype=torch.bool)
        no = ~yes
        state.contact(yes, yes, torch.tensor([.40, .40]))
        charge = state.step(torch.tensor([.60, .60]), flight, no, no, no, no, no)
        partial = state.step(torch.tensor([.52, .52]), flight, yes, no, no, no, no)
        self.assertTrue((partial > 0).all() and (charge + partial < 0).all())
        inflight = state.step(torch.zeros(2), flight, no, no, no, no, no)
        torch.testing.assert_close(inflight, torch.zeros(2))
        landed = torch.tensor([True, False])
        settled = state.step(torch.zeros(2), flight, no, landed, landed, ~landed, ~landed)
        net = charge + partial + settled
        torch.testing.assert_close(net[0], torch.tensor(0.), atol=1.e-6, rtol=0)
        self.assertLess(float(net[1]), 0.)
        self.assertFalse(state.debt.any() or state.tracking.any())

    def test_only_missed_continue_assignment_can_be_promoted(self):
        recovery = ABRecovery(5,'cpu',42,dict(retry_probability=.25,local_hold_probability=.25))
        recovery.assignment[:] = torch.tensor([0,0,2,0,1])
        recovery.start(torch.tensor([True,True,True,False,True]),
                       torch.tensor([True,False,True,True,True]))
        self.assertEqual(recovery.active.tolist(),[1,0,2,0,1])
        self.assertEqual(recovery.assignment.tolist(),[0,0,2,0,1])

    def test_axis_sweep_rotates_with_heading(self):
        leg=torch.tensor([[0.,0.,.27]]*2)
        vel=torch.tensor([[.38,0.,-3.],[0.,.38,-3.]])
        omega=torch.tensor([[0.,2.,0.]]*2)
        quat=torch.tensor([[1.,0.,0.,0.],[2**-.5,0.,0.,2**-.5]])
        v=projected_release_at_contact(leg,quat,omega,vel,.02,1.81)
        self.assertGreater(v[0,0],.47)
        torch.testing.assert_close(v[0].flip(0),v[1])
