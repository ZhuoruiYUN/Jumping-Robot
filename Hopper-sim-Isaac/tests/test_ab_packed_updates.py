import unittest


class PackedMemoryTest(unittest.TestCase):
    def test_unsorted_lengths_match_dense_ppo_outputs_and_gradients(self):
        from copy import deepcopy
        try:
            import torch
            from rsl_rl.networks.memory import Memory
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        from benchmark37.packed_recurrent import use_packed_updates

        torch.manual_seed(17)
        dense = Memory(9, hidden_size=16).double()
        packed = deepcopy(dense)
        use_packed_updates(packed)
        inputs = torch.randn(8, 3, 9, dtype=torch.double, requires_grad=True)
        other_inputs = inputs.detach().clone().requires_grad_(True)
        hidden = tuple(torch.randn(1, 3, 16, dtype=torch.double) for _ in range(2))
        mask = torch.arange(8)[:, None] < torch.tensor([5, 8, 3])
        reference = dense(inputs, mask, hidden)
        actual = packed(other_inputs, mask, hidden)
        weights = torch.randn_like(reference)
        (reference * weights).sum().backward()
        (actual * weights).sum().backward()
        pairs = [(reference, actual), (inputs.grad, other_inputs.grad),
                 (dense(inputs[0]), packed(other_inputs[0]))]
        pairs.extend((a.grad, b.grad) for a, b in zip(dense.parameters(), packed.parameters()))
        for expected, result in pairs:
            torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
