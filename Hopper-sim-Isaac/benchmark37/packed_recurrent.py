"""Skip right padding during recurrent PPO updates on the AB architecture comparison."""
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rsl_rl.utils import unpad_trajectories


def use_packed_updates(memory):
    inference_forward = memory.forward

    def forward(inputs, masks=None, hidden_states=None):
        if masks is None:
            return inference_forward(inputs, masks, hidden_states)
        if hidden_states is None:
            raise ValueError("PPO updates require the initial state of each trajectory")
        lengths = masks.sum(dim=0).cpu()
        packed = pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        outputs, _ = memory.rnn(packed, hidden_states)
        padded, _ = pad_packed_sequence(outputs, total_length=inputs.shape[0])
        return unpad_trajectories(padded, masks)

    memory.forward = forward
