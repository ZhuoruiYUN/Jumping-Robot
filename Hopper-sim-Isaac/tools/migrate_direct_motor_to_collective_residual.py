"""Convert a four-direct-motor recurrent checkpoint to collective/residual coordinates."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Quadhopper_Stable.action_parameterization import direct_motor_to_collective_residual_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error(f'Output already exists: {args.output}')

data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
state = copy.deepcopy(data['model_state_dict'])
weight_key, bias_key = 'actor.4.weight', 'actor.4.bias'
if state[weight_key].shape[0] != 4 or state[bias_key].shape[0] != 4:
    raise ValueError('Expected four direct-motor actor outputs')
transform = direct_motor_to_collective_residual_matrix(dtype=state[weight_key].dtype)
state[weight_key] = transform @ state[weight_key]
state[bias_key] = transform @ state[bias_key]
# The LSTM receives five four-action history slots at observations [17:37].
# Old motor actions y are reconstructed from new coordinates x by y=P x.
# Transform those input columns for both actor and critic recurrent encoders so
# their latent state is unchanged for every compatible action history.
basis = torch.tensor(
    ((1.0, 1.0, -1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (1.0, -1.0, 1.0)),
    dtype=state[weight_key].dtype,
)
reconstruct = torch.cat((torch.ones((4, 1), dtype=basis.dtype), basis), dim=1)
for key in ('memory_a.rnn.weight_ih_l0', 'memory_c.rnn.weight_ih_l0'):
    if key not in state:
        raise ValueError(f'Missing recurrent input weights: {key}')
    original = state[key].clone()
    for start in range(17, 37, 4):
        state[key][:, start:start + 4] = original[:, start:start + 4] @ reconstruct
data['model_state_dict'] = state
infos = copy.deepcopy(data.get('infos') or {})
contract = copy.deepcopy(infos.get('tracking_recovery_contract') or {})
contract.update(
    version=13,
    protocol='collective_residual_v1_physical_1m',
    actions=4,
    action_parameterization='collective_residual_v1',
    action_history='five_raw_collective_residual_actions_oldest_to_newest',
    deployment_action_shaping='none',
    v12_launcher_compatible=False,
    migration=(
        'linear direct-motor to collective/residual warm start; the transform '
        'is exact only when neither coordinate system saturates, so this '
        'checkpoint is training-only until it passes a fresh evaluation'
    ),
)
infos['tracking_recovery_contract'] = contract
data['infos'] = infos
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, args.output)
print(f'[COLLECTIVE-RESIDUAL] migrated={args.checkpoint} output={args.output}')
