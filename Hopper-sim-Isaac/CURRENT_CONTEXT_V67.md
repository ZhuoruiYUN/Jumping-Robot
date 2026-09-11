# CURRENT CONTEXT V67 — physical baseline lineage audit

## Confirmed mismatch

The canonical stable checkpoint
`logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt`
was trained before the 2026-09-05 physical correction. It used:

- `Izz = 2.305957e-3 kg m^2`
- `Ktau = 5.4e-2`

The current asset and actuator use:

- `Izz = 7.931903e-4 kg m^2` through `OriginJumpHopperAsset.usda`
- `Ktau = 1.72e-2`

New/old Izz is 0.344 (external yaw-disturbance acceleration sensitivity is
2.907 times larger). `Ktau/Izz` is 21.685 now versus 23.418 before, so nominal
commanded yaw authority is similar, but disturbance dynamics are not.

The V62 model used by the failed transition stages descends from the
2026-09-06 43-D random-two-hop from-scratch run. It was not initialized from
the 37-D stable `model_498.pt`. Therefore it does not preserve the prior
baseline-trained vertical rhythm and recurrent state.

## Correct lineage

1. Evaluate old `model_498.pt` in the current corrected stable environment.
2. Migrate that checkpoint 37-D to 43-D using the baseline warm-start stage,
   nearly stationary 0--0.02 m commands, baseline-compatible stance height,
   and planner blend 0.
3. Expand XY distance only after the migrated policy preserves repeated hops.

The corrected-physics evaluation passed without any stable-task fine-tuning:

- full length and deploy success: 1.0
- death rate: 0
- good jump rate: 0.999696
- mean apex error: 0.0344 m
- mean landing XY error: 0.0411 m

Therefore stable-task fine-tuning is unnecessary and risks damaging an already
valid controller.

The unrestricted 2026-09-01 37-D to 43-D PPO run is also not safe to repeat.
Its apex fell to roughly 0.26 m and hit rate reached zero within 12 updates.
The old checkpoint contains no serialized observation-normalizer buffers, while
RSL-RL updates actor normalization every rollout. That changed all original 37
inputs at once; unrestricted PPO also updated all actor weights.

The new warm-start freezes the actor MLP, recurrent weights/biases, action std,
and the first 37 columns of the recurrent input. Only columns 37:43 learn. Actor
normalization remains the identity used by the passing evaluation. The critic
remains trainable.

Commands:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  eval_stable.py --headless --num_envs 256 --max_steps 3000 \
  --checkpoint logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt

bash run_eval_baseline_in_two_hop_env.sh 256 1500

bash run_train_two_hop_from_baseline.sh 256
```

The former V3 script that continued from V62 and the unnecessary corrected
stable fine-tune script have been removed.
