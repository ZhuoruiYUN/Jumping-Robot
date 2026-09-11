# CURRENT_CONTEXT_V58

Date: 2026-09-01

## Main Diagnosis

The current random two-hop planner problem is not that the stable baseline
cannot jump. The stable baseline still contains useful hopping behavior after
observation-width migration.

The failure mode is that several recent PPO fine-tuning branches destroy the
baseline behavior within a few updates. This is visible when direct play from
the migrated baseline has nonzero landing accuracy, but the same checkpoint
after 20 training iterations collapses to zero current hit rate and very low
apex height.

Use `[EVAL-DIRECT]` as the primary validation block. The regular `[EVAL]`
block often reports zero live metrics from reset/logging artifacts and is less
useful for checkpoint selection.

## Baseline Migration Check

Stable baseline checkpoint:

```text
logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt
```

`play_planner_circular.py` was updated to migrate 37-D recurrent checkpoints to
the current planner observation width for play/eval.

Observed migrated-baseline eval on near-stationary commands `0.00-0.02 m`:

```text
target_hit_rate                 0.405448
prepared_landing_rate           0.405448
episode_touchdown_error_m       0.116305
touchdown_along_error_m        -0.021062
touchdown_lateral_abs_error_m   0.053931
touchdown_attitude_error_rad    0.051348
touchdown_next_velocity_error   0.114141
last_apex_height_m              0.956868
max_consecutive_hits            3.066406
successful_waypoints            6.046875
deploy_success_rate             0.855469
death_rate                      0.144531
```

Conclusion: the 37-D baseline is a valid warm-start source. It is not enough
for deployment because it still has tilt deaths, but it is much better than the
post-training collapsed policies.

## Failed Training Evidence

Recent `static_apex_finetune` / `trajectory_landing_finetune` attempts from
baseline or planner checkpoints repeatedly degraded quickly:

```text
fixed 0.25 m:
  target_hit_rate               0.07-0.08
  touchdown_error_m             0.25-0.27
  touchdown_along_error_m      -0.16 to -0.22
```

This indicates systematic under-jump. Adaptive apex height around
`1.14-1.15 m` improved vertical command following but did not fix XY landing.

Small-distance pvelobs/no-impulse branch still works partially:

```text
0.04-0.08 m:
  target_hit_rate               0.46-0.48
  touchdown_error_m             ~0.112
  max_consecutive_hits          ~3.8-3.9
```

Intermediate distances degrade:

```text
0.08-0.14 m:
  best A2 model_80 target_hit   ~0.293
  touchdown_error_m             ~0.149

0.12-0.18 m:
  A3 model_39 target_hit        ~0.181
  landing_compensation=0.10     ~0.208
  touchdown_error_m             ~0.167
```

Longer `0.24-0.26 m` tests remain poor:

```text
target_hit_rate                 ~0.07-0.08
touchdown_error_m               ~0.25-0.27
```

## Important Failure Pattern

After starting from a useful migrated baseline, a 20-iteration training run
collapsed:

```text
Mean action noise std           0.03
Mean episode length             ~907
target_hit_rate                 0.0000
prepared_landing_rate           0.0000
mean_cycle_apex_height_m        ~0.27
episode_touchdown_error_m       ~0.233
```

At the same time, EMA metrics were still high:

```text
target_hit_rate_ema             ~0.288
prepared_landing_rate_ema       ~0.286
live_max_consecutive_hits       14
```

Interpretation: the initial policy was still good at the beginning, then PPO
updates quickly pushed it out of the stable hopping basin.

Likely causes:

- action std `0.03` is too large for preserving a migrated recurrent baseline
- `static_apex_finetune` removes or gates too much dense XY and touchdown credit
- `target_miss`, `low_apex`, and apex penalties dominate before the policy has
  a usable two-hop landing signal
- `static_apex_height_probe` is diagnostic only; it zeros most landing/planner
  rewards and should not be used as a training stage
- adding new observation dimensions to a recurrent policy can make PPO updates
  fragile even when zero-padded migration loads successfully

## Code Changes

`play_planner_circular.py`:

- added automatic recurrent checkpoint migration for play
- supports 37-D stable baseline checkpoints in 43-D/47-D planner eval
- writes migrated play checkpoints under:

```text
outputs/planner_migrations/
```

`train_planner_circular.py`:

- added `--baseline_warmstart_finetune`
- new experiment route version: `v67`
- intent: preserve the stable baseline jump contract while adding only small
  landing/trajectory hints

Warm-start branch behavior:

```text
randomize_dynamics              False
randomize_action_delay          False
randomize_route_phase           False
terminate_on_target_miss        False
force_full_planner              False
stance_reference_uses_apex      True
planner_reference_blend         0.0 by default
require_apex_tolerance_for_hit  False
gate_touchdown_rewards_by_apex  False
gate_dense_xy_rewards_by_apex   False
target_miss_penalty_scale       0.0
low_apex_touchdown penalties    0.0
action std                      0.003
learning_rate                   2e-6
entropy_coef                    1e-6
num_steps_per_env               256
num_mini_batches                16
save_interval                   5
```

## Current Recommendation

Do not continue the failed 20-iteration run. It already showed destructive
updates.

Next controlled experiment:

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

BASE=/home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac/logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt

/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh train_planner_circular.py \
  --headless \
  --checkpoint "$BASE" \
  --route random_two_hop \
  --distance_stage custom \
  --short_radius_min 0.00 --short_radius_max 0.02 \
  --long_radius_min 0.00 --long_radius_max 0.02 \
  --height_stage high \
  --height_high 1.0 \
  --relative_next_hop \
  --baseline_warmstart_finetune \
  --target_tolerance 0.10 \
  --iterations 20
```

Eval immediately:

```bash
RUN=$(ls -td logs/rsl_rl/quadhopper_planner_random_two_hop_v67_custom_relative_next_baseline_warmstart_*/* | head -1)

for M in 0 5 10 19; do
  echo "=== baseline_warmstart model_$M eval 0.00-0.02 ==="
  /home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_planner_circular.py \
    --headless \
    --checkpoint "$RUN/model_${M}.pt" \
    --route random_two_hop \
    --distance_stage custom \
    --short_radius_min 0.00 --short_radius_max 0.02 \
    --long_radius_min 0.00 --long_radius_max 0.02 \
    --height_stage high \
    --height_high 1.0 \
    --relative_next_hop \
    --baseline_compatible_stance_reference \
    --planner_reference_blend 0.0 \
    --eval_steps 1700
done
```

Pass condition:

```text
model_0 keeps target_hit_rate around 0.40
model_5/10/19 do not collapse to zero
mean_cycle_apex_height stays near baseline, not 0.27-0.35 m
```

If this still collapses, the next diagnosis is no longer reward scaling. The
next likely issue is PPO update instability after recurrent observation-width
migration. Then try one of:

- freeze recurrent layers briefly and train only input adapters/head
- initialize new observation columns to exact zeros and keep lr below `1e-6`
- imitation/distillation from baseline before PPO
- train from baseline with no new observation dimensions first, then add
  planner observations in a second migration
