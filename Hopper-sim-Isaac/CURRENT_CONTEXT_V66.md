# CURRENT CONTEXT V66 — V2 transition reward regression and V3 correction

> Superseded after lineage audit: the V62 source is a corrected-physics 43-D
> from-scratch lineage, not a migration from the canonical 37-D stable
> baseline. Do not run the former V3 transition recipe. Rebuild the lineage as
> documented in `CURRENT_CONTEXT_V67.md`.

## What the 1500-step evaluation proves

- 2892 completed episodes, 3120 touchdowns: `1.0788 touchdown/episode`.
- Only 99 long-phase touchdowns versus 3021 short-phase touchdowns.
- 2887/2892 completed episodes ended in tilt death.
- Reset roll/pitch angular-speed sum was 65.70 for a six-reset log batch,
  approximately `10.95 rad/s/reset`; yaw was only `1.75 rad/s/reset`.

The failure is still the spring-rebound transition after the first touchdown.

## Root configuration regression

V1/V2 combined `--long_hop_curriculum` and
`--two_hop_transition_curriculum`. The former set
`long_hop_stability_only=True`, so the first/short hop did not receive the new
airborne attitude, roll/pitch rate, action-spread, or near-ground tilt shaping.
The latter simultaneously asked the first hop to:

- pre-tilt 3 degrees toward hop 2 during late descent/stance;
- land with 0.10 m/s forward velocity toward hop 2;
- earn prepared-landing reward based on that transition state.

Actual projection reached about 0.15 m/s in training and 0.23 m/s in direct
evaluation. Touchdown attitude was 0.133--0.200 rad, versus 0.068 rad for the
previous successful m79 model. Historical V59 already recorded that touchdown
velocity and anticipatory-attitude shaping degraded hit rate and attitude.

This is not evidence that ordinary attitude penalties are too high. The old
successful safety model used stronger global stability penalties. V2 instead
removed the added stability terms from hop 1 while positively rewarding a
non-neutral rebound state.

## V3 correction

`train_planner_circular.py` now makes transition mode:

- set `long_hop_stability_only=False`, applying stability shaping to both hops;
- remove setup forward-velocity and lateral-velocity shaping;
- remove anticipatory pre-tilt reward/penalty;
- remove prepared-landing reward;
- retain pair-hit/streak credit and termination penalty.

`run_train_two_hop_transition_v3.sh`:

- branches from V62 isolated-hop `model_99.pt`, never V1/V2;
- fixes second-hop geometry at 0.05--0.15 m and 15 degrees;
- lowers PPO action noise from 0.06 to 0.03 and LR to 2e-5;
- runs 150 iterations; geometry expansion is deferred.

Run:

```bash
bash run_train_two_hop_transition_v3.sh 1024
```

Check by iteration 30--50. Continue only if touchdowns/reset rises above 1.10,
long touchdowns remain nonzero, and reset XY angular speed declines. A useful
transition checkpoint should then approach 2 touchdowns/reset before any
distance expansion.
