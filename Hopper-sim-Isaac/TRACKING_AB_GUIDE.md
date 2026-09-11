# Direct motor PPO: practical current-target vs preview A/B (v2)

This is an isolated experiment. Existing training scripts/configs and robot
physics are unchanged. Both arms output four normalized motor commands at
100 Hz, followed by the existing motor lag and calibrated thrust curve.
There is no LQR, attitude controller, or frozen residual teacher.

## Common corrected contract

- Certified source: `2026-09-07_22-13-09/model_10.pt`, 43-D recurrent policy,
  relative next-target observations, original XY command scale 0.20 m.
- The source actor and critic next-XY input columns 39:41 are preserved. A
  `current` masks those observation slots at runtime; B `preview` receives
  them unchanged. Both arms load the same checkpoint parameters. Their
  initial actions intentionally differ because this v2 test measures the
  practical value of the already learned preview pathway. `initial.pt` is
  saved before any PPO update (unlike RSL-RL's post-update `model_0.pt`).
- A `current`: next-XY observation slots stay zero. B `preview`: those slots
  contain the relative next-target vector. Same architecture, optimizer,
  number of steps, dynamics, reward and route distribution.
- Actual identity actor/critic normalization, fixed LR 1e-5, fixed std 0.03,
  256 steps/rollout, 120 iterations, checkpoints every 10 iterations.
- Command scale remains 0.20 m when testing different distances. No legacy
  checkpoint with an uncertified scale is accepted by the new train entry.
- Continuous queue: the announced next target becomes the current target
  exactly. At a miss it remains an absolute physical target; it is not
  silently moved to keep a nominal hop distance. Turns are bounded by 30°
  between commanded segments. Actual corrective directions may be larger.
- Train on 10–20 cm; separate evaluation on that range and fixed20 cm.
- Target tolerance 10 cm and height tolerance ±15 cm around 1 m. Remove the
  35-cm goal bonus; narrow inherited XY reward width to 15 cm; gate landing
  quality reward by height validity; remove zero-velocity trajectory reward,
  descent velocity shaping, and target-switch progress reward. Keep modest
  source height shaping rather than the failed large-penalty intervention.
- Dynamics remain nominal with Tm=0.0674. Legacy `play_action_delay=0`
  indexing is deliberately retained for source compatibility (it currently
  selects the oldest of five samples, not actual zero delay). This experiment
  does not validate that delay for hardware; correcting it is a separate
  physics transfer experiment.

## Commands

From the project directory, the orchestration script uses standard Python;
it invokes Isaac Python itself. `ISAAC_SIM_ROOT` can override the install.

```bash
python run_tracking_ab.py train --root outputs/tracking_ab/compare_v2
python run_tracking_ab.py eval --root outputs/tracking_ab/compare_v2
```

Training runs both arms sequentially, seed42. Screening evaluates every
10-iteration checkpoint on range commands with seed142, rejects checkpoints
whose death rate exceeds their own initial arm by more than 3 percentage
points when a safe candidate exists, then maximizes strict valid10. Final
evaluation uses fresh seeds242 and342 on both distance sets. It evaluates
separate `initial_current` and `initial_preview` baselines plus both selected
policies. Raw logs are saved to `*.console.log`, manifests and embedded
checkpoint contracts record configuration, and the final report is
`outputs/tracking_ab/compare_v2/comparison.csv`; all checkpoint screens are in
`checkpoint_selection.csv`. Only progress and result
summaries are printed. Existing outputs are never overwritten; incomplete
runs fail explicitly rather than silently being reported as successful.

Run a small GPU smoke test before the main experiment:

```bash
python run_tracking_ab.py train --root outputs/tracking_ab/smoke_v2 --num-envs 4 --iterations 1
python run_tracking_ab.py eval --root outputs/tracking_ab/smoke_v2 --num-envs 4 --steps 400 --screen-seed 142 --eval-seeds 242
```

For replication, use `--seeds 42 43 44` for BOTH train and eval under a new
root. Default one-seed training is a pilot, not evidence of universal
superiority. Each arm receives the same seed, but state-dependent resets
can lead to different later sampled routes; evaluate across seeds.

## Read the comparison

`valid10` means XY error <10 cm AND apex error ≤15 cm. `pair10` requires
both hops to meet those conditions. Counters accumulate at touchdown before
reset, including failed episodes. Pair denominator is pairs started at a
first touchdown: deaths between hops remain failures, and unfinished pairs
at the horizon are conservatively included (also reported as pending).
Deaths before the first touchdown are represented by the separate death
metric. XY-only hit rates at5/10/15 cm are also saved.

Compare each trained arm with its corresponding initial arm under the v2
contract. Compare A and B at fixed20 cm first: valid10, pair10, signed along-error, all-touchdown
apex error and fraction of environments with a physical death. Look at
per-evaluation-seed rows as well as the printed means. A small gain on one
seed is inconclusive; do not trade a large death increase for more hits.
If B improves pair completion consistently, keep preview. If A matches or
beats B, start with current-target control. If neither improves over initial,
stop further distance expansion and diagnose the common objective/control
limitations. This experiment does not attribute improvements to any one of
the shared fixes and its stricter metrics are not comparable to old printed
hit rates.

## Local validation

CPU tests cover preservation and behavioral use of the real source
checkpoint's actor/critic preview columns, queue persistence across both phase boundaries,
turn bounds, latching the completed landing target before route advancement,
statistics surviving resets, pair failures across death, and height-invalid
landings not counting as valid hits. All 12 tests, including the existing
command tests, pass. Python compilation and orchestration dry runs pass.
GPU simulation/training remains unverified in this execution environment
because the NVIDIA driver is unavailable. No training gain is claimed yet.
