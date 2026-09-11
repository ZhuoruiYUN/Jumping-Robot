# CURRENT CONTEXT V65 — second-hop failure is a transition-state problem

## V2 result at iteration 74/75

The short-to-long bridge still fails while the second command is only
`0.0966--0.2031 m` with a maximum turn of `24.69 deg`:

- reset batch count: `4.1758`
- short touchdowns: `4.1641`
- long touchdowns: `0.1523`
- resets while in long phase: `4.0117` (`96.1%` of resets)
- tilt-over-limit resets: `4.1758` (all resets)
- pair hits: `0`
- short hit EMA: `0.9572`
- long hit EMA: `0`

This rules out long-hop geometry as the immediate bottleneck. The controller
usually completes hop 1, advances the command, and tips before completing hop
2 even when hop 2 is nearly stationary.

## Leading hypothesis

The 43-D observation has no explicit hop-boundary flag. The recurrent policy,
five-action history, and motor state all continue across touchdown. An isolated
hop starts with recurrent history from the random-drop/settling sequence, while
hop 2 starts with recurrent history from a complete directional flight. The
same small target therefore reaches the actor with a very different hidden
state. Reward and distance curricula cannot directly remove that mismatch.

## Decisive evaluation ablation

`play_planner_circular.py` now supports `--reset_rnn_on_touchdown`. It clears
only actor/critic recurrent state after touchdown; physics, motor lag, delayed
actions, and the 43-D observation remain unchanged. It also always resets RNN
state on episode dones, matching training behavior (the old play loop omitted
this and contaminated recurrent multi-episode evaluation).

Run the two cases with the same seed and checkpoint:

```bash
bash run_eval_two_hop_rnn_ablation.sh baseline 512 1500
bash run_eval_two_hop_rnn_ablation.sh touchdown-reset 512 1500
```

Compare `[EVAL-DEPLOY] touchdowns_per_episode`, `long_touchdowns`, and tilt
deaths. A large gain from `touchdown-reset` proves cross-hop recurrent memory is
the blocker. If it does not improve, the next ablation should preserve RNN state
and reset only the five-action observation/delay history; that separates policy
memory from actuator/history carryover.

Do not start another PPO curriculum before this A/B result. V1 and V2 already
show that additional reward shaping and easier geometry do not create a usable
second-hop transition.

## Validation

- `python -m compileall -q play_planner_circular.py Quadhopper_Planner_Circular Quadhopper_Planner_Random`
- `bash -n run_eval_two_hop_rnn_ablation.sh`
- `git diff --check -- play_planner_circular.py run_eval_two_hop_rnn_ablation.sh`

No Isaac Sim/GPU evaluation was launched in this change.
