# Current arbitrary-direction Stage 2 task

This directory implements the current path-free two-target observation and repeated-hop task in Isaac Sim. It also contains older route generators and diagnostic variants; those are historical experiments, not the selected V38 controller.

## Training contract

- **Policy:** recurrent RSL-RL PPO with direct four-motor actions. `history_foot_v19` stacks six 56-value frames (current plus five previous) into a 336-value observation. The current and next ground targets are visible in the state, together with measured foot geometry.
- **Route and height:** 0.12–0.50 m hops and a 1.0 m world-Z apex command. The training sampler includes arbitrary headings, reverse turns and zero-distance targets. The current task uses `--path-free-stage2-v3`; the older 0.5–1.0 m random-route curriculum and its 43-value observation are separate, historical tasks.
- **Plant:** the 11 October 2026 mounted SysID update is enabled by `--sysid-20261011-plant`. It modifies the simulated mass/inertia and motor response, but retains a spring-leg proxy and does not prove real-hardware fidelity.
- **Episodes:** V38 nominal and robust adaptation use `--two-hop-episode-v38` to reset after two hops. V38 chain short, mid, bridge and final remove that flag and use 12, 24, 40 and 61 s episode limits respectively.
- **Scale:** completed formal V37–V40 training used 1,024 parallel environments and 256 policy steps per environment per PPO iteration. The selected V38 final-chain run completed 200 iterations after the shorter curriculum stages. Checkpoints are transferred as weights; optimizer state is not resumed across stages.

The source entry points are `../run_tracking_recovery.py` and `../train_tracking_recovery.py`; V37–V40 shell wrappers provide the recorded parameters and use the installed Isaac Sim launcher. A later wrapper requires a completed local checkpoint from its prior stage. Generated checkpoints, manifests, evaluation JSON and local handoff documents are excluded from Git.

## Result and limit

The selected research checkpoint is V38 final-chain `model_199.pt`. Under seed 583 with 100% wider hardware stress, its strict foot-pair 10 cm plus 1.0 m apex rates were 54.75% arbitrary-direction, 77.15% in-place and 68.38% reverse during 61 s continuous tests. Each scenario used 16 environments × 30,000 policy steps. A 301 s stress run gave 52.60%, 82.25% and 74.17% respectively. These are simulation measurements, not hardware validation or a deployment qualification. See the [repository README](../../README.md) for the training-stage and evaluation-condition table.
