# Jumping Robot: reinforcement learning for a hybrid hopper

This repository contains a planar hybrid-dynamics hopper, early MuJoCo studies, and an active 3D Isaac Sim research task. The current Isaac task learns foot placement and attitude control over repeated jumps between ground targets. **All current performance results below are from simulation; no physical-robot success or deployment reliability is claimed.**

## Current Isaac Sim method (16 October 2026)

- **Environment and control:** Isaac Sim 5.1 with an Isaac Lab `DirectRLEnv`; four direct motor commands at a 100 Hz policy rate. Contact, stance and ballistic flight are handled by the physics simulation. The current plant is a spring-leg proxy updated with the 11 October 2026 mounted-system identification parameters; it is still an approximation of the hardware.
- **Policy:** recurrent RSL-RL PPO. Its `history_foot_v19` observation has 336 values: the current 56-value frame plus five previous frames. It includes measured robot state, foot geometry, and body-frame information for the current and next ground targets. The learned controller outputs motor commands directly. The active path-free Stage 2 task does **not** use a prescribed Raibert or direct-collocation trajectory during training or inference.
- **Task:** successive arbitrary-direction targets at 0.12–0.50 m, including short/long hops, reverse turns, and zero-distance targets. The apex target is 1.0 m in world Z. Foot landing position, apex, posture and continuity across jumps are evaluated separately.
- **Training:** the formal V37–V40 runs used **1,024 parallel environments**, 256 policy steps per environment per PPO iteration, and weights-only continuation between stages. The V38 nominal and robust stages reset an environment after two hops; the subsequent continuous-chain stages do not reset after every two hops. A 61 s episode limit is therefore compatible with the earlier two-hop curriculum.

### Training that actually ran

| Stage | Environments | PPO iterations | Episode limit | Reset scheme |
|---|---:|---:|---:|---|
| V37 SysID adaptation | 1,024 | 100 | 61 s | Continuous |
| V38 nominal two-hop | 1,024 | 100 | 12 s | After two hops |
| V38 robust two-hop | 1,024 | 150 | 12 s | After two hops; hardware perturbation ramps from 0 to 45% |
| V38 chain short / mid / bridge / final | 1,024 each | 50 / 50 / 80 / 200 | 12 / 24 / 40 / 61 s | Continuous |
| V39 direction-mix control / variant | 1,024 each | 50 each | 61 s | Continuous; variant did not advance |
| V40 foot-width variant | 1,024 | 50 | 61 s | Continuous; variant did not advance |

The selected research checkpoint is **V38 final-chain `model_199.pt`**. These are separate completed training runs, not one uninterrupted run or a claim that 1,024 robots were tested physically. Small smoke tests used fewer environments and are excluded from the table. Run-specific manifests, checkpoints and console output stay local under `Hopper-sim-Isaac/outputs/` and are not committed.

### Frozen-policy evaluation

The strict two-hop measure requires **both** foot placements within 10 cm of their targets and **both** apexes at or above 1.0 m; the denominator includes started pairs that fail or remain incomplete. The V38 checkpoint was evaluated in 16 simulation environments for 30,000 control steps per continuous scenario:

| Evaluation setting | Arbitrary direction | In-place | Reverse |
|---|---:|---:|---:|
| Seed 483, 61 s episodes, 45% training-distribution hardware perturbation | 68.11% | 82.89% | 80.95% |
| Seed 583, 61 s episodes, 100% wider hardware stress | 54.75% | 77.15% | 68.38% |
| Seed 583, 301 s episodes, 100% wider hardware stress | 52.60% | 82.25% | 74.17% |

The 301 s stress test extends beyond the 61 s training limit. No physical death was observed in these finite frozen-policy evaluations, while the 200-iteration V38 final-chain training recorded seven physical death events. Zero observed deaths in a finite simulation test does not establish hardware safety. Accuracy under strong arbitrary-direction stress remains below the project's goal. Matched V39 direction-mix and V40 wider foot-reward variants were evaluated and **not selected** because they did not improve the primary arbitrary-direction result.

## Reproducing the current task

The Stage 2 implementation is in [`Hopper-sim-Isaac/Quadhopper_Planner_Random/`](Hopper-sim-Isaac/Quadhopper_Planner_Random/README.md). Training and evaluation enter through [`run_tracking_recovery.py`](Hopper-sim-Isaac/run_tracking_recovery.py), [`train_tracking_recovery.py`](Hopper-sim-Isaac/train_tracking_recovery.py), and the V37–V40 shell wrappers in `Hopper-sim-Isaac/`. Use the installed Isaac Sim `python.sh` through those wrappers. Later stages require the local checkpoint produced by the preceding stage; this Git repository provides the source and configuration, not the training weights or generated reports.

## Earlier work

The planar QuadHopper uses a custom hybrid stance/flight model and PPO implementations in Stable-Baselines3 and CUDA-native PyTorch. A five-target demonstration and energy plots are retained below. Separate MuJoCo experiments studied ballistic launch timing and landing attitude. Their settings and results should not be attributed to the current Isaac Sim policy.

| Planar five-hop rollout | Planar phase and energy diagnostics |
|---|---|
| <img src="docs/images/quadhopper-strict-five-hop.gif" alt="Planar five-target hopping"> | <img src="docs/images/quadhopper-phase-aligned-five-hop.png" alt="Planar phase diagnostics"><br><img src="docs/images/quadhopper-mechanical-energy-five-hop.png" alt="Planar mechanical energy"> |

| Early MuJoCo rollout | Energy and limit-cycle diagnostics |
|---|---|
| <img src="docs/images/mujoco-hopper-2d.gif" alt="Early MuJoCo hopper" width="360"> | <img src="docs/images/ppo-energy-analysis.png" alt="MuJoCo energy analysis" width="230"><br><img src="docs/images/ppo-limit-cycle.png" alt="MuJoCo limit cycle" width="230"> |

## Repository layout

- `Hopper-sim-Isaac/`: active 3D Isaac Sim environments, recurrent PPO training and evaluation.
- `Hopper_simu/`: planar hybrid-dynamics studies.
- `model-free-mujoco-RL/`: earlier MuJoCo control experiments.
- `Jumping-Robot-Obd/`: hardware-side notes and code.
- `docs/images/`: selected project visualizations.
