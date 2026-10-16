"""Train/evaluate continuous direct-motor tracking with untimed spatial guidance."""
import argparse
import csv
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('train', 'eval'), default='train')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--iterations', type=int, default=240)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--distance-min', type=float, default=0.20)
parser.add_argument('--distance-max', type=float, default=0.25)
parser.add_argument('--short-distance-min', type=float,
                    help=('Optional short-hop radius lower bound. Supply all four '
                          'short/long distance flags to enable an alternating two-circle route.'))
parser.add_argument('--short-distance-max', type=float)
parser.add_argument('--long-distance-min', type=float)
parser.add_argument('--long-distance-max', type=float)
parser.add_argument('--distance-curriculum-iterations', type=float, default=0.0,
                    help='PPO updates over which two-circle radius ranges expand from explicit start ranges.')
parser.add_argument('--distance-curriculum-start-short-min', type=float)
parser.add_argument('--distance-curriculum-start-short-max', type=float)
parser.add_argument('--distance-curriculum-start-long-min', type=float)
parser.add_argument('--distance-curriculum-start-long-max', type=float)
parser.add_argument('--distance-curriculum-start-turn-angle-deg', type=float,
                    help='Initial turn limit used by the distance curriculum; set equal to the final limit to vary only distance.')
parser.add_argument('--hard-distance-min', type=float, default=0.25,
                    help='Boundary between replay and hard distance intervals.')
parser.add_argument('--hard-sample-probability', type=float, default=0.0,
                    help='Probability of sampling [hard-distance-min, distance-max].')
parser.add_argument('--max-turn-angle-deg', type=float, default=30.0,
                    help='Maximum signed heading change between consecutive waypoints.')
parser.add_argument('--zero-hop-probability', type=float, default=0.0,
                    help='Probability that an appended waypoint is exactly at the current landing point.')
parser.add_argument('--reverse-turn-probability', type=float, default=0.0,
                    help='Probability of a heading reversal around 180 degrees after a hop.')
parser.add_argument('--reverse-turn-halfwidth-deg', type=float, default=30.0,
                    help='Half-width of the reversal band around +/- 180 degrees.')
parser.add_argument('--fixed-initial-heading-deg', type=float,
                    help='Evaluation diagnostic only: initial world heading; 0 gives +X.')
parser.add_argument('--direction-curriculum-iterations', type=float, default=0.0,
                    help=('PPO updates over which signed turn range and zero-hop probability '
                          'ramp from their direction-curriculum start values to the requested '
                          'final task; 0 keeps the requested task fixed.'))
parser.add_argument('--direction-curriculum-start-turn-angle-deg', type=float,
                    help='Initial maximum signed turn angle for the direction curriculum.')
parser.add_argument('--direction-curriculum-start-zero-hop-probability', type=float,
                    help='Initial exact-zero-hop probability for the direction curriculum.')
parser.add_argument('--landing-precision-width', type=float,
                    help='Gaussian touchdown-precision width in metres; unset preserves the checkpoint curriculum default.')
parser.add_argument('--sysid-20261011-plant', action='store_true',
                    help='Use the 2026-10-11 mounted SysID spring-leg proxy plant.')
parser.add_argument('--target-hit-reward-scale', type=float,
                    help='Override sparse touchdown reward; unset preserves the default.')
parser.add_argument('--landing-precision-reward-scale', type=float,
                    help='Override dense touchdown-precision reward; unset preserves the default.')
parser.add_argument('--landing-error-penalty-scale', type=float,
                    help='Override radial touchdown-error penalty; unset preserves the default.')
parser.add_argument('--landing-lateral-error-penalty-scale', type=float,
                    help='Override lateral touchdown-error penalty; unset preserves the default.')
parser.add_argument('--robust-dynamics', action='store_true',
                    help='Enable mass/inertia, delay, motor-response, and per-motor thrust randomization.')
parser.add_argument('--hardware-robust', action='store_true',
                    help='Train through the wider low-thrust and slow-motor envelope measured during deployment.')
parser.add_argument('--comprehensive-hardware-randomization', action='store_true',
                    help=('Also randomize common thrust, roll/pitch effectiveness cross-coupling, '
                          'and spring stiffness/damping. Implies --hardware-robust; direct F1--F4 '
                          'actions, 52-D observations, and the 100 Hz deployment contract stay unchanged.'))
parser.add_argument('--hardware-randomization-probability', type=float, default=1.0,
                    help='Fraction of reset environments receiving hardware perturbations.')
parser.add_argument('--hardware-randomization-probability-start', type=float,
                    help='Initial perturbation fraction for a nominal-to-robust curriculum.')
parser.add_argument('--hardware-randomization-curriculum-iterations', type=float,
                    help='PPO iterations over which hardware perturbation fraction reaches its final value.')
parser.add_argument('--comprehensive-global-thrust-scale-min', type=float,
                    help='Optional lower bound for training-only common motor thrust scaling.')
parser.add_argument('--comprehensive-global-thrust-scale-max', type=float,
                    help='Optional upper bound for training-only common motor thrust scaling.')
parser.add_argument('--comprehensive-com-offset-x-max-mm', type=float,
                    help='Maximum absolute Body CoM offset along body X for hard training environments.')
parser.add_argument('--comprehensive-com-offset-y-max-mm', type=float,
                    help='Maximum absolute Body CoM offset along body Y for hard training environments.')
parser.add_argument('--comprehensive-com-offset-z-max-mm', type=float,
                    help='Maximum absolute Body CoM offset along body Z for hard training environments.')
parser.add_argument('--comprehensive-yaw-effectiveness-min', type=float,
                    help='Lower yaw reaction-torque effectiveness for hard environments.')
parser.add_argument('--comprehensive-yaw-effectiveness-max', type=float,
                    help='Upper yaw reaction-torque effectiveness for hard environments.')
parser.add_argument('--comprehensive-yaw-bias-torque-max-mnm', type=float,
                    help='Maximum absolute collective-dependent yaw bias torque in mN*m.')
parser.add_argument('--observation-noise-std', type=float,
                    help='Override policy observation noise standard deviation.')
parser.add_argument('--initial-yaw-perturb-deg', type=float,
                    help='Symmetric reset yaw perturbation around world +X in degrees.')
parser.add_argument('--initial-angular-velocity-perturb-rad-s', type=float,
                    help='Symmetric reset body angular-velocity perturbation in rad/s.')
parser.add_argument('--touchdown-wrench-probability', type=float,
                    help='Training-only probability of a small contact wrench after a hard-environment touchdown.')
parser.add_argument('--touchdown-roll-wrench-max-nm', type=float,
                    help='Training-only maximum absolute post-touchdown roll wrench in N*m.')
parser.add_argument('--touchdown-pitch-wrench-max-nm', type=float,
                    help='Training-only maximum absolute post-touchdown pitch wrench in N*m.')
parser.add_argument('--touchdown-yaw-wrench-max-nm', type=float,
                    help='Training-only maximum absolute post-touchdown yaw wrench in N*m.')
parser.add_argument('--touchdown-wrench-window-s', type=float,
                    help='Duration of the training-only post-touchdown wrench window in seconds.')
parser.add_argument('--touchdown-wrench-persistence', type=float,
                    help='Cross-hop correlation in [0,1) for the training-only touchdown wrench.')
parser.add_argument('--touchdown-wrench-on-moving-long-hop', action='store_true',
                    help='Also perturb physical moving long-hop touchdowns during training.')
parser.add_argument('--moving-long-recovery-window-s', type=float,
                    help='Physical-contact reward window after a moving long-hop touchdown.')
parser.add_argument('--moving-long-recovery-angle-deadband-deg', type=float,
                    help='Roll/pitch deadband during moving long-hop contact.')
parser.add_argument('--moving-long-recovery-rate-deadband-rad-s', type=float,
                    help='Roll/pitch angular-rate deadband during moving long-hop contact.')
parser.add_argument('--moving-long-recovery-angle-penalty-scale', type=float,
                    help='Non-positive moving long-hop contact attitude penalty.')
parser.add_argument('--moving-long-recovery-rate-penalty-scale', type=float,
                    help='Non-positive moving long-hop contact angular-rate penalty.')
parser.add_argument('--airborne-attitude-penalty-scale', type=float,
                    help='Non-positive all-airborne roll/pitch attitude penalty scale.')
parser.add_argument('--airborne-angvel-xy-penalty-scale', type=float,
                    help='Non-positive all-airborne roll/pitch rate penalty scale.')
parser.add_argument('--airborne-yaw-penalty-scale', type=float,
                    help='Non-positive all-airborne world-yaw-to-zero penalty scale.')
parser.add_argument('--airborne-yaw-rate-penalty-scale', type=float,
                    help='Non-positive all-airborne yaw-rate penalty scale.')
parser.add_argument('--airborne-action-spread-penalty-scale', type=float,
                    help='Non-positive all-airborne motor-spread penalty scale.')
parser.add_argument('--airborne-cross-track-tilt-penalty-scale', type=float,
                    help='Non-positive flight penalty for tilt perpendicular to the target direction.')
parser.add_argument('--airborne-cross-track-tilt-penalty-start', type=float,
                    help='Cross-track thrust-axis XY deadband; 0.05 is about 2.9 deg near level.')
parser.add_argument('--airborne-cross-track-velocity-penalty-scale', type=float,
                    help='Non-positive flight penalty for velocity perpendicular to the target direction.')
parser.add_argument('--airborne-cross-track-velocity-penalty-start-mps', type=float,
                    help='Cross-track world-velocity deadband in m/s.')
parser.add_argument('--airborne-tilt-penalty-start', type=float,
                    help='Deadband for qx^2+qy^2 in the inherited airborne attitude cost.')
parser.add_argument('--airborne-angvel-xy-penalty-start', type=float,
                    help='Airborne roll/pitch angular-rate deadband in rad/s.')
parser.add_argument('--airborne-yaw-deadband-deg', type=float,
                    help='World-yaw deadband around the board arrow pointing +X.')
parser.add_argument('--airborne-yaw-rate-penalty-start', type=float,
                    help='Airborne yaw-rate deadband in rad/s.')
parser.add_argument('--airborne-action-spread-penalty-start', type=float,
                    help='Airborne normalized motor-command spread deadband.')
parser.add_argument('--stationary-hold-radius', type=float,
                    help='Apply stationary stability terms only this close to an exact zero-hop target.')
parser.add_argument('--stationary-tilt-penalty-scale', type=float,
                    help='Soft post-apex tilt penalty for settled exact zero hops.')
parser.add_argument('--stationary-yaw-rate-penalty-scale', type=float,
                    help='Soft post-apex yaw-rate penalty for settled exact zero hops.')
parser.add_argument('--stationary-action-spread-penalty-scale', type=float,
                    help='Soft post-apex motor-spread penalty for settled exact zero hops.')
parser.add_argument('--stationary-pitch-deadband-deg', type=float,
                    help='Pitch deadband in degrees for exact zero-hop flight.')
parser.add_argument('--stationary-pitch-penalty-scale', type=float,
                    help='Non-positive exact-zero-hop pitch penalty scale.')
parser.add_argument('--stationary-pitch-rate-deadband-rad-s', type=float,
                    help='Pitch-rate deadband for exact zero-hop flight.')
parser.add_argument('--stationary-pitch-rate-penalty-scale', type=float,
                    help='Non-positive exact-zero-hop pitch-rate penalty scale.')
parser.add_argument('--stationary-airborne-differential-action-rate-deadband', type=float,
                    help='Allowed per-tick RMS change of the zero-mean four-motor action in flight.')
parser.add_argument('--stationary-airborne-differential-action-rate-penalty-scale', type=float,
                    help='Non-positive airborne differential-action rate penalty scale.')
parser.add_argument('--stationary-contact-recovery-window-s', type=float,
                    help='Physical stationary-touchdown recovery window; 0 disables the new terms.')
parser.add_argument('--stationary-contact-recovery-spread-deadband', type=float,
                    help='Allowed delayed-target differential spread in the recovery window.')
parser.add_argument('--stationary-contact-recovery-spread-penalty-scale', type=float,
                    help='Non-positive differential-spread penalty inside the recovery window.')
parser.add_argument('--stationary-contact-recovery-delta-deadband', type=float,
                    help='Allowed per-tick zero-mean motor-target change in the recovery window.')
parser.add_argument('--stationary-contact-recovery-delta-penalty-scale', type=float,
                    help='Non-positive differential-change penalty inside the recovery window.')
parser.add_argument('--stationary-apex-width', type=float,
                    help='Apex-quality width for exact zero-hop refinement.')
parser.add_argument('--stationary-apex-reward-scale', type=float,
                    help='Measured-apex quality reward for exact zero hops.')
parser.add_argument('--stationary-apex-error-penalty-scale', type=float,
                    help='Measured-apex absolute-error penalty for exact zero hops.')
parser.add_argument('--stationary-apex-shortfall-penalty-scale', type=float,
                    help='Measured-apex low-height penalty for exact zero hops.')
parser.add_argument('--stationary-apex-progress-scale', type=float,
                    help='Bounded reward for measured maximum-height gains on exact zero hops.')
parser.add_argument('--stationary-apex-progress-width', type=float,
                    help='Shortfall range over which stationary maximum-height progress is shaped.')
parser.add_argument('--height-progress-reward-scale', type=float,
                    help='Measured symmetric apex-progress reward per metre; baseline is 3.')
parser.add_argument('--stationary-ascent-support-scale', type=float,
                    help='Per-second reward for a stationary hop still ascending below its apex target.')
parser.add_argument('--stationary-ascent-collective-scale', type=float,
                    help='Collective-thrust reward during the valid stationary ascent phase.')
parser.add_argument('--stationary-predicted-apex-width', type=float,
                    help='Ballistic predicted-apex error width during a valid stationary ascent.')
parser.add_argument('--stationary-predicted-apex-reward-scale', type=float,
                    help='Per-second predicted-apex reward during a valid stationary ascent.')
parser.add_argument('--ascent-predicted-apex-width', type=float,
                    help='All-hop ascent predicted-apex reward width.')
parser.add_argument('--ascent-predicted-apex-reward-scale', type=float,
                    help='Per-second all-hop ascent predicted-apex reward.')
parser.add_argument('--apex-event-height-width', type=float,
                    help='All-hop physical apex-event quality width in metres.')
parser.add_argument('--apex-event-height-reward-scale', type=float,
                    help='Small bounded all-hop apex-event quality reward.')
parser.add_argument('--apex-event-shortfall-penalty-scale', type=float,
                    help='All-hop apex-event penalty per metre below the active target.')
parser.add_argument('--simulation-only-phase-collective-baseline', action='store_true',
                    help='Research-only takeoff/ascent collective floor; exported deployment is forbidden.')
parser.add_argument('--phase-contact-collective-floor-pwm', type=float,
                    help='Simulation-only mean-PWM floor while spring contact is active.')
parser.add_argument('--phase-ascent-collective-floor-pwm', type=float,
                    help='Simulation-only mean-PWM floor during valid positive-vz ascent.')
parser.add_argument('--diagnostic-ascent-pulse-pwm', type=float,
                    help='Evaluation-only symmetric PWM increment during the first 0.20 s after liftoff.')
parser.add_argument('--simulation-only-phase-collective-controller', action='store_true',
                    help='Research-only fixed collective plus bounded zero-mean motor residual.')
parser.add_argument('--phase-contact-collective-pwm', type=float)
parser.add_argument('--phase-ascent-collective-pwm', type=float)
parser.add_argument('--phase-descent-collective-pwm', type=float)
parser.add_argument('--phase-contact-differential-limit-pwm', type=float)
parser.add_argument('--phase-ascent-differential-limit-pwm', type=float)
parser.add_argument('--phase-descent-differential-limit-pwm', type=float)
parser.add_argument('--simulation-only-apex-bias-controller', action='store_true',
                    help='Research-only per-hop apex feedback into a symmetric contact PWM bias.')
parser.add_argument('--apex-bias-initial-pwm', type=float)
parser.add_argument('--apex-bias-gain-pwm-per-m', type=float)
parser.add_argument('--apex-bias-max-pwm', type=float)
parser.add_argument('--apex-bias-feedback', choices=('apex', 'liftoff_vz'), default='apex')
parser.add_argument('--apex-bias-liftoff-vz-target', type=float)
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--target-tolerance', type=float, default=0.05)
parser.add_argument('--apex-tolerance', type=float,
                    help='Apex tolerance used by target-relative height gates; unset preserves the task default.')
parser.add_argument('--target-height', type=float, default=1.000,
                    help='Absolute simulation/policy root-Z apex command in metres.')
parser.add_argument('--raibert-hermite-guidance', action='store_true',
                    help=('Use a bounded Raibert next-foothold correction and ballistic '
                          'velocity-scaled Hermite tangents. Requires history_foot_v19.'))
parser.add_argument('--path-free-stage2', action='store_true',
                    help='Remove curve inputs/rewards; refine physical landings with a small energy cost.')
parser.add_argument('--path-free-stage2-v2', action='store_true',
                    help='Foot progress and staged nominal/robust/energy refinement.')
parser.add_argument('--path-free-stage2-v3', action='store_true',
                    help='Strict-apex success refinement without added energy cost.')
parser.add_argument('--raibert-stance-time-s', type=float, default=0.045,
                    help='Nominal stance duration used only by the Raibert foothold planner.')
parser.add_argument('--raibert-velocity-gain-s', type=float, default=0.08,
                    help='Velocity-error gain for the goal-anchored Raibert correction.')
parser.add_argument('--raibert-max-foothold-shift-m', type=float, default=0.04,
                    help='Maximum guidance-foothold displacement from the scored waypoint.')
parser.add_argument('--raibert-preview-blend', type=float, default=0.5,
                    help='Blend from current-hop velocity to next-hop velocity at touchdown.')
parser.add_argument('--raibert-max-horizontal-speed-m-s', type=float, default=2.5,
                    help='Safety cap for measured and planned horizontal velocities.')
parser.add_argument('--disable-initial-drop', action='store_true',
                    help='Reset on the ground instead of training-only random aerial drops.')
parser.add_argument('--episode-length-s', type=float,
                    help='Override episode duration; use a value longer than the eval horizon for continuous playback.')
parser.add_argument('--successful-hops-per-episode', type=int,
                    help='Override success-reset hop count; continuous eval must exceed the expected number of hops.')
parser.add_argument('--play-motor-time-constant', type=float,
                    help='Fixed nominal motor time constant for deterministic plant identification.')
parser.add_argument('--play-thrust-scale', type=float,
                    help='Fixed nominal per-motor thrust multiplier for plant identification.')
parser.add_argument('--height-curriculum-start', type=float,
                    help='Start a cosine curriculum from this policy/sim apex height to --target-height.')
parser.add_argument('--height-curriculum-iterations', type=float,
                    help='Number of PPO iterations over which the height curriculum reaches --target-height.')
parser.add_argument('--termination-penalty-scale', type=float,
                    help='Override the terminal fall penalty; use a negative value to protect reliability.')
parser.add_argument('--target-relative-apex-gate', action='store_true',
                    help='Gate touchdown success/rewards by current target height minus apex tolerance.')
parser.add_argument('--actuator-mode',
                    choices=('parity', 'corrected_history', 'baseline', 'zero_delay_raw_history'),
                    default='parity',
                    help=('Actuator/history contract. parity matches deployment; '
                          'corrected_history disables PWM shaping only; baseline restores '
                          'the source checkpoint\'s legacy delay and raw-action history; '
                          'zero_delay_raw_history isolates the delay-index fix while preserving '
                          'the source checkpoint\'s raw-action observation history.'))
parser.add_argument('--action-parameterization', choices=('direct_motor', 'collective_residual_v1'),
                    default='direct_motor')
parser.add_argument('--observation-layout', choices=('spatial52', 'history_foot_v19'), default='spatial52')
parser.add_argument('--collective-preserving-spread-limit-pwm', type=float, default=0.0,
                    help='Limit only zero-mean four-motor differential; 0 disables the constraint.')
parser.add_argument('--legacy-v11-contract', action='store_true',
                    help=('Write an explicit v11 direct-motor contract. This is only valid with '
                          '--actuator-mode baseline and is incompatible with the v12 launcher.'))
parser.add_argument('--stability-profile',
                    choices=('baseline', 'takeoff_v9', 'launch_robust_v10', 'yaw_event_v11',
                             'touchdown_robust_v12', 'height_robust_v13',
                             'cross_hop_v14', 'cross_hop_v15', 'cross_hop_v15_1',
                             'cross_hop_v16', 'stationary_corridor_v17',
                             'two_circle_v18'), default='baseline',
                    help='Opt-in versioned stability training; v10-v18 preserve direct four-motor control.')
parser.add_argument('--allow-compatible-checkpoint', action='store_true',
                    help='Evaluate an older compatible 52-D recovery checkpoint under this environment.')
parser.add_argument('--resume-runner-state', action='store_true',
                    help=('For train mode, restore the PPO optimizer and iteration counter from '
                          '--checkpoint rather than loading policy weights only.'))
parser.add_argument('--moving-handoff-v11', action='store_true',
                    help='Opt-in moving short/long contact and early-launch reward shaping.')
parser.add_argument('--landing-recovery-v12', action='store_true',
                    help='Opt-in prelanding safety and near-ground recovery reward extension.')
parser.add_argument('--contact-readiness-v13', action='store_true',
                    help='Opt-in earlier danger forecast and low-altitude handoff reward.')
parser.add_argument('--apex-event-fix-v15', action='store_true',
                    help='Opt-in v12 apex velocity snapshot and once-per-hop event repair.')
parser.add_argument('--terminal-penalty-fix-v16', action='store_true',
                    help='Opt-in v15 repair aligning terminal reward with physical death.')
parser.add_argument('--v3-accounting-fix', action='store_true',
                    help='V3-only apex event and physical-death reward accounting repair.')
parser.add_argument('--foot-pair-v33', action='store_true',
                    help='Replace root pair bonus with strict foot-pair and touchdown posture quality.')
parser.add_argument('--two-hop-episode-v38', action='store_true',
                    help='Train two attempted hops per reset with scenario buckets and landing settle check.')
parser.add_argument('--stage2-foot-precision-scale', type=float,
                    help='V3 foot touchdown Gaussian reward scale; default 60.')
parser.add_argument('--stage2-foot-precision-width-m', type=float,
                    help='V3 foot touchdown Gaussian width in metres; default 0.06.')
parser.add_argument('--pair-xy-quality-weight-v36', type=float,
                    help='Within strict V33 foot pairs, weight smooth two-foot XY quality in [0, 1].')
parser.add_argument('--moving-launch-rate-v34', action='store_true',
                    help='V33 pair task: penalize excess roll/pitch rate in moving-hop late stance and first 0.16 s of flight.')
parser.add_argument('--liftoff-cross-track-penalty-scale', type=float,
                    help='V30 V3-accounting event penalty for lateral Body velocity at liftoff; 0 control, negative experiment.')
parser.add_argument('--hardware-severity-curriculum', action='store_true',
                    help='V3 accounting training: hard hardware offsets use 0.5/0.75/1.0 strength.')
parser.add_argument('--hardware-severity-stage-iterations', type=int, default=30,
                    help='PPO updates at each partial hardware-strength stage.')
parser.add_argument('--route-recovery-mode', choices=('advance', 'retry_large'),
                    help='Evaluation-only frozen-policy A/B after a target miss.')
parser.add_argument('--route-recovery-severe-miss-m', type=float, default=0.20,
                    help='Foot miss threshold for retry_large; the same value is logged in both arms.')
parser.add_argument('--deferred-landing-v17', action='store_true',
                    help='Opt-in v16 experiment deferring half the foot bonus to the next measured apex.')
parser.add_argument('--low-hop-recovery-v18', action='store_true',
                    help='Opt-in v16 measured-height debt and repayment after a low moving hop.')
parser.add_argument('--attitude-recovery-v14', action='store_true',
                    help='Opt-in quaternion risk forecast and attitude-gated recovery release.')
parser.add_argument('--visualize', action='store_true',
                    help='Enable Isaac debug visualization during evaluation.')
parser.add_argument('--control-diagnostics', action='store_true',
                    help='Record policy/RNN and pre/post-step plant state during one-environment evaluation.')
parser.add_argument('--failure-trace', action='store_true',
                    help='Evaluation-only first-death time windows; supports vector environments.')
parser.add_argument('--takeoff-velocity-audit', action='store_true',
                    help='Evaluation-only all-hop liftoff velocity and P1 landing audit; requires --failure-trace.')
parser.add_argument('--contact-phase-audit', action='store_true',
                    help='Evaluation-only touchdown/stance proxies; requires --takeoff-velocity-audit.')
parser.add_argument('--closed-loop-p2-env-id', type=int)
parser.add_argument('--closed-loop-p2-step', type=int)
parser.add_argument('--closed-loop-p2-mode', choices=('original', 'left90', 'right90', 'reverse'))
parser.add_argument('--closed-loop-p2-expected-sha256')
parser.add_argument('--failure-trace-window-s', type=float, default=0.5,
                    help='Seconds before/after first death to retain (0 < value <= 8).')
parser.add_argument('--next-hop-probe-env-ids',
                    help='Read-only next-target actor probe, comma-separated vector environment IDs.')
parser.add_argument('--next-hop-probe-stride', type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.route_recovery_mode and (args.mode != 'eval' or not args.v3_accounting_fix):
    parser.error('--route-recovery-mode requires eval and --v3-accounting-fix')
if args.diagnostic_ascent_pulse_pwm is not None and (
    args.mode != 'eval' or not args.v3_accounting_fix or args.route_recovery_mode
    or args.simulation_only_phase_collective_baseline
    or args.simulation_only_phase_collective_controller
):
    parser.error('--diagnostic-ascent-pulse-pwm requires standalone V3 accounting eval without phase floors')
if args.diagnostic_ascent_pulse_pwm is not None and not 0 <= args.diagnostic_ascent_pulse_pwm <= 100:
    parser.error('--diagnostic-ascent-pulse-pwm must be in [0, 100]')
if args.hardware_severity_curriculum and (
    args.mode != 'train' or not args.v3_accounting_fix
    or not args.comprehensive_hardware_randomization or args.resume_runner_state
):
    parser.error('--hardware-severity-curriculum requires train, V3 accounting fix, comprehensive hardware randomization, and weights-only initialization')
if args.hardware_severity_stage_iterations <= 0:
    parser.error('--hardware-severity-stage-iterations must be positive')
if not 0.10 < args.route_recovery_severe_miss_m <= 0.50:
    parser.error('--route-recovery-severe-miss-m must be in (0.10, 0.50]')
if args.v3_accounting_fix and (
    not args.path_free_stage2_v3 or args.moving_handoff_v11 or args.landing_recovery_v12
    or args.contact_readiness_v13 or args.attitude_recovery_v14
    or args.apex_event_fix_v15 or args.terminal_penalty_fix_v16
    or args.deferred_landing_v17 or args.low_hop_recovery_v18
    or args.simulation_only_apex_bias_controller
):
    parser.error('--v3-accounting-fix requires standalone v3 and no other recovery tasks')
if args.foot_pair_v33 and (
    not args.v3_accounting_fix or args.hardware_severity_curriculum
    or args.route_recovery_mode or args.diagnostic_ascent_pulse_pwm is not None
):
    parser.error('--foot-pair-v33 requires standalone V3 accounting without diagnostic task overrides')
if args.two_hop_episode_v38 and (
    not args.foot_pair_v33 or not args.sysid_20261011_plant
    or args.resume_runner_state
):
    parser.error('--two-hop-episode-v38 requires SysID foot-pair weights without runner-state resume')
if args.moving_launch_rate_v34 and not args.foot_pair_v33:
    parser.error('--moving-launch-rate-v34 requires --foot-pair-v33')
if args.sysid_20261011_plant and not (
    args.path_free_stage2_v3 and args.v3_accounting_fix and args.foot_pair_v33
    and args.observation_layout == 'history_foot_v19'
    and args.action_parameterization == 'direct_motor'
):
    parser.error('--sysid-20261011-plant requires V3 accounting, foot pair, history_foot_v19, direct_motor')
if args.stage2_foot_precision_scale is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.stage2_foot_precision_scale)
    or not 0.0 <= args.stage2_foot_precision_scale <= 200.0
):
    parser.error('--stage2-foot-precision-scale requires --foot-pair-v33 and a finite value in [0, 200]')
if args.stage2_foot_precision_width_m is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.stage2_foot_precision_width_m)
    or not 0.0 < args.stage2_foot_precision_width_m <= 0.20
):
    parser.error('--stage2-foot-precision-width-m requires --foot-pair-v33 and a finite value in (0, 0.20]')
if args.pair_xy_quality_weight_v36 is not None and (
    not args.foot_pair_v33 or not math.isfinite(args.pair_xy_quality_weight_v36)
    or not 0.0 <= args.pair_xy_quality_weight_v36 <= 1.0
):
    parser.error('--pair-xy-quality-weight-v36 requires --foot-pair-v33 and a finite value in [0, 1]')
if args.liftoff_cross_track_penalty_scale is not None and (
    not args.v3_accounting_fix
    or not math.isfinite(args.liftoff_cross_track_penalty_scale)
    or not -20.0 <= args.liftoff_cross_track_penalty_scale <= 0.0
):
    parser.error('--liftoff-cross-track-penalty-scale requires V3 accounting and a finite value in [-20, 0]')
if args.deferred_landing_v17 and args.low_hop_recovery_v18:
    parser.error('--low-hop-recovery-v18 is a separate experiment from --deferred-landing-v17')
if args.low_hop_recovery_v18 and not args.terminal_penalty_fix_v16:
    parser.error('--low-hop-recovery-v18 requires --terminal-penalty-fix-v16')
if args.deferred_landing_v17 and not args.terminal_penalty_fix_v16:
    parser.error('--deferred-landing-v17 requires --terminal-penalty-fix-v16')
if args.terminal_penalty_fix_v16 and not args.apex_event_fix_v15:
    parser.error('--terminal-penalty-fix-v16 requires --apex-event-fix-v15')
if args.apex_event_fix_v15:
    if not args.landing_recovery_v12 or args.contact_readiness_v13 or args.attitude_recovery_v14:
        parser.error('--apex-event-fix-v15 requires v12 and excludes v13/v14')
    if args.simulation_only_apex_bias_controller:
        parser.error('--apex-event-fix-v15 excludes the simulation-only apex controller')
if args.attitude_recovery_v14 and not args.contact_readiness_v13:
    parser.error('--attitude-recovery-v14 requires --contact-readiness-v13')
if args.contact_readiness_v13 and not args.landing_recovery_v12:
    parser.error('--contact-readiness-v13 requires --landing-recovery-v12')
if args.landing_recovery_v12 and not (args.moving_handoff_v11 and args.path_free_stage2_v3):
    parser.error('--landing-recovery-v12 requires --moving-handoff-v11 and --path-free-stage2-v3')
if args.moving_handoff_v11 and not args.path_free_stage2_v3:
    parser.error('--moving-handoff-v11 requires --path-free-stage2-v3')
if args.path_free_stage2_v2:
    args.path_free_stage2 = True
    if args.resume_runner_state:
        parser.error('Stage2 v2 uses weights-only initialization; exact curriculum resume is unsupported')
if args.path_free_stage2_v3:
    args.path_free_stage2 = True
    if args.resume_runner_state:
        parser.error('Stage2 v3 uses weights-only initialization; exact curriculum resume is unsupported')
if args.path_free_stage2 and (args.raibert_hermite_guidance
                            or args.observation_layout != 'history_foot_v19'):
    parser.error('path-free-stage2 requires history_foot_v19 and no Raibert guidance')
if args.observation_layout == 'history_foot_v19' and args.stability_profile != 'two_circle_v18':
    parser.error('history_foot_v19 requires the two_circle_v18 task profile')
if args.raibert_hermite_guidance and args.observation_layout != 'history_foot_v19':
    parser.error('Raibert-Hermite guidance requires --observation-layout history_foot_v19')
if args.raibert_stance_time_s < 0.0 or args.raibert_velocity_gain_s < 0.0:
    parser.error('Raibert stance time and velocity gain must be non-negative')
if not 0.0 <= args.raibert_preview_blend <= 1.0:
    parser.error('raibert-preview-blend must be in [0, 1]')
if (args.raibert_hermite_guidance
        and not 0.0 <= args.raibert_max_foothold_shift_m < args.target_tolerance):
    parser.error('Raibert foothold shift must be non-negative and below target tolerance')
if args.raibert_max_horizontal_speed_m_s <= 0.0:
    parser.error('raibert-max-horizontal-speed-m-s must be positive')
if args.stability_profile != 'baseline' and (
    args.actuator_mode != 'parity' or args.legacy_v11_contract
    or args.action_parameterization != 'direct_motor'
    or args.collective_preserving_spread_limit_pwm != 0.0
):
    parser.error('stability profiles require parity/direct_motor and no extra spread limiter')
if args.comprehensive_hardware_randomization:
    # The comprehensive layer complements, rather than replaces, the existing
    # per-motor/mass/inertia hardware envelope.
    args.hardware_robust = True
source = Path(args.checkpoint).expanduser().resolve()
output = Path(args.output).expanduser().resolve()
if not source.is_file():
    parser.error(f'Missing checkpoint: {source}')
if output.exists():
    parser.error(f'Output already exists: {output}')
if min(args.num_envs, args.iterations, args.steps) <= 0:
    parser.error('num_envs, iterations and steps must be positive')
if not (0 <= args.distance_min <= args.distance_max and args.distance_max > 0.0) or args.target_tolerance <= 0:
    parser.error('Require 0 <= distance-min <= distance-max, distance-max > 0, and positive target-tolerance')
two_circle_range_values = (
    args.short_distance_min, args.short_distance_max,
    args.long_distance_min, args.long_distance_max,
)
custom_two_circle_ranges = any(value is not None for value in two_circle_range_values)
if custom_two_circle_ranges:
    if any(value is None for value in two_circle_range_values):
        parser.error('short-distance-min/max and long-distance-min/max must be supplied together')
    if not (
        0.0 <= args.short_distance_min <= args.short_distance_max
        and 0.0 <= args.long_distance_min <= args.long_distance_max
        and args.short_distance_max > 0.0
        and args.long_distance_max > 0.0
    ):
        parser.error('Each short/long distance range must satisfy 0 <= min <= max and max > 0')
short_distance_range = (
    (args.short_distance_min, args.short_distance_max)
    if custom_two_circle_ranges else (args.distance_min, args.distance_max)
)
long_distance_range = (
    (args.long_distance_min, args.long_distance_max)
    if custom_two_circle_ranges else (args.distance_min, args.distance_max)
)
if args.stability_profile == 'two_circle_v18' and not custom_two_circle_ranges:
    parser.error('two_circle_v18 requires explicit short/long distance ranges')
distance_curriculum_values = (
    args.distance_curriculum_start_short_min,
    args.distance_curriculum_start_short_max,
    args.distance_curriculum_start_long_min,
    args.distance_curriculum_start_long_max,
    args.distance_curriculum_start_turn_angle_deg,
)
if args.distance_curriculum_iterations < 0.0:
    parser.error('distance-curriculum-iterations must be non-negative')
if args.distance_curriculum_iterations == 0.0:
    if any(value is not None for value in distance_curriculum_values):
        parser.error('distance curriculum start values require positive distance-curriculum-iterations')
else:
    if args.mode != 'train':
        parser.error('distance curriculum is training-only; evaluate the final radius ranges directly')
    if not custom_two_circle_ranges:
        parser.error('distance curriculum requires explicit final short/long distance ranges')
    if any(value is None for value in distance_curriculum_values):
        parser.error('distance curriculum requires all four start ranges and the start turn angle')
    if not (
        0.0 <= args.distance_curriculum_start_short_min
        <= args.distance_curriculum_start_short_max
        and 0.0 <= args.distance_curriculum_start_long_min
        <= args.distance_curriculum_start_long_max
        and args.distance_curriculum_start_short_max > 0.0
        and args.distance_curriculum_start_long_max > 0.0
    ):
        parser.error('distance curriculum start ranges must satisfy 0 <= min <= max and max > 0')
    if not (0.0 < args.distance_curriculum_start_turn_angle_deg <= args.max_turn_angle_deg):
        parser.error('distance curriculum start turn must lie in (0, max-turn-angle-deg]')
if args.learning_rate <= 0.0:
    parser.error('learning-rate must be positive')
if args.target_height <= 0.38:
    parser.error('target-height must exceed the 0.380 m simulation landing root height')
if args.episode_length_s is not None and args.episode_length_s <= 0.0:
    parser.error('episode-length-s must be positive')
if args.successful_hops_per_episode is not None and args.successful_hops_per_episode <= 0:
    parser.error('successful-hops-per-episode must be positive')
if args.apex_tolerance is not None and not (0.01 <= args.apex_tolerance <= 0.30):
    parser.error('apex-tolerance must be in [0.01, 0.30] metres')
if args.play_motor_time_constant is not None and not (0.01 <= args.play_motor_time_constant <= 0.20):
    parser.error('play-motor-time-constant must be in [0.01, 0.20] seconds')
if args.play_thrust_scale is not None and not (0.50 <= args.play_thrust_scale <= 1.20):
    parser.error('play-thrust-scale must be in [0.50, 1.20]')
if (args.height_curriculum_start is None) != (args.height_curriculum_iterations is None):
    parser.error('height-curriculum-start and height-curriculum-iterations must be supplied together')
if args.height_curriculum_start is not None:
    if not (0.38 < args.height_curriculum_start <= args.target_height):
        parser.error('height-curriculum-start must be in (0.380, target-height]')
    if args.height_curriculum_iterations <= 0.0:
        parser.error('height-curriculum-iterations must be positive')
if not 0.0 <= args.hard_sample_probability <= 1.0:
    parser.error('hard-sample-probability must be in [0, 1]')
if not 0.0 <= args.hardware_randomization_probability <= 1.0:
    parser.error('hardware-randomization-probability must be in [0, 1]')
if ((args.hardware_randomization_probability_start is None)
        != (args.hardware_randomization_curriculum_iterations is None)):
    parser.error(
        'hardware-randomization-probability-start and '
        'hardware-randomization-curriculum-iterations must be supplied together'
    )
if args.hardware_randomization_probability_start is not None:
    if not 0.0 <= args.hardware_randomization_probability_start <= 1.0:
        parser.error('hardware-randomization-probability-start must be in [0, 1]')
    if args.hardware_randomization_curriculum_iterations <= 0.0:
        parser.error('hardware-randomization-curriculum-iterations must be positive')
if (args.comprehensive_global_thrust_scale_min is None) != (
    args.comprehensive_global_thrust_scale_max is None
):
    parser.error(
        'comprehensive-global-thrust-scale-min and '
        'comprehensive-global-thrust-scale-max must be supplied together'
    )
if args.comprehensive_global_thrust_scale_min is not None:
    if not args.comprehensive_hardware_randomization:
        parser.error('common thrust scaling overrides require --comprehensive-hardware-randomization')
    if not (
        0.60 <= args.comprehensive_global_thrust_scale_min
        <= args.comprehensive_global_thrust_scale_max <= 1.20
    ):
        parser.error('comprehensive global thrust scale range must lie in [0.60, 1.20]')
com_offset_values = (
    args.comprehensive_com_offset_x_max_mm,
    args.comprehensive_com_offset_y_max_mm,
    args.comprehensive_com_offset_z_max_mm,
)
if any(value is not None for value in com_offset_values):
    if not args.comprehensive_hardware_randomization:
        parser.error('CoM offset overrides require --comprehensive-hardware-randomization')
    if any(value is None for value in com_offset_values):
        parser.error('all three comprehensive CoM offset maxima must be supplied together')
    if any(not 0.0 <= value <= 30.0 for value in com_offset_values):
        parser.error('comprehensive CoM offset maxima must lie in [0, 30] mm')
yaw_effectiveness_values = (
    args.comprehensive_yaw_effectiveness_min,
    args.comprehensive_yaw_effectiveness_max,
)
if any(value is not None for value in yaw_effectiveness_values):
    if not args.comprehensive_hardware_randomization:
        parser.error('yaw effectiveness overrides require --comprehensive-hardware-randomization')
    if any(value is None for value in yaw_effectiveness_values):
        parser.error('both comprehensive yaw effectiveness bounds must be supplied')
    if not (0.5 <= yaw_effectiveness_values[0] <= yaw_effectiveness_values[1] <= 1.5):
        parser.error('comprehensive yaw effectiveness range must lie in [0.5, 1.5]')
if args.comprehensive_yaw_bias_torque_max_mnm is not None:
    if not args.comprehensive_hardware_randomization:
        parser.error('yaw bias torque requires --comprehensive-hardware-randomization')
    if not 0.0 <= args.comprehensive_yaw_bias_torque_max_mnm <= 10.0:
        parser.error('comprehensive yaw bias torque must lie in [0, 10] mN*m')
if args.initial_yaw_perturb_deg is not None and not 0.0 <= args.initial_yaw_perturb_deg <= 45.0:
    parser.error('initial-yaw-perturb-deg must lie in [0, 45]')
if (args.initial_angular_velocity_perturb_rad_s is not None
        and not 0.0 <= args.initial_angular_velocity_perturb_rad_s <= 5.0):
    parser.error('initial-angular-velocity-perturb-rad-s must lie in [0, 5]')
touchdown_wrench_bounds = (
    args.touchdown_roll_wrench_max_nm,
    args.touchdown_pitch_wrench_max_nm,
    args.touchdown_yaw_wrench_max_nm,
)
if args.touchdown_wrench_probability is not None:
    if not 0.0 <= args.touchdown_wrench_probability <= 1.0:
        parser.error('touchdown-wrench-probability must lie in [0, 1]')
    if any(value is None for value in touchdown_wrench_bounds):
        parser.error('touchdown-wrench-probability requires all three touchdown wrench bounds')
    if args.touchdown_wrench_window_s is None or args.touchdown_wrench_persistence is None:
        parser.error('touchdown-wrench-probability requires window and persistence values')
elif any(value is not None for value in touchdown_wrench_bounds) or any(
    value is not None for value in (
        args.touchdown_wrench_window_s, args.touchdown_wrench_persistence,
    )
):
    parser.error('touchdown wrench values require --touchdown-wrench-probability')
if any(value is not None and not 0.0 <= value <= 0.05 for value in touchdown_wrench_bounds):
    parser.error('touchdown wrench bounds must lie in [0, 0.05] N*m')
if (args.touchdown_wrench_window_s is not None
        and not 0.0 <= args.touchdown_wrench_window_s <= 0.20):
    parser.error('touchdown-wrench-window-s must lie in [0, 0.20] seconds')
if (args.touchdown_wrench_persistence is not None
        and not 0.0 <= args.touchdown_wrench_persistence < 1.0):
    parser.error('touchdown-wrench-persistence must lie in [0, 1)')
if args.touchdown_wrench_probability and args.touchdown_wrench_window_s <= 0.0:
    parser.error('nonzero touchdown-wrench-probability requires a positive window')
if args.touchdown_wrench_on_moving_long_hop and not args.touchdown_wrench_probability:
    parser.error('moving long-hop touchdown wrench requires positive touchdown-wrench-probability')
if (args.moving_long_recovery_window_s is not None
        and not 0.0 <= args.moving_long_recovery_window_s <= 0.20):
    parser.error('moving-long-recovery-window-s must lie in [0, 0.20]')
for flag, value in (
    ('moving-long-recovery-angle-deadband-deg', args.moving_long_recovery_angle_deadband_deg),
    ('moving-long-recovery-rate-deadband-rad-s', args.moving_long_recovery_rate_deadband_rad_s),
):
    if value is not None and value < 0.0:
        parser.error(f'{flag} must be non-negative')
for flag, value in (
    ('moving-long-recovery-angle-penalty-scale', args.moving_long_recovery_angle_penalty_scale),
    ('moving-long-recovery-rate-penalty-scale', args.moving_long_recovery_rate_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
if (any(value is not None and value < 0.0 for value in (
        args.moving_long_recovery_angle_penalty_scale,
        args.moving_long_recovery_rate_penalty_scale,
    )) and not args.moving_long_recovery_window_s):
    parser.error('moving long-hop recovery penalties require a positive window')
for flag, value in (
    ('airborne-attitude-penalty-scale', args.airborne_attitude_penalty_scale),
    ('airborne-angvel-xy-penalty-scale', args.airborne_angvel_xy_penalty_scale),
    ('airborne-yaw-penalty-scale', args.airborne_yaw_penalty_scale),
    ('airborne-yaw-rate-penalty-scale', args.airborne_yaw_rate_penalty_scale),
    ('airborne-action-spread-penalty-scale', args.airborne_action_spread_penalty_scale),
    ('airborne-cross-track-tilt-penalty-scale', args.airborne_cross_track_tilt_penalty_scale),
    ('airborne-cross-track-velocity-penalty-scale', args.airborne_cross_track_velocity_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
for flag, value, maximum in (
    ('airborne-tilt-penalty-start', args.airborne_tilt_penalty_start, 0.5),
    ('airborne-angvel-xy-penalty-start', args.airborne_angvel_xy_penalty_start, 10.0),
    ('airborne-yaw-deadband-deg', args.airborne_yaw_deadband_deg, 90.0),
    ('airborne-yaw-rate-penalty-start', args.airborne_yaw_rate_penalty_start, 10.0),
    ('airborne-action-spread-penalty-start', args.airborne_action_spread_penalty_start, 1.0),
    ('airborne-cross-track-tilt-penalty-start', args.airborne_cross_track_tilt_penalty_start, 1.0),
    ('airborne-cross-track-velocity-penalty-start-mps', args.airborne_cross_track_velocity_penalty_start_mps, 10.0),
):
    if value is not None and not 0.0 <= value <= maximum:
        parser.error(f'{flag} must lie in [0, {maximum:g}]')
if not 0.0 < args.max_turn_angle_deg <= 180.0:
    parser.error('max-turn-angle-deg must be in (0, 180]')
if not 0.0 <= args.zero_hop_probability <= 1.0:
    parser.error('zero-hop-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_probability <= 1.0:
    parser.error('reverse-turn-probability must be in [0, 1]')
if not 0.0 <= args.reverse_turn_halfwidth_deg <= 180.0:
    parser.error('reverse-turn-halfwidth-deg must be in [0, 180]')
if args.fixed_initial_heading_deg is not None and not math.isfinite(args.fixed_initial_heading_deg):
    parser.error('fixed-initial-heading-deg must be finite')
if args.reverse_turn_probability > 0.0 and (
    args.max_turn_angle_deg < 180.0 - args.reverse_turn_halfwidth_deg
):
    parser.error('max-turn-angle-deg must include the configured reverse-turn band')
if args.direction_curriculum_iterations < 0.0:
    parser.error('direction-curriculum-iterations must be non-negative')
direction_curriculum_values = (
    args.direction_curriculum_start_turn_angle_deg,
    args.direction_curriculum_start_zero_hop_probability,
)
if args.direction_curriculum_iterations == 0.0:
    if any(value is not None for value in direction_curriculum_values):
        parser.error('direction curriculum start values require positive direction-curriculum-iterations')
else:
    if args.direction_curriculum_start_turn_angle_deg is None:
        parser.error('direction curriculum requires direction-curriculum-start-turn-angle-deg')
    if not 0.0 < args.direction_curriculum_start_turn_angle_deg <= args.max_turn_angle_deg:
        parser.error('direction curriculum start turn must lie in (0, max-turn-angle-deg]')
    if args.direction_curriculum_start_zero_hop_probability is None:
        args.direction_curriculum_start_zero_hop_probability = 0.0
    if not (0.0 <= args.direction_curriculum_start_zero_hop_probability
            <= args.zero_hop_probability):
        parser.error('direction curriculum zero-hop start must lie in [0, zero-hop-probability]')
if args.landing_precision_width is not None and args.landing_precision_width <= 0.0:
    parser.error('landing-precision-width must be positive')
if args.observation_noise_std is not None and args.observation_noise_std < 0.0:
    parser.error('observation-noise-std must be non-negative')
if args.stationary_hold_radius is not None and args.stationary_hold_radius <= 0.0:
    parser.error('stationary-hold-radius must be positive')
if (args.stationary_pitch_deadband_deg is not None
        and not 0.0 <= args.stationary_pitch_deadband_deg <= 30.0):
    parser.error('stationary-pitch-deadband-deg must be in [0, 30]')
if (args.stationary_pitch_rate_deadband_rad_s is not None
        and not 0.0 <= args.stationary_pitch_rate_deadband_rad_s <= 10.0):
    parser.error('stationary-pitch-rate-deadband-rad-s must be in [0, 10]')
if (args.stationary_airborne_differential_action_rate_deadband is not None
        and not 0.0 <= args.stationary_airborne_differential_action_rate_deadband <= 2.0):
    parser.error('stationary airborne differential action-rate deadband must be in [0, 2]')
for flag, value in (
    ('stationary-pitch-penalty-scale', args.stationary_pitch_penalty_scale),
    ('stationary-pitch-rate-penalty-scale', args.stationary_pitch_rate_penalty_scale),
    ('stationary-airborne-differential-action-rate-penalty-scale',
     args.stationary_airborne_differential_action_rate_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
if (args.stationary_contact_recovery_window_s is not None
        and not 0.0 <= args.stationary_contact_recovery_window_s <= 0.50):
    parser.error('stationary-contact-recovery-window-s must be in [0, 0.50] seconds')
for flag, value in (
    ('stationary-contact-recovery-spread-deadband', args.stationary_contact_recovery_spread_deadband),
    ('stationary-contact-recovery-delta-deadband', args.stationary_contact_recovery_delta_deadband),
):
    if value is not None and not 0.0 <= value <= 1.0:
        parser.error(f'{flag} must be in [0, 1]')
for flag, value in (
    ('stationary-contact-recovery-spread-penalty-scale', args.stationary_contact_recovery_spread_penalty_scale),
    ('stationary-contact-recovery-delta-penalty-scale', args.stationary_contact_recovery_delta_penalty_scale),
):
    if value is not None and value > 0.0:
        parser.error(f'{flag} must be non-positive')
if args.stationary_apex_width is not None and args.stationary_apex_width <= 0.0:
    parser.error('stationary-apex-width must be positive')
if args.stationary_apex_progress_width is not None and args.stationary_apex_progress_width <= 0.0:
    parser.error('stationary-apex-progress-width must be positive')
if (args.height_progress_reward_scale is not None
        and not 0.0 <= args.height_progress_reward_scale <= 100.0):
    parser.error('height-progress-reward-scale must be in [0, 100]')
if args.stationary_ascent_support_scale is not None and args.stationary_ascent_support_scale < 0.0:
    parser.error('stationary-ascent-support-scale must be non-negative')
if args.stationary_ascent_collective_scale is not None and args.stationary_ascent_collective_scale < 0.0:
    parser.error('stationary-ascent-collective-scale must be non-negative')
if args.stationary_predicted_apex_width is not None and args.stationary_predicted_apex_width <= 0.0:
    parser.error('stationary-predicted-apex-width must be positive')
if args.ascent_predicted_apex_width is not None and args.ascent_predicted_apex_width <= 0.0:
    parser.error('ascent-predicted-apex-width must be positive')
if (args.stationary_predicted_apex_reward_scale is not None
        and args.stationary_predicted_apex_reward_scale < 0.0):
    parser.error('stationary-predicted-apex-reward-scale must be non-negative')
if (args.ascent_predicted_apex_reward_scale is not None
        and args.ascent_predicted_apex_reward_scale < 0.0):
    parser.error('ascent-predicted-apex-reward-scale must be non-negative')
if args.apex_event_height_width is not None and args.apex_event_height_width <= 0.0:
    parser.error('apex-event-height-width must be positive')
if (args.apex_event_height_reward_scale is not None
        and args.apex_event_height_reward_scale < 0.0):
    parser.error('apex-event-height-reward-scale must be non-negative')
if (args.apex_event_shortfall_penalty_scale is not None
        and args.apex_event_shortfall_penalty_scale > 0.0):
    parser.error('apex-event-shortfall-penalty-scale must be non-positive')
if args.legacy_v11_contract and args.actuator_mode != 'baseline':
    parser.error('--legacy-v11-contract requires --actuator-mode baseline')
if args.resume_runner_state and args.mode != 'train':
    parser.error('--resume-runner-state is only valid in train mode')
if args.action_parameterization == 'collective_residual_v1' and args.actuator_mode != 'baseline':
    parser.error('collective_residual_v1 requires --actuator-mode baseline')
if not 0.0 <= args.collective_preserving_spread_limit_pwm <= 1000.0:
    parser.error('collective-preserving-spread-limit-pwm must be in [0, 1000]')
if args.simulation_only_phase_collective_baseline:
    if args.phase_contact_collective_floor_pwm is None or args.phase_ascent_collective_floor_pwm is None:
        parser.error('simulation-only phase baseline requires both phase collective floor PWM values')
    if not (0.0 <= args.phase_contact_collective_floor_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_floor_pwm <= 1000.0):
        parser.error('phase collective floor PWM values must be in [0, 1000]')
elif (args.phase_contact_collective_floor_pwm is not None
      or args.phase_ascent_collective_floor_pwm is not None):
    parser.error('phase collective floor PWM values require --simulation-only-phase-collective-baseline')
if args.simulation_only_phase_collective_baseline and args.simulation_only_phase_collective_controller:
    parser.error('choose either the phase collective floor or the fixed collective controller, not both')
controller_values = (
    args.phase_contact_collective_pwm,
    args.phase_ascent_collective_pwm,
    args.phase_descent_collective_pwm,
    args.phase_contact_differential_limit_pwm,
    args.phase_ascent_differential_limit_pwm,
    args.phase_descent_differential_limit_pwm,
)
if args.simulation_only_phase_collective_controller:
    if any(value is None for value in controller_values):
        parser.error('phase collective controller requires both collective and both differential-limit PWM values')
    if not (0.0 <= args.phase_contact_collective_pwm <= 1000.0
            and 0.0 <= args.phase_ascent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_descent_collective_pwm <= 1000.0
            and 0.0 <= args.phase_contact_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_ascent_differential_limit_pwm <= 500.0
            and 0.0 <= args.phase_descent_differential_limit_pwm <= 500.0):
        parser.error('phase controller PWM values must be collective [0,1000], differential limit [0,500]')
elif any(value is not None for value in controller_values):
    parser.error('phase controller PWM values require --simulation-only-phase-collective-controller')
apex_bias_values = (args.apex_bias_initial_pwm, args.apex_bias_gain_pwm_per_m, args.apex_bias_max_pwm)
if args.simulation_only_apex_bias_controller:
    if any(value is None for value in apex_bias_values):
        parser.error('apex bias controller requires initial, gain, and max PWM values')
    if not (0.0 <= args.apex_bias_initial_pwm <= args.apex_bias_max_pwm <= 400.0
            and 0.0 < args.apex_bias_gain_pwm_per_m <= 2000.0):
        parser.error('apex bias requires 0 <= initial <= max <= 400 and gain in (0, 2000]')
    if args.apex_bias_feedback == 'liftoff_vz':
        if args.apex_bias_liftoff_vz_target is None or args.apex_bias_liftoff_vz_target <= 0.0:
            parser.error('liftoff_vz feedback requires a positive --apex-bias-liftoff-vz-target')
    elif args.apex_bias_liftoff_vz_target is not None:
        parser.error('apex-bias-liftoff-vz-target requires --apex-bias-feedback liftoff_vz')
elif any(value is not None for value in apex_bias_values):
    parser.error('apex bias values require --simulation-only-apex-bias-controller')
elif args.apex_bias_liftoff_vz_target is not None:
    parser.error('apex-bias-liftoff-vz-target requires --simulation-only-apex-bias-controller')
if args.hard_sample_probability > 0.0 and not (
    args.distance_min < args.hard_distance_min < args.distance_max
):
    parser.error('hard-distance-min must lie strictly inside the distance range')
if custom_two_circle_ranges and args.hard_sample_probability > 0.0:
    parser.error(
        'Alternating two-circle ranges currently require hard-sample-probability 0; '
        'one hard-distance-min cannot safely split two independent radius ranges'
    )
if args.mode == 'eval' and args.hard_sample_probability != 0.0:
    parser.error('Evaluation must use uniform distances; set hard-sample-probability to 0')
if args.visualize and args.mode != 'eval':
    parser.error('--visualize is only supported in eval mode')
if args.control_diagnostics and (args.mode != 'eval' or args.num_envs != 1):
    parser.error('--control-diagnostics requires eval mode with --num_envs 1')
if args.failure_trace and args.mode != 'eval':
    parser.error('--failure-trace requires eval mode')
if args.takeoff_velocity_audit and not args.failure_trace:
    parser.error('--takeoff-velocity-audit requires --failure-trace')
if args.contact_phase_audit and not args.takeoff_velocity_audit:
    parser.error('--contact-phase-audit requires --takeoff-velocity-audit')
if args.closed_loop_p2_mode is not None:
    if args.mode != 'eval' or not args.failure_trace or not args.takeoff_velocity_audit:
        parser.error('--closed-loop-p2-mode requires eval with failure trace and takeoff audit')
    if args.closed_loop_p2_env_id is None or args.closed_loop_p2_step is None:
        parser.error('--closed-loop-p2-mode requires environment ID and step')
    if not 0 <= args.closed_loop_p2_env_id < args.num_envs or not 1 <= args.closed_loop_p2_step < args.steps:
        parser.error('P2 branch environment or step out of range')
elif any(value is not None for value in (args.closed_loop_p2_env_id,
                                         args.closed_loop_p2_step,
                                         args.closed_loop_p2_expected_sha256)):
    parser.error('P2 branch options require --closed-loop-p2-mode')
if args.next_hop_probe_env_ids and args.mode != 'eval':
    parser.error('--next-hop-probe-env-ids requires eval mode')
if args.next_hop_probe_stride < 1:
    parser.error('--next-hop-probe-stride must be positive')
probe_env_ids = []
if args.next_hop_probe_env_ids:
    try:
        probe_env_ids = [int(part) for part in args.next_hop_probe_env_ids.split(',')]
    except ValueError:
        parser.error('--next-hop-probe-env-ids must contain comma-separated integers')
    if not probe_env_ids or len(set(probe_env_ids)) != len(probe_env_ids) or any(
            env_id < 0 or env_id >= args.num_envs for env_id in probe_env_ids):
        parser.error('--next-hop-probe-env-ids must be distinct and within num_envs')
    if args.observation_layout != 'history_foot_v19':
        parser.error('--next-hop-probe-env-ids requires history_foot_v19')
if not 0 < args.failure_trace_window_s <= 8:
    parser.error('--failure-trace-window-s must be in (0, 8]')
if args.failure_trace_window_s != 0.5 and not args.failure_trace:
    parser.error('--failure-trace-window-s requires --failure-trace')
app = AppLauncher(args).app

import os
import random
import sys
import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat
import isaaclab

sys.path.insert(0, os.path.join(os.path.dirname(isaaclab.__file__), 'source', 'isaaclab_rl'))
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.spatial_tracking_env import (
    SpatialTrackingEnv,
    SpatialTrackingPhysical1mEnvCfg,
)
from Quadhopper_Planner_Random.stationary_corridor_v17_env import (
    StationaryCorridorV17Env,
    StationaryCorridorV17EnvCfg,
)
from Quadhopper_Planner_Random.spatial_path import expand_policy_inputs

ROOT = Path(__file__).resolve().parent
CONTRACT = {
    'version': 12,
    'protocol': 'physical_1m_actuator_parity_stationary_precision',
    'source': str(source),
    'obs_dim': 52,
    'actions': 4,
    'preview': True,
    'distance_range_m': [args.distance_min, args.distance_max],
    'max_turn_angle_deg': args.max_turn_angle_deg,
    'zero_hop_probability': args.zero_hop_probability,
    'reverse_turn_probability': args.reverse_turn_probability,
    'reverse_turn_halfwidth_deg': args.reverse_turn_halfwidth_deg,
    'fixed_initial_heading_deg': args.fixed_initial_heading_deg,
    'direction_curriculum': {
        'iterations': args.direction_curriculum_iterations,
        'start_max_turn_angle_deg': args.direction_curriculum_start_turn_angle_deg,
        'end_max_turn_angle_deg': args.max_turn_angle_deg,
        'start_zero_hop_probability': args.direction_curriculum_start_zero_hop_probability,
        'end_zero_hop_probability': args.zero_hop_probability,
        'end_reverse_turn_probability': args.reverse_turn_probability,
        'reverse_turn_halfwidth_deg': args.reverse_turn_halfwidth_deg,
        'reverse_activation': 'after current maximum turn contains the full reverse band',
    },
    'distance_curriculum': {
        'iterations': args.distance_curriculum_iterations,
        'start_short_hop_radius_m': (
            None if args.distance_curriculum_iterations == 0.0 else [
                args.distance_curriculum_start_short_min,
                args.distance_curriculum_start_short_max,
            ]
        ),
        'end_short_hop_radius_m': list(short_distance_range),
        'start_long_hop_radius_m': (
            None if args.distance_curriculum_iterations == 0.0 else [
                args.distance_curriculum_start_long_min,
                args.distance_curriculum_start_long_max,
            ]
        ),
        'end_long_hop_radius_m': list(long_distance_range),
        'start_max_turn_angle_deg': args.distance_curriculum_start_turn_angle_deg,
        'end_max_turn_angle_deg': args.max_turn_angle_deg,
        'schedule': 'linear by PPO update; synchronized when a route command is sampled',
    },
    'distance_sampling': (
        {
            'mode': 'weighted_two_interval',
            'replay_range_m': [args.distance_min, args.hard_distance_min],
            'hard_range_m': [args.hard_distance_min, args.distance_max],
            'hard_probability': args.hard_sample_probability,
        }
        if args.hard_sample_probability > 0.0
        else {'mode': 'uniform', 'range_m': [args.distance_min, args.distance_max]}
    ),
    'target_tolerance_m': args.target_tolerance,
    'waypoint_observation_scale_m': 0.20,
    'path_corridor_m': 0.08,
    'path_lookahead_m': 0.10,
    'spatial_progress_reward_per_hop': 12.0,
    'path_tracking_reward_per_second': 16.0,
    'path_tracking_width_m': 0.08,
    # XY progress weights are populated from the actual cfg below, rather
    # than recording a number that can disagree with inherited defaults.
    'xy_progress_form': 'signed_distance_difference_m',
    'xy_progress_gating': 'same_target_consecutive_airborne_steps',
    'two_hop_pair_definition': 'second_phase_touchdown_and_two_consecutive_5cm_hits',
    'target_hit_reward': 120.0,
    'landing_precision_reward': 60.0,
    'landing_error_penalty': -120.0,
    'landing_lateral_error_penalty': -60.0,
    'projected_landing_reward_per_second': 30.0,
    'projected_landing_error_penalty_per_second': -30.0,
    'stable_bonus_profile': {
        'survival': 2.0, 'distance_to_z': 1.0,
        'energy_tracking': 3.0, 'apex': 3.0,
    },
    'path_penalty_max_per_second': 1.0,
    'landing_hint_penalty_max_per_second': 0.0,
    'target_hit_requires_minimum_apex': False,
    'target_hit_requires_apex_tolerance': False,
    'landing_precision_gated_by_apex': False,
    'start_from_random_drop': not args.disable_initial_drop,
    'initial_drop_height_m': [0.65, 1.05] if not args.disable_initial_drop else None,
    'guidance_clock': False,
    'episode_length_s': args.episode_length_s or 15.0,
    'appended_observations': ['carrot_error_body_xyz_m', 'path_tangent_body_xyz', 'landing_up_hint_body_xyz'],
    'learning_rate': args.learning_rate,
    'schedule': 'fixed',
    'std': 0.04,
    'motor_tc_s': args.play_motor_time_constant or 0.0674,
    'nominal_thrust_scale': args.play_thrust_scale or 1.0,
    'policy_target_root_z_m': args.target_height,
    'physical_ground_root_z_m': 0.285,
    # 0.380 m is a spatial-planner landing reference, not a conversion
    # between simulator and Vicon root-Z coordinates.  The native v11
    # observation and physical target use the same root-Z command.
    'spatial_planner_landing_reference_z_m': 0.380,
    'physical_target_root_z_m': args.target_height,
    'action_delay': 'zero_delay_is_newest_command; optional delays count prior 10ms commands',
    'action_history': 'five_executed_post_limit_motor_actions_oldest_to_newest',
    'deployment_action_shaping': {
        'version': 'height_spread_yaw_v1',
        'max_pwm_spread': 600.0,
        'airborne_yaw_diagonal_pwm_diff': 140.0,
        'grounded_yaw_diagonal_pwm_diff': 220.0,
        'constraint_projection_iterations': 2,
        'pwm_quantization': 'floor_to_integer',
    },
    'seed': args.seed,
}
if args.stability_profile == 'takeoff_v9':
    CONTRACT.update(
        version=15,
        protocol='physical_1m_takeoff_stability_yaw220',
        v12_launcher_compatible=False,
    )
    CONTRACT['deployment_action_shaping']['airborne_yaw_diagonal_pwm_diff'] = 220.0
elif args.stability_profile == 'launch_robust_v10':
    CONTRACT.update(
        version=12,
        protocol='physical_1m_launch_robust_v10_yaw140',
        v12_launcher_compatible=True,
    )
elif args.stability_profile == 'yaw_event_v11':
    CONTRACT.update(
        version=12,
        protocol='physical_1m_yaw_event_v11_yaw140',
        v12_launcher_compatible=True,
    )
elif args.stability_profile == 'touchdown_robust_v12':
    CONTRACT.update(
        version=12,
        protocol='physical_1m_touchdown_robust_v12_yaw140',
        v12_launcher_compatible=True,
    )
elif args.stability_profile == 'height_robust_v13':
    CONTRACT.update(
        version=12,
        protocol='physical_1m_height_robust_v13_yaw140',
        v12_launcher_compatible=True,
    )
elif args.stability_profile == 'cross_hop_v14':
    CONTRACT.update(
        version=12,
        protocol='physical_1m_cross_hop_robust_v14_yaw140',
        v12_launcher_compatible=True,
    )
elif args.stability_profile == 'cross_hop_v15':
    CONTRACT.update(
        version=16,
        protocol='physical_1m_cross_hop_recovery_v15_yaw140_launch450',
        v12_launcher_compatible=False,
    )
    CONTRACT['deployment_action_shaping'].update({
        'version': 'height_spread_yaw_launch_recovery_v16',
        'launch_recovery_window_s': 0.16,
        'launch_recovery_max_pwm_spread': 450.0,
    })
elif args.stability_profile == 'cross_hop_v15_1':
    CONTRACT.update(
        version=17,
        protocol='physical_1m_cross_hop_recovery_v15_1_yaw140_ramp450to600',
        v12_launcher_compatible=False,
    )
    CONTRACT['deployment_action_shaping'].update({
        'version': 'height_spread_yaw_launch_recovery_ramp_v17',
        'launch_recovery_window_s': 0.16,
        'launch_recovery_hold_s': 0.08,
        'launch_recovery_initial_max_pwm_spread': 450.0,
        'launch_recovery_final_max_pwm_spread': 600.0,
    })
elif args.stability_profile == 'cross_hop_v16':
    # v16 changes training-only persistent uncertainty and rewards.  It uses
    # the accepted v17 deployed actuator envelope exactly, so exports retain
    # the same versioned launcher requirements as v15.1.
    CONTRACT.update(
        version=17,
        protocol='physical_1m_cross_hop_v16_persistent_motor_yaw_touchdown_v17',
        v12_launcher_compatible=False,
    )
    CONTRACT['deployment_action_shaping'].update({
        'version': 'height_spread_yaw_launch_recovery_ramp_v17',
        'launch_recovery_window_s': 0.16,
        'launch_recovery_hold_s': 0.08,
        'launch_recovery_initial_max_pwm_spread': 450.0,
        'launch_recovery_final_max_pwm_spread': 600.0,
    })
elif args.stability_profile == 'stationary_corridor_v17':
    # v17 stationary corridor: 15 cm tolerance + attitude priority + slow recenter
    # Uses the v17 actuator envelope (80ms/450PWM + 80ms/600PWM ramp)
    CONTRACT.update(
        version=17,
        protocol='physical_1m_stationary_corridor_v17_yaw140_ramp450to600',
        v12_launcher_compatible=False,
    )
    CONTRACT['deployment_action_shaping'].update({
        'version': 'height_spread_yaw_launch_recovery_ramp_v17',
        'launch_recovery_window_s': 0.16,
        'launch_recovery_hold_s': 0.08,
        'launch_recovery_initial_max_pwm_spread': 450.0,
        'launch_recovery_final_max_pwm_spread': 600.0,
    })
elif args.stability_profile == 'two_circle_v18':
    # This is a task-route refinement of the accepted v15.1 deployment
    # actuator envelope. It deliberately changes neither the 52-D policy
    # interface nor the calibrated plant; only the rolling waypoint geometry
    # is versioned separately so a checkpoint cannot be evaluated as v15.1.
    if not custom_two_circle_ranges:
        raise ValueError('two_circle_v18 requires explicit short/long distance ranges')
    CONTRACT.update(
        version=17,
        protocol='physical_1m_two_circle_v18_yaw140_ramp450to600',
        v12_launcher_compatible=False,
        two_circle_route={
            'short_hop_radius_m': list(short_distance_range),
            'long_hop_radius_m': list(long_distance_range),
            'queue': ('rolling current/next targets; touchdown advances P_t '
                      'to P_(t+1) without an RNN/history reset'),
            'alternation': 'short,long,short,long',
        },
    )
    CONTRACT['deployment_action_shaping'].update({
        'version': 'height_spread_yaw_launch_recovery_ramp_v17',
        'launch_recovery_window_s': 0.16,
        'launch_recovery_hold_s': 0.08,
        'launch_recovery_initial_max_pwm_spread': 450.0,
        'launch_recovery_final_max_pwm_spread': 600.0,
    })
if args.legacy_v11_contract:
    # The accepted 40_v1/mixed family observes raw actions and applies the
    # five-sample legacy delay. Do not label that stream as v12's executed and
    # shaped motor-command history.
    CONTRACT.update(
        version=11,
        protocol='legacy_v11_physical_1m_height_completion',
        action_delay='legacy_indexing_setting_0; five-sample raw-action history selects oldest',
        action_history='five_raw_policy_actions_oldest_to_newest',
        deployment_action_shaping='none',
        v12_launcher_compatible=False,
    )
elif args.actuator_mode == 'zero_delay_raw_history':
    CONTRACT.update(
        version=14,
        protocol='direct_motor_zero_delay_raw_history',
        action_delay='zero_delay_selects_newest_current_10ms_command',
        action_history='five_raw_policy_actions_oldest_to_newest',
        deployment_action_shaping='none',
        v12_launcher_compatible=False,
    )
if args.action_parameterization == 'collective_residual_v1':
    CONTRACT.update(
        version=13,
        protocol='collective_residual_v1_physical_1m',
        action_history='five_raw_collective_residual_actions_oldest_to_newest',
        deployment_action_shaping='none',
        action_parameterization='collective_residual_v1',
        v12_launcher_compatible=False,
    )


if args.observation_layout == 'history_foot_v19':
    from Quadhopper_Planner_Random.history_foot import OBS_DIM, PROTOCOL, expand_from_spatial52
    from Quadhopper_Planner_Random.history_foot_env import HistoryFootEnv, HistoryFootEnvCfg
    CONTRACT.update(
        version=19, protocol=PROTOCOL, obs_dim=OBS_DIM,
        observation_layout=args.observation_layout,
        observation_history=dict(past_frames=5, current_frame=True, frame_dim=56,
                                 order='current,t-1,t-2,t-3,t-4,t-5', sample_period_s=0.01,
                                 reset='repeat_first_frame; reset_only_on_episode_reset'),
        foot_observation=dict(offset_xy_slice=[52,54], target_error_xy_slice=[54,56],
                              frame='world', units='m', error_clip=[-1,1],
                              measurement='existing noisy quaternion and spring_position',
                              landing_reward_reference='body_link_origin_unchanged'),
        deployment_action_contract_version=17,
    )
    if args.raibert_hermite_guidance:
        from Quadhopper_Planner_Random.raibert_hermite import RAIBERT_HERMITE_PROTOCOL
        CONTRACT.update(
            version=20,
            protocol=RAIBERT_HERMITE_PROTOCOL,
            guidance_planner={
                'type': 'goal_anchored_raibert_foothold_plus_ballistic_hermite',
                'scoring_target': 'unmodified_commanded_circle_centre',
                'stance_time_s': args.raibert_stance_time_s,
                'velocity_error_gain_s': args.raibert_velocity_gain_s,
                'max_foothold_shift_m': args.raibert_max_foothold_shift_m,
                'touchdown_next_hop_preview_blend': args.raibert_preview_blend,
                'max_horizontal_speed_m_s': args.raibert_max_horizontal_speed_m_s,
                'full_trajectory_generated_by_raibert': False,
            },
        )


if args.path_free_stage2:
    from Quadhopper_Planner_Random.path_free_stage2 import (
        PROTOCOL as PATH_FREE_PROTOCOL, VERSION as PATH_FREE_VERSION,
        SETTINGS as PATH_FREE_SETTINGS, apply_settings as apply_path_free_settings,
        PROTOCOL as PATH_FREE_V1_PROTOCOL,
        PROTOCOL_V2 as PATH_FREE_V2_PROTOCOL,
    )
    from Quadhopper_Planner_Random.path_free_stage2_env import PathFreeStage2Env
    if args.path_free_stage2_v2:
        from Quadhopper_Planner_Random.path_free_stage2 import (
            PROTOCOL_V2 as PATH_FREE_PROTOCOL, VERSION_V2 as PATH_FREE_VERSION,
            SETTINGS_V2 as PATH_FREE_SETTINGS,
        )
        from Quadhopper_Planner_Random.path_free_stage2_v2_env import (
            PathFreeStage2V2Env as PathFreeStage2Env,
        )
    if args.path_free_stage2_v3:
        from Quadhopper_Planner_Random.path_free_stage2 import (
            PROTOCOL_V3 as PATH_FREE_PROTOCOL, VERSION_V3 as PATH_FREE_VERSION,
            SETTINGS_V3 as PATH_FREE_SETTINGS,
        )
        from Quadhopper_Planner_Random.path_free_stage2_v3_env import (
            PathFreeStage2V3Env as PathFreeStage2Env,
        )
    if args.v3_accounting_fix:
        from Quadhopper_Planner_Random.path_free_stage2_v3_accounting_env import (
            PathFreeStage2V3AccountingEnv as PathFreeStage2Env,
        )
        CONTRACT['v3_accounting_fix'] = True
        CONTRACT['v3_accounting_fix_settings'] = {
            'apex_event': 'owned_velocity_snapshot_once_per_hop',
            'terminal_penalty': 'physical_death_counter_delta',
            'base_task': 'path_free_stage2_v3',
        }
    if args.foot_pair_v33:
        from Quadhopper_Planner_Random.foot_pair_reward_v33 import (
            FOOT_RADIUS_M, TILT_WIDTH_RAD, UPRIGHT_WEIGHT, PAIR_XY_WIDTH_M,
        )
        from Quadhopper_Planner_Random.path_free_stage2_v33_env import (
            PathFreeStage2V33Env as PathFreeStage2Env,
        )
        CONTRACT['foot_pair_v33'] = {
            'replaces': 'root_xy_consecutive_pair_bonus_only',
            'event': 'second_hop_touchdown',
            'first_and_second_hop': 'foot_error_below_radius_and_apex_at_or_above_target',
            'foot_radius_m': FOOT_RADIUS_M,
            'upright_tilt_width_rad': TILT_WIDTH_RAD,
            'upright_quality_weight': UPRIGHT_WEIGHT,
            'pair_reward_scale': PATH_FREE_SETTINGS['two_hop_pair_reward_scale'],
            'xy_quality_width_m': PAIR_XY_WIDTH_M,
        }
    if args.two_hop_episode_v38:
        from Quadhopper_Planner_Random.two_hop_episode_v38 import (
            TwoHopEpisodeV38Env as PathFreeStage2Env,
            SETTLE_TIME_S, SETTLE_TILT_DEG, SETTLE_XY_RATE_RAD_S,
        )
        CONTRACT['two_hop_episode_v38'] = {
            'hops': 'two_attempted_touchdowns_after_initial_drop',
            'scenario_fractions': {'uniform': 0.50, 'reverse': 0.25, 'zero_patterns': 0.25},
            'zero_patterns': ['zero_zero', 'zero_moving', 'moving_zero'],
            'settle_time_s': SETTLE_TIME_S,
            'settle_tilt_deg': SETTLE_TILT_DEG,
            'settle_xy_rate_rad_s': SETTLE_XY_RATE_RAD_S,
        }
    if args.hardware_severity_curriculum:
        from Quadhopper_Planner_Random.path_free_stage2_v3_severity_env import (
            PathFreeStage2V3SeverityEnv as PathFreeStage2Env,
        )
        CONTRACT['hardware_severity_curriculum'] = {
            'hard_episode_probability': args.hardware_randomization_probability,
            'severity_stages': [0.5, 0.75, 1.0],
            'updates_per_partial_stage': args.hardware_severity_stage_iterations,
            'applies_at': 'hard_episode_reset_only',
        }
    if args.route_recovery_mode:
        from Quadhopper_Planner_Random.path_free_stage2_v3_route_eval_env import (
            PathFreeStage2V3RouteEvalEnv as PathFreeStage2Env,
        )
        CONTRACT['route_recovery_eval'] = {
            'mode': args.route_recovery_mode,
            'severe_miss_foot_error_m': args.route_recovery_severe_miss_m,
            'policy': 'frozen_v3_accounting_checkpoint',
            'hit_definitions': 'task_hit_and_foot_10cm_and_strict_1m',
        }
    if args.diagnostic_ascent_pulse_pwm is not None:
        from Quadhopper_Planner_Random.ascent_pulse_v26 import PULSE_DURATION_S
        from Quadhopper_Planner_Random.path_free_stage2_v3_pulse_eval_env import (
            PathFreeStage2V3PulseEvalEnv as PathFreeStage2Env,
        )
        CONTRACT['diagnostic_ascent_pulse'] = {
            'pwm': args.diagnostic_ascent_pulse_pwm,
            'duration_s': PULSE_DURATION_S,
            'gate': 'airborne_positive_vz_and_cycle_time_since_liftoff',
        }
    if args.landing_recovery_v12:
        from Quadhopper_Planner_Random.path_free_stage2_v12_env import (
            PathFreeStage2V12Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.landing_recovery_v12 import SETTINGS_V12
        CONTRACT['landing_recovery_v12'] = True
        CONTRACT['landing_recovery_v12_settings'] = dict(SETTINGS_V12)
    if args.contact_readiness_v13:
        from Quadhopper_Planner_Random.path_free_stage2_v13_env import (
            PathFreeStage2V13Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.contact_readiness_v13 import SETTINGS_V13
        CONTRACT['contact_readiness_v13'] = True
        CONTRACT['contact_readiness_v13_settings'] = dict(SETTINGS_V13)
    if args.attitude_recovery_v14:
        from Quadhopper_Planner_Random.path_free_stage2_v14_env import (
            PathFreeStage2V14Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.attitude_recovery_v14 import SETTINGS_V14
        CONTRACT['attitude_recovery_v14'] = True
        CONTRACT['attitude_recovery_v14_settings'] = dict(SETTINGS_V14)
    if args.apex_event_fix_v15:
        from Quadhopper_Planner_Random.path_free_stage2_v15_env import (
            PathFreeStage2V15Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.apex_event_fix_v15 import SETTINGS_V15
        CONTRACT['apex_event_fix_v15'] = True
        CONTRACT['apex_event_fix_v15_settings'] = dict(SETTINGS_V15)
    if args.terminal_penalty_fix_v16:
        from Quadhopper_Planner_Random.path_free_stage2_v16_env import (
            PathFreeStage2V16Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.terminal_penalty_fix_v16 import SETTINGS_V16
        CONTRACT['terminal_penalty_fix_v16'] = True
        CONTRACT['terminal_penalty_fix_v16_settings'] = dict(SETTINGS_V16)
    if args.deferred_landing_v17:
        from Quadhopper_Planner_Random.path_free_stage2_v17_env import (
            PathFreeStage2V17Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.deferred_landing_v17 import SETTINGS_V17
        CONTRACT['deferred_landing_v17'] = True
        CONTRACT['deferred_landing_v17_settings'] = dict(SETTINGS_V17)
    if args.low_hop_recovery_v18:
        from Quadhopper_Planner_Random.path_free_stage2_v18_env import (
            PathFreeStage2V18Env as PathFreeStage2Env,
        )
        from Quadhopper_Planner_Random.low_hop_recovery_v18 import SETTINGS_V18
        CONTRACT['low_hop_recovery_v18'] = True
        CONTRACT['low_hop_recovery_v18_settings'] = dict(SETTINGS_V18)
    CONTRACT.update(
        version=PATH_FREE_VERSION, protocol=PATH_FREE_PROTOCOL,
        guidance_planner={'type': 'none'},
        path_free_stage2=dict(PATH_FREE_SETTINGS),
        appended_observations=['nine_reserved_zeros_per_frame'],
        path_tracking_reward_per_second=0.0, spatial_progress_reward_per_hop=0.0,
        path_penalty_max_per_second=0.0,
        projected_landing_reward_per_second=0.0,
        projected_landing_error_penalty_per_second=0.0,
        path_guidance_observation='frame[43:52]=0 for all six history frames',
    )


if args.sysid_20261011_plant:
    # The observation width and task protocol are unchanged, but a new
    # physical plant must never pass as the historical v36 checkpoint contract.
    CONTRACT.update(version=24, sysid_20261011_plant=True)
if args.two_hop_episode_v38:
    # Version 26 separates the one-sample touchdown boundary from the old
    # 0.20 s hold in version 25.  Version 27 identifies a genuinely ramped
    # hardware curriculum; older runs declared a ramp but sampled its final
    # hardware probability from the first iteration.
    ramped_hardware = (
        args.hardware_randomization_probability_start is not None
        and args.hardware_randomization_curriculum_iterations is not None
        and args.hardware_randomization_probability_start
        != args.hardware_randomization_probability
    )
    CONTRACT.update(version=27 if ramped_hardware else 26)


class RecoveryRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos={'tracking_recovery_contract': CONTRACT})


DIAGNOSTIC_STATE_NAMES = (
    'X', 'Y', 'Z', 'Vx', 'Vy', 'Vz', 'Wx_b', 'Wy_b', 'Wz_b',
    'Qw', 'Qx', 'Qy', 'Qz', 'Roll_rad', 'Pitch_rad', 'Yaw_rad',
    'Spring_Pos', 'Spring_Vel', 'Is_Contact',
)


def _state_snapshot(base) -> list[float]:
    """Return the physical state for env zero without changing the environment."""
    pos = base._robot.data.root_pos_w[0]
    vel = base._robot.data.root_lin_vel_w[0]
    ang = base._robot.data.root_ang_vel_b[0]
    quat = base._robot.data.root_quat_w[0]
    qw, qx, qy, qz = quat.unbind()
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    joint_pos = base._robot.data.joint_pos[0, base._spring_joint_id]
    joint_vel = base._robot.data.joint_vel[0, base._spring_joint_id]
    values = torch.cat((
        pos, vel, ang, quat,
        torch.stack((roll, pitch, yaw, joint_pos, joint_vel,
                     (joint_pos > 0.002).to(joint_pos.dtype))),
    ))
    return values.detach().cpu().tolist()


def _clone_actor_hidden(policy_model):
    hidden = policy_model.memory_a.hidden_states
    if hidden is None:
        return None
    if not isinstance(hidden, tuple):
        hidden = (hidden,)
    return tuple(value.detach().clone() for value in hidden)


def _hidden_metrics(prefix: str, before, after) -> dict[str, float]:
    """Summarize LSTM state and its one-tick change for a single environment."""
    result: dict[str, float] = {}
    labels = ('H', 'C')
    for index, label in enumerate(labels):
        current = after[index].reshape(-1)
        previous = torch.zeros_like(current) if before is None else before[index].reshape(-1)
        delta = current - previous
        denominator = torch.linalg.vector_norm(current) * torch.linalg.vector_norm(previous)
        cosine = torch.where(
            denominator > 0.0,
            torch.dot(current, previous) / denominator,
            torch.zeros_like(denominator),
        )
        result.update({
            f'{prefix}{label}_Pre_Norm': float(torch.linalg.vector_norm(previous).cpu()),
            f'{prefix}{label}_Post_Norm': float(torch.linalg.vector_norm(current).cpu()),
            f'{prefix}{label}_Delta_Norm': float(torch.linalg.vector_norm(delta).cpu()),
            f'{prefix}{label}_Post_MaxAbs': float(current.abs().max().cpu()),
            f'{prefix}{label}_Cosine_Pre_Post': float(cosine.cpu()),
        })
    return result


def _motor_metrics(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1).cpu()
    result = {f'{prefix}{index + 1}': float(value) for index, value in enumerate(values)}
    result[f'{prefix}Collective'] = float(values.mean())
    result[f'{prefix}Spread'] = float(values.max() - values.min())
    return result


def _rate_metrics(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """Record a four-channel rate plus magnitude summaries."""
    result = _motor_metrics(prefix, values)
    flat = values.detach().reshape(-1).cpu()
    result[f'{prefix}AbsMean'] = float(flat.abs().mean())
    result[f'{prefix}AbsMax'] = float(flat.abs().max())
    return result


def main():
    from Quadhopper_Planner_Random.raibert_hermite import RAIBERT_HERMITE_PROTOCOL
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = torch.load(source, map_location='cpu', weights_only=False)
    state = data['model_state_dict']
    infos = data.get('infos') or {}
    prior = infos.get('tracking_contract')
    recovery = infos.get('tracking_recovery_contract', {})
    if args.path_free_stage2:
        allowed = {PATH_FREE_PROTOCOL}
        if args.mode == 'train' and not args.resume_runner_state:
            allowed.update((PROTOCOL, RAIBERT_HERMITE_PROTOCOL))
            if args.path_free_stage2_v2:
                allowed.add(PATH_FREE_V1_PROTOCOL)
            if args.path_free_stage2_v3:
                allowed.update((PATH_FREE_V1_PROTOCOL, PATH_FREE_V2_PROTOCOL))
        if recovery.get('protocol') not in allowed:
            raise ValueError('Stage2 requires a history-foot or Raibert weights-only warm start; '
                             'eval/resume requires the path-free protocol')
    if args.observation_layout == 'history_foot_v19':
        source_width = state['memory_a.rnn.weight_ih_l0'].shape[1]
        if source_width == 52:
            if (args.mode != 'train' or args.resume_runner_state
                    or recovery.get('protocol') != 'physical_1m_two_circle_v18_yaw140_ramp450to600'):
                raise ValueError('v19 expansion requires weights-only training from a v18 spatial52 checkpoint')
            state = expand_from_spatial52(state)
        else:
            expected_protocol = (
                PATH_FREE_PROTOCOL if args.path_free_stage2 else
                (RAIBERT_HERMITE_PROTOCOL if args.raibert_hermite_guidance else PROTOCOL)
            )
            weights_only_raibert_warm_start = (
                args.raibert_hermite_guidance
                and args.mode == 'train'
                and not args.resume_runner_state
                and recovery.get('protocol') in (PROTOCOL, RAIBERT_HERMITE_PROTOCOL)
            )
            if (source_width != OBS_DIM
                    or (recovery.get('protocol') != expected_protocol
                        and not weights_only_raibert_warm_start
                        and not args.path_free_stage2)):
                raise ValueError('Incompatible history-foot checkpoint observation/guidance contract')
    elif recovery.get('obs_dim', 52) != 52:
        raise ValueError('Non-52D checkpoints require their explicit observation layout')
    if args.mode == 'train':
        if prior and prior.get('version') == 2 and prior.get('variant') == 'preview':
            state = expand_policy_inputs(state)
        elif recovery.get('version') not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27):
            raise ValueError('Curriculum training requires a compatible 52-D spatial checkpoint')
        state['std'].fill_(CONTRACT['std'])
    else:
        spatial = (data.get('infos') or {}).get('tracking_recovery_contract', {})
        current_contract = (
            spatial.get('protocol') == CONTRACT['protocol']
            and spatial.get('version') == CONTRACT['version']
        )
        compatible_prior = (
            spatial.get('version') in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                       15, 16, 17, 25, 26, 27)
            or (args.two_hop_episode_v38 and spatial.get('version') == 24
                and spatial.get('sysid_20261011_plant'))
        )
        if not current_contract and not (args.allow_compatible_checkpoint and compatible_prior):
            raise ValueError('Evaluation checkpoint does not match the selected actuator/profile contract; '
                             'use --allow-compatible-checkpoint only for an intentional cross-contract test')
    if args.resume_runner_state and recovery.get('version') != CONTRACT['version']:
        raise ValueError('Cross-contract training must initialize weights only, not resume optimizer state')

    # Select environment class based on stability profile
    if args.observation_layout == 'history_foot_v19':
        cfg = HistoryFootEnvCfg()
    elif args.stability_profile == 'stationary_corridor_v17':
        cfg = StationaryCorridorV17EnvCfg()
    else:
        cfg = SpatialTrackingPhysical1mEnvCfg()
    if args.actuator_mode == 'corrected_history':
        cfg.apply_deployment_action_shaping = False
    elif args.actuator_mode == 'baseline':
        cfg.corrected_action_delay_indexing = False
        cfg.record_executed_action_history = False
        cfg.apply_deployment_action_shaping = False
    elif args.actuator_mode == 'zero_delay_raw_history':
        cfg.corrected_action_delay_indexing = True
        cfg.record_executed_action_history = False
        cfg.apply_deployment_action_shaping = False
    cfg.target_height = args.target_height
    cfg.deployment_action_target_height = args.target_height
    cfg.use_raibert_hermite_guidance = args.raibert_hermite_guidance
    cfg.raibert_stance_time_s = args.raibert_stance_time_s
    cfg.raibert_velocity_gain_s = args.raibert_velocity_gain_s
    cfg.raibert_max_foothold_shift_m = args.raibert_max_foothold_shift_m
    cfg.raibert_preview_blend = args.raibert_preview_blend
    cfg.raibert_max_horizontal_speed_m_s = args.raibert_max_horizontal_speed_m_s
    if args.disable_initial_drop:
        cfg.start_from_random_drop = False
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    if args.successful_hops_per_episode is not None:
        cfg.successful_hops_per_episode = args.successful_hops_per_episode
    cfg.action_parameterization = args.action_parameterization
    cfg.collective_preserving_spread_limit_pwm = args.collective_preserving_spread_limit_pwm
    if args.apex_tolerance is not None:
        cfg.apex_tolerance = args.apex_tolerance
    if args.play_motor_time_constant is not None:
        cfg.play_motor_time_constant = args.play_motor_time_constant
    if args.play_thrust_scale is not None:
        cfg.play_thrust_scale = args.play_thrust_scale
    cfg.simulation_only_phase_collective_baseline = args.simulation_only_phase_collective_baseline
    cfg.diagnostic_ascent_pulse_pwm = args.diagnostic_ascent_pulse_pwm or 0.0
    if args.simulation_only_phase_collective_baseline:
        cfg.phase_contact_collective_floor_pwm = args.phase_contact_collective_floor_pwm
        cfg.phase_ascent_collective_floor_pwm = args.phase_ascent_collective_floor_pwm
    cfg.simulation_only_phase_collective_controller = args.simulation_only_phase_collective_controller
    if args.simulation_only_phase_collective_controller:
        cfg.phase_contact_collective_pwm = args.phase_contact_collective_pwm
        cfg.phase_ascent_collective_pwm = args.phase_ascent_collective_pwm
        cfg.phase_descent_collective_pwm = args.phase_descent_collective_pwm
        cfg.phase_contact_differential_limit_pwm = args.phase_contact_differential_limit_pwm
        cfg.phase_ascent_differential_limit_pwm = args.phase_ascent_differential_limit_pwm
        cfg.phase_descent_differential_limit_pwm = args.phase_descent_differential_limit_pwm
    cfg.simulation_only_apex_bias_controller = args.simulation_only_apex_bias_controller
    if args.simulation_only_apex_bias_controller:
        cfg.apex_bias_initial_pwm = args.apex_bias_initial_pwm
        cfg.apex_bias_gain_pwm_per_m = args.apex_bias_gain_pwm_per_m
        cfg.apex_bias_max_pwm = args.apex_bias_max_pwm
        cfg.apex_bias_feedback_mode = args.apex_bias_feedback
        if args.apex_bias_feedback == 'liftoff_vz':
            cfg.apex_bias_liftoff_vz_target_mps = args.apex_bias_liftoff_vz_target
    if args.height_curriculum_start is not None:
        cfg.fixed_height_curriculum = True
        cfg.height_curriculum_start = args.height_curriculum_start
        cfg.height_curriculum_end = args.target_height
        cfg.height_curriculum_iterations = args.height_curriculum_iterations
        cfg.height_curriculum_iteration_offset = 0.0
    cfg.short_hop_radius_min, cfg.short_hop_radius_max = short_distance_range
    cfg.long_hop_radius_min, cfg.long_hop_radius_max = long_distance_range
    cfg.hard_radius_split = args.hard_distance_min
    cfg.hard_radius_probability = args.hard_sample_probability
    cfg.max_turn_angle_deg = args.max_turn_angle_deg
    cfg.zero_hop_probability = args.zero_hop_probability
    cfg.reverse_turn_probability = args.reverse_turn_probability
    cfg.pair_scenario_groups = args.two_hop_episode_v38
    cfg.reverse_turn_halfwidth_deg = args.reverse_turn_halfwidth_deg
    cfg.fixed_initial_heading_deg = args.fixed_initial_heading_deg
    cfg.distance_curriculum_iterations = args.distance_curriculum_iterations
    if args.distance_curriculum_iterations > 0.0:
        cfg.curriculum_short_radius_min = args.distance_curriculum_start_short_min
        cfg.curriculum_short_radius_max = args.distance_curriculum_start_short_max
        cfg.curriculum_long_radius_min = args.distance_curriculum_start_long_min
        cfg.curriculum_long_radius_max = args.distance_curriculum_start_long_max
        cfg.curriculum_max_turn_angle_deg = args.distance_curriculum_start_turn_angle_deg
    cfg.direction_curriculum_iterations = args.direction_curriculum_iterations
    if args.direction_curriculum_iterations > 0.0:
        cfg.direction_curriculum_start_turn_angle_deg = (
            args.direction_curriculum_start_turn_angle_deg
        )
        cfg.direction_curriculum_start_zero_hop_probability = (
            args.direction_curriculum_start_zero_hop_probability
        )
    reward_overrides = {
        'landing_precision_width': args.landing_precision_width,
        'target_hit_reward_scale': args.target_hit_reward_scale,
        'landing_precision_reward_scale': args.landing_precision_reward_scale,
        'landing_error_penalty_scale': args.landing_error_penalty_scale,
        'landing_lateral_error_penalty_scale': args.landing_lateral_error_penalty_scale,
    }
    for name, value in reward_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.robust_dynamics or args.hardware_robust:
        # Real hops arrive with per-motor thrust, bandwidth, and timing
        # differences.  Keep these ranges mild enough to retain the measured
        # calibrated nominal plant in the curriculum centre, while ensuring a
        # policy cannot rely on perfectly matched rotors.
        cfg.randomize_dynamics = True
        cfg.randomize_action_delay = True
        cfg.randomize_motor_time_constant = True
        cfg.randomize_thrust_curve = True
        cfg.train_thrust_scale_min = 0.93
        cfg.train_thrust_scale_max = 1.07
        cfg.train_thrust_curve_shape_min = 0.96
        cfg.train_thrust_curve_shape_max = 1.04
    if args.hardware_robust:
        # Start from the flight-data nominal plant and cover the remaining
        # hardware uncertainty without asking every rollout to solve an
        # implausible worst-case combination.  The previous 0.78 thrust / 110
        # ms / 1.25 inertia envelope destroyed the already-good 1 m hopping
        # behaviour, rather than making it robust.  This range still includes
        # lower rebound authority and slower motors seen after impact.
        cfg.train_mass_multi_min = 0.97
        cfg.train_mass_multi_max = 1.07
        cfg.train_inertia_multi_min = 0.88
        cfg.train_inertia_multi_max = 1.15
        cfg.train_motor_time_constant_min = 0.054
        cfg.train_motor_time_constant_max = 0.090
        cfg.train_thrust_scale_min = 0.87
        cfg.train_thrust_scale_max = 1.06
        cfg.train_thrust_curve_shape_min = 0.94
        cfg.train_thrust_curve_shape_max = 1.06
        # Flight logs show no persistent multi-step command delay, so do not
        # trade away nominal takeoff height for an artificial delay model.
        cfg.randomize_action_delay = False
    cfg.comprehensive_hardware_randomization = args.comprehensive_hardware_randomization
    cfg.hardware_randomization_probability = args.hardware_randomization_probability
    if args.hardware_randomization_probability_start is not None:
        cfg.hardware_randomization_probability_start = (
            args.hardware_randomization_probability_start
        )
        cfg.hardware_randomization_curriculum_iterations = (
            args.hardware_randomization_curriculum_iterations
        )
    if args.comprehensive_global_thrust_scale_min is not None:
        cfg.train_global_thrust_scale_min = args.comprehensive_global_thrust_scale_min
        cfg.train_global_thrust_scale_max = args.comprehensive_global_thrust_scale_max
    if args.comprehensive_com_offset_x_max_mm is not None:
        cfg.train_body_com_offset_x_max_m = args.comprehensive_com_offset_x_max_mm / 1000.0
        cfg.train_body_com_offset_y_max_m = args.comprehensive_com_offset_y_max_mm / 1000.0
        cfg.train_body_com_offset_z_max_m = args.comprehensive_com_offset_z_max_mm / 1000.0
    if args.comprehensive_yaw_effectiveness_min is not None:
        cfg.train_yaw_effectiveness_min = args.comprehensive_yaw_effectiveness_min
        cfg.train_yaw_effectiveness_max = args.comprehensive_yaw_effectiveness_max
    if args.comprehensive_yaw_bias_torque_max_mnm is not None:
        cfg.train_yaw_bias_torque_max_nm = (
            args.comprehensive_yaw_bias_torque_max_mnm / 1000.0
        )
    if args.observation_noise_std is not None:
        cfg.observation_noise_std = args.observation_noise_std
    if args.initial_yaw_perturb_deg is not None:
        cfg.initial_yaw_perturb_rad = math.radians(args.initial_yaw_perturb_deg)
    if args.initial_angular_velocity_perturb_rad_s is not None:
        cfg.initial_angular_velocity_perturb_radps = (
            args.initial_angular_velocity_perturb_rad_s
        )
    if args.touchdown_wrench_probability is not None:
        cfg.train_touchdown_wrench_probability = args.touchdown_wrench_probability
        cfg.train_touchdown_roll_wrench_max_nm = args.touchdown_roll_wrench_max_nm
        cfg.train_touchdown_pitch_wrench_max_nm = args.touchdown_pitch_wrench_max_nm
        cfg.train_touchdown_yaw_wrench_max_nm = args.touchdown_yaw_wrench_max_nm
        cfg.train_touchdown_wrench_window_s = args.touchdown_wrench_window_s
        cfg.train_touchdown_wrench_persistence = args.touchdown_wrench_persistence
    cfg.train_touchdown_wrench_on_moving_long_hop = (
        args.touchdown_wrench_on_moving_long_hop
    )
    moving_long_recovery_overrides = {
        'moving_long_recovery_window_s': args.moving_long_recovery_window_s,
        'moving_long_recovery_angle_deadband_rad': (
            None if args.moving_long_recovery_angle_deadband_deg is None
            else math.radians(args.moving_long_recovery_angle_deadband_deg)
        ),
        'moving_long_recovery_rate_deadband_rad_s': (
            args.moving_long_recovery_rate_deadband_rad_s
        ),
        'moving_long_recovery_angle_penalty_scale': (
            args.moving_long_recovery_angle_penalty_scale
        ),
        'moving_long_recovery_rate_penalty_scale': (
            args.moving_long_recovery_rate_penalty_scale
        ),
    }
    for name, value in moving_long_recovery_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    airborne_overrides = {
        'airborne_attitude_penalty_scale': args.airborne_attitude_penalty_scale,
        'airborne_angvel_xy_penalty_scale': args.airborne_angvel_xy_penalty_scale,
        'airborne_yaw_penalty_scale': args.airborne_yaw_penalty_scale,
        'airborne_yaw_rate_penalty_scale': args.airborne_yaw_rate_penalty_scale,
        'airborne_action_spread_penalty_scale': args.airborne_action_spread_penalty_scale,
        'airborne_cross_track_tilt_penalty_scale': args.airborne_cross_track_tilt_penalty_scale,
        'airborne_cross_track_tilt_penalty_start': args.airborne_cross_track_tilt_penalty_start,
        'airborne_cross_track_velocity_penalty_scale': args.airborne_cross_track_velocity_penalty_scale,
        'airborne_cross_track_velocity_penalty_start_mps': args.airborne_cross_track_velocity_penalty_start_mps,
        'airborne_tilt_penalty_start': args.airborne_tilt_penalty_start,
        'airborne_angvel_xy_penalty_start': args.airborne_angvel_xy_penalty_start,
        'airborne_yaw_penalty_start': (
            None if args.airborne_yaw_deadband_deg is None
            else math.radians(args.airborne_yaw_deadband_deg)
        ),
        'airborne_yaw_rate_penalty_start': args.airborne_yaw_rate_penalty_start,
        'airborne_action_spread_penalty_start': args.airborne_action_spread_penalty_start,
    }
    for name, value in airborne_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.termination_penalty_scale is not None:
        cfg.termination_penalty_scale = args.termination_penalty_scale
    if args.target_relative_apex_gate:
        cfg.target_relative_valid_apex = True
        cfg.require_minimum_apex_for_hit = True
        cfg.gate_touchdown_rewards_by_apex = True
    stationary_overrides = {
        'stationary_hold_radius_m': args.stationary_hold_radius,
        'stationary_tilt_penalty_scale': args.stationary_tilt_penalty_scale,
        'stationary_yaw_rate_penalty_scale': args.stationary_yaw_rate_penalty_scale,
        'stationary_action_spread_penalty_scale': args.stationary_action_spread_penalty_scale,
        'stationary_pitch_deadband_rad': (
            None if args.stationary_pitch_deadband_deg is None
            else math.radians(args.stationary_pitch_deadband_deg)
        ),
        'stationary_pitch_penalty_scale': args.stationary_pitch_penalty_scale,
        'stationary_pitch_rate_deadband_rad_s': args.stationary_pitch_rate_deadband_rad_s,
        'stationary_pitch_rate_penalty_scale': args.stationary_pitch_rate_penalty_scale,
        'stationary_airborne_differential_action_rate_deadband': (
            args.stationary_airborne_differential_action_rate_deadband
        ),
        'stationary_airborne_differential_action_rate_penalty_scale': (
            args.stationary_airborne_differential_action_rate_penalty_scale
        ),
        'stationary_contact_recovery_window_s': args.stationary_contact_recovery_window_s,
        'stationary_contact_recovery_spread_deadband': args.stationary_contact_recovery_spread_deadband,
        'stationary_contact_recovery_spread_penalty_scale': args.stationary_contact_recovery_spread_penalty_scale,
        'stationary_contact_recovery_delta_deadband': args.stationary_contact_recovery_delta_deadband,
        'stationary_contact_recovery_delta_penalty_scale': args.stationary_contact_recovery_delta_penalty_scale,
        'stationary_apex_width_m': args.stationary_apex_width,
        'stationary_apex_reward_scale': args.stationary_apex_reward_scale,
        'stationary_apex_error_penalty_scale': args.stationary_apex_error_penalty_scale,
        'stationary_apex_shortfall_penalty_scale': args.stationary_apex_shortfall_penalty_scale,
        'stationary_apex_progress_scale': args.stationary_apex_progress_scale,
        'stationary_apex_progress_width_m': args.stationary_apex_progress_width,
        'stationary_ascent_support_scale': args.stationary_ascent_support_scale,
        'stationary_ascent_collective_scale': args.stationary_ascent_collective_scale,
        'stationary_predicted_apex_width_m': args.stationary_predicted_apex_width,
        'stationary_predicted_apex_reward_scale': args.stationary_predicted_apex_reward_scale,
        'ascent_predicted_apex_width_m': args.ascent_predicted_apex_width,
        'ascent_predicted_apex_reward_scale': args.ascent_predicted_apex_reward_scale,
        'apex_event_height_width_m': args.apex_event_height_width,
        'apex_event_height_reward_scale': args.apex_event_height_reward_scale,
        'apex_event_shortfall_penalty_scale': args.apex_event_shortfall_penalty_scale,
    }
    for name, value in stationary_overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.height_progress_reward_scale is not None:
        cfg.height_progress_reward_scale = args.height_progress_reward_scale
        CONTRACT['height_progress_reward_scale'] = cfg.height_progress_reward_scale
    if args.liftoff_cross_track_penalty_scale is not None:
        from Quadhopper_Planner_Random.liftoff_cross_track_v30 import (
            DEADBAND_M_S, NORMALIZER_M_S, MAX_NORMALIZED_EXCESS,
        )
        cfg.liftoff_cross_track_penalty_scale = args.liftoff_cross_track_penalty_scale
        CONTRACT['v30_liftoff_cross_track'] = {
            'penalty_scale': cfg.liftoff_cross_track_penalty_scale,
            'deadband_m_s': DEADBAND_M_S,
            'normalizer_m_s': NORMALIZER_M_S,
            'max_normalized_excess': MAX_NORMALIZED_EXCESS,
            'event': 'liftoff_only_except_initial_drop',
            'direction': 'actual_foot_to_current_P1_world_xy',
            'velocity': 'Body_rigid_COM_world_xy',
        }
    from Quadhopper_Planner_Random.stability_profile import apply_stability_profile
    profile_settings = apply_stability_profile(cfg, args.stability_profile)
    if args.moving_launch_rate_v34:
        cfg.moving_launch_stability_window_s = 0.16
        cfg.moving_launch_rate_deadband_rad_s = 0.8
        cfg.moving_launch_rate_penalty_scale = -8.0
        CONTRACT['moving_launch_rate_v34'] = {
            'event': 'moving_hop_late_stance_and_early_flight',
            'window_s': cfg.moving_launch_stability_window_s,
            'rate_deadband_rad_s': cfg.moving_launch_rate_deadband_rad_s,
            'penalty_scale': cfg.moving_launch_rate_penalty_scale,
            'cost': 'sum_clamped_squared_excess_roll_pitch_rate',
        }
    if args.moving_handoff_v11:
        from Quadhopper_Planner_Random.moving_handoff import apply_settings as apply_handoff
        apply_handoff(cfg)
    CONTRACT['moving_handoff_v11'] = args.moving_handoff_v11
    CONTRACT['moving_handoff_settings'] = {
        'moving_contact_recovery_all_hops': cfg.moving_contact_recovery_all_hops,
        'moving_launch_stability_window_s': cfg.moving_launch_stability_window_s,
        'moving_launch_rate_deadband_rad_s': cfg.moving_launch_rate_deadband_rad_s,
        'moving_launch_rate_penalty_scale': cfg.moving_launch_rate_penalty_scale,
    }
    if args.path_free_stage2:
        apply_path_free_settings(
            cfg, v2=args.path_free_stage2_v2, v3=args.path_free_stage2_v3)
        if args.stage2_foot_precision_scale is not None:
            cfg.stage2_foot_precision_scale = args.stage2_foot_precision_scale
            CONTRACT['path_free_stage2']['stage2_foot_precision_scale'] = (
                cfg.stage2_foot_precision_scale
            )
        if args.stage2_foot_precision_width_m is not None:
            cfg.stage2_foot_precision_width_m = args.stage2_foot_precision_width_m
            CONTRACT['path_free_stage2']['stage2_foot_precision_width_m'] = (
                cfg.stage2_foot_precision_width_m
            )
        if args.pair_xy_quality_weight_v36 is not None:
            cfg.stage2_pair_xy_quality_weight = args.pair_xy_quality_weight_v36
            CONTRACT['path_free_stage2']['stage2_pair_xy_quality_weight'] = (
                cfg.stage2_pair_xy_quality_weight
            )
        if args.path_free_stage2_v2:
            cfg.stage2_curriculum_training = args.mode == 'train'
            cfg.stage2_training_iterations = args.iterations
            cfg.curriculum_steps_per_iteration = 256
            # The v2 environment owns the held/ramped schedule. Evaluation
            # uses exactly the probability requested by the evaluation CLI.
            cfg.hardware_randomization_curriculum_iterations = 0.0
            CONTRACT['path_free_stage2'].update(
                curriculum_training=cfg.stage2_curriculum_training,
                training_iterations=args.iterations if args.mode == 'train' else None,
                curriculum_steps_per_iteration=256,
                hardware_probability_final=cfg.hardware_randomization_probability,
                progress_reference='measured_foot_xy_signed_distance_change',
            )
        if args.path_free_stage2_v3:
            # Preserve an explicitly requested nominal-to-robust schedule.
            # Equal start/end probabilities remain constant as in v37.
            if (args.hardware_randomization_probability_start is None or
                    args.hardware_randomization_probability_start ==
                    args.hardware_randomization_probability):
                cfg.hardware_randomization_curriculum_iterations = 0.0
            CONTRACT['path_free_stage2'].update(
                curriculum_training=(
                    cfg.hardware_randomization_curriculum_iterations > 0.0
                ),
                hardware_probability_final=cfg.hardware_randomization_probability,
                progress_reference='measured_foot_xy_signed_distance_change',
                strict_success='foot_xy_inside_tolerance_and_measured_apex_at_or_above_target',
                added_energy_penalty='disabled',
            )
        CONTRACT['path_free_stage2'].update(
            baseline_power_penalty_scale=cfg.power_penalty_scale,
            baseline_flight_power_penalty_scale=cfg.flight_power_penalty_scale,
            baseline_synergy_reward_scale=cfg.synergy_reward_scale,
            baseline_anti_synergy_penalty_scale=cfg.anti_synergy_penalty_scale,
            energy_measurement='existing_TorchScript_power_estimate_clamped_nonnegative',
        )
    if args.stability_profile in ('cross_hop_v15', 'cross_hop_v15_1', 'cross_hop_v16',
                                  'two_circle_v18'):
        CONTRACT['deployment_action_shaping'].update({
            'launch_recovery_window_s': cfg.deployment_launch_recovery_window_s,
        })
        if args.stability_profile == 'cross_hop_v15':
            CONTRACT['deployment_action_shaping']['launch_recovery_max_pwm_spread'] = (
                cfg.deployment_launch_recovery_spread_limit_pwm
            )
        else:
            CONTRACT['deployment_action_shaping'].update({
                'launch_recovery_hold_s': cfg.deployment_launch_recovery_hold_s,
                'launch_recovery_initial_max_pwm_spread': cfg.deployment_launch_recovery_spread_limit_pwm,
                'launch_recovery_final_max_pwm_spread': cfg.deployment_launch_recovery_spread_limit_end_pwm,
            })
    CONTRACT.update(stability_profile=args.stability_profile,
                    stability_profile_settings=profile_settings)
    cfg.target_tolerance = args.target_tolerance
    cfg.collect_tracking_metrics = args.mode == 'eval'
    if args.hardware_severity_curriculum:
        cfg.hardware_severity_stage_iterations = args.hardware_severity_stage_iterations
    if args.route_recovery_mode:
        cfg.advance_route_on_miss = True
        cfg.route_recovery_mode = args.route_recovery_mode
        cfg.route_recovery_severe_miss_m = args.route_recovery_severe_miss_m
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = args.visualize
    cfg.power_model_path = str(ROOT / 'Quadhopper_Stable/model/quadhopper_memory_power.pt')
    cfg.csv_log_path = str(output / 'single_env.csv')
    if args.sysid_20261011_plant:
        from Quadhopper_Planner_Random.sysid_20261011 import NOMINAL
        asset = ROOT / 'Quadhopper_Planner_Random/assets/SysID20261011MountedSpringProxy.usda'
        source_json = ROOT / 'Quadhopper_Planner_Random/assets/domain_randomization_20261011_source.json'
        if not asset.is_file() or not source_json.is_file():
            raise FileNotFoundError('Mounted SysID asset or source parameter JSON is missing')
        cfg.robot = cfg.robot.replace(
            spawn=cfg.robot.spawn.replace(usd_path=str(asset))
        )
        cfg.sysid_20261011_plant = True
        cfg.play_motor_time_constant = NOMINAL.motor_time_constant_s
        cfg.sysid_voltage_reference_v = NOMINAL.voltage_reference_v
        cfg.sysid_voltage_exponent = NOMINAL.voltage_exponent
        cfg.sysid_yaw_kqf_m = NOMINAL.yaw_torque_per_thrust_m
        # Start with the report's narrow physical curriculum. Broad B-class
        # stress ranges are not silently combined with an untested new plant.
        cfg.train_mass_multi_min = 0.190 / NOMINAL.total_mass_kg
        cfg.train_mass_multi_max = 0.198 / NOMINAL.total_mass_kg
        cfg.train_inertia_multi_min = 0.95
        cfg.train_inertia_multi_max = 1.05
        cfg.train_motor_time_constant_min = 0.060
        cfg.train_motor_time_constant_max = 0.080
        cfg.train_thrust_scale_min = 0.95
        cfg.train_thrust_scale_max = 1.05
        cfg.train_thrust_curve_shape_min = 1.0
        cfg.train_thrust_curve_shape_max = 1.0
        cfg.randomize_motor_time_constant = False
        cfg.randomize_thrust_curve = False
        cfg.randomize_action_delay = False
        cfg.train_global_thrust_scale_min = 0.95
        cfg.train_global_thrust_scale_max = 1.05
        cfg.train_rp_effectiveness_diag_min = 1.0
        cfg.train_rp_effectiveness_diag_max = 1.0
        cfg.train_rp_effectiveness_rotation_deg = 0.0
        cfg.train_rp_effectiveness_cross_max = 0.0
        cfg.train_yaw_effectiveness_min = 1.0
        cfg.train_yaw_effectiveness_max = 1.0
        cfg.train_yaw_bias_torque_max_nm = 0.0
        cfg.train_yaw_reaction_imbalance_max = 0.05
        CONTRACT['motor_tc_s'] = NOMINAL.motor_time_constant_s
        CONTRACT['sysid_20261011_plant'] = True
        CONTRACT['sysid_20261011'] = {
            'source_markdown_sha256': '077e663eac06116cd4d097d9dec42b455b1e5544035d873115ebc72c6fd28170',
            'source_sha256': hashlib.sha256(source_json.read_bytes()).hexdigest(),
            'usd_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
            'asset': str(asset),
            'mounted_total_mass_kg': NOMINAL.total_mass_kg,
            'body_inertia_kg_m2': list(NOMINAL.inertia_kg_m2),
            'motor_arm_component_m': NOMINAL.motor_arm_component_m,
            'motor_time_constant_s': NOMINAL.motor_time_constant_s,
            'pwm_compare_divisor': 999,
            'voltage_reference_v': NOMINAL.voltage_reference_v,
            'loaded_voltage_train_range_v': [3.5, 4.0],
            'voltage_exponent_nominal': NOMINAL.voltage_exponent,
            'voltage_exponent_train_range': [1.3, 1.6],
            'thrust_polynomial_n': [NOMINAL.thrust_quadratic_n, NOMINAL.thrust_linear_n, 0.0],
            'yaw_kqf_m_nominal': NOMINAL.yaw_torque_per_thrust_m,
            'yaw_kqf_m_initial_train_range': [0.008, 0.012],
            'interface': 'four_direct_motor_actions_100hz',
            'leg_model': 'existing_spring_proxy_not_rigid_leg',
            'mass_inertia_randomization': 'force_and_moment_equivalent_scaling',
        }
    if cfg.xy_progress_reward_scale != 0.0:
        raise ValueError('Legacy XY progress must stay disabled to avoid duplicate/queue-jump rewards')
    CONTRACT.update(
        actuator_mode=args.actuator_mode,
        action_parameterization=cfg.action_parameterization,
        collective_preserving_spread_limit_pwm=cfg.collective_preserving_spread_limit_pwm,
        deployment_launch_recovery_window_s=cfg.deployment_launch_recovery_window_s,
        deployment_launch_recovery_spread_limit_pwm=(
            cfg.deployment_launch_recovery_spread_limit_pwm
        ),
        deployment_launch_recovery_hold_s=cfg.deployment_launch_recovery_hold_s,
        deployment_launch_recovery_spread_limit_end_pwm=cfg.deployment_launch_recovery_spread_limit_end_pwm,
        launch_recovery_raw_spread_penalty_scale=(
            cfg.launch_recovery_raw_spread_penalty_scale
        ),
        xy_potential_progress_scale=cfg.spatial_xy_progress_reward_scale,
        legacy_xy_progress_reward_scale=cfg.xy_progress_reward_scale,
        two_hop_pair_reward=cfg.two_hop_pair_reward_scale,
        landing_precision_width_m=cfg.landing_precision_width,
        target_hit_reward=cfg.target_hit_reward_scale,
        landing_precision_reward=cfg.landing_precision_reward_scale,
        landing_error_penalty=cfg.landing_error_penalty_scale,
        landing_lateral_error_penalty=cfg.landing_lateral_error_penalty_scale,
        target_hit_requires_minimum_apex=cfg.require_minimum_apex_for_hit,
        target_hit_requires_apex_tolerance=cfg.require_apex_tolerance_for_hit,
        landing_precision_gated_by_apex=cfg.gate_touchdown_rewards_by_apex,
        robust_dynamics=cfg.randomize_dynamics,
        hardware_robust=args.hardware_robust,
        comprehensive_hardware_randomization=cfg.comprehensive_hardware_randomization,
        hardware_randomization_probability=cfg.hardware_randomization_probability,
        hardware_randomization_probability_start=(
            cfg.hardware_randomization_probability_start
        ),
        hardware_randomization_curriculum_iterations=(
            cfg.hardware_randomization_curriculum_iterations
        ),
        randomize_action_delay=cfg.randomize_action_delay,
        randomize_motor_time_constant=cfg.randomize_motor_time_constant,
        motor_time_constant_range_s=[cfg.train_motor_time_constant_min, cfg.train_motor_time_constant_max],
        mass_multiplier_range=[cfg.train_mass_multi_min, cfg.train_mass_multi_max],
        inertia_multiplier_range=[cfg.train_inertia_multi_min, cfg.train_inertia_multi_max],
        randomize_thrust_curve=cfg.randomize_thrust_curve,
        thrust_scale_range=[cfg.train_thrust_scale_min, cfg.train_thrust_scale_max],
        thrust_curve_shape_range=[cfg.train_thrust_curve_shape_min, cfg.train_thrust_curve_shape_max],
        global_thrust_scale_range=(
            [cfg.train_global_thrust_scale_min, cfg.train_global_thrust_scale_max]
            if cfg.comprehensive_hardware_randomization else None
        ),
        rp_effectiveness=(
            {
                'diagonal_range': [
                    cfg.train_rp_effectiveness_diag_min,
                    cfg.train_rp_effectiveness_diag_max,
                ],
                'rotation_deg': cfg.train_rp_effectiveness_rotation_deg,
                'cross_abs_max': cfg.train_rp_effectiveness_cross_max,
            }
            if cfg.comprehensive_hardware_randomization else None
        ),
        yaw_effectiveness=(
            {
                'scale_range': [
                    cfg.train_yaw_effectiveness_min,
                    cfg.train_yaw_effectiveness_max,
                ],
                'collective_bias_torque_range_nm': [
                    -cfg.train_yaw_bias_torque_max_nm,
                    cfg.train_yaw_bias_torque_max_nm,
                ],
            }
            if cfg.comprehensive_hardware_randomization else None
        ),
        spring_randomization=(
            {
                'stiffness_range': [
                    cfg.train_spring_stiffness_scale_min,
                    cfg.train_spring_stiffness_scale_max,
                ],
                'damping_range': [
                    cfg.train_spring_damping_scale_min,
                    cfg.train_spring_damping_scale_max,
                ],
            }
            if cfg.comprehensive_hardware_randomization else None
        ),
        body_com_offset_range_m=(
            {
                'x': [-cfg.train_body_com_offset_x_max_m, cfg.train_body_com_offset_x_max_m],
                'y': [-cfg.train_body_com_offset_y_max_m, cfg.train_body_com_offset_y_max_m],
                'z': [-cfg.train_body_com_offset_z_max_m, cfg.train_body_com_offset_z_max_m],
            }
            if cfg.comprehensive_hardware_randomization else None
        ),
        stationary_airborne_differential_action_rate={
            'deadband_per_tick': cfg.stationary_airborne_differential_action_rate_deadband,
            'penalty_scale': cfg.stationary_airborne_differential_action_rate_penalty_scale,
            'collective_removed': True,
        },
        observation_noise_std=cfg.observation_noise_std,
        initial_yaw_perturb_deg=math.degrees(cfg.initial_yaw_perturb_rad),
        initial_angular_velocity_perturb_rad_s=(
            cfg.initial_angular_velocity_perturb_radps
        ),
        airborne_stability={
            'attitude_penalty_scale': cfg.airborne_attitude_penalty_scale,
            'angvel_xy_penalty_scale': cfg.airborne_angvel_xy_penalty_scale,
            'yaw_to_world_x_penalty_scale': cfg.airborne_yaw_penalty_scale,
            'yaw_rate_penalty_scale': cfg.airborne_yaw_rate_penalty_scale,
            'action_spread_penalty_scale': cfg.airborne_action_spread_penalty_scale,
            'tilt_start_qxy_squared': cfg.airborne_tilt_penalty_start,
            'angvel_xy_start_rad_s': cfg.airborne_angvel_xy_penalty_start,
            'yaw_deadband_deg': math.degrees(cfg.airborne_yaw_penalty_start),
            'yaw_rate_start_rad_s': cfg.airborne_yaw_rate_penalty_start,
            'action_spread_start': cfg.airborne_action_spread_penalty_start,
        },
        corrected_action_delay_indexing=cfg.corrected_action_delay_indexing,
        simulation_only_phase_collective_baseline=cfg.simulation_only_phase_collective_baseline,
        phase_contact_collective_floor_pwm=cfg.phase_contact_collective_floor_pwm,
        phase_ascent_collective_floor_pwm=cfg.phase_ascent_collective_floor_pwm,
        simulation_only_phase_collective_controller=cfg.simulation_only_phase_collective_controller,
        phase_contact_collective_pwm=cfg.phase_contact_collective_pwm,
        phase_ascent_collective_pwm=cfg.phase_ascent_collective_pwm,
        phase_descent_collective_pwm=cfg.phase_descent_collective_pwm,
        phase_contact_differential_limit_pwm=cfg.phase_contact_differential_limit_pwm,
        phase_ascent_differential_limit_pwm=cfg.phase_ascent_differential_limit_pwm,
        phase_descent_differential_limit_pwm=cfg.phase_descent_differential_limit_pwm,
        simulation_only_apex_bias_controller=cfg.simulation_only_apex_bias_controller,
        apex_bias_initial_pwm=cfg.apex_bias_initial_pwm,
        apex_bias_gain_pwm_per_m=cfg.apex_bias_gain_pwm_per_m,
        apex_bias_max_pwm=cfg.apex_bias_max_pwm,
        apex_bias_feedback_mode=cfg.apex_bias_feedback_mode,
        apex_bias_liftoff_vz_target_mps=cfg.apex_bias_liftoff_vz_target_mps,
        fixed_height_curriculum=cfg.fixed_height_curriculum,
        height_curriculum_start_m=cfg.height_curriculum_start,
        height_curriculum_end_m=cfg.height_curriculum_end,
        height_curriculum_iterations=cfg.height_curriculum_iterations,
        termination_penalty_scale=cfg.termination_penalty_scale,
        moving_long_recovery_window_s=cfg.moving_long_recovery_window_s,
        moving_long_recovery_angle_deadband_deg=math.degrees(
            cfg.moving_long_recovery_angle_deadband_rad
        ),
        moving_long_recovery_rate_deadband_rad_s=(
            cfg.moving_long_recovery_rate_deadband_rad_s
        ),
        moving_long_recovery_angle_penalty_scale=(
            cfg.moving_long_recovery_angle_penalty_scale
        ),
        moving_long_recovery_rate_penalty_scale=(
            cfg.moving_long_recovery_rate_penalty_scale
        ),
        touchdown_wrench_on_moving_long_hop=(
            cfg.train_touchdown_wrench_on_moving_long_hop
        ),
        target_relative_apex_gate=cfg.target_relative_valid_apex,
        require_minimum_apex_for_hit=cfg.require_minimum_apex_for_hit,
        gate_touchdown_rewards_by_apex=cfg.gate_touchdown_rewards_by_apex,
        apex_tolerance_m=cfg.apex_tolerance,
        stationary_hold_radius_m=cfg.stationary_hold_radius_m,
        stationary_tilt_penalty_scale=cfg.stationary_tilt_penalty_scale,
        stationary_yaw_rate_penalty_scale=cfg.stationary_yaw_rate_penalty_scale,
        stationary_action_spread_penalty_scale=cfg.stationary_action_spread_penalty_scale,
        stationary_pitch_deadband_deg=math.degrees(cfg.stationary_pitch_deadband_rad),
        stationary_pitch_penalty_scale=cfg.stationary_pitch_penalty_scale,
        stationary_pitch_rate_deadband_rad_s=cfg.stationary_pitch_rate_deadband_rad_s,
        stationary_pitch_rate_penalty_scale=cfg.stationary_pitch_rate_penalty_scale,
        stationary_contact_recovery_window_s=cfg.stationary_contact_recovery_window_s,
        stationary_contact_recovery_spread_deadband=cfg.stationary_contact_recovery_spread_deadband,
        stationary_contact_recovery_spread_penalty_scale=cfg.stationary_contact_recovery_spread_penalty_scale,
        stationary_contact_recovery_delta_deadband=cfg.stationary_contact_recovery_delta_deadband,
        stationary_contact_recovery_delta_penalty_scale=cfg.stationary_contact_recovery_delta_penalty_scale,
        stationary_apex_width_m=cfg.stationary_apex_width_m,
        stationary_apex_reward_scale=cfg.stationary_apex_reward_scale,
        stationary_apex_error_penalty_scale=cfg.stationary_apex_error_penalty_scale,
        stationary_apex_shortfall_penalty_scale=cfg.stationary_apex_shortfall_penalty_scale,
        stationary_apex_progress_scale=cfg.stationary_apex_progress_scale,
        stationary_apex_progress_width_m=cfg.stationary_apex_progress_width_m,
        height_progress_reward_scale=cfg.height_progress_reward_scale,
        stationary_ascent_support_scale=cfg.stationary_ascent_support_scale,
        stationary_ascent_collective_scale=cfg.stationary_ascent_collective_scale,
        stationary_predicted_apex_width_m=cfg.stationary_predicted_apex_width_m,
        stationary_predicted_apex_reward_scale=cfg.stationary_predicted_apex_reward_scale,
        ascent_predicted_apex_width_m=cfg.ascent_predicted_apex_width_m,
        ascent_predicted_apex_reward_scale=cfg.ascent_predicted_apex_reward_scale,
        apex_event_height_width_m=cfg.apex_event_height_width_m,
        apex_event_height_reward_scale=cfg.apex_event_height_reward_scale,
        apex_event_shortfall_penalty_scale=cfg.apex_event_shortfall_penalty_scale,
    )
    output.mkdir(parents=True)
    print('[RECOVERY] constructing vector environment', flush=True)
    # Select environment class based on stability profile
    if args.observation_layout == 'history_foot_v19':
        env_class = PathFreeStage2Env if args.path_free_stage2 else HistoryFootEnv
        env = RslRlVecEnvWrapper(env_class(cfg))
        CONTRACT['foot_geometry'] = env.unwrapped.foot_geometry
    elif args.stability_profile == 'stationary_corridor_v17':
        env = RslRlVecEnvWrapper(StationaryCorridorV17Env(cfg))
    else:
        env = RslRlVecEnvWrapper(SpatialTrackingEnv(cfg))
    print('[RECOVERY] vector environment ready', flush=True)

    rcfg = PlannerCircularPPORunnerCfg()
    rcfg.experiment_name = 'quadhopper_pair_precision_guidance'
    if args.observation_layout == 'history_foot_v19':
        rcfg.experiment_name = 'quadhopper_two_circle_history_foot_v19'
    if args.path_free_stage2:
        rcfg.experiment_name = 'quadhopper_two_circle_path_free_stage2'
        if args.path_free_stage2_v2:
            rcfg.experiment_name += '_v2'
        if args.path_free_stage2_v3:
            rcfg.experiment_name += '_v3'
        if args.v3_accounting_fix:
            rcfg.experiment_name += '_accounting_fix'
        if args.apex_event_fix_v15:
            rcfg.experiment_name += '_v15_apex_event_fix'
        if args.terminal_penalty_fix_v16:
            rcfg.experiment_name += '_v16_terminal_penalty_fix'
        if args.deferred_landing_v17:
            rcfg.experiment_name += '_v17_deferred_landing'
        if args.low_hop_recovery_v18:
            rcfg.experiment_name += '_v18_low_hop_recovery'
    rcfg.seed = args.seed
    rcfg.num_steps_per_env = 256
    rcfg.save_interval = 10
    rcfg.algorithm.schedule = 'fixed'
    rcfg.algorithm.learning_rate = CONTRACT['learning_rate']
    rcfg.algorithm.num_mini_batches = min(16, args.num_envs)
    rcfg.algorithm.entropy_coef = 0.0
    rcfg.empirical_normalization = False
    rcfg.policy.actor_obs_normalization = False
    rcfg.policy.critic_obs_normalization = False
    runner = RecoveryRunner(env, rcfg.to_dict(), log_dir=str(output), device=args.device)
    print('[RECOVERY] PPO runner ready', flush=True)
    resume_checkpoint_iteration = None
    if args.resume_runner_state:
        runner.load(str(source), load_optimizer=True, map_location=args.device)
        resume_checkpoint_iteration = runner.current_learning_iteration
        # RSL checkpoints are written after the numbered update.  Continue at
        # the following update so tensorboard/checkpoint indices are never
        # duplicated and ``iterations`` means additional PPO updates.
        runner.current_learning_iteration += 1
    else:
        runner.alg.policy.load_state_dict(state)
    runner.alg.policy.std.requires_grad_(False)
    if env.num_actions != 4 or env.get_observations()['policy'].shape[-1] != CONTRACT['obs_dim']:
        raise RuntimeError('Observation/action dimensions disagree with the selected contract')
    manifest = dict(
        CONTRACT,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        resumed_runner_state=args.resume_runner_state,
        resume_checkpoint_iteration=resume_checkpoint_iteration,
        resume_start_iteration=(
            runner.current_learning_iteration if args.resume_runner_state else None
        ),
        cli=vars(args), environment=cfg.to_dict(), runner=rcfg.to_dict(),
    )
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print('[RECOVERY] ' + json.dumps(CONTRACT), flush=True)
    diagnostic_stream = None
    probe_stream = None
    failure_trace_session = None
    velocity_audit = None
    closed_loop_p2_probe = None
    try:
        if args.mode == 'train':
            if not args.resume_runner_state:
                torch.save({'model_state_dict': runner.alg.policy.state_dict(), 'iter': 0,
                            'infos': {'tracking_recovery_contract': CONTRACT}}, output / 'initial.pt')
            runner.learn(args.iterations, init_at_random_ep_len=False)
            (output / 'train_complete.json').write_text(json.dumps({
                'iterations_requested': args.iterations,
                'final_iteration': runner.current_learning_iteration,
                'resumed_runner_state': args.resume_runner_state,
            }))
        else:
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = env.reset()
            base = env.unwrapped
            if args.failure_trace:
                from Quadhopper_Planner_Random.failure_trace import EvaluationFailureTrace
                failure_trace_session = EvaluationFailureTrace(
                    base, window_s=args.failure_trace_window_s,
                    record_next_target=args.takeoff_velocity_audit)
                print(f'[SPATIAL-EVAL] failure_trace: all vector envs, pre-reset samples, '
                      f'+/-{args.failure_trace_window_s:g} s', flush=True)
                if args.takeoff_velocity_audit:
                    from Quadhopper_Planner_Random.takeoff_velocity_audit import TakeoffVelocityAudit
                    velocity_audit = TakeoffVelocityAudit(
                        base.num_envs, base.step_dt, contact_phase=args.contact_phase_audit)
                    print('[SPATIAL-EVAL] takeoff_velocity_audit: all vector envs, '
                          'paired liftoff to touchdown', flush=True)
            if args.closed_loop_p2_mode is not None:
                from Quadhopper_Planner_Random.closed_loop_p2_probe import ClosedLoopP2Probe
                closed_loop_p2_probe = ClosedLoopP2Probe(
                    base, runner.alg.policy, args.closed_loop_p2_env_id,
                    args.closed_loop_p2_step, args.closed_loop_p2_mode,
                    expected_sha256=args.closed_loop_p2_expected_sha256,
                    episode_getter=lambda: failure_trace_session.trace.episodes[
                        args.closed_loop_p2_env_id])
                closed_loop_p2_probe.install()
                print(f'[P2-BRANCH] env={args.closed_loop_p2_env_id} '
                      f'step={args.closed_loop_p2_step} mode={args.closed_loop_p2_mode}', flush=True)
            diagnostic_writer = None
            previous_policy_action = None
            previous_motor_pwm = None
            if args.control_diagnostics:
                diagnostic_path = output / 'control_diagnostics.csv'
                diagnostic_stream = diagnostic_path.open('w', newline='')
                print(f'[SPATIAL-EVAL] control_diagnostics={diagnostic_path}', flush=True)
            if probe_env_ids:
                from Quadhopper_Planner_Random.next_hop_probe import probe_actor
                probe_path = output / 'next_hop_probe.jsonl'
                probe_stream = probe_path.open('w')
                print(f'[SPATIAL-EVAL] next_hop_probe={probe_path}', flush=True)
            progress_interval = max(1, min(1000, args.steps // 10))
            started_at = time.monotonic()
            hard_attitude_samples = []
            evaluation_done_count = torch.zeros((), dtype=torch.long, device=base.device)
            with torch.inference_mode():
                for step in range(args.steps):
                    should_probe = probe_stream is not None and step % args.next_hop_probe_stride == 0
                    if diagnostic_stream is not None or should_probe:
                        hidden_before = _clone_actor_hidden(runner.alg.policy)
                    if diagnostic_stream is not None:
                        pre_state = _state_snapshot(base)
                        policy_obs = obs['policy'][0].detach().cpu().tolist()
                    actions = policy(obs)
                    if should_probe and hidden_before is not None:
                        selected = [env_id for env_id in probe_env_ids if
                                    bool(base._cycle_active[env_id]) and
                                    not bool(base._initial_drop_pending[env_id])]
                        if selected:
                            for row in probe_actor(runner.alg.policy, obs['policy'],
                                                   hidden_before, selected, actions):
                                env_id = row['env_id']
                                spring = float(base._robot.data.joint_pos[env_id, base._spring_joint_id])
                                vz = float(base._robot.data.root_lin_vel_w[env_id, 2])
                                row.update(step=step, time_s=step * base.step_dt,
                                           phase=('contact' if spring > 0.002 else
                                                  'ascent' if vz > 0 else 'descent'),
                                           root_z_m=float(base._robot.data.root_pos_w[env_id, 2]),
                                           vertical_velocity_m_s=vz,
                                           spring_m=spring,
                                           foot_error_xy_m=obs['policy'][env_id, 54:56].detach().cpu().tolist(),
                                           actual_next_segment_body_normalized=obs['policy'][env_id, 39:41].detach().cpu().tolist())
                                probe_stream.write(json.dumps(row) + '\n')
                    if failure_trace_session is not None:
                        failure_trace_session.begin_step(actions)
                    if closed_loop_p2_probe is not None:
                        closed_loop_p2_probe.begin_step(obs['policy'], actions)
                    if diagnostic_stream is not None:
                        hidden_after = _clone_actor_hidden(runner.alg.policy)
                    obs, _, dones, _ = env.step(actions)
                    if closed_loop_p2_probe is not None:
                        closed_loop_p2_probe.after_step(obs['policy'], dones)
                    if velocity_audit is not None:
                        if failure_trace_session.trace.last_step != step + 1:
                            raise RuntimeError('takeoff audit missing pre-reset trace sample')
                        velocity_audit.observe(failure_trace_session.trace.history[-1],
                                               failure_trace_session.trace.fields)
                    evaluation_done_count += dones.long().sum()
                    if args.stability_profile in ('launch_robust_v10', 'yaw_event_v11',
                                                  'touchdown_robust_v12',
                                                  'height_robust_v13',
                                                  'cross_hop_v14', 'cross_hop_v15',
                                                  'cross_hop_v15_1', 'cross_hop_v16',
                                                  'two_circle_v18') and args.num_envs > 1:
                        hard_airborne = (
                            base._hardware_hard_mask
                            & base._cycle_active
                            & ~base._initial_drop_pending
                            & (base._robot.data.joint_pos[:, base._spring_joint_id] <= 0.002)
                        )
                        quat = base._robot.data.root_quat_w
                        _, pitch, yaw = euler_xyz_from_quat(quat)
                        tilt = torch.acos((1.0 - 2.0 * (quat[:, 1].square()
                            + quat[:, 2].square())).clamp(-1.0, 1.0))
                        hard_attitude_samples.append(torch.stack((
                            torch.rad2deg(pitch.abs()),
                            torch.rad2deg(yaw.abs()),
                            torch.rad2deg(tilt),
                            hard_airborne.float(),
                        ), dim=-1).detach())
                    if diagnostic_stream is not None:
                        post_state = _state_snapshot(base)
                        row: dict[str, float | int] = {
                            'Step': step,
                            'Pre_Time_s': step * base.step_dt,
                            'Post_Time_s': (step + 1) * base.step_dt,
                            'Done': int(bool(dones[0].detach().cpu())),
                        }
                        row.update({
                            f'Pre_{name}': value
                            for name, value in zip(DIAGNOSTIC_STATE_NAMES, pre_state, strict=True)
                        })
                        row.update({
                            f'Post_{name}': value
                            for name, value in zip(DIAGNOSTIC_STATE_NAMES, post_state, strict=True)
                        })
                        row.update({f'Obs{index:02d}': value for index, value in enumerate(policy_obs)})
                        if hasattr(base, 'foot_position_w'):
                            foot = base.foot_position_w()[0].detach().cpu().tolist()
                            row.update({f'Post_Foot_{axis}': value for axis, value in zip(('X','Y','Z'), foot)})
                        row.update(_motor_metrics('PolicyAction', actions[0]))
                        policy_action_rate = (
                            torch.zeros_like(actions[0])
                            if previous_policy_action is None
                            else (actions[0] - previous_policy_action) / base.step_dt
                        )
                        row.update(_rate_metrics('PolicyActionRate_per_s', policy_action_rate))
                        clamped_actions = actions[0].clamp(-1.0, 1.0)
                        row.update(_motor_metrics('ClampedAction', clamped_actions))
                        row['PolicyActionSaturationFraction'] = float(
                            (actions[0].abs() >= 1.0).float().mean().detach().cpu()
                        )
                        row.update(_motor_metrics('RawCmd_PWM', base._raw_target_u[0] * 1000.0))
                        row.update(_motor_metrics('ShapedCmd_PWM', base._shaped_target_u[0] * 1000.0))
                        motor_pwm = base._motor_u[0] * 1000.0
                        row.update(_motor_metrics('Motor_PWM', motor_pwm))
                        motor_pwm_rate = (
                            torch.zeros_like(motor_pwm)
                            if previous_motor_pwm is None
                            else (motor_pwm - previous_motor_pwm) / base.step_dt
                        )
                        row.update(_rate_metrics('Motor_PWM_Rate_per_s', motor_pwm_rate))
                        row.update(_hidden_metrics('RNN_', hidden_before, hidden_after))
                        row.update({
                            'Target_X': float(base._desired_pos_w[0, 0].detach().cpu()),
                            'Target_Y': float(base._desired_pos_w[0, 1].detach().cpu()),
                            'Target_Z': float(base._desired_pos_w[0, 2].detach().cpu()),
                            'CycleActive': int(bool(base._cycle_active[0].detach().cpu())),
                            'InitialDropPending': int(bool(base._initial_drop_pending[0].detach().cpu())),
                            'TouchdownEvent': int(bool(base._touchdown_event[0].detach().cpu())),
                            'LiftoffEvent': int(bool(base._liftoff_event[0].detach().cpu())),
                            'ApexEvent': int(bool(base._apex_event[0].detach().cpu())),
                        })
                        if diagnostic_writer is None:
                            diagnostic_writer = csv.DictWriter(diagnostic_stream, fieldnames=list(row))
                            diagnostic_writer.writeheader()
                        diagnostic_writer.writerow(row)
                        # Do not count an episode reset discontinuity as an action-rate spike.
                        if bool(dones[0].detach().cpu()):
                            previous_policy_action = None
                            previous_motor_pwm = None
                        else:
                            previous_policy_action = actions[0].detach().clone()
                            previous_motor_pwm = motor_pwm.detach().clone()
                        if (step + 1) % 100 == 0:
                            diagnostic_stream.flush()
                    runner.alg.policy.reset(dones)
                    if (step + 1) % progress_interval == 0 or step + 1 == args.steps:
                        if probe_stream is not None:
                            probe_stream.flush()
                        print(
                            f'[SPATIAL-EVAL] progress={step + 1}/{args.steps} '
                            f'elapsed_s={time.monotonic() - started_at:.1f}',
                            flush=True,
                        )
            result = base.tracking_metrics.summary()
            if velocity_audit is not None:
                audit_report = velocity_audit.export(output / 'takeoff_velocity_audit')
                result['takeoff_velocity_audit'] = audit_report
                print(f'[TAKEOFF-AUDIT] {output / "takeoff_velocity_audit" / "index.json"}',
                      flush=True)
            if closed_loop_p2_probe is not None:
                branch_report = closed_loop_p2_probe.finish(velocity_audit.rows)
                branch_path = output / 'closed_loop_p2_probe.json'
                branch_path.write_text(json.dumps(branch_report, indent=2, allow_nan=False) + '\n')
                result['closed_loop_p2_probe'] = dict(
                    path=str(branch_path), env_id=args.closed_loop_p2_env_id,
                    step=args.closed_loop_p2_step, mode=args.closed_loop_p2_mode,
                    state_sha256=branch_report['branch']['state_sha256'],
                    first_hop_completed=branch_report['first_hop_completed'],
                    second_hop_completed=branch_report['second_hop_completed'])
                print(f'[P2-BRANCH] {branch_path}', flush=True)
            if probe_stream is not None:
                probe_stream.flush()
                result['next_hop_probe'] = dict(
                    path=str(output / 'next_hop_probe.jsonl'),
                    environment_ids=probe_env_ids,
                    stride=args.next_hop_probe_stride,
                    variants=['actual', 'left90', 'reverse', 'stop'],
                    changed_observation_slice='current_frame[39:41] only',
                    actor_hidden_state='same pre-action state for all variants',
                    simulated_actions='actual only',
                )
            if failure_trace_session is not None:
                trace_directory = output / 'failure_traces'
                trace_report = failure_trace_session.close(trace_directory)
                failure_trace_session = None
                result['failure_trace'] = dict(
                    index=str(trace_directory / 'index.json'),
                    captured_environments=len(trace_report['events']))
                result['strict_hit_references'] = trace_report['strict_hit_references']
                print(f'[FAILURE-TRACE] {trace_directory / "index.json"}', flush=True)
            if hasattr(base, 'foot_summary'):
                result.update(base.foot_summary())
            if getattr(base, 'failure_diagnostics', None) is not None:
                # Captured inside _get_dones before Isaac resets a failed env.
                # Only the first physical death per environment is retained;
                # aggregate death_event_count above still counts every death.
                result['first_physical_death_by_env'] = (
                    base.failure_diagnostics.summary()
                )
            if hard_attitude_samples:
                samples = torch.cat(hard_attitude_samples, dim=0)
                hard_samples = samples[samples[:, 3] > 0.5, :3]
                result.update(
                    hard_airborne_samples=int(hard_samples.shape[0]),
                    hard_airborne_pitch_p90_deg=(
                        float(torch.quantile(hard_samples[:, 0], .9).cpu())
                        if len(hard_samples) else None
                    ),
                    hard_airborne_yaw_p90_deg=(
                        float(torch.quantile(hard_samples[:, 1], .9).cpu())
                        if len(hard_samples) else None
                    ),
                    hard_airborne_tilt_p90_deg=(
                        float(torch.quantile(hard_samples[:, 2], .9).cpu())
                        if len(hard_samples) else None
                    ),
                )
            guide = base._guide_sum
            raibert = base._raibert_sum
            result.update(checkpoint=str(source), steps=args.steps, num_envs=args.num_envs, seed=args.seed,
                          evaluation_done_count=int(evaluation_done_count.item()),
                          checkpoint_contract=spatial,
                          evaluation_contract=CONTRACT,
                          death_rate=(base._physical_death_total > 0).float().mean().item(),
                          death_env_count=int((base._physical_death_total > 0).sum().item()),
                          death_event_count=int(base._physical_death_total.sum().item()),
                          low_height_death_event_count=int(base._low_height_death_total.sum().item()),
                          tilt_death_event_count=int(base._tilt_death_total.sum().item()),
                          far_xy_death_event_count=int(base._far_xy_death_total.sum().item()),
                          high_angvel_death_event_count=int(base._high_angvel_death_total.sum().item()),
                          mean_path_distance_m=(guide[0]/guide[2].clamp_min(1)).item(),
                          path_corridor_rate=(guide[1]/guide[2].clamp_min(1)).item(),
                          mean_spatial_progress=(guide[3]/guide[2].clamp_min(1)).item(),
                          mean_raibert_foothold_shift_m=(raibert[0]/raibert[5].clamp_min(1)).item(),
                          mean_raibert_velocity_error_m_s=(raibert[1]/raibert[5].clamp_min(1)).item(),
                          mean_raibert_desired_speed_m_s=(raibert[2]/raibert[5].clamp_min(1)).item(),
                          mean_raibert_touchdown_speed_m_s=(raibert[3]/raibert[5].clamp_min(1)).item(),
                          mean_raibert_flight_time_s=(raibert[4]/raibert[5].clamp_min(1)).item(),
                          xy_progress_reward_per_env=(base._xy_progress_sum[0]/args.num_envs).item(),
                          xy_approach_reward_per_env=(base._xy_progress_sum[1]/args.num_envs).item(),
                          xy_retreat_penalty_per_env=(base._xy_progress_sum[2]/args.num_envs).item())
            if args.path_free_stage2:
                result.update(base.stage2_summary())
                # There is no reference curve; zero distance/100% corridor
                # would be misleading evidence of "perfect tracking".
                for key in tuple(result):
                    if key.startswith('mean_raibert_') or key in (
                        'mean_path_distance_m', 'path_corridor_rate', 'mean_spatial_progress'):
                        result[key] = None
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print('[SPATIAL-EVAL] ' + json.dumps(result), flush=True)
    finally:
        if failure_trace_session is not None:
            failure_trace_session.close(output / 'failure_traces')
        if diagnostic_stream is not None:
            diagnostic_stream.close()
        if probe_stream is not None:
            probe_stream.close()
        env.close()


try:
    main()
except BaseException:
    # Most failures happen after ``main`` creates ``output``.  Keep the
    # diagnostic beside the requested output when an earlier compatibility or
    # checkpoint check fails, so the original exception is not masked by a
    # second FileNotFoundError.
    failure_path = (
        output / 'startup_exception.txt'
        if output.exists()
        else output.with_suffix('.startup_exception.txt')
    )
    failure_path.write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    app.close()
