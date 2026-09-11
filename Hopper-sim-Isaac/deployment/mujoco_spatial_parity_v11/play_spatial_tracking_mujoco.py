"""MuJoCo playback for the Isaac spatial-v11 ``model_39.pt`` contract.

This is a self-contained macOS/Linux player: numpy, PyTorch, MuJoCo,
Matplotlib, tqdm and Pillow are the only dependencies.  It deliberately does
not import Isaac, Isaac Lab, Gymnasium or RSL-RL.

It reproduces the *policy contract* from the accepted Isaac run: its rolling
two-waypoint command, untimed Hermite carrot, 52-D observation, motor delay
and lag, and the 2026-09-05 identified yaw parameters.  MuJoCo and PhysX use
different contact solvers, so it is a close cross-engine validation rather
than a bitwise dynamics replay.
"""
from pathlib import Path
import sys
import time
import struct

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'model_39.pt'
DURATION = 20.0  # s, simulation
DT = 0.001  # s, physics step
CONTROL_DT = 0.01  # s, policy step
MOTOR_TIME_CONSTANT = 0.0674  # s
ACTION_DELAY_SAMPLES = 5  # legacy Isaac ``play_action_delay=0`` selects oldest
WAYPOINT_OBS_SCALE = 0.20  # m
LANDING_ROOT_HEIGHT = 0.38  # m
TARGET_APEX_HEIGHT = 1.00  # m; current and next command
DISTANCE_MIN, DISTANCE_MAX = 0.20, 0.30  # m, accepted model contract
MAX_TURN_ANGLE_DEG = 30.0
TARGET_TOLERANCE = 0.05  # m
INITIAL_DROP_MIN, INITIAL_DROP_MAX = 0.65, 1.05  # above grounded root height
LANDING_HINT_TILT_DEG = 4.0
SEED = 242  # evaluation seed
# ``offscreen_gif`` preserves the professor's original Matplotlib/Pillow flow.
# Switch this one value to ``native`` for a live MuJoCo window, or ``plots``
# for PNG/interactive plots without creating a GIF.
RENDER_PROFILE = 'offscreen_gif'  # one of: offscreen_gif, plots, native
if RENDER_PROFILE not in {'offscreen_gif', 'plots', 'native'}:
    raise ValueError('RENDER_PROFILE must be offscreen_gif, plots, or native')
SHOW_PLOTS = RENDER_PROFILE in {'offscreen_gif', 'plots'}
SAVE_ANIMATION = RENDER_PROFILE == 'offscreen_gif'
REALTIME_VIEWER = RENDER_PROFILE == 'native'
FPS = 25  # Hz, animation
OUTPUT_DIR = ROOT / 'outputs/mujoco'

import numpy as np
from numpy import *
import torch
import mujoco
import mujoco.viewer
from tqdm import tqdm


def build_model():
    """Build the two-body hopper from the same CAD placement and inertias as USD."""
    # Transform the STL vertices in memory; no generated mesh files are needed.
    records = []
    triangle_type = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)),
                              ('attribute', '<u2')])
    for name in ('Jump Base.STL', 'Jump Leg.STL'):
        raw = (ROOT / 'Quadhopper_Stable/model' / name).read_bytes()
        records.append(np.frombuffer(raw, dtype=triangle_type, offset=84).copy())
    vertices = concatenate([r['vertices'].reshape(-1, 3) for r in records])
    anchor = (vertices.min(0) + vertices.max(0)) / 2  # mm, shared CAD origin
    rotation = array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])  # Rz(90) Ry(-90)
    for record, z in zip(records, (-0.075, -0.100)):
        record['vertices'] = (record['vertices'] - anchor) * 0.001 @ rotation.T + [0, 0, z]
        # Some CAD STL facets contain invalid/overflowed stored normals.  MuJoCo
        # only needs a valid triangle normal, so recompute it from transformed
        # vertices instead of rotating the unreliable file value.
        edge_1 = record['vertices'][:, 1] - record['vertices'][:, 0]
        edge_2 = record['vertices'][:, 2] - record['vertices'][:, 0]
        normals = cross(edge_1, edge_2)
        lengths = linalg.norm(normals, axis=1, keepdims=True)
        record['normal'] = divide(normals, lengths, out=zeros_like(normals), where=lengths > 1e-12)
    leg_vertices = records[1]['vertices'].reshape(-1, 3)
    foot = (leg_vertices.min(0) + leg_vertices.max(0)) / 2
    foot[2] = leg_vertices[:, 2].min() + 0.002 - 0.14  # m, rod tip
    records[1]['vertices'] -= foot
    assets = {name: b'\0' * 80 + struct.pack('<I', len(record)) + record.tobytes()
              for name, record in zip(('base.stl', 'leg.stl'), records)}
    motors = [(-0.0813, 0.0813), (-0.0813, -0.0813),
              (0.0813, -0.0813), (0.0813, 0.0813)]
    motor_xml = ''.join(f'<geom type="cylinder" pos="{x} {y} 0" size="0.015 0.0025" '
                        f'rgba="{0.8 if i % 2 else 0.15} 0.15 0.15 1"/>'
                        for i, (x, y) in enumerate(motors))
    xml = f'''<mujoco model="quadhopper">
      <compiler inertiafromgeom="false"/>
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
      <visual><global offwidth="640" offheight="480"/><headlight ambient="0.5 0.5 0.5"/></visual>
      <default><geom friction="1.5 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
        <joint solreflimit="0.004 1"/></default>
      <asset><mesh name="base" file="base.stl"/><mesh name="leg" file="leg.stl"/>
        <texture name="floor" type="2d" builtin="checker" rgb1="0.82 0.84 0.86"
                 rgb2="0.92 0.93 0.94" width="128" height="128"/>
        <material name="floor" texture="floor" texrepeat="8 8"/></asset>
      <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="10 10 0.1" material="floor"/>
        <site name="target" type="sphere" size="0.025" pos="0 0 0.015" rgba="0.1 0.7 0.2 1"/>
        <site name="next_target" type="sphere" size="0.018" pos="0 0 0.015" rgba="0.8 0.6 0.1 1"/>
        <site name="reference" type="sphere" size="0.018" rgba="0.7 0.2 0.7 0.7"/>
        <body name="Body" pos="0 0 1">
          <freejoint/>
          <inertial pos="0 0 0" mass="0.173" diaginertia="0.001231252 0.001286169 0.0007931903"/>
          <geom type="box" size="0.02 0.02 0.006" rgba="0.2 0.5 0.8 1"/>
          <geom type="cylinder" fromto="-0.0813 -0.0813 0 0.0813 0.0813 0" size="0.003"/>
          <geom type="cylinder" fromto="-0.0813 0.0813 0 0.0813 -0.0813 0" size="0.003"/>
          {motor_xml}
          <geom type="mesh" mesh="base" rgba="0.15 0.15 0.18 1"/>
          <body name="SpringLeg" pos="{' '.join(map(str, foot))}">
            <joint name="center_spring_joint" type="slide" axis="0 0 1" range="0 0.08"
                   stiffness="604" damping="2" springref="0"/>
            <inertial pos="0 0 0" mass="0.01" diaginertia="0.00001 0.00001 0.00001"/>
            <geom name="foot" type="sphere" size="0.012" rgba="0.8 0.15 0.1 1"/>
            <geom type="cylinder" fromto="0 0 0 0 0 0.14" size="0.003" rgba="0.6 0.6 0.6 1"/>
            <geom type="mesh" mesh="leg" rgba="0.9 0.75 0.1 1"/>
          </body>
        </body>
      </worldbody>
      <contact><exclude body1="Body" body2="SpringLeg"/></contact>
    </mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return model, mujoco.MjData(model), float(-foot[2] + 0.012)


def load_policy(checkpoint):
    """Load a supported 43-D or 52-D recurrent actor without Isaac/RSL-RL."""
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint_data['model_state_dict']
    input_weight = state['memory_a.rnn.weight_ih_l0']
    obs_dim = int(input_weight.shape[1])
    hidden_size = int(input_weight.shape[0] // 4)
    if hidden_size != 256 or obs_dim not in (43, 52):
        raise ValueError(
            f'Unsupported policy contract: obs={obs_dim}, hidden={hidden_size}; '
            'this player supports 43-D and model_39-style 52-D policies.'
        )
    memory = torch.nn.LSTM(obs_dim, hidden_size, 1)
    memory.load_state_dict({k.removeprefix('memory_a.rnn.'): v for k, v in state.items()
                            if k.startswith('memory_a.rnn.')})
    actor = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 4))
    actor.load_state_dict({k.removeprefix('actor.'): v for k, v in state.items()
                           if k.startswith('actor.')})
    return memory.eval(), actor.eval(), obs_dim


def _unit_xy(vector):
    norm = linalg.norm(vector[:2])
    return vector[:2] / norm if norm > 1e-8 else array([1.0, 0.0])


def hop_curve(start, landing, following, apex_height, nodes=33):
    """Numpy transcription of Quadhopper_Planner_Random.spatial_path.hop_curve."""
    delta = landing - start
    outgoing = following - landing
    outgoing[2] = 0.0
    outgoing /= linalg.norm(outgoing) + 1e-8
    incoming = delta.copy()
    incoming[2] = 0.0
    length = linalg.norm(incoming)
    incoming /= length + 1e-8
    landing_direction = 0.75 * incoming + 0.25 * outgoing
    landing_direction /= linalg.norm(landing_direction) + 1e-8
    peak = (start + landing) * 0.5
    peak[2] = apex_height
    t0 = 0.25 * length * incoming
    t0[2] = 2.0 * max(apex_height - start[2], 0.0)
    tm = 0.65 * length * incoming
    t1 = 0.25 * length * landing_direction
    t1[2] = -2.0 * max(apex_height - landing[2], 0.0)
    u = linspace(0.0, 1.0, nodes // 2 + 1)[:, None]

    def segment(a, b, ta, tb):
        return ((2*u**3 - 3*u**2 + 1)*a + (u**3 - 2*u**2 + u)*ta
                + (-2*u**3 + 3*u**2)*b + (u**3 - u**2)*tb)

    return concatenate((segment(start, peak, t0, tm), segment(peak, landing, tm, t1)[1:]))


def project_path(position, curve, descending, minimum_fraction, lookahead_m=0.10):
    """Numpy transcription of the untimed spatial path projection."""
    start = curve[:-1]
    segment = curve[1:] - start
    length = linalg.norm(segment, axis=1).clip(1e-8)
    fraction = clip(((position - start) * segment).sum(1) / square(length), 0.0, 1.0)
    projected = start + fraction[:, None] * segment
    distance_sq = square(position - projected).sum(1)
    indices = arange(len(segment))
    allowed = indices >= len(segment) // 2 if descending else indices < len(segment) // 2
    chosen = where(allowed, distance_sq, inf).argmin()
    arc_starts = concatenate(([0.0], cumsum(length)[:-1]))
    total = length.sum()
    raw_arc = arc_starts[chosen] + fraction[chosen] * length[chosen]
    arc = maximum(raw_arc, clip(minimum_fraction, 0.0, 1.0) * total)
    forward_arc = minimum(arc + lookahead_m, total)
    forward = minimum((cumsum(length) < forward_arc).sum(), len(segment) - 1)
    forward_fraction = clip((forward_arc - arc_starts[forward]) / length[forward], 0.0, 1.0)
    carrot = start[forward] + forward_fraction * segment[forward]
    tangent = segment[forward] / length[forward]
    return carrot, tangent, sqrt(distance_sq[chosen]), raw_arc / max(total, 1e-8)


class RollingTwoHopRoute:
    """The v11 20--30 cm, +/-30 degree rolling waypoint distribution."""
    def __init__(self, rng, anchor):
        self.rng = rng
        self.anchor = asarray(anchor, dtype=float).copy()
        self.target = self.anchor.copy()
        self.next_target = self.anchor.copy()
        self.reset(anchor)

    def _radius(self):
        return self.rng.uniform(DISTANCE_MIN, DISTANCE_MAX)

    def _offset(self, heading):
        return self._radius() * array([cos(heading), sin(heading)])

    def reset(self, anchor):
        self.anchor = asarray(anchor, dtype=float).copy()
        heading = self.rng.uniform(-pi, pi)
        first = self._offset(heading)
        second_heading = heading + self.rng.uniform(-radians(MAX_TURN_ANGLE_DEG),
                                                    radians(MAX_TURN_ANGLE_DEG))
        self.target = self.anchor + first
        self.next_target = self.target + self._offset(second_heading)

    def advance(self):
        incoming = self.next_target - self.target
        heading = arctan2(incoming[1], incoming[0])
        self.anchor = self.target.copy()
        self.target = self.next_target.copy()
        heading += self.rng.uniform(-radians(MAX_TURN_ANGLE_DEG), radians(MAX_TURN_ANGLE_DEG))
        self.next_target = self.target + self._offset(heading)


def body_velocity(model, data, body_id):
    """Return root linear and angular velocity in the body frame."""
    velocity = empty(6, dtype=float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 1)
    return velocity[3:], velocity[:3]


def observation(model, data, body_id, target, next_target, desired, carrot, tangent,
                landing_hint, history):
    """Exact 52-D observation layout of SpatialTrackingEnv model_39."""
    rotation = data.xmat[body_id].reshape(3, 3)  # body -> world
    pos = data.qpos[:3]
    linear_velocity_b, angular_velocity_b = body_velocity(model, data, body_id)
    current_error = rotation.T @ append(target - pos[:2], 0.0)
    next_displacement = rotation.T @ append(next_target - target, 0.0)
    stable = concatenate((linear_velocity_b, angular_velocity_b, data.qpos[3:7],
                          rotation.T @ (desired - pos), [pos[2], data.qpos[7] > 0.002,
                          data.qpos[7], data.qvel[6]], history.ravel()))
    planner_obs = concatenate((current_error[:2] / WAYPOINT_OBS_SCALE,
                               next_displacement[:2] / WAYPOINT_OBS_SCALE,
                               [TARGET_APEX_HEIGHT / 2.0, TARGET_APEX_HEIGHT / 2.0]))
    carrot_error_b = clip(rotation.T @ (carrot - pos), -1.0, 1.0)
    return concatenate((stable, planner_obs, carrot_error_b, rotation.T @ tangent,
                        rotation.T @ landing_hint)).astype(float32)


def motor_wrench(u):
    thrust = maximum(-0.2371 * u**2 + 0.8130 * u + 0.0113, 0)  # N, per motor
    f1, f2, f3, f4 = thrust
    torque = array([0.0813 * (f1 + f4 - f2 - f3),
                    0.0813 * (f1 + f2 - f3 - f4),
                    -1.720762152765e-02 * (u[0]**2 + u[2]**2 - u[1]**2 - u[3]**2)])  # N m
    return thrust, torque


# %% Load the model and the v11 policy contract.
if __name__ == '__main__':
    assert DURATION > 0 and DT > 0 and CONTROL_DT >= DT
    assert isclose(CONTROL_DT / DT, round(CONTROL_DT / DT))
    assert ACTION_DELAY_SAMPLES == 5 and MOTOR_TIME_CONSTANT >= 0
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model, data, rest_height = build_model()
    memory, actor, obs_dim = load_policy(CHECKPOINT)
    if obs_dim != 52:
        raise ValueError(f'This player implements the spatial-v11 52-D contract, got {obs_dim}-D')
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'Body')
    hidden = None  # -, LSTM state
    route = RollingTwoHopRoute(rng, array([0.0, 0.0]))
    data.qpos[:3] = [0.0, 0.0, LANDING_ROOT_HEIGHT + rng.uniform(INITIAL_DROP_MIN, INITIAL_DROP_MAX)]
    mujoco.mj_forward(model, data)
    print(f'Checkpoint: {CHECKPOINT}\nPolicy observations: {obs_dim}-D\n'
          f'MuJoCo {mujoco.__version__}; geometric rest height: {rest_height:.4f} m\n'
          f'Contract: 20--30 cm, +/-{MAX_TURN_ANGLE_DEG:.0f} deg, 5 cm, untimed spatial path')
    history = zeros((5, 4), dtype=float32)  # -, applied policy history
    motor_u = zeros(4)  # -, filtered throttle
    waypoint_index = 0  # touchdown count
    active = False  # -, flight phase
    confirmed = False  # -, observed spring contact
    previous_contact = False  # -, previous control sample
    drop_pending = True
    apex = 0.0  # m, measured flight apex
    touchdowns = 0  # -, completed flights
    descending = False
    path_fraction = 0.0
    prior_vertical_velocity = 0.0
    curve = None
    records = []
    poses = []
    observations = []
    stop_reason = 'duration reached'
    steps = int(round(DURATION / CONTROL_DT))

    def replan():
        global curve, descending, path_fraction
        start = data.qpos[:3].copy()
        landing = append(route.target, LANDING_ROOT_HEIGHT)
        following = append(route.next_target, LANDING_ROOT_HEIGHT)
        curve = hop_curve(start, landing, following, TARGET_APEX_HEIGHT)
        descending = False
        path_fraction = 0.0

    replan()

    viewer = None
    if REALTIME_VIEWER:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except RuntimeError as error:
            if sys.platform == 'darwin' and 'mjpython' in str(error):
                raise SystemExit(
                    'MuJoCo native live viewer on macOS must use mjpython.\n'
                    f'Run:\n  cd "{ROOT}"\n  ./.venv/bin/mjpython play_spatial_tracking_mujoco.py\n'
                    'PyCharm\'s ordinary Python Run button cannot launch this native viewer.'
                ) from error
            raise
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -18
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = data.qpos[:3]

    # %% Simulate. Every row records the pre-action state and the ensuing held force.
    for step in tqdm(range(steps), desc='MuJoCo', unit='step'):
        wall_step_start = time.perf_counter()
        if viewer is not None and not viewer.is_running():
            stop_reason = 'viewer closed'
            break
        contact = data.qpos[7] > 0.002
        liftoff = confirmed and previous_contact and not contact
        touchdown = active and not previous_contact and contact
        confirmed = confirmed or contact
        if drop_pending and contact:
            drop_pending = False
            replan()
        if liftoff:
            active = True
            apex = data.qpos[2]
            replan()
        if active:
            apex = maximum(apex, data.qpos[2])
            if prior_vertical_velocity > 0.0 and data.qvel[2] <= 0.0:
                descending = True
        if touchdown:
            touchdowns += 1
            error = linalg.norm(data.qpos[:2] - route.target)
            hit = error < TARGET_TOLERANCE
            tqdm.write(f't={data.time:.2f} s: landing error={error:.3f} m, apex={apex:.3f} m, hit={hit}')
            waypoint_index += 1
            # v11 consumes P_t after *every* physical touchdown, including a miss.
            route.advance()
            active = False
            replan()

        obs_target, obs_next = route.target, route.next_target
        if drop_pending:
            obs_target = obs_next = route.anchor
        desired = append(obs_target, TARGET_APEX_HEIGHT)
        carrot, tangent, _, raw_fraction = project_path(
            data.qpos[:3], curve, descending, path_fraction
        )
        path_fraction = maximum(path_fraction, raw_fraction)
        next_direction = _unit_xy(route.next_target - route.target)
        landing_hint = append(sin(radians(LANDING_HINT_TILT_DEG)) * next_direction,
                              cos(radians(LANDING_HINT_TILT_DEG)))
        obs = observation(model, data, body_id, obs_target, obs_next, desired, carrot,
                          tangent, landing_hint, history)
        observations.append(obs.copy())
        with torch.inference_mode():
            latent, hidden = memory(torch.from_numpy(obs).reshape(1, 1, obs_dim), hidden)
            actions = clip(actor(latent[0])[0].numpy(), -1, 1)
        history = roll(history, -1, axis=0)
        history[-1] = actions
        target_u = clip(0.5 * history[-ACTION_DELAY_SAMPLES] + 0.5, 0, 1)
        alpha = CONTROL_DT / (MOTOR_TIME_CONSTANT + CONTROL_DT)
        motor_u += alpha * (target_u - motor_u)
        thrust, torque = motor_wrench(motor_u)
        w, x, y, z = data.qpos[3:7]
        attitude = array([arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                          arcsin(clip(2*(w*y-z*x), -1, 1)),
                          arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])
        records.append(concatenate(([data.time], data.qpos[:3], desired, data.qvel[:3],
                        zeros(3), attitude, data.qvel[3:6], data.qpos[7:8],
                        data.qvel[6:7], [contact], actions, motor_u, thrust,
                        route.target, route.next_target,
                        [TARGET_APEX_HEIGHT, TARGET_APEX_HEIGHT], [active, touchdown, waypoint_index])))
        poses.append(data.qpos.copy())
        if not isfinite(records[-1]).all():
            raise FloatingPointError(f'Non-finite state at t={data.time:.3f} s')
        if data.qpos[2] < 0.05 or x*x + y*y > 0.5:
            stop_reason = 'robot fell (height or tilt limit)'
            break
        for substep in range(int(round(CONTROL_DT / DT))):
            rotation = data.xmat[body_id].reshape(3, 3)
            data.xfrc_applied[body_id, :3] = rotation @ array([0, 0, thrust.sum()])
            data.xfrc_applied[body_id, 3:] = rotation @ torque
            # Match the small velocity damping in the source asset.
            data.xfrc_applied[body_id, :3] -= 0.05 * model.body_mass[body_id] * data.qvel[:3]
            data.xfrc_applied[body_id, 3:] -= rotation @ (0.05 * model.body_inertia[body_id] * data.qvel[3:6])
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        previous_contact = contact
        prior_vertical_velocity = data.qvel[2]
        if viewer is not None:
            model.site_pos[0] = append(route.target, 0.02)
            model.site_pos[1] = append(route.next_target, 0.02)
            model.site_pos[2] = carrot
            viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.5]
            viewer.sync()
            remaining = CONTROL_DT - (time.perf_counter() - wall_step_start)
            if remaining > 0:
                time.sleep(remaining)

    if viewer is not None:
        viewer.close()

    # %% Save all histories and draw the key variables versus time.
    records = array(records)
    poses = array(poses)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%m%d_%H%M%S') + f'_{time.time_ns() % 1000000:06d}'
    prefix = OUTPUT_DIR / f'planner_{stamp}'
    columns = ('time_s x_m y_m z_m ref_x_m ref_y_m ref_z_m vx_mps vy_mps vz_mps '
               'ref_vx_mps ref_vy_mps ref_vz_mps roll_rad pitch_rad yaw_rad '
               'wx_radps wy_radps wz_radps spring_m spring_mps contact '
               'a1 a2 a3 a4 u1 u2 u3 u4 f1_N f2_N f3_N f4_N '
               'target_x_m target_y_m next_x_m next_y_m apex_command_m next_apex_command_m '
               'flight touchdown waypoint_index')
    savetxt(str(prefix) + '.csv', records, delimiter=',', header=columns.replace(' ', ','), comments='')
    savez_compressed(str(prefix) + '.npz', history=records, qpos=poses,
                     observations=array(observations), columns=columns.split(),
                     checkpoint=str(CHECKPOINT), dt=DT, control_dt=CONTROL_DT,
                     motor_time_constant=MOTOR_TIME_CONSTANT,
                     action_delay=ACTION_DELAY_SAMPLES,
                     target_tolerance_m=TARGET_TOLERANCE, stop_reason=stop_reason)
    print(f'{stop_reason}; {touchdowns} touchdowns; {waypoint_index} waypoint hits.\nSaved: {prefix}')

    import matplotlib
    matplotlib.use('MacOSX' if SHOW_PLOTS and sys.platform == 'darwin' else 'Agg')
    from matplotlib.pyplot import *
    from matplotlib.animation import FuncAnimation
    figure(figsize=(12, 9))
    t = records[:, 0]
    for i, (actual, desired, unit, labels) in enumerate([
        (slice(1, 4), slice(4, 7), 'Position [m]', ('x', 'y', 'z')),
        (slice(7, 10), slice(10, 13), 'Velocity [m/s]', ('vx', 'vy', 'vz')),
        (slice(13, 16), None, 'Attitude [rad]', ('roll', 'pitch', 'yaw')),
        (slice(16, 19), None, 'Angular velocity [rad/s]', ('wx', 'wy', 'wz')),
        (slice(19, 20), None, 'Spring displacement [m]', ('spring',)),
        (slice(20, 21), None, 'Spring velocity [m/s]', ('spring',)),
        (slice(30, 34), None, 'Motor thrust [N]', ('F1', 'F2', 'F3', 'F4')),
        (slice(26, 30), None, 'Motor input [-]', ('u1', 'u2', 'u3', 'u4')),
    ]):
        subplot(4, 2, i + 1)
        lines = plot(t, records[:, actual])
        for line, label in zip(lines, labels):
            line.set_label(label)
        if desired is not None:
            for j, line in enumerate(lines):
                plot(t, records[:, desired][:, j], '--', color=line.get_color(), linewidth=0.9)
        xlabel('Time [s]')
        ylabel(unit)
        legend(fontsize=7, loc='upper right')
    tight_layout()
    savefig(str(prefix) + '.png', dpi=160)

    # %% Replay the actual MuJoCo poses. Markers: green target, gold next, purple reference.
    if SHOW_PLOTS or SAVE_ANIMATION:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=480, width=640)
        replay = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation, camera.distance = 135, -18, 2.5
        frames = unique(append(arange(0, len(records), int(maximum(1, round(1 / (FPS * CONTROL_DT))))), len(records)-1))
        animation_figure = figure(figsize=(8, 6))
        picture = imshow(zeros((480, 640, 3), dtype=uint8))
        axis('off')
        tight_layout(pad=0)

        def draw_frame(index):
            replay.qpos[:] = poses[index]
            model.site_pos[0] = append(records[index, 34:36], 0.02)
            model.site_pos[1] = append(records[index, 36:38], 0.02)
            model.site_pos[2] = records[index, 4:7]
            mujoco.mj_forward(model, replay)
            camera.lookat[:] = [poses[index, 0], poses[index, 1], 0.5]
            renderer.update_scene(replay, camera=camera)
            picture.set_data(renderer.render())
            return (picture,)

        if SAVE_ANIMATION:
            images = []
            for index in tqdm(frames, desc='Rendering animation', unit='frame'):
                draw_frame(index)
                images.append(Image.fromarray(asarray(picture.get_array())).convert('P', palette=Image.Palette.ADAPTIVE))
            images[0].save(str(prefix) + '.gif', save_all=True, append_images=images[1:],
                           duration=1000 / FPS, loop=0)
            del images
        if SHOW_PLOTS:
            animation = FuncAnimation(animation_figure, draw_frame, frames=frames,
                                      interval=1000 / FPS, blit=False, repeat=True)
            pause(1e-5)
            show()
        renderer.close()
    if not SHOW_PLOTS:
        close('all')
