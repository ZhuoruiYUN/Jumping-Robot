# MuJoCo spatial-v11 parity player

This package runs the accepted Isaac checkpoint `model_39.pt` with a MuJoCo
implementation of its v11 policy contract.  It is intended for qualitative
cross-engine playback on macOS and Linux, not as a replacement for the Isaac
evaluation gate.

The player reproduces the following parts of the accepted contract:

- 52-D recurrent observation with the exact feature order;
- 20--30 cm rolling two-waypoint route and a maximum 30 degree turn;
- no-clock Hermite path, path projection, carrot and tangent;
- 1.00 m current/next apex command and the 4 degree landing-up hint;
- initial drop from `0.38 + U(0.65, 1.05)` m;
- 5 cm hit reporting and advance-on-miss route semantics;
- five-sample legacy action delay, 67.4 ms motor lag, calibrated thrust curve;
- `Izz = 7.931903e-4 kg m^2` and `Ktau = 1.720762152765e-2`.

MuJoCo contact constraints and PhysX contact constraints differ.  A successful
run should look qualitatively similar and receive the same policy inputs and
actuator contract, but is not expected to match Isaac frame-for-frame.

## macOS setup

```bash
cd mujoco_spatial_parity_v11
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python play_spatial_tracking_mujoco.py
```

The default `RENDER_PROFILE = 'offscreen_gif'` preserves the professor's
original workflow: Matplotlib analysis figures, a Pillow GIF, CSV and NPZ are
saved under `outputs/mujoco/`.  Change that one setting near the top of the
script as follows:

- `'native'`: use the live MuJoCo window; run with `.venv/bin/mjpython` on macOS.
- `'plots'`: PNG and Matplotlib plots only; no GIF.
- `'offscreen_gif'`: PNG, Matplotlib plots and GIF; this is the default.

Each run writes CSV, PNG and NPZ files under `outputs/mujoco/`.  The NPZ file
includes a `observations` array with shape `[policy_steps, 52]`, the exact input
fed to the LSTM before every action.  It is the artifact to compare with an
Isaac single-environment observation recording when diagnosing remaining
cross-engine differences.

## Contract lock

This package is specific to `model_39.pt`:

```text
checkpoint SHA-256: 157e2ab1b1ae362aff1a72d02a6b961ab0e8451e614ce4364f3862f2ea62b89a
observation width: 52
actions: 4
LSTM hidden width: 256
```

Do not replace `model_39.pt` with an arbitrary checkpoint.  A new policy may
have a different observation contract, route distribution, motor lag or action
delay.  Update this player from that checkpoint's manifest before using it.
