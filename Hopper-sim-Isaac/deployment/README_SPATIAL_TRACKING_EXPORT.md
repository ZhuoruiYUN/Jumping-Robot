# 52-D spatial-tracking policy export

This package contains a deterministic recurrent actor from a selected
spatial-tracking checkpoint. `policy_metadata.json` is the authoritative
machine-readable input, output, checkpoint, hash, and actuator contract for
the checkpoint copied into the same export directory.

The ONNX model has three inputs (`obs`, `h_in`, `c_in`) and three outputs
(`actions`, `h_out`, `c_out`).  Call it at 100 Hz and carry both LSTM states
between calls.  Reset them on environment/mission reset, disarm, estimator
reset, or fall detection.  Clamp actions to `[-1, 1]`, then compute normalized
motor targets with `u = clip(0.5 * action + 0.5, 0, 1)` in motor order
`F1,F2,F3,F4`.

Install the lightweight inference dependencies and run the included check:

```bash
python -m pip install -r requirements-runtime.txt
python spatial_tracking_onnx_inference.py quadhopper_spatial_tracking_policy.onnx
```

The ONNX file contains only the policy actor.  It does not generate the 52-D
observation.  A complete controller must reproduce the exact ordering and
scaling in `policy_metadata.json`, including the untimed Hermite path, path
projection/carrot, current and next waypoint commands, contact/apex events,
five-action history, body/world transforms, and `wxyz` quaternion convention.
It also needs state estimation, actuator handling, and independent safety
limits.

To reproduce the Isaac Sim visualization, use the full Hopper-sim-Isaac source
tree and the `.pt` checkpoint in the matching Isaac Sim 5.1/Isaac Lab setup.
ONNX Runtime alone does not contain the simulator, robot USD, environment, or
target generator.
