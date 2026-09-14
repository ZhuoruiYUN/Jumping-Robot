# Retained experiment anchors

- `outputs/tracking_recovery/precision_zero40_08_12_turn30_nominal_v1/model_119.pt` — hardware-proven v11 40_v1 source checkpoint.
- `outputs/tracking_recovery/isaac_40v1_stationary_1m/` — native-coordinate 1 m stationary Isaac evaluation and its CSV analysis.
- `deployment/spatial_tracking_export_precision_zero40_v1/` — deployable 40_v1 export.
- `deployment/spatial_tracking_export_hardware_mixed_v1/` — retained mixed-hardware comparison export.

All old calibration, failed height-reward, smoke, curriculum and release artifacts were intentionally purged on 2026-09-14. New native-Isaac evaluations should use `./run_eval_40v1_isaac.sh`.
