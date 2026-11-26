"""Loop the archived V88 +X first hop with the T gate in Isaac Sim."""
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (ROOT / "outputs/benchmark73_precision10_v50/v88_continuation_forecast"
           / "train_seed42_batch_20261126_014822_649684173Z/acceptance/cardinal")
SCENE = ROOT / "benchmark37/TGate_V88_preview.usda"


def load_first_moving_hop():
    import numpy as np

    case_id = 0
    events = json.loads((ARCHIVE / "straight_visual_events.json").read_text())[str(case_id)]
    moving_index = next(i for i, event in enumerate(events) if event["moving"])
    touchdown = events[moving_index]["touchdown_step"]
    start = events[moving_index - 1]["touchdown_step"] - 8 if moving_index else 1
    with np.load(ARCHIVE / "straight_visual_trace.npz") as archive:
        column = list(archive["columns"]).index
        samples = archive["trace"][:, list(archive["case_ids"]).index(case_id)]
    take = (samples[:, column("step")] >= start) & (samples[:, column("step")] <= touchdown + 1)
    samples = samples[take]
    positions = samples[:, [column(key) for key in ("root_x_m", "root_y_m", "height_m")]]
    attitudes = samples[:, [column(key) for key in ("q_w", "q_x", "q_y", "q_z")]]
    return positions, attitudes


def author_replay(stage, positions, attitudes):
    """Animate the recorded root pose in the session layer at half speed."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    frame_rate = 50  # Archive poses are sampled at 100 Hz.
    hopper = stage.GetPrimAtPath("/World/Hopper")
    if not hopper:
        raise RuntimeError("The opened stage does not contain /World/Hopper")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in Usd.PrimRange(hopper):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

        transform = UsdGeom.Xformable(hopper)
        translate, = transform.GetOrderedXformOps()
        orient = transform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        for frame, (p, q) in enumerate(zip(positions, attitudes, strict=True)):
            translate.Set(Gf.Vec3d(*map(float, p)), frame)
            quat = Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))
            orient.Set(quat.GetNormalized(), frame)

        end_frame = len(positions) - 1 + frame_rate // 2
        stage.SetTimeCodesPerSecond(frame_rate)
        stage.SetFramesPerSecond(frame_rate)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(end_frame)
    return end_frame / frame_rate


def main():
    positions, attitudes = load_first_moving_hop()
    from isaacsim import SimulationApp

    # Isaac Sim 5.1 otherwise creates a blank stage after opening open_usd.
    app = SimulationApp({"headless": False, "open_usd": str(SCENE), "create_new_stage": False})
    try:
        import omni.timeline
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Sdf

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac Sim could not load {SCENE}")
        duration = author_replay(stage, positions, attitudes)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac Sim did not create a viewport for the replay")
        viewport.camera_path = Sdf.Path("/World/PreviewCamera")
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(duration)
        timeline.set_current_time(0)
        timeline.set_looping(True)
        timeline.play()
        print(f"V88 T-gate replay ready: {len(positions)} poses at 0.5x speed", flush=True)
        while app.is_running():
            app.update()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise SystemExit(1) from None
    finally:
        app.close()


if __name__ == "__main__":
    main()
