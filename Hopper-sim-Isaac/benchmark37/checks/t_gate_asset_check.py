"""Open the actual USD and check the V88 T-frame target against its colliders on CPU."""
from pathlib import Path

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ..crossing_obs73_quat import command_from_t_frame
from ..t_frame import DEFAULT_ENVELOPE


def main():
    asset = Path(__file__).resolve().parents[1] / "TGate_20mm.usda"
    stage = Usd.Stage.CreateInMemory()
    gate = UsdGeom.Xform.Define(stage, "/Fixture")
    assert gate.GetPrim().GetReferences().AddReference(str(asset))
    # A displaced and rotated environment catches silently hard-coded world axes.
    gate.AddTranslateOp().Set(Gf.Vec3d(3, 2, 0.95))
    gate.AddRotateXYZOp().Set(Gf.Vec3d(0, -12, 90))
    cache = UsdGeom.XformCache()
    transform = cache.GetLocalToWorldTransform(gate.GetPrim())
    frame = transform.ExtractRotationQuat()
    pos = torch.tensor([list(transform.ExtractTranslation())])
    quat = torch.tensor([[frame.GetReal(), *frame.GetImaginary()]])
    aligned = gate.GetPrim().GetAttribute("gate:alignedRootOffsetT").Get()
    assert Gf.IsClose(aligned, Gf.Vec3d(0), 1e-12)
    zero = torch.zeros_like(pos)
    target, attitude, _, _ = command_from_t_frame(
        pos, quat, torch.tensor([list(aligned)]), torch.tensor([[1., 0., 0., 0.]]), zero, zero)
    assert torch.allclose(target, pos)
    assert torch.allclose((attitude * quat).sum(-1).abs(), torch.ones(1))
    bounds = []
    for child in gate.GetPrim().GetChildren():
        cube = UsdGeom.Cube(child)
        assert cube and child.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
        inverse = cache.GetLocalToWorldTransform(child).GetInverse()
        bounds.append((inverse, cube.GetSizeAttr().Get() / 2))
    assert len(bounds) == 8
    top = UsdGeom.Xformable(gate.GetPrim().GetChild("Top")).GetLocalTransformation()
    shoulder = UsdGeom.Xformable(gate.GetPrim().GetChild("LeftShoulder")).GetLocalTransformation()
    opening_top = top.Transform(Gf.Vec3d(0, 0, -0.5))[2]
    opening_bottom = shoulder.Transform(Gf.Vec3d(0, 0, 0.5))[2]
    assert abs(opening_top - 0.035) < 1e-8 and abs(opening_bottom + 0.035) < 1e-8

    def hits_frame(x, y, z):
        point = transform.Transform(Gf.Vec3d(x, y, z))
        return any(max(map(abs, inv.Transform(point))) <= half for inv, half in bounds)

    # The aligned rotor tips and extended leg fit the nominal opening.
    top = aligned[2] + DEFAULT_ENVELOPE["top_z_A_m"]
    assert not hits_frame(0, 0.19, top)
    assert not hits_frame(0, 0, top - 0.30)
    # Root has passed the back face. A trailing point on the crossbar is
    # still inside the slab and, after a lateral offset, hits its end strip.
    assert not hits_frame(0.01, 0.03, aligned[2])
    assert hits_frame(0.00, 0.03 + 0.19, aligned[2])
    assert hits_frame(0.004, 0.06, -0.235)
    assert not hits_frame(0.006, 0.06, -0.235)
    assert not hits_frame(0.00, 0.50, 0.00)  # clear exterior allows bypass
    print("Passed: 8 static colliders, crossbar origin equals A target, nominal fit, "
          "root-past/edge-hit counterexample, finite thickness and exterior.")


if __name__ == "__main__":
    main()
