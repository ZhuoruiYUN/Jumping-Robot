"""Static T aperture for the V88 single-gate course, in gate-local metres.

The prim origin is the centre of the open crossbar, at mid-thickness.
Use its transform for both the crossing command and the collision sweep.
"""
import argparse
import math

from .t_frame import DEFAULT_ENVELOPE, boxes


def gate_layout(envelope, clearance_m, thickness_m, rim_width_m):
    boxes(envelope)  # Validate the source robot envelope.
    if not all(math.isfinite(v) and v > 0 for v in (clearance_m, thickness_m, rim_width_m)):
        raise ValueError("Gate clearance, axial thickness and rim width must be positive metres")
    half_width = envelope["width_m"] / 2 + clearance_m
    half_stem = envelope["stem_width_m"] / 2 + clearance_m
    top = envelope["bar_height_m"] / 2 + clearance_m
    bottom = top - envelope["height_m"] - 2 * clearance_m
    shoulder = -top
    rim = rim_width_m
    if half_stem >= half_width or shoulder - rim <= bottom:
        raise ValueError("The aperture needs distinct shoulders and a positive lower stem")

    # Rectangular offsets give the two inside shoulders square corners.
    # Each rectangle is a physical strip, with an open exterior for bypass tests.
    strips = {
        "Top": (-half_width - rim, half_width + rim, top, top + rim),
        "Bottom": (-half_stem - rim, half_stem + rim, bottom - rim, bottom),
    }
    for side, sign in (("Left", -1), ("Right", 1)):
        for label, inner, outer, low, high in (
            ("BarEnd", half_width, half_width + rim, shoulder - rim, top),
            ("Shoulder", half_stem, half_width, shoulder - rim, shoulder),
            ("Stem", half_stem, half_stem + rim, bottom, shoulder - rim),
        ):
            y_min, y_max = sorted((sign * inner, sign * outer))
            strips[side + label] = (y_min, y_max, low, high)
    return strips


def define_t_gate(stage, path, *, envelope=None, clearance_m=0.02,
                  thickness_m=0.01, rim_width_m=0.02):
    """Define an unscaled static gate before environment cloning, on a Z-up metre stage."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    envelope = DEFAULT_ENVELOPE if envelope is None else envelope
    strips = gate_layout(envelope, clearance_m, thickness_m, rim_width_m)
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("The T gate requires a Z-up stage in metres")
    if stage.GetPrimAtPath(path):
        raise ValueError(f"Gate path already exists: {path}")
    root = UsdGeom.Xform.Define(stage, path)
    prim = root.GetPrim()
    # Upright aligned body only. For a new prescribed pose re-evaluate the fit.
    offset_z = envelope["bar_height_m"] / 2 - envelope["top_z_A_m"]
    prim.CreateAttribute("gate:alignedRootOffsetT", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, offset_z))
    prim.CreateAttribute("gate:clearanceM", Sdf.ValueTypeNames.Double).Set(clearance_m)
    prim.CreateAttribute("gate:axialThicknessM", Sdf.ValueTypeNames.Double).Set(thickness_m)
    for name, (left, right, low, high) in strips.items():
        cube = UsdGeom.Cube.Define(stage, prim.GetPath().AppendChild(name))
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(0, (left + right) / 2, (low + high) / 2))
        cube.AddScaleOp().Set(Gf.Vec3f(thickness_m, right - left, high - low))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.35, 0.04)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return root


if __name__ == "__main__":
    from pxr import Usd, UsdGeom

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Output USD asset path")
    parser.add_argument("--clearance-m", type=float, default=0.02)
    args = parser.parse_args()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = define_t_gate(stage, "/TGate", clearance_m=args.clearance_m)
    stage.SetDefaultPrim(root.GetPrim())
    if not stage.GetRootLayer().Export(args.output):
        raise OSError(f"Could not export {args.output}")
