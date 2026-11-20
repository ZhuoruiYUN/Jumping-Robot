"""Provenance and immutable run folders; imports no simulator."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Quadhopper_PPO_Benchmark37"
SYSID_PACKAGE = ROOT / "Quadhopper_PPO_Reference_SysID"
SYSID_HANDOFF = Path("/home/terry/Downloads/参数辨识交接文档.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_sources():
    lock = json.loads((ROOT / "benchmark37/source_lock.json").read_text())
    for name, expected in lock["files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Frozen reference changed: {name}; create a new profile, do not overwrite this one")
    return lock


def verify_sysid_sources():
    """Pin the reviewed physical replacement and its identification handoff."""
    lock = json.loads((ROOT / "benchmark37/sysid_lock.json").read_text())
    for name, expected in lock["files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"SysID source changed: {name}; review and create a new physics lock")
    if sha(SYSID_HANDOFF) != lock["handoff_sha256"]:
        raise RuntimeError("SysID handoff changed; review the physical parameters before launch")
    return lock


def run_folder(mode, seed, series="benchmark37"):
    if series not in ("benchmark37", "benchmark37_stage2a", "benchmark37_stage2b",
                      "benchmark37_stage2c", "benchmark37_stage2c_apex2cm",
                      "benchmark37_stage2c_energy2cm", "benchmark37_stage2d_release_tilt",
                      "benchmark37_stage2d_attitude", "benchmark37_stage2d_goal10",
                      "benchmark37_stage2e_pretouch", "benchmark37_stage3a_foot_event",
                      "benchmark37_stage3b_point_short", "benchmark37_stage3b_point_edge",
                      "benchmark37_stage3b_point_energy4cm",
                      "benchmark37_stage3b_first_two_control", "benchmark37_stage3b_first_two_focus",
                      "benchmark37_stage4a_two_ground_short", "benchmark37_stage4a_calibrated_20261028",
                      "benchmark37_stage4a_calibrated_20261028_v2", "benchmark37_stage4b_range08",
                      "benchmark37_stage4c_smooth_control", "benchmark37_stage4c_smooth_penalty",
                      "benchmark37_stage4d_foot_tail_control", "benchmark37_stage4d_foot_tail_continuous",
                      "benchmark37_fixed_case_rewards", "benchmark37_stage4e_flight_power_control",
                      "benchmark37_stage4e_flight_power_relaxed", "benchmark37_stage4f_apex_deficit_control",
                      "benchmark37_stage4f_apex_deficit_penalty", "benchmark37_stage4g_first_hop_margin_control",
                      "benchmark37_stage4g_first_hop_margin_margin3cm", "benchmark37_stage4h_long_pair_control",
                      "benchmark37_stage4h_long_pair_longmix", "benchmark37_stage4i_foot_excess_control",
                      "benchmark37_stage4i_foot_excess_penalty", "benchmark37_stage4j_entropy_control",
                      "benchmark37_stage4j_entropy_low", "benchmark37_stage4k_precision_refine",
                      "benchmark37_stage4l_landing_gain", "benchmark37_stage4m_height_refine"):
        raise ValueError(f"Unknown benchmark series: {series}")
    path = ROOT / "outputs" / series / (
        f"{mode}_seed{seed}_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def runtime_info():
    import torch
    import gymnasium
    import rsl_rl
    versions = {}
    for name in ("rsl-rl-lib", "isaaclab", "gymnasium", "tensordict"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not registered; see source paths"
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                torch_cuda_build=torch.version.cuda, torch_file=torch.__file__,
                gymnasium_file=gymnasium.__file__, rsl_rl_file=rsl_rl.__file__, versions=versions)


def clean(value):
    """Serialize configs without silently dropping unresolved fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def snapshot_code(folder, include_sysid=False):
    """Keep evaluator/adapter hashes with every run, not just task hashes."""
    paths = [ROOT / "benchmark37_runner.py", ROOT / "run_benchmark37.sh",
             ROOT / "run_benchmark37_smoothness_ab.sh", ROOT / "run_benchmark37_landing_tail_ab.sh",
             ROOT / "run_benchmark37_fixed_case_rewards.sh", ROOT / "run_benchmark37_flight_power_ab.sh",
             ROOT / "run_benchmark37_apex_deficit_ab.sh", ROOT / "run_benchmark37_first_hop_margin_ab.sh",
             ROOT / "run_benchmark37_long_pair_ab.sh", ROOT / "run_benchmark37_next_target_mask.sh",
             ROOT / "run_benchmark37_foot_excess.sh", ROOT / "run_benchmark37_entropy_refine.sh",
             ROOT / "run_benchmark37_precision_refine.sh", ROOT / "run_benchmark37_landing_gain.sh",
             ROOT / "run_benchmark37_height_refine.sh"]
    paths += sorted((ROOT / "benchmark37").glob("*.py"))
    paths += sorted((ROOT / "benchmark37").glob("*.json"))
    paths += sorted((ROOT / "benchmark37").glob("*.usda"))
    paths += [ROOT / name for name in verify_sources()["files"]]
    if include_sysid:
        paths += [ROOT / name for name in verify_sysid_sources()["files"]]
    target = folder / "source"
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(relative)] = sha(path)
    return hashes
