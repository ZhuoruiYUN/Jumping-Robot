#!/usr/bin/env python3
"""Extract per-hop takeoff evidence from deployed Quadhopper CSV logs.

This is intentionally an identification report, not a controller tuner.  PWM,
voltage, spring compression, and feedback actions are correlated during a hop;
the report therefore never recommends an automatic voltage-compensation gain.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


REQUIRED = {"Time_s", "Z", "Obs_Teacher_IsContact", "Obs_Teacher_JointPos", "VBat_V", "Current_A"}
PWM_FINAL = tuple(f"PWM_FinalFloat_m{i}" for i in range(1, 5))
PWM_RAW = tuple(f"PWM_Raw_m{i}" for i in range(1, 5))


def number(row: dict[str, str], name: str) -> float:
    try:
        value = float(row.get(name, "nan"))
    except ValueError:
        value = float("nan")
    return value


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def average(values: list[float]) -> float:
    values = finite(values)
    return mean(values) if values else float("nan")


def quantile(values: list[float], probability: float) -> float:
    values = sorted(finite(values))
    if not values:
        return float("nan")
    index = (len(values) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def correlation(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    x_bar = mean(x for x, _ in pairs)
    y_bar = mean(y for _, y in pairs)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - x_bar) ** 2 for x, _ in pairs) * sum((y - y_bar) ** 2 for _, y in pairs)
    )
    return numerator / denominator if denominator > 1.0e-12 else float("nan")


def pwm_stats(rows: list[dict[str, str]], columns: tuple[str, ...]) -> tuple[float, float, float]:
    samples = []
    spreads = []
    for row in rows:
        motors = [number(row, column) for column in columns]
        if all(math.isfinite(value) for value in motors):
            samples.append(mean(motors))
            spreads.append(max(motors) - min(motors))
    return average(samples), max(samples, default=float("nan")), average(spreads)


def velocity_at(rows: list[dict[str, str]], index: int) -> float:
    for name in ("Policy_Vel_Z", "EKF_Vel_Z", "Vz"):
        value = number(rows[index], name)
        if math.isfinite(value):
            return value
    return float("nan")


def tilt_peak(rows: list[dict[str, str]]) -> float:
    values = [number(row, "Deploy_Tilt_Deg") for row in rows]
    values = finite(values)
    return max(values, default=float("nan"))


def extract(path: Path, min_contact_s: float, max_contact_s: float) -> list[dict[str, float | int | str | bool]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if not set(PWM_FINAL).issubset(rows[0]):
        raise ValueError(f"{path} is missing final PWM columns")

    contact = [number(row, "Obs_Teacher_IsContact") > 0.5 for row in rows]
    starts = [index for index in range(1, len(rows)) if contact[index] and not contact[index - 1]]
    samples: list[dict[str, float | int | str | bool]] = []
    for hop_index, start in enumerate(starts):
        liftoff = next((index for index in range(start + 1, len(rows)) if not contact[index]), None)
        if liftoff is None:
            continue
        touchdown = next((index for index in range(liftoff + 1, len(rows)) if contact[index]), len(rows))
        stance = rows[start:liftoff]
        flight = rows[liftoff:touchdown]
        if not stance or not flight:
            continue
        apex_offset = max(range(len(flight)), key=lambda offset: number(flight[offset], "Z"))
        apex_index = liftoff + apex_offset
        contact_s = number(rows[liftoff], "Time_s") - number(rows[start], "Time_s")
        flight_s = number(rows[touchdown - 1], "Time_s") - number(rows[liftoff], "Time_s")
        final_mean, final_peak, final_spread = pwm_stats(stance, PWM_FINAL)
        raw_mean, _, raw_spread = pwm_stats(stance, PWM_RAW)
        compression = [number(row, "Obs_Teacher_JointPos") for row in stance]
        volts = [number(row, "VBat_V") for row in stance]
        currents = [number(row, "Current_A") for row in stance]
        apex = number(rows[apex_index], "Z")
        valid = (
            min_contact_s <= contact_s <= max_contact_s
            and 0.25 <= flight_s <= 1.50
            and 0.40 <= apex <= 1.20
        )
        samples.append({
            "source": path.name,
            "hop": hop_index,
            "start_s": number(rows[start], "Time_s"),
            "contact_s": contact_s,
            "flight_s": flight_s,
            "apex_s": number(rows[apex_index], "Time_s"),
            "apex_z_m": apex,
            "liftoff_z_m": number(rows[liftoff], "Z"),
            "liftoff_vz_mps": velocity_at(rows, liftoff),
            "spring_peak_m": max(finite(compression), default=float("nan")),
            "final_collective_pwm": final_mean,
            "final_collective_peak_pwm": final_peak,
            "final_spread_pwm": final_spread,
            "raw_collective_pwm": raw_mean,
            "raw_spread_pwm": raw_spread,
            "vbat_mean_v": average(volts),
            "vbat_min_v": min(finite(volts), default=float("nan")),
            "current_mean_a": average(currents),
            "current_peak_a": max(finite(currents), default=float("nan")),
            "flight_tilt_peak_deg": tilt_peak(flight),
            "target_z_m": average([number(row, "Target_Z") for row in stance]),
            "valid_takeoff_sample": valid,
        })
    return samples


def summary(samples: list[dict[str, float | int | str | bool]]) -> dict[str, object]:
    valid = [sample for sample in samples if sample["valid_takeoff_sample"]]
    metrics = (
        "apex_z_m", "liftoff_vz_mps", "spring_peak_m", "final_collective_pwm",
        "final_spread_pwm", "vbat_mean_v", "vbat_min_v", "current_mean_a", "contact_s",
    )
    report: dict[str, object] = {
        "all_detected_hops": len(samples),
        "valid_takeoff_samples": len(valid),
        "filter": "0.04<=contact_s<=0.10, 0.25<=flight_s<=1.50, 0.40<=apex_z_m<=1.20",
        "metrics": {},
        "apex_correlations": {},
        "liftoff_vz_correlations": {},
        "interpretation": (
            "Correlations are observational only: controller feedback, spring compression, and PWM are "
            "simultaneously coupled. Do not derive a voltage compensation gain from this report alone."
        ),
    }
    for metric in metrics:
        values = [float(sample[metric]) for sample in valid]
        report["metrics"][metric] = {
            "mean": average(values), "median": median(finite(values)) if finite(values) else float("nan"),
            "p10": quantile(values, 0.10), "p90": quantile(values, 0.90),
        }
    predictors = ("final_collective_pwm", "final_spread_pwm", "vbat_mean_v", "vbat_min_v",
                  "current_mean_a", "spring_peak_m", "contact_s", "liftoff_vz_mps")
    for predictor in predictors:
        values = [float(sample[predictor]) for sample in valid]
        report["apex_correlations"][predictor] = correlation(
            values, [float(sample["apex_z_m"]) for sample in valid]
        )
        report["liftoff_vz_correlations"][predictor] = correlation(
            values, [float(sample["liftoff_vz_mps"]) for sample in valid]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="Deployment CSV file(s).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-contact-s", type=float, default=0.040)
    parser.add_argument("--max-contact-s", type=float, default=0.100)
    args = parser.parse_args()
    if not (0.0 < args.min_contact_s < args.max_contact_s):
        parser.error("Require 0 < min-contact-s < max-contact-s")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    all_samples = []
    for path in args.csv:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            parser.error(f"Missing CSV: {resolved}")
        all_samples.extend(extract(resolved, args.min_contact_s, args.max_contact_s))
    output.mkdir(parents=True)
    fields = list(all_samples[0]) if all_samples else ["source", "hop"]
    with (output / "per_hop.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_samples)
    report = summary(all_samples)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[TAKEOFF-ID] output={output}")


if __name__ == "__main__":
    main()
