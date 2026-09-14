"""Summarize contact-to-liftoff impulse from a one-environment diagnostic CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


parser = argparse.ArgumentParser()
parser.add_argument("csv_path", type=Path)
parser.add_argument("--mode", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

with args.csv_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit(f"No rows in {args.csv_path}")


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


contact = [int(float(row["Is_Contact"])) > 0 for row in rows]
events: list[dict[str, float]] = []
index = 1
while index < len(rows):
    if not (contact[index - 1] and not contact[index]):
        index += 1
        continue
    start = index - 1
    while start > 0 and contact[start - 1]:
        start -= 1
    if index - start < 3:
        index += 1
        continue
    end = index
    while end < len(rows) and not contact[end]:
        end += 1
    flight = rows[index:end]
    if not flight:
        index += 1
        continue
    liftoff = rows[index]
    liftoff_vz = value(liftoff, "Vz")
    if liftoff_vz > 0.20:
        contact_rows = rows[start:index]
        z0 = value(liftoff, "Z")
        apex_end = index + 1
        while apex_end < end and value(rows[apex_end], "Vz") > 0.0:
            apex_end += 1
        ascent_rows = rows[index:apex_end]
        if not ascent_rows:
            ascent_rows = [liftoff]
        # The motor lag is comparable to the short contact phase.  Whole-hop
        # averages hide whether thrust arrives before or after the spring has
        # released, so expose the last 40 ms before liftoff and first 100 ms
        # after it separately.
        contact_exit_rows = contact_rows[-4:]
        ascent_head_rows = ascent_rows[:10]
        raw_mean = fmean(fmean(value(row, f"RawCmd{i}") for i in range(1, 5)) for row in contact_rows)
        shaped_mean = fmean(fmean(value(row, f"ShapedCmd{i}") for i in range(1, 5)) for row in contact_rows)
        actual_mean = fmean(fmean(value(row, f"M{i}") for i in range(1, 5)) for row in contact_rows)
        events.append({
            "liftoff_time_s": value(liftoff, "Time_s"),
            "contact_s": value(rows[index - 1], "Time_s") - value(rows[start], "Time_s") + 0.01,
            "spring_peak_m": max(value(row, "Spring_Pos") for row in contact_rows),
            "liftoff_z_m": z0,
            "liftoff_vz_mps": liftoff_vz,
            "observed_apex_m": max(value(row, "Z") for row in ascent_rows),
            "unpowered_ballistic_apex_m": z0 + liftoff_vz * liftoff_vz / (2.0 * 9.81),
            "ascent_s": value(ascent_rows[-1], "Time_s") - value(liftoff, "Time_s") + 0.01,
            "raw_collective_pwm": raw_mean,
            "shaped_collective_pwm": shaped_mean,
            "motor_collective_pwm": actual_mean,
            "shape_collective_loss_pwm": raw_mean - shaped_mean,
            "pre_liftoff_raw_collective_pwm": fmean(
                fmean(value(row, f"RawCmd{i}") for i in range(1, 5)) for row in contact_exit_rows
            ),
            "pre_liftoff_shaped_collective_pwm": fmean(
                fmean(value(row, f"ShapedCmd{i}") for i in range(1, 5)) for row in contact_exit_rows
            ),
            "pre_liftoff_motor_collective_pwm": fmean(
                fmean(value(row, f"M{i}") for i in range(1, 5)) for row in contact_exit_rows
            ),
            "ascent_raw_collective_pwm": fmean(
                fmean(value(row, f"RawCmd{i}") for i in range(1, 5)) for row in ascent_rows
            ),
            "ascent_shaped_collective_pwm": fmean(
                fmean(value(row, f"ShapedCmd{i}") for i in range(1, 5)) for row in ascent_rows
            ),
            "ascent_motor_collective_pwm": fmean(
                fmean(value(row, f"M{i}") for i in range(1, 5)) for row in ascent_rows
            ),
            "early_ascent_raw_collective_pwm": fmean(
                fmean(value(row, f"RawCmd{i}") for i in range(1, 5)) for row in ascent_head_rows
            ),
            "early_ascent_shaped_collective_pwm": fmean(
                fmean(value(row, f"ShapedCmd{i}") for i in range(1, 5)) for row in ascent_head_rows
            ),
            "early_ascent_motor_collective_pwm": fmean(
                fmean(value(row, f"M{i}") for i in range(1, 5)) for row in ascent_head_rows
            ),
        })
    index = max(end, index + 1)

summary: dict[str, object] = {"mode": args.mode, "csv": str(args.csv_path), "hops": events}
if events:
    keys = [key for key in events[0] if key != "liftoff_time_s"]
    summary["mean"] = {key: fmean(event[key] for event in events) for key in keys}
else:
    summary["mean"] = None

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
if args.quiet:
    print(json.dumps({"mode": args.mode, "hops": len(events), "mean": summary["mean"]}))
else:
    print(json.dumps(summary, indent=2))
