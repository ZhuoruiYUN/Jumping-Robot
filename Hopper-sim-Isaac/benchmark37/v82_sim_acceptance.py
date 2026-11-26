"""Recheck a completed pair of 0.50 m straight simulations on the V82 gates."""

import argparse
import json
import math
import operator
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
V82_COURSE = ROOT / "training_config_v51/ab_ws_first_precision/course.json"
SUITES = {"cardinal": "EWNS", "opposite": "WESN"}
DIRECTIONS = {"E": "+X", "W": "-X", "N": "+Y", "S": "-Y"}
SOURCE = {
    "cardinal": "benchmark37.waypoint_stream:cardinal_half_metre",
    "opposite": "benchmark37.waypoint_stream:opposite_cardinal_half_metre",
}
CASE_COUNT = 128
FIRST_HIT_MIN = .90


def read_json(path):
    return json.loads(path.read_text())


def by_case(path):
    rows = read_json(path)
    mapped = {row["case_id"]: row for row in rows}
    if len(rows) != CASE_COUNT or mapped.keys() != set(range(CASE_COUNT)):
        raise ValueError(f"{path} must contain each of the {CASE_COUNT} case IDs exactly once")
    return mapped


def load_suites(folder, seed):
    suites = {}
    for name in SUITES:
        path = folder / name
        m = read_json(path / "manifest.json")
        requirement = {
            "status": "complete", "mode": "eval", "optimizer_updates": 0,
            "seed": seed, "num_envs": CASE_COUNT, "task_episode_horizon_s": 60.,
            "waypoint_source": SOURCE[name],
            "canonical_axis": "dominant horizontal waypoint axis, dual recurrent histories",
            "evaluation_scope": "online_waypoints",
        }
        for key, value in requirement.items():
            if m.get(key) != value:
                raise ValueError(f"{path}/manifest.json {key} must be {value!r}")
        if m.get("evaluation_reset_rng") is not None:
            reset_calls = m.get("evaluation_reset_rng_audit", {}).get("calls")
            if type(reset_calls) is not int or reset_calls < 0:
                raise ValueError(f"{path}/manifest.json is missing a valid evaluation reset audit")
        cases = read_json(path / "cases.json")
        if len(cases["cases"]) != CASE_COUNT or cases["seed"] != seed:
            raise ValueError(f"{path}/cases.json does not match the declared protocol")
        suites[name] = dict(
            manifest=m, cases=cases, episodes=by_case(path / "episodes.json"),
            yaw=by_case(path / "yaw_by_case.json"),
            rpy=by_case(path / "roll_pitch_by_case.json"),
            training_trend=read_json(path / "summary.json").get("training_trend"),
        )
    checkpoints = {suite["manifest"].get("checkpoint") for suite in suites.values()}
    if len(checkpoints) != 1 or None in checkpoints:
        raise ValueError("Both suites must evaluate the same named checkpoint")
    reset_protocols = {suite["manifest"].get("evaluation_reset_rng") for suite in suites.values()}
    if len(reset_protocols) != 1:
        raise ValueError("Both suites must use the same evaluation reset RNG protocol")
    return suites


def verify_pair(reference, candidate):
    training_response_keys = {'training_roll_pitch_gain_range', 'training_tail_fraction_of_weaker', 'training'}
    for name in SUITES:
        a, b = reference[name], candidate[name]
        if a["cases"] != b["cases"]:
            raise ValueError(f"{name} initial conditions differ from the reference")
        for key in ("physics_revision", "imu_link_plant_audit", "observation_contract",
                    "additional_response_randomization", "rigid_body_randomization", "evaluation_reset_rng"):
            baseline, tested = (suite["manifest"].get(key) for suite in (a, b))
            if key == "additional_response_randomization" and isinstance(baseline, dict) and isinstance(tested, dict):
                baseline, tested = ({k: v for k, v in config.items() if k not in training_response_keys}
                                    for config in (baseline, tested))
            if baseline != tested:
                raise ValueError(f"{name} protocol differs from the reference in {key}")


def fraction(count, total):
    return dict(count=count, total=total, rate=count / total if total else None)


def p95(values):
    if not values:
        return None
    ordered = sorted(values)
    rank = .95 * (len(ordered) - 1)
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def landing_metrics(events, course, total=None):
    radius = course["omni"]["strict_foot_m"]
    bottom = course["omni"]["strict_height_m"]
    top = course["overheight_band"]["upper_m"]
    tilt_limit = course["omni"]["strict_tilt_deg"]
    total = len(events) if total is None else total
    recorded = len(events)
    events = [event for event in events if event["valid_flight"]]
    errors = [e["foot_error_m"] for e in events]
    cross = [abs(e["foot_error_cross_m"]) for e in events]
    height = [e["apex_root_link_world_m"] for e in events]
    release_cross = [abs(e["launch_error_cross_m_s"]) for e in events if "launch_error_cross_m_s" in e]
    ascent_cross = [e["ascent_cross_pitch_peak_deg"] for e in events if e.get("ascent_cross_pitch_samples", 0) > 0]
    strict = sum(e["strict"] for e in events)
    return {
        "scored": len(events), "missing_landings": total - len(events),
        "invalid_flight_events": recorded - len(events), "unrecorded_landings": total - recorded,
        "foot_5cm": fraction(sum(error <= radius for error in errors), total),
        "foot_error_p95_m": p95(errors), "absolute_cross_error_p95_m": p95(cross),
        "strict_height_foot_tilt": fraction(strict, total),
        "apex_p95_m": p95(height), "apex_max_m": max(height, default=None),
        "apex_at_least_1m": fraction(sum(h >= bottom for h in height), len(events)),
        "apex_over_1p10m": fraction(sum(h > top for h in height), len(events)),
        "apex_1_to_1p10m": fraction(sum(bottom <= h <= top for h in height), len(events)),
        "joint_height_band_foot_tilt": fraction(sum(
            bottom <= h <= top and error <= radius and e["touchdown_tilt_deg"] <= tilt_limit
            for e, error, h in zip(events, errors, height)), total),
        "touchdown_tilt_peak_deg": max((e["touchdown_tilt_deg"] for e in events), default=None),
        "release_cross_velocity_error_m_s": dict(
            samples=len(release_cross), median=median(release_cross) if release_cross else None,
            p95=p95(release_cross)),
        "ascent_cross_lean_peak_deg": dict(
            samples=len(ascent_cross), median=median(ascent_cross) if ascent_cross else None,
            p95=p95(ascent_cross)),
    }


def yaw_metrics(rows, peak_limit):
    n = sum(row["samples"] for row in rows)
    if not n or any(row["samples"] <= 0 for row in rows):
        raise ValueError("All routes must have all-phase yaw samples")
    return {
        "samples": n,
        "actual_rms_deg": math.sqrt(sum(r["samples"] * r["rms_deg"] ** 2 for r in rows) / n),
        "actual_peak_deg": max(r["peak_deg"] for r in rows),
        "forecast_peak_deg": max(r["predicted_peak_deg"] for r in rows),
        "routes_actual_over_limit": sum(r["peak_deg"] > peak_limit for r in rows),
        "routes_forecast_over_limit": sum(r["predicted_peak_deg"] > peak_limit for r in rows),
        "routes_actual_or_forecast_over_limit": sum(
            r["peak_deg"] > peak_limit or r["predicted_peak_deg"] > peak_limit for r in rows),
    }


def posture_metrics(episodes, attitude, mode):
    totals = [r["flight_plane_totals"][int(mode == "travel")] for r in episodes]
    n, angle, rate, speed = map(sum, zip(*(row[:4] for row in totals)))
    if not n:
        return {"samples": 0, "roll_pitch_euler": None}
    rpy = [r[f"{mode}_free_flight"] for r in attitude]
    attitude_n = sum(r["samples"] for r in rpy)
    axes = {}
    if attitude_n:
        for axis in ("roll", "pitch", "tilt"):
            measured = [(r["samples"], r["angles"][axis]) for r in rpy if r["samples"]]
            axes[axis] = dict(
                rms_deg=math.sqrt(sum(n * a["rms_deg"] ** 2 for n, a in measured) / attitude_n),
                peak_deg=max(a["abs_peak_deg"] for _, a in measured),
            )
    return {
        "samples": int(n), "attitude_samples": attitude_n, "roll_pitch_euler": axes,
        "lean_rms_deg": math.degrees(math.sqrt(angle / n)),
        "lean_peak_deg": math.degrees(max(row[4] for row in totals)),
        "rate_rms_rad_s": math.sqrt(rate / n), "leg_speed_rms_m_s": math.sqrt(speed / n),
    }


def read_routes(suites):
    routes = []
    for name, direction_order in SUITES.items():
        suite = suites[name]
        for case, episode in suite["episodes"].items():
            direction = direction_order[case % len(direction_order)]
            moving = []
            if episode["first_moving_direction"] != direction:
                raise ValueError(f"{name} case {case} has an unexpected first moving direction")
            for e in episode["events"]:
                if not e["moving"]:
                    continue
                if (e["direction"] != direction or abs(e["commanded_edge_m"] - .50) > 1e-5
                        or e["first_attempt_for_target"] != (e["attempt_for_target"] == 1)):
                    raise ValueError(f"{name} case {case} has an unexpected straight event")
                moving.append(e)
            keys = [(e["target_index"], e["attempt_for_target"]) for e in moving]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} case {case} has duplicate moving attempts")
            routes.append((name, case, direction, episode, suite["yaw"][case], suite["rpy"][case], moving))
    return routes


def summarize(routes, course):
    yaw = course["moving_yaw_prediction"]
    events = [event for route in routes for event in route[-1]]
    first = [e for e in events if e["first_attempt_for_target"]]
    route_first = [e for e in first if e["target_index"] == 1]
    retries = [e for e in events if not e["first_attempt_for_target"]]
    if sum(e["strict"] for e in route_first) != sum(r[3]["first_moving_strict"] for r in routes):
        raise ValueError("First moving strict events do not match the per-route counts")
    return {
        "route_first": landing_metrics(route_first, course, len(routes)),
        "target_first": landing_metrics(first, course),
        "retry": landing_metrics(retries, course),
        "yaw": yaw_metrics([r[4] for r in routes], yaw["peak_heading_goal_deg"]),
        "travel": posture_metrics([r[3] for r in routes], [r[5] for r in routes], "travel"),
        "hold": posture_metrics([r[3] for r in routes], [r[5] for r in routes], "hold"),
        "physical_deaths": [[r[0], r[1]] for r in routes if r[3]["physical_death"]],
        "routes_with_three_strict_moving_hits": sum(r[3]["three_moving_hits"] for r in routes),
        "routes_first_hold_strict": sum(r[3]["first_stationary_strict"] for r in routes),
    }


def evaluate(suites, course):
    routes = read_routes(suites)
    overall = summarize(routes, course)
    groups = {name: summarize([r for r in routes if r[2] == code], course)
              for code, name in DIRECTIONS.items()}
    gates = {}

    def check(name, value, comparison, limit):
        compare = {"<=": operator.le, ">=": operator.ge}[comparison]
        gates[name] = dict(value=value, comparison=comparison, limit=limit,
                           passed=value is not None and compare(value, limit))

    promotion = course["omni"]["promotion"]
    first = overall["target_first"]
    check("target_first_foot_p95_m", first["foot_error_p95_m"], "<=", promotion["foot_error_p95_m_max"])
    for name in ("route_first", "target_first"):
        check(f"{name}_foot_5cm", overall[name]["foot_5cm"]["rate"], ">=", FIRST_HIT_MIN)
        check(f"{name}_strict", overall[name]["strict_height_foot_tilt"]["rate"],
              ">=", promotion["moving_first_attempt_strict_min"])
    check("physical_deaths", len(overall["physical_deaths"]), "<=", promotion["physical_deaths_max"])
    for name in SUITES:
        suite_routes = [r for r in routes if r[0] == name]
        check(f"{name}.three_strict_moving_routes", sum(r[3]["three_moving_hits"] for r in suite_routes),
              ">=", promotion["episodes_with_three_moving_hits_min"])
    yaw_limits = course["moving_yaw_prediction"]
    for name, key in (("actual_rms_deg", "rms_heading_goal_deg"),
                      ("actual_peak_deg", "peak_heading_goal_deg"),
                      ("forecast_peak_deg", "peak_heading_goal_deg")):
        check(f"yaw.{name}", overall["yaw"][name], "<=", yaw_limits[key])
    for name, group in groups.items():
        strict = group["route_first"]["strict_height_foot_tilt"]
        check(f"{name}.route_first_sample_count", strict["total"], ">=", promotion["per_direction_samples_min"])
        check(f"{name}.route_first_strict", strict["rate"], ">=", promotion["per_direction_strict_min"])
    modes = {"hold": overall["hold"], **{f"{k}.travel": group["travel"] for k, group in groups.items()}}
    for mode, values in modes.items():
        check(f"{mode}.euler_samples", values.get("attitude_samples", 0), ">=", 1)
        for key, limit in course["flight_plane_limits"].items():
            check(f"{mode}.{key}", values.get(key), "<=", limit)
    return dict(
        passed=all(g["passed"] for g in gates.values()),
        failed_checks=[name for name, gate in gates.items() if not gate["passed"]],
        gates=gates, overall=overall, directions=groups,
        training_trend=suites["cardinal"]["training_trend"],
    )


def summarize_response(suites, course):
    config = course["vertical_pretraining"]["response_robustness"]
    low, high = config["roll_pitch_gain_range"]
    bottom_quartile = low + .25 * (high - low)
    gains = {(name, case["case_id"]): case["response_gain"]
             for name, suite in suites.items() for case in suite["cases"]["cases"]}
    groups = {"nominal": [], "lowest_weak_quartile": [], "remaining_weaker": []}
    for route in read_routes(suites):
        gain = gains[route[:2]]
        if gain == 1.:
            groups["nominal"].append(route)
        elif low <= gain < bottom_quartile:
            groups["lowest_weak_quartile"].append(route)
        elif bottom_quartile <= gain <= high:
            groups["remaining_weaker"].append(route)
        else:
            raise ValueError("Saved response gain is outside the fixed evaluation range")
    if any(not rows for rows in groups.values()):
        raise ValueError("All fixed response groups need evaluation coverage")
    return dict(lowest_weak_quartile_range=[low, bottom_quartile],
                groups={name: dict(routes=len(rows), **summarize(rows, course)) for name, rows in groups.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path, help="Parent of the cardinal and opposite folders")
    parser.add_argument("--reference", type=Path, help="Paired baseline with the exact same saved cases")
    parser.add_argument("--seed", type=int, default=7001, help="Expected shared case seed")
    parser.add_argument("--output", type=Path, help="Write to this new JSON path")
    parser.add_argument("--require-pass", action="store_true", help="Exit 1 if any hard gate failed")
    parser.add_argument("--response-strata", action="store_true",
                        help="Include metrics by fixed actuator response group")
    args = parser.parse_args()
    try:
        course = read_json(V82_COURSE)
        suites = load_suites(args.evaluation, args.seed)
        report = {
            "protocol": dict(seed=args.seed, routes_per_suite=CASE_COUNT, step_m=.50, horizon_s=60,
                             evaluation_reset_rng=suites["cardinal"]["manifest"].get("evaluation_reset_rng")),
            "scope": "Straight simulated routes and their opening holds.",
            "travel_attitude": "Use world route-normal tilt, angular rate about the route axis and foot side speed. "
                               "Euler Roll/Pitch are diagnostics. X travel primarily uses Pitch; Y uses Roll.",
            "diagnostics_without_aggregate_gate": [
                "Apex in 1.00–1.10 m and joint height/foot/tilt rates have no specified minimum pass rate.",
                "Retry 5 cm rate and backward or mixed-axis routes need separate acceptance evidence.",
                "Standalone stationary/omnidirectional promotion and real hardware are outside this protocol.",
            ],
            "checkpoint": suites["cardinal"]["manifest"]["checkpoint"],
            "candidate": evaluate(suites, course),
        }
        if args.response_strata:
            report["response_strata"] = {"candidate": summarize_response(suites, course)}
        if args.reference:
            reference = load_suites(args.reference, args.seed)
            verify_pair(reference, suites)
            report["reference_checkpoint"] = reference["cardinal"]["manifest"]["checkpoint"]
            report["reference"] = evaluate(reference, course)
            if args.response_strata:
                report["response_strata"]["reference"] = summarize_response(reference, course)
        data = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as f:
                f.write(data)
        else:
            print(data, end="")
    except (KeyError, OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    return int(args.require_pass and not report["candidate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
