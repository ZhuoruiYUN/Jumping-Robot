"""Quantify and plot an Isaac spatial-tracking ``single_env.csv``.

The historical filename is retained for compatibility. This is no longer a
MuJoCo player: it reads the CSV emitted by ``run_tracking_recovery.py eval``
and writes a clean, per-hop analysis directory alongside the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path', type=Path, help='Isaac eval single_env.csv to analyse.')
    parser.add_argument('--output-dir', type=Path, help='Defaults to <csv parent>/analysis.')
    parser.add_argument('--target-height', type=float,
                        help='Strict apex target; defaults to median Target_Z in the CSV.')
    parser.add_argument('--target-tolerance', type=float, default=0.05,
                        help='Secondary height/landing tolerance in metres (default: 0.05).')
    parser.add_argument('--window-start-s', type=float,
                        help='Start time in continuous evaluation time. Defaults to the final window.')
    parser.add_argument('--window-duration-s', type=float, default=30.0,
                        help='Detail-window duration in seconds (default: 30).')
    parser.add_argument('--show', action='store_true', help='Open figures after writing them.')
    args = parser.parse_args()
    if not args.csv_path.is_file():
        parser.error(f'CSV does not exist: {args.csv_path}')
    if args.target_tolerance <= 0.0 or args.window_duration_s <= 0.0:
        parser.error('--target-tolerance and --window-duration-s must be positive')
    return args


def value(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except KeyError as error:
        raise ValueError(f'Missing required Isaac CSV column: {column}') from error


def attitude_components_deg(row: dict[str, str]) -> tuple[float, float]:
    """Return signed roll and pitch from Isaac scalar-first root quaternion."""
    qw, qx, qy, qz = (value(row, key) for key in ('Qw', 'Qx', 'Qy', 'Qz'))
    roll = np.arctan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2.0 * (qw*qy - qz*qx), -1.0, 1.0))
    return float(np.degrees(roll)), float(np.degrees(pitch))


def attitude_deg(row: dict[str, str]) -> tuple[float, float]:
    roll, pitch = attitude_components_deg(row)
    return abs(roll), abs(pitch)


def continuous_time(rows: list[dict[str, str]]) -> np.ndarray:
    """Make 15 s Isaac episode resets monotonic without altering raw Time_s."""
    output: list[float] = []
    offset, previous = 0.0, -np.inf
    for row in rows:
        current = value(row, 'Time_s')
        if current < previous:
            offset += previous
        output.append(offset + current)
        previous = current
    return np.asarray(output)


def extract_hops(rows: list[dict[str, str]], target: float, tolerance: float) -> list[dict[str, float]]:
    """Extract actual liftoff-to-touchdown hops while excluding reset drops."""
    hops: list[dict[str, float]] = []
    previous_time = -np.inf
    previous_contact: bool | None = None
    armed = False
    flight: list[dict[str, str]] | None = None
    for row in rows:
        time_s = value(row, 'Time_s')
        contact = bool(int(value(row, 'Is_Contact')))
        if time_s < previous_time:
            previous_contact, armed, flight = None, False, None
        if contact:
            armed = True
        if armed and previous_contact is True and not contact:
            flight = [row]
        elif flight is not None:
            flight.append(row)
            if contact:
                air = flight[:-1]
                if len(air) > 2:
                    touchdown, last_air = flight[-1], air[-1]
                    apex = max(value(sample, 'Z') for sample in air)
                    roll_peak, pitch_peak = zip(*(attitude_deg(sample) for sample in air))
                    collective = [np.mean([value(sample, key) for key in ('U1', 'U2', 'U3', 'U4')])
                                  for sample in air]
                    spread = [np.ptp([value(sample, key) for key in ('U1', 'U2', 'U3', 'U4')])
                              for sample in air]
                    dx = value(touchdown, 'X') - value(last_air, 'Target_X')
                    dy = value(touchdown, 'Y') - value(last_air, 'Target_Y')
                    landing_error = float(np.hypot(dx, dy))
                    hops.append({
                        'hop_index': float(len(hops) + 1),
                        'liftoff_time_s': value(air[0], 'Time_s'),
                        'touchdown_time_s': value(touchdown, 'Time_s'),
                        'flight_time_s': value(touchdown, 'Time_s') - value(air[0], 'Time_s'),
                        'apex_m': apex, 'target_apex_m': target, 'apex_error_m': apex - target,
                        'height_shortfall_m': max(target - apex, 0.0),
                        'height_ge_target': float(apex >= target),
                        'height_ge_target_minus_tolerance': float(apex >= target - tolerance),
                        'landing_error_m': landing_error,
                        'landing_hit_tolerance': float(landing_error <= tolerance),
                        'peak_roll_deg': max(roll_peak), 'peak_pitch_deg': max(pitch_peak),
                        'peak_tilt_deg': max(max(roll_peak), max(pitch_peak)),
                        'mean_motor_collective': float(np.mean(collective)),
                        'peak_motor_spread': float(max(spread)),
                    })
                flight = None
        previous_time, previous_contact = time_s, contact
    return hops


def describe(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {key: None for key in ('mean', 'std', 'median', 'p10', 'p90', 'min', 'max')}
    return {
        'mean': float(np.mean(values)), 'std': float(np.std(values)),
        'median': float(np.median(values)), 'p10': float(np.percentile(values, 10)),
        'p90': float(np.percentile(values, 90)), 'min': float(np.min(values)), 'max': float(np.max(values)),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    with args.csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'No data rows in {args.csv_path}')
    target = args.target_height
    if target is None:
        target = float(np.median([value(row, 'Target_Z') for row in rows]))
    output_dir = (args.output_dir or args.csv_path.parent / 'analysis').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hops = extract_hops(rows, target, args.target_tolerance)
    columns = list(hops[0]) if hops else [
        'hop_index', 'liftoff_time_s', 'touchdown_time_s', 'flight_time_s', 'apex_m',
        'target_apex_m', 'apex_error_m', 'height_shortfall_m', 'height_ge_target',
        'height_ge_target_minus_tolerance', 'landing_error_m', 'landing_hit_tolerance',
        'peak_roll_deg', 'peak_pitch_deg', 'peak_tilt_deg', 'mean_motor_collective',
        'peak_motor_spread',
    ]
    with (output_dir / 'isaac_hops.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(hops)

    def values(column: str) -> np.ndarray:
        return np.asarray([hop[column] for hop in hops], dtype=float)

    apex = values('apex_m') if hops else np.empty(0)
    landing = values('landing_error_m') if hops else np.empty(0)
    tilt = values('peak_tilt_deg') if hops else np.empty(0)
    collective = values('mean_motor_collective') if hops else np.empty(0)
    spread = values('peak_motor_spread') if hops else np.empty(0)
    summary = {
        'source_csv': str(args.csv_path.resolve()), 'rows': len(rows), 'completed_hops': len(hops),
        'target_apex_m': target, 'target_tolerance_m': args.target_tolerance,
        'strict_height_ge_target_rate': float(np.mean(apex >= target)) if len(apex) else 0.0,
        'height_ge_target_minus_tolerance_rate': float(np.mean(apex >= target - args.target_tolerance)) if len(apex) else 0.0,
        'landing_hit_tolerance_rate': float(np.mean(landing <= args.target_tolerance)) if len(landing) else 0.0,
        'apex_m': describe(apex), 'landing_error_m': describe(landing),
        'peak_tilt_deg': describe(tilt), 'mean_motor_collective': describe(collective),
        'apex_correlation_with_peak_tilt': correlation(apex, tilt),
        'apex_correlation_with_collective': correlation(apex, collective),
        'apex_correlation_with_motor_spread': correlation(apex, spread),
        'apex_linear_drift_m_per_hop': float(np.polyfit(np.arange(len(apex)), apex, 1)[0]) if len(apex) > 1 else None,
    }
    (output_dir / 'isaac_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    matplotlib.use('TkAgg' if args.show else 'Agg')
    from matplotlib import pyplot as plt
    global_time = continuous_time(rows)
    z = np.asarray([value(row, 'Z') for row in rows])
    target_z = np.asarray([value(row, 'Target_Z') for row in rows])
    contact = np.asarray([value(row, 'Is_Contact') > 0.5 for row in rows])
    figure, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=False)
    axes[0].plot(global_time, z, label='Root Z')
    axes[0].plot(global_time, target_z, '--', label='Target Z')
    axes[0].scatter(global_time[contact], z[contact], s=3, c='tab:cyan', label='Contact')
    axes[0].set_ylabel('Height [m]')
    axes[0].legend(ncol=3, fontsize=8)
    if hops:
        index = values('hop_index')
        axes[1].plot(index, apex, 'o-', markersize=3, label='Measured apex')
        axes[1].axhline(target, color='tab:orange', linestyle='--', label='Strict target')
        axes[1].axhline(target - args.target_tolerance, color='0.4', linestyle=':', label='Target - tolerance')
        axes[1].set_ylabel('Apex [m]')
        axes[1].legend(ncol=3, fontsize=8)
        axes[2].plot(index, 100.0 * (apex - target), 'o-', markersize=3)
        axes[2].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
        axes[2].set_ylabel('Apex error [cm]')
        axes[3].plot(index, 100.0 * landing, 'o-', markersize=3, color='tab:red')
        axes[3].axhline(100.0 * args.target_tolerance, color='k', linestyle='--', linewidth=0.8)
        axes[3].set_ylabel('Landing error [cm]')
        axes[4].plot(index, tilt, 'o-', markersize=3, label='Peak tilt [deg]')
        axes[4].plot(index, collective, 'o-', markersize=3, label='Mean collective')
        axes[4].set_ylabel('Tilt / collective')
        axes[4].legend(fontsize=8)
        axes[4].set_xlabel('Completed hop')
    else:
        for axis in axes[1:]:
            axis.text(0.5, 0.5, 'No completed liftoff-to-touchdown hop', ha='center', va='center')
            axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / 'isaac_hop_metrics.png', dpi=170)
    if args.show:
        plt.show()
    plt.close(figure)

    # This five-panel time window mirrors the hardware plot so a policy can be
    # inspected identically in Isaac: vertical cyan lines are physical
    # touchdown events. Isaac logs model power but has no battery-current
    # channel, so the final panel intentionally contains power only.
    roll_pitch = np.asarray([attitude_components_deg(row) for row in rows])
    motor_u = np.asarray([[value(row, key) for key in ('U1', 'U2', 'U3', 'U4')] for row in rows])
    xy_error = np.asarray([np.hypot(value(row, 'X') - value(row, 'Target_X'),
                                    value(row, 'Y') - value(row, 'Target_Y')) for row in rows])
    power = np.asarray([value(row, 'Power_W') for row in rows])
    contact = np.asarray([value(row, 'Is_Contact') > 0.5 for row in rows])
    touchdown = np.zeros(len(rows), dtype=bool)
    previous_contact = False
    previous_raw_time = -np.inf
    reset_drop_intervals: list[tuple[float, float]] = []
    reset_start: float | None = None
    for index, row in enumerate(rows):
        raw_time = value(row, 'Time_s')
        if raw_time < previous_raw_time:
            previous_contact = False
            reset_start = float(global_time[index])
        touchdown[index] = contact[index] and not previous_contact
        if reset_start is not None and contact[index]:
            # The evaluation reset intentionally starts from a random aerial
            # drop. It is not a policy-produced hop and is excluded by
            # extract_hops; shade it so the raw Z trace cannot be misread.
            reset_drop_intervals.append((reset_start, float(global_time[index])))
            reset_start = None
        previous_contact, previous_raw_time = contact[index], raw_time
    window_start = args.window_start_s
    if window_start is None:
        window_start = max(float(global_time[0]), float(global_time[-1] - args.window_duration_s))
    window_end = window_start + args.window_duration_s
    window = (global_time >= window_start) & (global_time <= window_end)
    if not np.any(window):
        raise ValueError(f'No samples in requested detail window [{window_start}, {window_end}] s')
    detail_time = global_time[window]
    detail_figure, detail_axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    detail_figure.suptitle('Quadhopper Isaac-simulation detail window', fontsize=15, fontweight='bold')
    detail_axes[0].plot(detail_time, z[window], linewidth=2.0, label='Measured z')
    detail_axes[0].plot(detail_time, target_z[window], '--', color='tab:orange', linewidth=1.8,
                        label='Target z')
    detail_axes[0].set_ylabel('Height [m]')
    detail_axes[0].legend(fontsize=9)
    detail_axes[1].plot(detail_time, roll_pitch[window, 1], color='mediumpurple', label='Pitch')
    detail_axes[1].plot(detail_time, roll_pitch[window, 0], color='seagreen', label='Roll')
    detail_axes[1].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    detail_axes[1].set_ylabel('Attitude [deg]')
    detail_axes[1].legend(fontsize=9)
    for motor in range(4):
        detail_axes[2].plot(detail_time, motor_u[window, motor], linewidth=1.0, label=f'M{motor + 1}')
    detail_axes[2].set_ylim(-0.02, 1.05)
    detail_axes[2].set_ylabel('Motor command\n[0–1]')
    detail_axes[2].legend(fontsize=8, ncol=4)
    detail_axes[3].plot(detail_time, xy_error[window], color='tab:red', linewidth=1.8,
                        label='XY target error')
    detail_axes[3].axhline(args.target_tolerance, color='k', linestyle='--', linewidth=0.8,
                           label=f'{args.target_tolerance:.2f} m tolerance')
    detail_axes[3].set_ylabel('XY error [m]')
    detail_axes[3].legend(fontsize=9)
    detail_axes[4].plot(detail_time, power[window], color='tab:blue', linewidth=1.5,
                        label='Model electrical power')
    detail_axes[4].set_ylabel('Power [W]')
    detail_axes[4].set_xlabel('Continuous evaluation time [s]')
    detail_axes[4].legend(fontsize=9)
    for axis in detail_axes:
        reset_label_used = False
        for start, end in reset_drop_intervals:
            if end >= window_start and start <= window_end:
                axis.axvspan(start, end, color='0.75', alpha=0.28,
                            label='Episode reset/drop' if not reset_label_used else None)
                reset_label_used = True
        for event_time in global_time[touchdown & window]:
            axis.axvline(event_time, color='cyan', linewidth=0.8, alpha=0.8)
        axis.grid(alpha=0.22)
        axis.set_xlim(window_start, window_end)
    detail_figure.tight_layout()
    detail_figure.savefig(output_dir / 'isaac_detail_window.png', dpi=180)
    if args.show:
        plt.show()
    plt.close(detail_figure)
    print(json.dumps(summary, indent=2))
    print(f'Analysis written to: {output_dir}')


if __name__ == '__main__':
    main()
