#!/usr/bin/env python3
"""Per-hop analysis of the 1-env diagnostic CSV (on_quadhopper_sim.csv).

Segments hops by spring-contact transitions, classifies each hop by the
target radius measured at liftoff (short <= 0.30 / long > 0.30), estimates
the env route phase (parity of touchdown index within the episode, adjusted
for a drop-bounce first touchdown), and reports landing error, apex, and the
lenient hit predicate (err < 0.10 AND apex >= 0.58).
"""
import csv
import math
import sys
from collections import defaultdict

CSV_PATH = (
    "/home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac/"
    "outputs/planner_random_two_hop/on_quadhopper_sim.csv"
)

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "t": float(r["Time_s"]),
            "x": float(r["X"]),
            "y": float(r["Y"]),
            "z": float(r["Z"]),
            "tx": float(r["Target_X"]),
            "ty": float(r["Target_Y"]),
            "contact": int(float(r["Is_Contact"])),
        })

if len(rows) < 10:
    print(f"CSV too short ({len(rows)} rows)")
    sys.exit(1)

# Episode boundaries: Time_s rewinds.
episode_of = [0] * len(rows)
ep = 0
for i in range(1, len(rows)):
    if rows[i]["t"] < rows[i - 1]["t"] - 0.05:
        ep += 1
    episode_of[i] = ep

# Contact transitions.
events = []  # (idx, kind)
for i in range(1, len(rows)):
    prev, cur = rows[i - 1], rows[i]
    if prev["contact"] == 1 and cur["contact"] == 0:
        events.append((i, "liftoff"))
    elif prev["contact"] == 0 and cur["contact"] == 1:
        events.append((i, "touchdown"))

# Build hops: liftoff -> next touchdown.
hops = []  # dicts
open_hop = None
for idx, kind in events:
    if kind == "liftoff":
        open_hop = {"lo": idx}
    else:
        if open_hop is not None:
            open_hop["td"] = idx
            hops.append(open_hop)
            open_hop = None

if not hops:
    print("no hops found")
    sys.exit(1)

# First touchdown per episode = drop-bounce heuristic: duration < 0.9 s.
episode_first_td = {}
for h in hops:
    ep = episode_of[h["td"]]
    if ep not in episode_first_td:
        episode_first_td[ep] = h["td"]

bounce_td = set()
for h in hops:
    dur = rows[h["td"]]["t"] - rows[h["lo"]]["t"]
    if dur < 0.9 and episode_first_td.get(episode_of[h["td"]]) == h["td"]:
        bounce_td.add(h["td"])

# Classify each hop.
stats = defaultdict(lambda: {"n": 0, "errs": [], "apexes": [], "hits": 0, "phases": []})
detail = []
td_count = defaultdict(int)  # per episode, # of touchdowns seen (incl bounce)
for h in hops:
    lo, td = h["lo"], h["td"]
    ep = episode_of[td]
    is_bounce = td in bounce_td
    # env phase = parity of touchdown index within episode, starting 0
    phase = td_count[ep] % 2
    td_count[ep] += 1
    r_lo = rows[lo]
    r_td = rows[td]
    radius = math.hypot(r_lo["tx"] - r_lo["x"], r_lo["ty"] - r_lo["y"])
    err = math.hypot(r_td["x"] - r_td["tx"], r_td["y"] - r_td["ty"])
    seg = rows[lo:td + 1]
    apex = max(s["z"] for s in seg)
    dur = r_td["t"] - r_lo["t"]
    kind = "short" if radius <= 0.30 else "long"
    hit = err < 0.10 and apex >= 0.58
    if not is_bounce:
        s = stats[kind]
        s["n"] += 1
        s["errs"].append(err)
        s["apexes"].append(apex)
        s["hits"] += int(hit)
        s["phases"].append(phase)
        detail.append((ep, kind, phase, radius, err, apex, dur, hit))

for kind in ("short", "long"):
    s = stats[kind]
    if s["n"] == 0:
        print(f"{kind}: no hops")
        continue
    errs, apexes = s["errs"], s["apexes"]
    print(
        f"{kind} ({s['n']} hops): err mean {sum(errs)/len(errs):.3f} "
        f"[<0.10: {sum(1 for e in errs if e < 0.10)}], "
        f"apex mean {sum(apexes)/len(apexes):.3f} [>=0.58: {sum(1 for a in apexes if a >= 0.58)}], "
        f"hits {s['hits']} rate {s['hits']/s['n']:.3f}, "
        f"phase parity: even {sum(1 for p in s['phases'] if p == 0)} / odd {sum(1 for p in s['phases'] if p == 1)}"
    )

print("\nper-hop detail (ep, kind, phase, radius, err, apex, dur, hit):")
for d in detail:
    print(f"  ep{d[0]} {d[1]:5s} phase={d[2]} rad={d[3]:.2f} err={d[4]:.3f} apex={d[5]:.3f} dur={d[6]:.2f} hit={int(d[7])}")
