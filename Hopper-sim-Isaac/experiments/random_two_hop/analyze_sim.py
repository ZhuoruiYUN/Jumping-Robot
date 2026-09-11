import csv, sys
import numpy as np

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows

def yaw_deg(qw,qx,qy,qz):
    return np.degrees(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))

def tilt_deg(qx,qy):
    return np.degrees(np.arccos(np.clip(1-2*(qx*qx+qy*qy), -1, 1)))

for path in sys.argv[1:]:
    rows = load(path)
    t = np.array([r['Time_s'] for r in rows])
    x = np.array([r['X'] for r in rows]); y = np.array([r['Y'] for r in rows])
    z = np.array([r['Z'] for r in rows])
    tx = np.array([r['Target_X'] for r in rows]); ty = np.array([r['Target_Y'] for r in rows])
    vx = np.array([r['Vx'] for r in rows]); vy = np.array([r['Vy'] for r in rows])
    qw = np.array([r['Qw'] for r in rows]); qx = np.array([r['Qx'] for r in rows])
    qy = np.array([r['Qy'] for r in rows]); qz = np.array([r['Qz'] for r in rows])
    contact = np.array([r['Is_Contact'] for r in rows])
    yaw = yaw_deg(qw,qx,qy,qz); tilt = tilt_deg(qx,qy)
    gz = np.degrees(np.gradient(np.unwrap(np.radians(yaw)), t))
    z_rest = float(np.median(z[contact > 0.5])) if np.any(contact > 0.5) else float(np.median(z[:100]))
    d = np.diff(contact)
    lift = np.where(d < 0)[0] + 1
    td = np.where(d > 0)[0] + 1
    hops = []
    for s in lift:
        e = td[td > s]
        if len(e) == 0: break
        hops.append((int(s), int(e[0])))
    print('='*90); print(path)
    print(f'  rows={len(rows)} dur={t[-1]-t[0]:.1f}s hops={len(hops)}')
    stats = []
    for k,(s,e) in enumerate(hops):
        seg = slice(s, e+1)
        zseg = z[seg]; apex_i = int(np.argmax(zseg))
        dz = float(zseg[apex_i] - z_rest)
        yseg = np.degrees(np.unwrap(np.radians(yaw[seg])))
        y_rot = float(yseg[-1] - yseg[0])
        ctx = 'S' if k%2==0 else 'L'
        td_err = float(np.hypot(x[e]-tx[e], y[e]-ty[e]))
        stats.append(dict(ctx=ctx, dz=dz, apex_tilt=float(tilt[s+apex_i]),
            peak_gz=float(np.abs(gz[seg]).max()), yaw_rot=abs(y_rot),
            td_vxy=float(np.hypot(vx[e],vy[e])), td_err=td_err, td_tilt=float(tilt[e])))
        if k < 12 or k >= len(hops)-4:
            print(f'  hop {k+1:3d} {ctx} dz={dz:6.3f} tilt={stats[-1]["apex_tilt"]:5.1f} peakgz={stats[-1]["peak_gz"]:5.1f} yawrot={y_rot:+6.1f} vxy={stats[-1]["td_vxy"]:5.2f} err={td_err:5.3f} tdtilt={stats[-1]["td_tilt"]:4.1f}')
    def summ(ss, name):
        if not ss: return
        print(f'  {name} n={len(ss)}: dz={np.mean([s["dz"] for s in ss]):.3f} apex_tilt={np.mean([s["apex_tilt"] for s in ss]):.1f} peak_gz={np.mean([s["peak_gz"] for s in ss]):.1f} |yaw_rot|={np.mean([s["yaw_rot"] for s in ss]):.1f} td_vxy={np.mean([s["td_vxy"] for s in ss]):.2f} td_err={np.mean([s["td_err"] for s in ss]):.3f} td_tilt={np.mean([s["td_tilt"] for s in ss]):.1f}')
    summ([s for s in stats if s['ctx']=='S'], 'SHORT')
    summ([s for s in stats if s['ctx']=='L'], 'LONG')
