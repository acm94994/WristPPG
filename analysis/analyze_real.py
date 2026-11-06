"""Summaries of the real-data runs. Subjects are the unit: 8 subjects, so intervals are wide."""
import csv, os
import numpy as np
from scipy.stats import spearmanr, wilcoxon

rd = lambda p: list(csv.DictReader(open(p)))
rng = np.random.default_rng(0)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def by_subject(rows, m, metric, base="ANC", acts=None):
    b = {r["record"]: num(r[metric]) for r in rows if r["method"] == base}
    per = {}
    for r in rows:
        if r["method"] == m and (acts is None or r["activity"] in acts):
            v = num(r[metric]) - b[r["record"]]
            if np.isfinite(v):
                per.setdefault(r["record"].split("_")[0], []).append(v)
    return {s: float(np.mean(v)) for s, v in per.items()}


def line(x):
    v = np.array(list(x.values()))
    if len(v) == 0:
        return "no data"
    ci = np.percentile(v[rng.integers(0, len(v), (5000, len(v)))].mean(1), [2.5, 97.5])
    nz = v[v != 0]
    p = wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan
    return f"{v.mean():+7.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  better in {(v < 0).sum()}/{len(v)}  wilcoxon p={p:.3f}"


print("=== COLLISION: census ===")
cen = rd("results/collision/census.csv")
sc = rd("results/collision/scores.csv")
anc = {r["record"]: num(r["mae"]) for r in sc if r["method"] == "ANC"}
print(f"{'record':26s}{'mu':>6s}{'%<15':>6s}{'%<6':>6s}{'n<6':>5s}{'med s':>7s}{'max s':>7s}{'ANC MAE':>9s}")
mu = {r["record"]: r["mu"] for r in sc}
for r in cen:
    print(f"{r['record']:26s}{mu[r['record']]:>6s}{100*num(r['frac_within_15']):6.0f}{100*num(r['frac_within_6']):6.0f}{r['n_within_6']:>5s}{num(r['median_dur_s']):7.0f}{num(r['max_dur_s']):7.0f}{anc[r['record']]:9.1f}")
f = np.array([num(r["frac_within_6"]) for r in cen])
m = np.array([anc[r["record"]] for r in cen])
print(f"mean time within 6 BPM {100*f.mean():.0f}%, within 15 BPM {100*np.mean([num(r['frac_within_15']) for r in cen]):.0f}%")
sr = spearmanr(f, m)
print(f"Spearman, time within 6 BPM vs NLMS-all MAE, 19 records: rho={sr.statistic:.2f} p={sr.pvalue:.3f}")
for a in ("walk", "run", "low_resistance_bike", "high_resistance_bike"):
    q = [r for r in cen if r["activity"] == a]
    print(f"  {a:22s} n={len(q)} within 6 BPM {100*np.mean([num(r['frac_within_6']) for r in q]):.0f}%  MAE {np.mean([anc[r['record']] for r in q]):.1f}")

print("\n=== COLLISION: methods, record mean level ===")
ms = ("ANC", "aware-only", "hold", "mu", "hold+aware", "mu+aware")
print(f"{'method':12s}{'MAE':>8s}{'zone MAE':>10s}{'zone F1':>9s}{'hold %':>8s}")
for x in ms:
    q = [r for r in sc if r["method"] == x]
    print(f"{x:12s}{np.nanmean([num(r['mae']) for r in q]):8.2f}{np.nanmean([num(r['mae_zone']) for r in q]):10.2f}{np.nanmean([num(r['f1_zone']) for r in q]):9.3f}{100*np.mean([num(r['hold_frac']) for r in q]):8.1f}")
print("\nchange vs ANC, by subject")
for metric in ("mae", "mae_zone", "f1_zone"):
    print(metric)
    for x in ms[1:]:
        print(f"  {x:12s} {line(by_subject(sc, x, metric))}")
print("\nrecords changed by more than 2 BPM, whole-record MAE")
for x in ms[1:]:
    d = {r["record"]: num(r["mae"]) - anc[r["record"]] for r in sc if r["method"] == x}
    print(f"  {x:12s} better {sum(v < -2 for v in d.values())}, worse {sum(v > 2 for v in d.values())}; largest worse {max(d, key=d.get)} {max(d.values()):+.1f}; largest better {min(d, key=d.get)} {min(d.values()):+.1f}")

if os.path.exists("results/collision/holdtest.csv"):
    h = rd("results/collision/holdtest.csv")
    print("\n=== held weights, away from collisions ===")
    for s in sorted({int(num(r["hold_s"])) for r in h}):
        q = [r for r in h if int(num(r["hold_s"])) == s]
        print(f"  hold {s:2d} s  records {len(q):2d}  leftover vs adaptive {np.median([num(r['left_vs_adaptive']) for r in q]):6.2f}  epsilon {np.median([num(r['epsilon']) for r in q]):.2f}")

print("\n=== IMU-gated WQN ===")
o = rd("results/wqn/scores.csv")
ev = rd("results/wqn/events.csv")
dur = sum(num(r["minutes"]) for r in cen)
print(f"events {len(ev)} over {dur:.0f} min = {len(ev)/dur:.2f}/min; no_reference {np.mean([r['status']=='no_reference' for r in ev])*100:.0f}%; anc_gain>1 in {np.mean([num(r.get('anc_gain')) > 1 for r in ev if r.get('anc_gain')])*100:.0f}%")
for x in ("ANC", "WQN-res", "WQN-imu", "WQN-both", "freeze", "freeze+WQN"):
    q = [r for r in o if r["method"] == x]
    print(f"  {x:11s} MAE {np.nanmean([num(r['mae']) for r in q]):6.2f}  event F1 {np.nanmean([num(r['beat_f1_events']) for r in q]):.3f}")
print("change vs ANC by subject, beat F1 near events")
for x in ("WQN-res", "WQN-imu", "WQN-both", "freeze+WQN"):
    print(f"  {x:11s} {line(by_subject(o, x, 'beat_f1_events'))}")
print("change vs ANC by subject, HR MAE")
for x in ("WQN-res", "WQN-imu", "WQN-both", "freeze+WQN"):
    print(f"  {x:11s} {line(by_subject(o, x, 'mae'))}")
j = rd("results/wqn/injected.csv")
print("injections (rmse vs same record un-injected)")
for x in ("ANC", "ANC+soft", "WQN-res", "WQN-imu", "WQN-both", "freeze", "freeze+WQN"):
    q = [r for r in j if r["method"] == x]
    print(f"  {x:11s} injected {np.mean([num(r['rmse_injected']) for r in q]):.3f}  elsewhere {np.mean([num(r['rmse_elsewhere']) for r in q]):.3f}")
for x in ("WQN-imu", "WQN-both", "freeze+WQN"):
    b = {r["record"]: num(r["rmse_injected"]) for r in j if r["method"] == "ANC"}
    per = {}
    for r in j:
        if r["method"] == x:
            per.setdefault(r["record"].split("_")[0], []).append((num(r["rmse_injected"]) / b[r["record"]] - 1) * 100)
    print(f"  {x:11s} injected error % by subject {line({s: float(np.mean(v)) for s, v in per.items()})}")
q = [r for r in j if r["method"] == "ANC"]
print(f"  detected {sum(num(r['n_detected']) for r in q):.0f} of {sum(num(r['n_injected']) for r in q):.0f} injected transients")
