"""Paired statistics on results/test.csv, plus a high-jerk stress run on the test seeds."""
import csv, json
from multiprocessing import Pool
import numpy as np
from experiments.run_wqn_sim import evaluate

rows = list(csv.DictReader(open("results/test.csv")))
for r in rows:
    for k, v in r.items():
        if k not in ("act", "method"):
            r[k] = float(v) if v not in ("", "nan") else np.nan
key = lambda r: (r["seed"], r["act"])
base = {key(r): r for r in rows if r["method"] == "ANC"}
rng = np.random.default_rng(0)


def paired(method, metric, acts=("walk", "run", "bike"), rel=False):
    """Mean change vs ANC with a seed-level bootstrap CI. Seeds, not windows, are the unit."""
    seeds = sorted({r["seed"] for r in rows})
    per = {}
    for r in rows:
        if r["method"] == method and r["act"] in acts:
            per.setdefault(r["seed"], []).append((r[metric], base[key(r)][metric]))
    per = {s: [(a, b) for a, b in v if np.isfinite(a - b)] for s, v in per.items()}
    seeds = [s for s in seeds if per.get(s)]
    x = np.array([np.mean([a - b for a, b in per[s]]) for s in seeds])
    b0 = np.array([np.mean([b for a, b in per[s]]) for s in seeds])
    idx = rng.integers(0, len(x), (5000, len(x)))
    f = (lambda i: 100 * x[i].mean(-1) / b0[i].mean(-1)) if rel else (lambda i: x[i].mean(-1))
    return float(f(np.arange(len(x)))), *np.percentile(f(idx), [2.5, 97.5]).tolist()


fmt = lambda t: f"{t[0]:+7.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
METHODS = ("ANC+soft", "WQN-res", "WQN-imu", "WQN-both", "freeze", "freeze+WQN", "WQN-oracle")
print("change vs ANC, mean [95% bootstrap CI over 40 seeds]")
for metric, rel in (("rmse_art", True), ("rmse_clean", True), ("mae", False), ("mae_art", False), ("f1_art", False), ("surge_gain", False)):
    print(f"\n{metric}" + (" (%)" if rel else ""))
    for m in METHODS:
        print(f"  {m:12s} {fmt(paired(m, metric, rel=rel))}")
print("\nby activity, WQN-both")
for a in ("walk", "run", "bike"):
    print(f"  {a:5s} rmse_art% {fmt(paired('WQN-both','rmse_art',(a,),True))}  f1 {fmt(paired('WQN-both','f1_art',(a,)))}  mae {fmt(paired('WQN-both','mae',(a,)))}")
print("\ndetection (fraction of events intersecting an interval)")
for m in ("WQN-res", "WQN-imu", "WQN-both"):
    rr = [r for r in rows if r["method"] == m]
    s = lambda k: sum(r[k] for r in rr)
    print(f"  {m:9s} coupled {s('hit_coupled')/s('n_coupled'):.3f}  ppg_only {s('hit_ppg')/s('n_ppg'):.3f}  imu_only {s('hit_imu')/s('n_imu'):.3f}  surge {s('hit_surge')/s('n_surge'):.3f}  intervals/rec {np.mean([r['n_runs'] for r in rr]):.1f}")
print("events:", {k: int(sum(r[k] for r in rows if r["method"] == "ANC")) for k in ("n_coupled", "n_ppg", "n_imu", "n_surge")})
print("records:", len(base))

cfg = json.load(open("results/locked.json"))


def _nanmean(v):
    """nan when every value is nan (no events at rate 0), without a RuntimeWarning."""
    v = np.asarray(v, float)
    return np.nanmean(v) if np.isfinite(v).any() else np.nan


def job(a):
    return evaluate(cfg["gate"], cfg["L"], cfg["mu"], a[0], a[1], ("ANC", "WQN-both", "WQN-res", "WQN-oracle"), jerks=a[2])


if __name__ == "__main__":
    print("\nstress: nominal rate vs measured rate (the simulator keeps events at least 12 s apart)")
    with Pool(2) as p:
        for rate in (0.0001, 2, 5, 10):
            jobs = [(s, a, rate) for s in range(1000, 1020) for a in ("walk", "run", "bike")]
            res = [r for rs in p.map(job, jobs) for r in rs]
            rr = [r for r in res if r["method"] == "ANC"]
            n = np.mean([r["n_coupled"] + r["n_ppg"] + r["n_imu"] + r["n_surge"] for r in rr]) / 4
            line = f"  nominal {rate:6.1f} measured {n:.1f}/min "
            for m in ("ANC", "WQN-both", "WQN-res", "WQN-oracle"):
                q = [r for r in res if r["method"] == m]
                line += f" {m}: mae {np.mean([r['mae'] for r in q]):.2f} f1 {_nanmean([r['f1_art'] for r in q]):.3f} |"
            print(line, flush=True)
