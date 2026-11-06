"""IMU-gated WQN on the real Wrist PPG During Exercise records.

  python -m experiments.run_wqn --data data --out results/wqn --loso results/loso_params.csv
  add --inject for known-truth injections
  python -m experiments.run_wqn --data data --out results/wqn --grid     # LOSO choice of k_imu, k_res

ECG is read only through src/extension_score.py.
"""
import argparse, csv, json, os
import numpy as np
from sim.semisynth import _burst, _ppg_transient
from src.filters import nlms
from src.gate import GateConfig, find_artifacts
from src.hr import track_hr
from src.load import list_records, load_record
from src.loso import loso_params
from src.wqn_pipeline import clean, jerk_z
from src.extension_score import beat_f1, beats, hr_mae, match_lag, windows_mask
from src.preprocess import _bandpass, _resample, preprocess
from src.wqn import universal_soft

fs = 64


def pipeline(rec, cfg, L, mu):
    """All compared methods on one record. Returns refs, outputs, and per-method runs/reports."""
    d, refs, _ = preprocess(rec)
    N = len(d)
    e0, _ = nlms(d, refs, L=L, mu=mu)
    zj = jerk_z(rec, fs, cfg)
    k = min(N, len(zj))
    zj = np.pad(zj[:k], (0, N - k), mode="edge")
    hr0, t = track_hr(e0, fs, motion=refs)
    outs, info = {"ANC": e0, "ANC+soft": universal_soft(e0)}, {}
    for name, mode in (("WQN-res", "res"), ("WQN-imu", "imu"), ("WQN-both", "both")):
        outs[name], runs, rep = clean(e0, refs, fs, zj, cfg, mode, hr=hr0, t_hr=t)
        info[name] = (runs, rep)
    r0 = find_artifacts(e0, zj, fs, cfg, "both")[0]
    m = np.full(N, float(mu))
    for a, b in r0:
        m[a:b] = 0.0
    e1, _ = nlms(d, refs, L=L, mu=m)
    outs["freeze"] = e1
    outs["freeze+WQN"], runs, rep = clean(e1, refs, fs, zj, cfg, "both", hr=hr0, t_hr=t)
    info["freeze+WQN"] = (runs, rep)
    return refs, outs, info, zj, t


def score_ecg(rec, refs, outs, info, t):
    """HR and beat scores against ECG, overall and around the events found by WQN-both."""
    N = len(next(iter(outs.values())))
    runs = info["WQN-both"][0]
    near = np.zeros(N, bool)
    for a, b in runs:
        near[max(0, a - 2 * fs):min(N, b + 2 * fs)] = True
    keep = np.array([near[max(0, int((c - 4) * fs)):int((c + 4) * fs)].any() for c in t])
    rows = []
    for m, e in outs.items():
        hr, _ = track_hr(e, fs, motion=refs)
        est = beats(e, fs, hr, t)
        ref = rec["rpeaks_s"]
        lag = match_lag(est, ref)
        sel = lambda b: b[near[np.clip((b * fs).astype(int), 0, N - 1)]]
        rows.append(dict(record=rec["name"], activity=rec["activity"], method=m,
                         mae=hr_mae(hr, t, ref), mae_events=hr_mae(hr, t, ref, keep),
                         beat_f1_events=beat_f1(sel(est - lag), sel(ref)),
                         n_events=len(runs), n_corrected=sum(r["status"] == "corrected" for r in info["WQN-both"][1]),
                         time_flagged=float(near.mean())))
    return rows


def inject(rec, seed, per_min=2.0):
    """Copy of rec with known PPG transients and matching IMU bursts. Returns (copy, times)."""
    rng = np.random.default_rng(seed)
    out = dict(rec)
    n = len(rec["ppg"])
    t = np.arange(n) / rec["fs"]
    T = n / rec["fs"]
    ppg, acc, gyro = rec["ppg"].copy(), rec["acc"].copy(), rec["gyro"].copy()
    level = _bandpass(_resample(rec["ppg"], rec["fs"], fs), fs).std()
    a_sd, g_sd = rec["acc"].std(0), rec["gyro"].std(0)
    times, tt = [], rng.uniform(25, 40)
    while tt < T - 25:
        ppg += _ppg_transient(rng, t, tt, level * rng.uniform(2, 6))
        for i in range(3):
            acc[:, i] += a_sd[i] * rng.uniform(4, 9) * _burst(rng, t, tt)
            gyro[:, i] += g_sd[i] * rng.uniform(4, 9) * _burst(rng, t, tt)
        times.append(float(tt))
        tt += max(12.0, rng.exponential(60 / per_min))
    out.update(ppg=ppg, acc=acc, gyro=gyro)
    return out, times


def score_inject(rec, cfg, L, mu, seed=0):
    """Truth = the same method on the record before injection. No ECG needed."""
    rec2, times = inject(rec, seed)
    if not times:
        return []
    _, base, _, _, _ = pipeline(rec, cfg, L, mu)
    _, outs, info, _, _ = pipeline(rec2, cfg, L, mu)
    s1 = _bandpass(_resample(rec["ppg"], rec["fs"], fs), fs).std()
    s2 = _bandpass(_resample(rec2["ppg"], rec["fs"], fs), fs).std()
    N = len(base["ANC"])
    M = windows_mask(N, fs, times, 0.75, 2.25)
    rows = []
    for m, e in outs.items():
        err = e * (s2 / s1) - base["ANC"]
        rows.append(dict(record=rec["name"], activity=rec["activity"], method=m,
                         rmse_injected=float(np.sqrt(np.mean(err[M] ** 2))),
                         rmse_elsewhere=float(np.sqrt(np.mean(err[~M] ** 2)))))
    runs = info["WQN-both"][0]
    det = np.zeros(N, bool)
    for a, b in runs:
        det[a:b] = True
    hits = sum(det[max(0, int((c - .75) * fs)):int((c + 2.25) * fs)].any() for c in times)
    for r in rows:
        r.update(n_injected=len(times), n_detected=int(hits), n_runs=len(runs))
    return rows


def grid(names, a, out):
    """Leave-one-subject-out choice of k_imu and k_res by injected-transient error."""
    recs = [load_record(os.path.join(a.data, n)) for n in names]
    res = {}
    for ki in (4, 6, 9):
        for kr in (3, 5, 8):
            cfg = GateConfig(k_imu=ki, k_res=kr)
            for r in recs:
                rows = score_inject(r, cfg, a.L, a.mu)
                res[(ki, kr, r["subject"], r["name"])] = next(
                    (x["rmse_injected"] for x in rows if x["method"] == "WQN-both"), np.nan)
    subjects = sorted({r["subject"] for r in recs})
    picks = []
    for s in subjects:
        keys = sorted({k[:2] for k in res})
        loss = {k: np.nanmean([v for q, v in res.items() if q[:2] == k and q[2] != s]) for k in keys}
        best = min(loss, key=loss.get)
        held = np.nanmean([v for q, v in res.items() if q[:2] == best and q[2] == s])
        picks.append(dict(subject=s, k_imu=best[0], k_res=best[1], held_out_rmse=float(held)))
    write(picks, os.path.join(out, "loso_picks.csv"))
    print("LOSO picks:", picks)


def write(rows, path):
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: next(i for i, r in enumerate(rows) if k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def summary(rows, keys):
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"{'method':14s}" + "".join(f"{k:>18s}" for k in keys))
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        print(f"{m:14s}" + "".join(f"{np.nanmean([r[k] for r in rr]):18.3f}" for k in keys))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results/wqn")
    p.add_argument("--records", default="")
    p.add_argument("--loso", default="", help="use each subject's held-out NLMS-all settings")
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--k-imu", type=float, default=None)
    p.add_argument("--k-res", type=float, default=None)
    p.add_argument("--inject", action="store_true")
    p.add_argument("--grid", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.records.split(",") if a.records else list_records(a.data)
    kw = {}
    if a.k_imu is not None:
        kw["k_imu"] = a.k_imu
    if a.k_res is not None:
        kw["k_res"] = a.k_res
    cfg = GateConfig(**kw)
    json.dump(dict(L=a.L, mu=a.mu, gate=vars(cfg)), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    if a.grid:
        grid(names, a, a.out)
    rows, ev, inj = [], [], []
    tuned = loso_params(a.loso) if a.loso else {}
    for n in names:
        rec = load_record(os.path.join(a.data, n))
        a.L, a.mu = tuned.get(rec["subject"], (a.L, a.mu))
        refs, outs, info, zj, t = pipeline(rec, cfg, a.L, a.mu)
        rows += score_ecg(rec, refs, outs, info, t)
        d = preprocess(rec)[0]
        for x in info["WQN-both"][1]:
            a0, a1 = int(x["t0"] * fs), int(x["t1"] * fs)
            # above 1 means ANC output is bigger than its input here: the canceller added energy
            x["anc_gain"] = float(outs["ANC"][a0:a1].std() / (d[a0:a1].std() + 1e-12))
            ev.append(dict(record=rec["name"], **x))
        if a.inject:
            inj += score_inject(rec, cfg, a.L, a.mu)
        print(n, "events", len(info["WQN-both"][1]), flush=True)
    write(rows, os.path.join(a.out, "scores.csv"))
    write(ev, os.path.join(a.out, "events.csv"))
    write(inj, os.path.join(a.out, "injected.csv"))
    summary(rows, ("mae", "mae_events", "beat_f1_events"))
    if inj:
        summary(inj, ("rmse_injected", "rmse_elsewhere", "n_detected", "n_injected"))
