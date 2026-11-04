"""Semi-synthetic experiment. Usage:
   python run_sim.py dev    (grid search on dev seeds, writes results/dev.csv)
   python run_sim.py test   (locked settings on untouched test seeds, writes results/test.csv)
"""
import csv, json, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.signal import butter
from sim.semisynth import make_record
from src.filters import nlms
from src.gate import GateConfig, find_artifacts
from src.hr import track_hr
from src.option3 import clean, jerk_z
from src.option3_eval import beat_f1, beats, hr_mae, match_lag, windows_mask
from src.preprocess import _bandpass, _resample, preprocess
from src.wqn import universal_soft

fs = 64
ACTS = ("walk", "run", "bike")
METHODS = ("ANC", "ANC+soft", "WQN-res", "WQN-imu", "WQN-both", "freeze", "freeze+WQN", "WQN-oracle")


def truth(rec, d_len):
    base = _bandpass(_resample(rec["ppg"], rec["fs"], fs), fs)
    s = _bandpass(_resample(rec["_s"], rec["fs"], fs), fs) / base.std()
    return s[:d_len]


def evaluate(cfg_kw, L, mu, seed, act, methods=METHODS, jerks=2.0):
    cfg = GateConfig(**cfg_kw)
    rec = make_record(seed, act, jerks_per_min=jerks)
    d, refs, _ = preprocess(rec)
    s = truth(rec, len(d))
    N = len(d)
    ev = rec["_events"]
    art_t = [x["t"] for x in ev if x["kind"] in ("coupled", "ppg_only")]
    coup_t = [x["t"] for x in ev if x["kind"] == "coupled"]
    surge_t = [x["t"] + 3 for x in ev if x["kind"] == "surge"]
    M_art = windows_mask(N, fs, art_t, 0.75, 2.25)
    M_any = windows_mask(N, fs, [x["t"] for x in ev], 1.0, 8.0)
    M_surge = windows_mask(N, fs, surge_t, 2.0, 2.0)

    e0, _ = nlms(d, refs, L=L, mu=mu)
    zj = jerk_z(rec, fs, cfg)
    k = min(N, len(zj))
    zj = np.pad(zj[:k], (0, N - k), mode="edge")
    hr0, t = track_hr(e0, fs, motion=refs)
    outs, runs = {}, {}
    for m in methods:
        if m == "ANC":
            outs[m], runs[m] = e0, []
        elif m == "ANC+soft":
            outs[m], runs[m] = universal_soft(e0), []
        elif m in ("WQN-res", "WQN-imu", "WQN-both"):
            mode = {"WQN-res": "res", "WQN-imu": "imu", "WQN-both": "both"}[m]
            outs[m], runs[m], _ = clean(e0, refs, fs, zj, cfg, mode, hr=hr0, t_hr=t)
        elif m in ("freeze", "freeze+WQN"):
            r0 = find_artifacts(e0, zj, fs, cfg, "both")[0]
            mu_n = np.full(N, float(mu))
            for a, b in r0:
                mu_n[a:b] = 0.0
            e1, _ = nlms(d, refs, L=L, mu=mu_n)
            if m == "freeze":
                outs[m], runs[m] = e1, r0
            else:
                outs[m], runs[m], _ = clean(e1, refs, fs, zj, cfg, "both", hr=hr0, t_hr=t)
        elif m == "WQN-oracle":
            orc = [(max(0, int((c - .75) * fs)), min(N, int((c + 2.25) * fs))) for c in art_t]
            outs[m], runs[m], _ = clean(e0, refs, fs, zj, cfg, "both", runs=orc, hr=hr0, t_hr=t)

    bt = rec["_beats_s"]
    bref = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] / fs + 1 / fs
    bref = bref[np.r_[True, np.diff(bref) > 0.25]]
    hr_true_keep = np.array([M_art[max(0, int((c - 4) * fs)):int((c + 4) * fs)].any() for c in t])
    rows = []
    for m, e in outs.items():
        hr, _ = track_hr(e, fs, motion=refs) if m != "ANC" else (hr0, t)
        est = beats(e, fs, hr, t)
        lag = match_lag(est, bref)
        est2 = est - lag
        inwin = lambda b: b[M_art[np.clip((b * fs).astype(int), 0, N - 1)]]
        rmse = lambda sel: float(np.sqrt(np.mean((e[sel] - s[sel]) ** 2))) if sel.any() else np.nan
        G = lambda sel: float((e[sel] @ s[sel]) / (s[sel] @ s[sel])) if sel.any() else np.nan
        outside = ~windows_mask(N, fs, [x["t"] for x in ev], 3, 6)
        changed = float(np.sum((e[outside] - e0[outside]) ** 2) / np.sum(e0[outside] ** 2))
        dets = np.zeros(N, bool)
        for a, b in runs[m]:
            dets[a:b] = True
        hit = lambda ts: [dets[max(0, int((c - 0.75) * fs)):int((c + 2.25) * fs)].any() for c in ts]
        imu_only = [x["t"] for x in ev if x["kind"] == "imu_only"]
        ppg_only = [x["t"] for x in ev if x["kind"] == "ppg_only"]
        surge = [x["t"] for x in ev if x["kind"] == "surge"]
        row = dict(seed=seed, act=act, method=m,
                   rmse_art=rmse(M_art), rmse_clean=rmse(~M_any), changed_outside=changed,
                   mae=hr_mae(hr, t, rec["rpeaks_s"]), mae_art=hr_mae(hr, t, rec["rpeaks_s"], hr_true_keep),
                   f1_art=beat_f1(inwin(est2), inwin(bref)), surge_gain=G(M_surge),
                   n_coupled=len(coup_t), hit_coupled=sum(hit(coup_t)),
                   n_ppg=len(ppg_only), hit_ppg=sum(hit(ppg_only)),
                   n_imu=len(imu_only), hit_imu=sum(hit(imu_only)),
                   n_surge=len(surge), hit_surge=sum(hit(surge)), n_runs=len(runs[m]))
        rows.append(row)
    return rows


def _job(a):
    return evaluate(*a)


def run(cfg_kw, L, mu, seeds, methods=METHODS):
    jobs = [(cfg_kw, L, mu, s, a, methods) for s in seeds for a in ACTS]
    with Pool(2) as p:
        return [r for rs in p.imap_unordered(_job, jobs, chunksize=2) for r in rs]


def write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(rows, keys=("rmse_art", "rmse_clean", "mae", "mae_art", "f1_art", "surge_gain", "changed_outside")):
    ms = []
    for r in rows:
        if r["method"] not in ms:
            ms.append(r["method"])
    print(f"{'method':12s}" + "".join(f"{k:>16s}" for k in keys))
    for m in METHODS:
        if m in ms:
            rr = [r for r in rows if r["method"] == m]
            print(f"{m:12s}" + "".join(f"{np.nanmean([r[k] for r in rr]):16.4f}" for k in keys))


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        seeds = range(0, 20)
        res = []
        for mu in (0.01, 0.05):
            rows = run({}, 16, mu, seeds, ("ANC",))
            print("mu", mu, "ANC mae", round(np.mean([r["mae"] for r in rows]), 2),
                  "rmse_clean", round(np.mean([r["rmse_clean"] for r in rows]), 3))
        for k_imu in (4, 6, 9):
            for k_res in (3, 5, 8):
                for q in (0.25, 0.35, 0.5):
                    cfg = dict(k_imu=k_imu, k_res=k_res, q_min=q)
                    rows = run(cfg, 16, 0.01, seeds, ("ANC", "WQN-both"))
                    b = [r for r in rows if r["method"] == "WQN-both"]
                    a = {(r["seed"], r["act"]): r for r in rows if r["method"] == "ANC"}
                    dr = np.mean([r["rmse_art"] - a[(r["seed"], r["act"])]["rmse_art"] for r in b])
                    dm = np.mean([r["mae"] - a[(r["seed"], r["act"])]["mae"] for r in b])
                    res.append(dict(k_imu=k_imu, k_res=k_res, q=q, d_rmse=dr, d_mae=dm,
                                    recall=sum(r["hit_coupled"] for r in b) / sum(r["n_coupled"] for r in b),
                                    fa_imu=sum(r["hit_imu"] for r in b) / max(1, sum(r["n_imu"] for r in b)),
                                    fa_surge=sum(r["hit_surge"] for r in b) / max(1, sum(r["n_surge"] for r in b))))
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[-1].items()}, flush=True)
        write(res, "results/dev.csv")
    else:
        cfg = json.load(open("results/locked.json"))
        rows = run(cfg["gate"], cfg["L"], cfg["mu"], range(1000, 1040))
        write(rows, "results/test.csv")
        table(rows)
