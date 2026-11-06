"""Collision experiment on synthetic records.
   python -m experiments.run_collision_sim dev    grid on dev seeds, all scenarios pooled
   python -m experiments.run_collision_sim test   locked settings on untouched test seeds
   python -m experiments.run_collision_sim leak   sweep of uncancellable motion, on further held-out seeds
   python -m experiments.run_collision_sim drift  sensitivity to coupling drift, on the test seeds
"""
import csv, json, sys
from multiprocessing import Pool
import numpy as np
from experiments.run_wqn_sim import fs, truth
from sim.collision_sim import make_collision
from src.collision import (CollisionConfig, cancelled_lines, hold_mask, mask_to_windows, nlms_hold,
                           run_collision, track_aware)
from src.filters import nlms
from src.hr import track_hr
from src.extension_score import beat_f1, beats, hr_mae, match_lag, ref_hr
from src.preprocess import preprocess

SCEN = {"crossing": dict(dwell=0), "dwell20": dict(dwell=20), "dwell40": dict(dwell=40),
        "lock20": dict(dwell=20, lock=True), "lock40": dict(dwell=40, lock=True),
        "near+8": dict(dwell=40, offset=8), "near-8": dict(dwell=40, offset=-8),
        "near+12": dict(dwell=40, offset=12), "near-12": dict(dwell=40, offset=-12),
        "control": dict(none=True)}
METHODS = ("ANC", "aware-only", "hold", "hold-last+aware", "mu+aware", "hold+aware", "hold+aware-oracle")


def evaluate(cfgkw, L, mu, passes, seed, act, scen, methods, drift=0.25, leak=0.0):
    cfg = CollisionConfig(**cfgkw)
    rec = make_collision(seed, act, drift=drift, leak=leak, **SCEN[scen])
    d, refs, _ = preprocess(rec)
    s = truth(rec, len(d))
    N = len(d)
    e0, _ = nlms(d, refs, L=L, mu=mu)
    h0, t = track_hr(e0, fs, motion=refs)
    true = ref_hr(rec["rpeaks_s"], t)
    tt = np.arange(N) / fs
    true_n = np.interp(tt, t, np.nan_to_num(true, nan=np.nanmedian(true)))
    step = np.interp(t, np.arange(len(rec["_step_bpm"])) / rec["fs"], rec["_step_bpm"])
    zone_w = np.abs(true - step) < 15
    step_n = np.interp(tt, t, step)
    zone_n = np.abs(true_n - step_n) < 15
    bref = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] / fs + 1 / fs
    bref = bref[np.r_[True, np.diff(bref) > 0.25]]
    rows = []
    lines = cancelled_lines(d - e0, fs, cfg)
    for m in methods:
        hold = np.zeros(N, bool)
        if m == "ANC":
            e, hr = e0, h0
        elif m == "aware-only":
            hold = hold_mask(h0, t, lines, N, fs, cfg)
            e, hr = e0, track_aware(e0, fs, refs, mask_to_windows(hold, t, fs))[0]
        elif m == "hold":
            hold = hold_mask(h0, t, lines, N, fs, cfg)
            e = nlms_hold(d, refs, hold, fs, L, mu, "avg", cfg)
            hr = track_hr(e, fs, motion=refs)[0]
        else:
            mode = {"hold-last+aware": "last", "mu+aware": "mu"}.get(m, "avg")
            th = np.nan_to_num(true, nan=np.nanmedian(true)) if m.endswith("oracle") else None
            o = run_collision(d, refs, fs, L, mu, cfg, mode, passes=passes, true_hr=th)
            e, hr, hold = o["e"], o["hr"], o["hold"]
        est = beats(e, fs, hr, t)
        est = est - match_lag(est, bref)
        pick = lambda b: b[zone_n[np.clip((b * fs).astype(int), 0, N - 1)]]
        rmse = lambda sel: float(np.sqrt(np.mean((e[sel] - s[sel]) ** 2))) if sel.any() else np.nan
        rows.append(dict(seed=seed, act=act, scen=scen, method=m, mae=hr_mae(hr, t, rec["rpeaks_s"]),
                         mae_zone=hr_mae(hr, t, rec["rpeaks_s"], zone_w) if zone_w.any() else np.nan,
                         rmse=rmse(np.ones(N, bool)), rmse_zone=rmse(zone_n),
                         f1_zone=beat_f1(pick(est), pick(bref)) if zone_n.any() else np.nan,
                         hold_frac=float(hold.mean())))
    return rows


def _job(a):
    return evaluate(*a)


def run(cfgkw, L, mu, passes, seeds, acts, scens, methods, drift=0.25, leak=0.0):
    jobs = [(cfgkw, L, mu, passes, s, a, c, methods, drift, leak) for s in seeds for a in acts for c in scens]
    with Pool(2) as p:
        return [r for rs in p.imap_unordered(_job, jobs, chunksize=4) for r in rs]


def write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1] == "dev":
        res = []
        for gate in (12, 15, 20):
            for passes in (1, 2):
                rows = run(dict(gate_bpm=gate), 16, 0.01, passes, range(0, 10), ("walk", "run"), list(SCEN), ("ANC", "hold+aware"))
                a = [r for r in rows if r["method"] == "ANC"]
                b = [r for r in rows if r["method"] == "hold+aware"]
                near = lambda rr: np.mean([r["mae"] for r in rr if r["scen"].startswith("near")])
                coll = lambda rr: np.nanmean([r["mae_zone"] for r in rr if r["scen"][:5] in ("cross", "dwell", "lock2", "lock4")])
                res.append(dict(gate=gate, passes=passes, mae_anc=np.mean([r["mae"] for r in a]), mae=np.mean([r["mae"] for r in b]),
                                coll_anc=coll(a), coll=coll(b), near_anc=near(a), near=near(b),
                                ctrl_anc=np.mean([r["mae"] for r in a if r["scen"] == "control"]),
                                ctrl=np.mean([r["mae"] for r in b if r["scen"] == "control"])))
                print({k: (round(float(v), 2)) for k, v in res[-1].items()}, flush=True)
        write(res, "results/collision_dev.csv")
    elif sys.argv[1] == "test":
        c = json.load(open("results/collision_locked.json"))
        rows = run(c["cfg"], c["L"], c["mu"], c["passes"], range(2000, 2040), ("walk", "run", "bike"), list(SCEN), METHODS)
        write(rows, "results/collision_test.csv")
    elif sys.argv[1] == "leak":
        c = json.load(open("results/collision_locked.json"))
        ms = ("ANC", "aware-only", "hold", "mu+aware", "hold+aware")
        for lk in (0.0, 0.5, 1.0, 2.0):
            rows = run(c["cfg"], c["L"], c["mu"], c["passes"], range(3000, 3020), ("walk", "run"), ("dwell40", "near+8", "near-8"), ms, leak=lk)
            write(rows, f"results/collision_leak{lk}.csv")
            for sc, name in ((("dwell40",), "collision"), (("near+8", "near-8"), "near miss")):
                line = f"leak {lk:.1f} {name:10s}"
                for m in ms:
                    q = [r for r in rows if r["method"] == m and r["scen"] in sc]
                    line += f" {m} {np.nanmean([r['mae'] for r in q]):.2f}/{np.nanmean([r['mae_zone'] for r in q]):.2f}"
                print(line, flush=True)
    else:
        c = json.load(open("results/collision_locked.json"))
        scens = ("dwell40", "lock40", "near+8", "near-8")
        for dr in (0.0, 0.25, 0.5):
            rows = run(c["cfg"], c["L"], c["mu"], c["passes"], range(2000, 2020), ("walk", "run"), scens, ("ANC", "hold+aware"), dr)
            for m in ("ANC", "hold+aware"):
                q = [r for r in rows if r["method"] == m]
                print(f"drift {dr:.2f} {m:11s} mae_zone {np.nanmean([r['mae_zone'] for r in q]):6.2f} rmse_zone {np.nanmean([r['rmse_zone'] for r in q]):.3f}", flush=True)
