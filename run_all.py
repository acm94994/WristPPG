"""Run every tuned method on its held-out subjects (main results).

    python run_all.py                  # needs results/loso_params.csv  (python tune.py)
    python run_all.py --methods NLMS-all CW-NLMS

For each method and LOSO fold, the config chosen on the training subjects runs once on the
held-out subject's records:
  split=subject           : all records of the held-out subject
  split=subject+activity  : the held-out subject's record of that activity
plus the semi-synthetic benchmark (E13) with the split=subject configs.

Writes to results/:
  main_windows.csv.gz   one row per (method, split, record, window)
  main_records.csv      one row per (method, split, record): runtime, flops, convergence, means
  synth_records.csv     semi-synthetic: NMSE, correlation, convergence vs the known pulse
  curves.npz            learning curves, 1 s moving average sampled at 4 Hz,
                        keys 'real|<method>|<record>' (err^2) and 'synth|<method>|<record>' ((e - s)^2)
"""
import argparse
import os
import numpy as np
import pandas as pd
from src import methods
from src.jobs import run_jobs, DATA
from src.load import list_records

OUT = "results"
CURVE_FS = 4


def _decimate(curve, fs):
    k = max(1, int(fs))
    sm = np.convolve(curve, np.ones(k) / k, mode="valid")
    return sm[::max(1, int(fs // CURVE_FS))].astype(np.float32)


def build_jobs(sel, names, records):
    by_subject = {}
    for r in records:
        by_subject.setdefault(r.split("_")[0], []).append(r)
    jobs = []
    for row in sel[sel["method"].isin(names)].itertuples():
        params = methods.parse_key(row.params)
        recs = [r for r in by_subject[row.held_out]
                if row.activity == "all" or r.split("_", 1)[1] == row.activity]
        for r in recs:
            tags = {"split": row.split}
            jobs.append((row.method, r, params, {"tags": tags, "keep_curve": row.split == "subject"}))
            if row.split == "subject":
                jobs.append((row.method, r, params, {"tags": tags, "keep_curve": True,
                                                     "synth_seed": records.index(r)}))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=methods.available())
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    args = ap.parse_args()

    sel = pd.read_csv(os.path.join(OUT, "loso_params.csv"))
    missing = set(args.methods) - set(sel["method"])
    if missing:
        raise SystemExit(f"not tuned yet: {sorted(missing)}; run python tune.py --methods ...")
    records = list_records(DATA)
    jobs = build_jobs(sel, args.methods, records)
    print(f"{len(jobs)} runs")
    res = run_jobs(jobs, args.jobs)

    real = [(j, w, r) for j, (w, r) in zip(jobs, res) if "synth_seed" not in j[3]]
    synth = [(j, w, r) for j, (w, r) in zip(jobs, res) if "synth_seed" in j[3]]

    curves = {}
    for kind, rows in (("real", real), ("synth", synth)):
        for _, _, r in rows:
            if "curve" in r:
                key = f"{kind}|{r['method']}|{r['record'].removesuffix('_syn')}"
                curves[key] = _decimate(r.pop("curve_clean", r["curve"]), r["fs"])
                r.pop("curve")

    # merge with earlier runs of other methods, so methods can be added one at a time
    def save(df, name):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old[~old["method"].isin(args.methods)], df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")

    save(pd.concat([w for _, w, _ in real], ignore_index=True), "main_windows.csv.gz")
    save(pd.DataFrame([r for _, _, r in real]), "main_records.csv")
    save(pd.DataFrame([r for _, _, r in synth]), "synth_records.csv")

    cpath = os.path.join(OUT, "curves.npz")
    if os.path.exists(cpath):
        old = {k: v for k, v in np.load(cpath).items() if k != "fs"}
        curves = {**{k: v for k, v in old.items() if k.split("|")[1] not in args.methods}, **curves}
    np.savez_compressed(cpath, fs=CURVE_FS, **curves)
    print(f"wrote {cpath} ({len(curves)} curves)")

    w = pd.read_csv(os.path.join(OUT, "main_windows.csv.gz"))
    print("\nMAE (BPM), split=subject, pooled windows:")
    print(w[w["split"] == "subject"].pivot_table(index="activity", columns="method",
                                                 values="abs_err", aggfunc="mean").round(1).to_string())
