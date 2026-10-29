"""Robustness tests and channel ablation (DECISIONS E8, E16), with the split=subject LOSO configs
(no retuning: a deployed filter would not be retuned after the sensor moved or a channel died).

    python robustness.py                   # all tuned methods, 10 rotations
    python robustness.py --n-rot 3         # cut order item 1

Conditions (test column):
  baseline                 unperturbed
  rotation    rot<k>       same random 3-D rotation of accel and gyro, k = 0..n_rot-1 (seeded)
  missing     zero_acc_<x|y|z>, no_gyro
  ablation    drop_<ch>    leave one of the 8 references out (multi-reference methods only)

Writes results/robustness_windows.csv.gz, results/robustness_records.csv
"""
import argparse
import os
import pandas as pd
from src import methods
from src.jobs import run_jobs, DATA
from src.load import list_records
from src.preprocess import REF_NAMES

OUT = "results"


def conditions(name, n_rot, ablation):
    yield "baseline", "baseline", None, {}
    for k in range(n_rot):
        yield "rotation", f"rot{k}", ("rotate", k), {}
    for i, ax in enumerate("xyz"):
        yield "missing", f"zero_acc_{ax}", ("zero_acc_axis", i), {}
    yield "missing", "no_gyro", ("no_gyro", None), {}
    if ablation and name in methods.MULTI_REF:
        for ch in REF_NAMES:
            yield "ablation", f"drop_{ch}", None, {"drop": (ch,)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=[m for m in methods.available() if m != "raw"])
    ap.add_argument("--n-rot", type=int, default=10)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    args = ap.parse_args()

    sel = pd.read_csv(os.path.join(OUT, "loso_params.csv"))
    sel = sel[(sel["split"] == "subject") & sel["method"].isin(args.methods)]
    chosen = {(r.method, r.held_out): methods.parse_key(r.params) for r in sel.itertuples()}

    jobs = []
    for name in args.methods:
        for test, cond, transform, extra in conditions(name, args.n_rot, not args.no_ablation):
            for r in list_records(DATA):
                opts = {"tags": {"test": test, "condition": cond}}
                if transform:
                    opts["transform"] = transform
                jobs.append((name, r, {**chosen[(name, r.split("_")[0])], **extra}, opts))
    print(f"{len(jobs)} runs")
    res = run_jobs(jobs, args.jobs)

    for df, fname in ((pd.concat([w for w, _ in res], ignore_index=True), "robustness_windows.csv.gz"),
                      (pd.DataFrame([r for _, r in res]), "robustness_records.csv")):
        path = os.path.join(OUT, fname)
        if os.path.exists(path):
            old = pd.read_csv(path)
            df = pd.concat([old[~old["method"].isin(args.methods)], df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")

    w = pd.read_csv(os.path.join(OUT, "robustness_windows.csv.gz"))
    print("\nMAE (BPM) by condition:")
    print(w.pivot_table(index=["test", "condition"], columns="method", values="abs_err",
                        aggfunc="mean", sort=False).round(1).to_string())
