"""Robustness tests (DECISIONS E8): NLMS-all and CW-NLMS with their LOSO configs, no retuning.

Cases per record:
  original          unchanged
  rot_0 .. rot_9    accel and gyro rotated by the same random rotation (fixed seed)
  forearm_zeroed    the forearm accelerometer axis set to zero (strongest single axis)
  no_gyro           all gyroscope channels set to zero (also zeroes |w|)

    python -m experiments.robustness                # needs results/loso_params.csv (python -m experiments.tune)
    python -m experiments.robustness --n-rot 3      # faster

Appends one row per (method, case, record) to results/robustness.csv as runs finish, so an
interrupted run picks up where it stopped. The summary table is T5 in tables.py.
"""
import argparse
import csv
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.transform import Rotation
from src.configs import parse_key
from src.jobs import DATA, record
from src.load import list_records
from src.methods import run_method
from src.preprocess import forearm_axis
from src.score import score

METHODS = ["NLMS-all", "CW-NLMS"]
FIELDS = ["method", "case", "record", "subject", "activity", "mae"]
OUT = "results/robustness.csv"


def perturb(rec, case, rots):
    """Applied to the raw record, so |a|, |w| and the forearm axis are recomputed afterwards."""
    r = dict(rec)
    if case.startswith("rot_"):
        R = rots[int(case[4:])].as_matrix()
        r["acc"], r["gyro"] = rec["acc"] @ R.T, rec["gyro"] @ R.T
    elif case == "forearm_zeroed":
        acc = rec["acc"].copy()
        acc[:, forearm_axis(rec["acc"])] = 0.0
        r["acc"] = acc
    elif case == "no_gyro":
        r["gyro"] = np.zeros_like(rec["gyro"])
    return r


def job(j):
    method, case, rec_name, params, n_rot = j
    rots = Rotation.random(n_rot, random_state=0)
    rec = record(rec_name)
    _, mae = score(run_method(method, perturb(rec, case, rots), **params), rec)
    return {"method": method, "case": case, "record": rec_name, "subject": rec["subject"],
            "activity": rec["activity"], "mae": mae}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--n-rot", type=int, default=10)
    a = p.parse_args()

    sel = pd.read_csv("results/loso_params.csv")
    sel = sel[sel["split"] == "subject"]
    params = {(r.held_out, r.method): parse_key(r.params) for r in sel.itertuples()}
    cases = ["original"] + [f"rot_{k}" for k in range(a.n_rot)] + ["forearm_zeroed", "no_gyro"]

    done = set()
    if os.path.exists(OUT):
        done = set(map(tuple, pd.read_csv(OUT)[["method", "case", "record"]].astype(str).values))
    jobs = [(m, c, r, params[(r.split("_")[0], m)], a.n_rot)
            for r in list_records(DATA) for m in METHODS for c in cases if (m, c, r) not in done]

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f, ProcessPoolExecutor(a.workers) as ex:
        out = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            out.writeheader()
        for k, fut in enumerate(as_completed([ex.submit(job, j) for j in jobs]), 1):
            out.writerow(fut.result())
            f.flush()
            print(f"\r{k}/{len(jobs)}", end="", flush=True)
    print()

    df = pd.read_csv(OUT)
    df = df[df.case.isin(cases)]
    df["group"] = df.case.where(~df.case.str.startswith("rot_"), "rotated")
    t = df.groupby(["method", "group"]).mae.agg(["mean", "median"]).round(1).unstack(0)
    print("\nRobustness: MAE (BPM) over records\n",
          t.reindex(["original", "rotated", "forearm_zeroed", "no_gyro"]))
