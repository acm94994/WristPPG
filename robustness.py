"""Step 4: robustness tests (DECISIONS E8) for NLMS-all and CW-NLMS with LOSO-chosen settings.

Cases per record:
  original          unchanged
  rot_0 .. rot_9    accel and gyro rotated by the same random rotation (fixed seed)
  forearm_zeroed    the forearm accelerometer axis set to zero (strongest single axis)
  no_gyro           all gyroscope channels set to zero (also zeroes |w|)

    python robustness.py
    python robustness.py --n-rot 3      # faster (cut order in DECISIONS.md)
Outputs: results/robustness.csv (per record), results/table_robustness.csv (summary)
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from src.runner import records, get_record, run_jobs, done_keys
from src.methods import run_method
from src.preprocess import forearm_axis
from src.score import score

METHODS = ["NLMS-all", "CW-NLMS"]
FIELDS = ["method", "case", "record", "subject", "activity", "mae"]
OUT = "results/robustness.csv"


def perturb(rec, case, rots):
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
    rec = get_record(rec_name)
    _, mae = score(run_method(method, perturb(rec, case, rots), **params), rec)
    return [{"method": method, "case": case, "record": rec_name, "subject": rec["subject"],
             "activity": rec["activity"], "mae": mae}]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--n-rot", type=int, default=10)
    a = p.parse_args()
    lo = pd.read_csv("results/loso.csv")
    P = {(r.subject, r.method): json.loads(r.params) for r in lo.itertuples()}
    cases = ["original"] + [f"rot_{k}" for k in range(a.n_rot)] + ["forearm_zeroed", "no_gyro"]
    done = done_keys(OUT, ["method", "case", "record"])
    jobs = []
    for r in records():
        s = get_record(r)["subject"]
        jobs += [(m, c, r, P[(s, m)], a.n_rot) for m in METHODS for c in cases if (m, c, r) not in done]
    run_jobs(job, jobs, OUT, FIELDS, a.workers)

    df = pd.read_csv(OUT)
    df = df[df.case.isin(cases)]
    df["group"] = df.case.where(~df.case.str.startswith("rot_"), "rotated")
    t = df.groupby(["method", "group"]).mae.agg(["mean", "median"]).round(1).unstack(0)
    t = t.reindex(["original", "rotated", "forearm_zeroed", "no_gyro"])
    t.to_csv("results/table_robustness.csv")
    print("\nRobustness: MAE (BPM) over records\n", t)
