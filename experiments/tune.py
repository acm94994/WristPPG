"""Leave-one-subject-out tuning (DECISIONS E4, E9, E10).

    python -m experiments.tune                          # all available methods
    python -m experiments.tune --methods RLS-all --force

1. Grid: every config of every method on every record, scored once and cached in
   results/grid/<method>.csv.gz (a filter's output does not depend on which subject is held out).
2. Selection, for each held-out subject:
     split=subject          : best config on the other 7 subjects, all activities
     split=subject+activity : best config on the other subjects' records of the same activity
   "best" = lowest mean over training subjects of each subject's MAE (E9).
   -> results/loso_params.csv
"""
import argparse
import os
import pandas as pd
from src.configs import GRIDS, METHODS
from src.jobs import run_jobs, DATA
from src.load import list_records

OUT = "results"
GRID_DIR = os.path.join(OUT, "grid")


def run_grid(name, records, n_workers, force=False):
    path = os.path.join(GRID_DIR, f"{name}.csv.gz")
    if os.path.exists(path) and not force:
        print(f"{name}: cached grid {path}")
        return pd.read_csv(path)
    jobs = [(name, r, p, {}) for p in GRIDS[name] for r in records]
    print(f"{name}: {len(jobs)} runs ({len(GRIDS[name])} configs x {len(records)} records)")
    res = run_jobs(jobs, n_workers)
    win = pd.concat([w for w, _ in res], ignore_index=True)
    os.makedirs(GRID_DIR, exist_ok=True)
    win.to_csv(path, index=False)
    pd.DataFrame([r for _, r in res]).to_csv(path.replace(".csv.gz", "_records.csv"), index=False)
    return win


def _objective(win):
    """E9: per config, mean over subjects of each subject's window MAE."""
    return win.groupby(["params", "subject"])["abs_err"].mean().groupby("params").mean()


def select(name, win):
    rows = []
    for s in sorted(win["subject"].unique()):
        train = win[win["subject"] != s]
        obj = _objective(train)
        rows.append({"method": name, "split": "subject", "held_out": s, "activity": "all",
                     "params": obj.idxmin(), "train_mae": obj.min(),
                     "n_train_subjects": train["subject"].nunique()})
        for a in sorted(win.loc[win["subject"] == s, "activity"].unique()):
            tr = train[train["activity"] == a]
            obj = _objective(tr)
            rows.append({"method": name, "split": "subject+activity", "held_out": s, "activity": a,
                         "params": obj.idxmin(), "train_mae": obj.min(),
                         "n_train_subjects": tr["subject"].nunique()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--force", action="store_true", help="recompute cached grids")
    args = ap.parse_args()

    records = list_records(DATA)
    sel_path = os.path.join(OUT, "loso_params.csv")
    old = pd.read_csv(sel_path) if os.path.exists(sel_path) else pd.DataFrame(columns=["method"])
    new = [select(m, run_grid(m, records, args.jobs, args.force)) for m in args.methods]
    sel = pd.concat([old[~old["method"].isin(args.methods)], *new], ignore_index=True)
    sel.to_csv(sel_path, index=False)

    print(f"\nwrote {sel_path}")
    chosen = sel[sel["split"] == "subject"].groupby(["method", "params"]).size().rename("folds")
    print(chosen.to_string())
