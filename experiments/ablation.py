"""Step-size control and ablations for CW-NLMS.
    python -m experiments.ablation          # needs results/grid/ and results/loso_params.csv (experiments.tune)

1. Step-size control. CW-NLMS adapts with mu[n] = mu0 * mean_k c_k, so its effective step size is
   smaller than mu0. Two NLMS-all controls without coherence:
     NLMS-all, matched mu : per record, constant mu = mean over samples of CW-NLMS's mu[n] (its held-out
                            setting, L included). Uses PPG + IMU only.
     NLMS-all, dense grid : mu in {0.01, 0.05, 0.1, 0.2, 0.3, 0.5}, L in {8, 16, 32}, LOSO-tuned.
2. Component ablation of CW-NLMS, each LOSO-tuned on the CW-NLMS grid (mu0 x L x tau):
     weights only : coherence-weighted references, fixed step size mu0     (vss=False)
     step only    : uniform references, step size mu0 * mean_k c_k        (weight_refs=False)
     both         : CW-NLMS as reported
     both, tau > 0: CW-NLMS with tau restricted to {0.2, 0.3, 0.5}
3. Channel ablation: NLMS-all and CW-NLMS with their held-out settings, each of the 8 references
   removed from the canceller in turn (the HR tracker still sees all IMU channels), no retuning.

Held-out results use split=subject. Every grid setting runs once on every record, so a held-out
result is the grid row of the setting chosen on the other 7 subjects, as in experiments.tune.
Writes results/ablation/: grid_<variant>.csv.gz (cached), heldout.csv.gz, channels.csv.gz, summary.md
"""
import argparse
import os
import numpy as np
import pandas as pd
from experiments.tune import select
from src.coherence import msc_weights, per_sample
from src.configs import GRIDS, ORDERS, parse_key
from src.jobs import DATA, record, run_jobs
from src.load import list_records
from src.preprocess import REF_NAMES, preprocess

OUT = os.path.join("results", "ablation")
DENSE_MU = (0.2, 0.3)                               # added to MU = (0.01, 0.05, 0.1, 0.5)


def grid(tag, method, configs, records, workers):
    """Every config on every record, cached. Returns the windows DataFrame with params as run."""
    path = os.path.join(OUT, f"grid_{tag}.csv.gz")
    if os.path.exists(path):
        return pd.read_csv(path)
    jobs = [(method, r, c, {}) for c in configs for r in records]
    print(f"{tag}: {len(jobs)} runs", flush=True)
    win = pd.concat([w for w, _ in run_jobs(jobs, workers)], ignore_index=True)
    win.to_csv(path, index=False)
    return win


def heldout(variant, win):
    """LOSO (split=subject) on a grid -> the held-out windows of each subject's chosen setting."""
    sel = select(variant, win)
    sel = sel[sel["split"] == "subject"]
    rows = [win[(win["subject"] == s.held_out) & (win["params"] == s.params)] for s in sel.itertuples()]
    return pd.concat(rows, ignore_index=True).assign(variant=variant), sel


def matched_mu(sel_cw, records):
    """Per record: CW-NLMS's held-out (L, mu0, tau) and the time average of its mu[n]."""
    params = {r.held_out: parse_key(r.params) for r in sel_cw.itertuples()}
    out = {}
    for name in records:
        rec = record(name)
        p = params[rec["subject"]]
        d, refs, fs = preprocess(rec)
        Wwin, starts, win = msc_weights(d, refs, fs, tau=p["tau"])
        W = per_sample(Wwin, starts, win, len(d), step=2 * fs)
        out[name] = dict(L=int(p["L"]), mu=round(float(p["mu"] * W.mean(1).mean()), 6))
    return out


def md(df):
    """DataFrame -> Markdown table (no extra dependency)."""
    df = df.reset_index()
    rows = [[str(c) for c in df.columns], ["---"] * df.shape[1]] + [[str(v) for v in r] for r in df.values]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def summarise(w):
    """Pooled MAE, mean over subjects, share within 10 BPM, pulse ratio (window means)."""
    g = w.groupby("variant", sort=False)
    return pd.DataFrame({
        "pooled MAE": g["abs_err"].mean(),
        "mean of subjects": w.groupby(["variant", "subject"], sort=False)["abs_err"].mean().groupby("variant", sort=False).mean(),
        "<=10 BPM (%)": g["abs_err"].apply(lambda x: 100 * (x.dropna() <= 10).mean()),
        "pulse ratio (dB)": g["pulse_ratio"].mean(),
    })


def per_subject(w, ref="CW-NLMS"):
    s = w.groupby(["variant", "subject"])["abs_err"].mean().unstack("subject")
    return s, (s.lt(s.loc[ref]).sum(1)).rename(f"subjects below {ref}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    records = list_records(DATA)

    # existing grids from experiments.tune
    g_nlms = pd.read_csv("results/grid/NLMS-all.csv.gz")
    g_cw = pd.read_csv("results/grid/CW-NLMS.csv.gz")

    # new grids
    cw_cfg = GRIDS["CW-NLMS"]
    g_wonly = grid("weights_only", "CW-NLMS", [dict(c, vss=False) for c in cw_cfg], records, a.workers)
    g_sonly = grid("step_only", "CW-NLMS", [dict(c, weight_refs=False) for c in cw_cfg], records, a.workers)
    g_dense = grid("nlms_dense", "NLMS-all", [dict(mu=m, L=L) for m in DENSE_MU for L in ORDERS], records, a.workers)
    g_dense = pd.concat([g_nlms, g_dense], ignore_index=True)

    parts, sels = [], {}
    for variant, win in [("NLMS-all", g_nlms), ("NLMS-all, dense mu grid", g_dense),
                         ("CW-NLMS", g_cw), ("CW: weights only", g_wonly), ("CW: step only", g_sonly),
                         ("CW-NLMS, tau > 0", g_cw[~g_cw["params"].str.endswith("tau=0.0")])]:
        h, sels[variant] = heldout(variant, win)
        parts.append(h)

    # matched-mu control: one run per record
    path = os.path.join(OUT, "grid_nlms_matched_mu.csv.gz")
    mm = matched_mu(sels["CW-NLMS"], records)
    if os.path.exists(path):
        g_mm = pd.read_csv(path)
    else:
        print(f"matched mu: {len(records)} runs", flush=True)
        g_mm = pd.concat([w for w, _ in run_jobs([("NLMS-all", r, mm[r], {}) for r in records], a.workers)],
                         ignore_index=True)
        g_mm.to_csv(path, index=False)
    parts.insert(1, g_mm.assign(variant="NLMS-all, CW-matched mu"))
    held = pd.concat(parts, ignore_index=True)
    held.to_csv(os.path.join(OUT, "heldout.csv.gz"), index=False)

    # channel ablation: held-out settings, one reference removed
    path = os.path.join(OUT, "channels.csv.gz")
    if os.path.exists(path):
        ch = pd.read_csv(path)
    else:
        jobs = []
        for m in ("NLMS-all", "CW-NLMS"):
            p = {s.held_out: parse_key(s.params) for s in sels[m].itertuples()}
            jobs += [(m, r, dict(p[r.split("_")[0]], drop=k), {"tags": {"dropped": REF_NAMES[k]}})
                     for k in range(len(REF_NAMES)) for r in records]
        print(f"channels: {len(jobs)} runs", flush=True)
        ch = pd.concat([w for w, _ in run_jobs(jobs, a.workers)], ignore_index=True)
        ch.to_csv(path, index=False)
    base = held[held["variant"].isin(["NLMS-all", "CW-NLMS"])].assign(dropped="none")
    chan = pd.concat([base.assign(m=base["variant"]), ch.assign(m=ch["method"])], ignore_index=True)
    chan_tab = chan.groupby(["dropped", "m"], sort=False)["abs_err"].mean().unstack("m")
    chan_subj = (chan.groupby(["dropped", "m", "subject"])["abs_err"].mean()
                 .groupby(["dropped", "m"]).mean().unstack("m"))

    # summary
    tab = summarise(held)
    subj, wins = per_subject(held)
    mu_tab = pd.DataFrame(mm).T.rename_axis("record")
    chosen = {v: s.groupby("params").size().to_dict() for v, s in sels.items()}
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("## Held-out results (split=subject)\n\n" + md(tab.round(2)) + "\n\n")
        f.write("## Per-subject MAE (BPM)\n\n" + md(subj.round(1)) + "\n\n")
        f.write("## Subjects with lower MAE than CW-NLMS\n\n" + md(wins) + "\n\n")
        f.write("## LOSO choices (setting: folds)\n\n" + "\n".join(f"- {v}: {c}" for v, c in chosen.items()) + "\n\n")
        f.write("## CW-matched constant mu per record\n\n" + md(mu_tab) + "\n\n")
        f.write("## Channel ablation, pooled MAE (BPM)\n\n" + md(chan_tab.round(2)) + "\n\n")
        f.write("## Channel ablation, mean of subjects (BPM)\n\n" + md(chan_subj.round(2)) + "\n")
    print(open(os.path.join(OUT, "summary.md")).read())
