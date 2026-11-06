"""All report tables, from the files run_all.py and robustness.py wrote. No recomputation.

    python -m analysis.tables        -> results/tables/T*.csv and results/tables/tables.md

T1 MAE per activity (both splits)
T2 MAE per held-out subject              T7 failure cases (E7)
T3 all real-data metrics per method      T8 severe motion (E15)
T4 semi-synthetic benchmark (E13)        T9 LOSO-chosen configs per fold
T5 robustness (rotation, forearm axis zeroed, no gyro)
"""
import os
import numpy as np
import pandas as pd
from src import methods
from src.score import failure_label

OUT = "results"
TAB = os.path.join(OUT, "tables")
ORDER = list(methods.METHODS)                      # column order = registry order
SEVERE_Q = 0.9                                     # E15: top 10 % of windows by motion
MSC_TAU = 0.3                                      # E7: "low MSC" if no CW-NLMS weight reaches this


def _cols(df):
    return df[[m for m in ORDER if m in df.columns]]


def _md(df, floatfmt="{:.1f}"):
    df = df.reset_index()
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) \
        else ("" if isinstance(v, float) else str(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def mae_by(w, index):
    return _cols(w.pivot_table(index=index, columns="method", values="abs_err", aggfunc="mean"))


def t1(w):
    parts = {s: mae_by(w[w["split"] == s], "activity") for s in ("subject", "subject+activity")}
    for s, df in parts.items():
        df.loc["all (pooled)"] = _cols(w[w["split"] == s].groupby("method")["abs_err"].mean().to_frame().T).iloc[0]
    return pd.concat(parts, names=["split"])


def t3(w, rec):
    w, rec = w[w["split"] == "subject"], rec[rec["split"] == "subject"]
    g = w.groupby("method")
    df = pd.DataFrame({
        "HR MAE (BPM)": g["abs_err"].mean(),
        "HR MAE, mean of subjects": w.groupby(["method", "subject"])["abs_err"].mean().groupby("method").mean(),
        "spectral-peak err (BPM)": g["peak_err"].mean(),
        "windows within 10 BPM (%)": g["abs_err"].apply(lambda e: 100 * (e <= 10).mean()),
        "SNR gain (dB)": g["snr_gain"].mean(),
        "pulse ratio (dB)": g["pulse_ratio"].mean(),
        "convergence (s)": rec.groupby("method")["conv_time_s"].median(),
        "multiplies / sample": rec.groupby("method")["flops"].median(),
        "runtime / signal-s (ms)": 1000 * rec.groupby("method")["rt_factor"].median(),
    })
    return df.loc[[m for m in ORDER if m in df.index]]


def t4(syn):
    g = syn.groupby("method")
    df = pd.DataFrame({"NMSE before (dB)": g["nmse_in_db"].mean(), "NMSE after (dB)": g["nmse_db"].mean(),
                       "correlation": g["corr"].mean(), "HR MAE (BPM)": g["mae"].mean(),
                       "SNR gain (dB)": g["snr_gain"].mean(), "pulse ratio (dB)": g["pulse_ratio"].mean(),
                       "convergence (s)": g["conv_time_clean_s"].median()})
    return df.loc[[m for m in ORDER if m in df.index]]


def t5(rob):
    """robustness.csv (robustness.py): MAE per record and case -> mean and median over records."""
    rob = rob.assign(group=rob["case"].where(~rob["case"].str.startswith("rot_"), "rotated"))
    t = rob.groupby(["group", "method"])["mae"].agg(["mean", "median"]).unstack("method")
    t.columns = [f"{m}, {stat}" for stat, m in t.columns]
    return t.reindex(["original", "rotated", "forearm_zeroed", "no_gyro"]).rename_axis("condition")


def t7(w):
    w = w[w["split"] == "subject"].assign(label=lambda x: failure_label(x, msc_tau=MSC_TAU))
    pct = w.groupby("method")["label"].value_counts(normalize=True).unstack().fillna(0) * 100
    cols = [c for c in ["ok", "cadence≈HR", "low MSC", "other", "no ECG"] if c in pct]
    return pct.loc[[m for m in ORDER if m in pct.index], cols].rename_axis("% of windows")


def t8(w):
    w = w[w["split"] == "subject"]
    thr = w.drop_duplicates(["record", "t"])["motion"].quantile(SEVERE_Q)
    df = pd.DataFrame({"MAE, severe motion": w[w["motion"] >= thr].groupby("method")["abs_err"].mean(),
                       "MAE, other windows": w[w["motion"] < thr].groupby("method")["abs_err"].mean()}).T
    df.index.name = f"motion >= {thr:.2f} m/s^2 (top {100 * (1 - SEVERE_Q):.0f} %)"
    return _cols(df)


def t9(sel):
    s = sel[sel["split"] == "subject"]
    return s.pivot(index="held_out", columns="method", values="params")[[m for m in ORDER if m in set(s["method"])]]


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    w = pd.read_csv(os.path.join(OUT, "main_windows.csv.gz"))
    rec = pd.read_csv(os.path.join(OUT, "main_records.csv"))
    syn = pd.read_csv(os.path.join(OUT, "synth_records.csv"))
    sel = pd.read_csv(os.path.join(OUT, "loso_params.csv"))
    rob = os.path.join(OUT, "robustness.csv")
    rob = pd.read_csv(rob) if os.path.exists(rob) else None

    tables = [
        ("T1", "HR MAE (BPM) per activity, held-out subjects, windows pooled", t1(w), "{:.1f}"),
        ("T2", "HR MAE (BPM) per held-out subject (split=subject)",
         mae_by(w[w["split"] == "subject"], "subject"), "{:.1f}"),
        ("T3", "Real-data metrics (split=subject)", t3(w, rec), "{:.2f}"),
        ("T4", "Semi-synthetic benchmark (known pulse, E13)", t4(syn), "{:.2f}"),
        ("T7", "Failure cases (E7)", t7(w), "{:.1f}"),
        ("T8", "Severe motion (E15)", t8(w), "{:.1f}"),
        ("T9", "LOSO-chosen config per held-out subject (split=subject)", t9(sel), "{}"),
    ]
    if rob is not None:
        tables += [("T5", "Robustness: HR MAE (BPM) per record, LOSO configs, no retuning", t5(rob), "{:.1f}")]
    tables.sort(key=lambda t: t[0])

    md = []
    for tid, title, df, fmt in tables:
        df.to_csv(os.path.join(TAB, f"{tid}.csv"))
        md += [f"## {tid}. {title}", "", _md(df, fmt), ""]
    with open(os.path.join(TAB, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
