"""All report figures, from the files run_all.py and robustness.py wrote. No recomputation.

    python plots.py         -> results/figures/F*.png

F1 MAE per activity, both splits          F4 robustness (rotation range, missing axes)
F2 HR tracks for one subject, 4 activities F5 channel ablation
F3 learning curves (semi-synthetic, real)  F6 MAE vs motion level (severe motion)

Colours are fixed per method (never by rank), so a method keeps its colour in every figure and
when A's methods are added. Exact numbers for every figure are in the matching tables (tables.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import methods

OUT = "results"
FIG = os.path.join(OUT, "figures")

# reference palette (dataviz skill), categorical slots 1-6 in order; raw = muted neutral
COLOR = {"raw": "#898781", "LMS-HK": "#008300", "NLMS-FV": "#e87ba4", "NLMS-1": "#eda100",
         "NLMS-all": "#1baf7a", "RLS-all": "#eb6834", "CW-NLMS": "#2a78d6"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ORDER = list(methods.METHODS)
ACT_LABEL = {"high_resistance_bike": "bike, high", "low_resistance_bike": "bike, low",
             "run": "run", "walk": "walk"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def present(df):
    return [m for m in ORDER if m in set(df["method"])]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def grouped_bars(ax, table, ms, labels, gap=0.02):
    """table: index = groups, columns = methods. Thin bars, small surface gap between neighbours."""
    n = len(ms)
    width = 0.8 / n
    x = np.arange(len(table))
    for i, m in enumerate(ms):
        ax.bar(x - 0.4 + width * (i + 0.5), table[m], width - gap, color=COLOR[m], label=m,
               edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x, labels)


def f1(w):
    ms = present(w)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, split in zip(axes, ["subject", "subject+activity"]):
        t = w[w["split"] == split].pivot_table(index="activity", columns="method", values="abs_err", aggfunc="mean")
        grouped_bars(ax, t, ms, [ACT_LABEL[a] for a in t.index])
        ax.set_title(f"Tuned on other subjects ({'all activities' if split == 'subject' else 'same activity'})")
    axes[0].set_ylabel("HR MAE (BPM)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(ms), loc="lower right",
               bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    save(fig, "F1_mae_by_activity.png")


def f2(w, subject="s3"):
    w = w[(w["split"] == "subject") & (w["subject"] == subject)]
    acts = [a for a in ACT_LABEL if a in set(w["activity"])]
    ms = present(w)
    fig, axes = plt.subplots(len(acts), 1, figsize=(9, 2.1 * len(acts)), sharex=False)
    for ax, a in zip(np.atleast_1d(axes), acts):
        x = w[w["activity"] == a]
        for m in ms:
            y = x[x["method"] == m]
            ax.plot(y["t"], y["hr_est"], color=COLOR[m], lw=1.5, label=m)
        ref = x[x["method"] == ms[0]]
        ax.plot(ref["t"], ref["hr_true"], color=INK, lw=2, label="ECG (truth)")
        ax.set_title(f"{subject}, {ACT_LABEL[a]}")
        ax.set_ylabel("HR (BPM)")
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    np.atleast_1d(axes)[0].legend(ncol=len(ms) + 1, loc="lower right", bbox_to_anchor=(1, 1.08))
    fig.tight_layout()
    save(fig, "F2_hr_tracks.png")


def f3(curves, horizon_s=120, smooth_s=5, skip_s=2):
    """Median over records of the stored curves, plus a smooth_s moving average for display.
    The first skip_s seconds (band-pass edge transient) are not drawn."""
    fs = float(curves["fs"])
    n = int(horizon_s * fs)
    k = int(smooth_s * fs)
    t = (np.arange(n - k + 1) + k - 1) / fs
    shown = t >= skip_s
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    titles = {"synth": "Semi-synthetic: (e - s)², known pulse s", "real": "Real: e², includes the pulse"}
    for ax, kind in zip(axes, ["synth", "real"]):
        for m in ORDER:
            cs = [curves[k][:n] for k in curves.files if k.startswith(f"{kind}|{m}|") and len(curves[k]) >= n]
            if not cs or m == "raw":
                continue
            med = np.convolve(np.median(np.array(cs), axis=0), np.ones(k) / k, mode="valid")
            ax.plot(t[shown], 10 * np.log10(med[shown] + 1e-12), color=COLOR[m], label=m)
        ax.set_title(titles[kind])
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("squared error, median over records (dB)")
    axes[0].legend()
    save(fig, "F3_learning_curves.png")


def f4(rw):
    rw = rw[rw["test"] != "ablation"]
    ms = present(rw)
    mae = rw.groupby(["method", "condition"])["abs_err"].mean().unstack("method")
    rot = mae.loc[mae.index.str.startswith("rot")]
    groups = ["baseline", "rotation", "zero_acc_x", "zero_acc_y", "zero_acc_z", "no_gyro"]
    labels = ["baseline", f"rotated\n(range of {len(rot)})", "acc x\nmissing", "acc y\nmissing",
              "acc z\nmissing", "no gyro"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    off = np.linspace(-0.25, 0.25, len(ms))
    for i, m in enumerate(ms):
        for j, g in enumerate(groups):
            if g == "rotation":
                ax.vlines(j + off[i], rot[m].min(), rot[m].max(), color=COLOR[m], lw=2)
                y = rot[m].mean()
            else:
                y = mae.loc[g, m]
            ax.plot(j + off[i], y, "o", ms=8, color=COLOR[m], mec=SURFACE, mew=2,
                    label=m if j == 0 else None)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("HR MAE (BPM)")
    ax.set_title("Robustness with LOSO configs, no retuning (rotation: mean and min-max)")
    fig.legend(*ax.get_legend_handles_labels(), ncol=len(ms), loc="lower right", bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    save(fig, "F4_robustness.png")


def f5(rw):
    rw = rw[rw["test"].isin(["baseline", "ablation"])]
    mae = rw.groupby(["method", "condition"])["abs_err"].mean().unstack("method")
    d = (mae - mae.loc["baseline"]).drop("baseline").dropna(axis=1, how="all")
    d.index = d.index.str.removeprefix("drop_")
    ms = [m for m in ORDER if m in d.columns]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    grouped_bars(ax, d, ms, list(d.index))
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("change in HR MAE (BPM)\n< 0: better without it")
    ax.set_title("Channel ablation: one reference left out")
    ax.legend(ncol=len(ms))
    save(fig, "F5_ablation.png")


def f6(w, n_bins=10):
    w = w[w["split"] == "subject"].copy()
    edges = np.quantile(w.drop_duplicates(["record", "t"])["motion"], np.linspace(0, 1, n_bins + 1))
    w["bin"] = np.clip(np.searchsorted(edges, w["motion"], side="right") - 1, 0, n_bins - 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for m in present(w):
        y = w[w["method"] == m].groupby("bin")["abs_err"].mean()
        ax.plot(y.index + 1, y.values, color=COLOR[m], marker="o", ms=5, label=m)
    ax.set_xticks(range(1, n_bins + 1))
    ax.set_xlabel("motion decile (std of |a| per window; 10 = most severe)")
    ax.set_ylabel("HR MAE (BPM)")
    ax.set_title("Error vs motion level")
    ax.legend(ncol=2)
    save(fig, "F6_mae_vs_motion.png")


if __name__ == "__main__":
    w = pd.read_csv(os.path.join(OUT, "main_windows.csv.gz"))
    f1(w)
    f2(w)
    f3(np.load(os.path.join(OUT, "curves.npz")))
    f6(w)
    rob = os.path.join(OUT, "robustness_windows.csv.gz")
    if os.path.exists(rob):
        rw = pd.read_csv(rob)
        f4(rw)
        f5(rw)
