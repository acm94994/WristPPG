"""Report figures (run after run_all.py and robustness.py).

    python -m figures.plots                    # learning curves for s3_run
    python -m figures.plots --record s4_run    # a different example record

Outputs (results/, PNG + PDF):
  fig_learning_curves   smoothed e^2 vs time, one line per method, for one record
  fig_robustness        MAE under rotation / dropped channels, NLMS-all vs CW-NLMS
For the spectrogram example figures use tools/diag.py (e.g. python -m tools.diag data/s3_walk).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.methods import METHODS

p = argparse.ArgumentParser()
p.add_argument("--record", default="s3_run")
a = p.parse_args()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"results/{name}.{ext}", dpi=200)
    print("saved", f"results/{name}.png/.pdf")


# 1. learning curves: stored as 1 s moving averages at fs Hz, smoothed here to 2 s
z = np.load("results/curves.npz")
fs = int(z["fs"])
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for m in METHODS:                                           # fixed order -> fixed colours
    key = f"real|{m}|{a.record}"
    if m == "RAW" or key not in z.files:
        continue
    c = np.convolve(z[key], np.ones(2 * fs) / (2 * fs), mode="valid")
    ax.plot(np.arange(len(c)) / fs, c, lw=1, label=m)
ax.set_yscale("log")
ax.set_xlabel("time (s)")
ax.set_ylabel("e² (2 s moving avg)")
ax.set_title(f"Residual power over time, {a.record}")
ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))   # outside the axes
save(fig, "fig_learning_curves")

# 2. robustness
r = pd.read_csv("results/robustness.csv")
r["group"] = r.case.where(~r.case.str.startswith("rot_"), "rotated")
order = ["original", "rotated", "forearm_zeroed", "no_gyro"]
labels = ["original", "axes\nrotated", "forearm axis\nzeroed", "gyro\nremoved"]
t = r.groupby(["group", "method"]).mae.median().unstack().reindex(order)
fig, ax = plt.subplots(figsize=(3.5, 2.4))
w = 0.38
for k, m in enumerate(t.columns):
    ax.bar([i + (k - 0.5) * w for i in range(len(order))], t[m], w, label=m)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("median MAE (BPM)")
ax.set_ylim(0, t.values.max() * 1.25)
ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
save(fig, "fig_robustness")
