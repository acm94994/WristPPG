"""Report figures (run after run_all.py and robustness.py).

    python plots.py                    # learning curves for s3_run
    python plots.py --record s4_run    # a different example record

Outputs (results/, PNG + PDF):
  fig_learning_curves   smoothed e^2 vs time, one line per method, for one record
  fig_robustness        MAE under rotation / dropped channels, NLMS-all vs CW-NLMS
For the spectrogram example figures use diag.py (e.g. python diag.py data/s3_walk).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
fig, ax = plt.subplots(figsize=(3.5, 2.4))                  # one IEEE column
for key in z.files:
    if not (key.startswith("real|") and key.endswith("|" + a.record)):
        continue
    m = key.split("|")[1]
    if m == "RAW":
        continue
    c = np.convolve(z[key], np.ones(2 * fs) / (2 * fs), mode="valid")
    ax.plot(np.arange(len(c)) / fs, c, lw=1, label=m)
ax.set_yscale("log")
ax.set_xlabel("time (s)")
ax.set_ylabel("e² (2 s moving avg)")
ax.set_title(f"Learning curves, {a.record}")
ax.legend(fontsize=6, ncol=2, frameon=False)
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
