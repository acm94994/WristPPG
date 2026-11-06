"""Change vs ANC as uncancellable motion grows: benefit in collision zones, cost on near misses."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LEAKS = (0.0, 0.5, 1.0, 2.0)
rng = np.random.default_rng(0)


def series(method, scens, metric):
    mean, lo, hi = [], [], []
    for lk in LEAKS:
        R = list(csv.DictReader(open(f"results/collision_leak{lk}.csv")))
        base = {(r["seed"], r["act"], r["scen"]): float(r[metric]) for r in R if r["method"] == "ANC"}
        per = {}
        for r in R:
            if r["method"] == method and r["scen"] in scens:
                per.setdefault(r["seed"], []).append(float(r[metric]) - base[(r["seed"], r["act"], r["scen"])])
        x = np.array([np.nanmean(v) for v in per.values()])
        b = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
        mean.append(x.mean())
        lo.append(np.percentile(b, 2.5))
        hi.append(np.percentile(b, 97.5))
    return map(np.array, (mean, lo, hi))


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, gridspec_kw=dict(wspace=.1))
panels = (("Inside collision zones", ("dwell40",), "mae_zone"), ("Whole record, near misses", ("near+8", "near-8"), "mae"))
x = np.arange(len(LEAKS))
for ax, (title, scens, metric) in zip(axes, panels):
    for m, c, name in (("hold+aware", BLUE, "Hold + tracker change"), ("hold", ORANGE, "Hold only")):
        y, lo, hi = series(m, scens, metric)
        ax.fill_between(x, lo, hi, color=c, alpha=.14, lw=0)
        ax.plot(x, y, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=name)
        ax.text(x[-1] + .08, y[-1], name, color=INK, fontsize=10, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x, ["0", "0.5", "1", "2"])
    ax.set_xlim(-.2, 4.6)
    ax.set_xlabel("Motion the canceller cannot remove (× main artifact line)")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Change in HR error vs NLMS alone (BPM)")
axes[1].legend(frameon=False, loc="lower right", fontsize=10)
fig.savefig("results/collision_simulation.png", dpi=200, bbox_inches="tight", facecolor=SURF)
