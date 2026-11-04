"""Verification figure: one median-difficulty event, and error change by method."""
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_sim import fs, truth
from sim.semisynth import make_record
from src.filters import nlms
from src.gate import GateConfig
from src.hr import track_hr
from src.option3 import clean, jerk_z
from src.preprocess import preprocess

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8983"
L = json.load(open("results/locked.json"))
cfg = GateConfig(**L["gate"])

cases = []
for seed in range(1000, 1040):
    rec = make_record(seed, "run")
    d, refs, _ = preprocess(rec)
    s = truth(rec, len(d))
    N = len(d)
    e0, _ = nlms(d, refs, L=L["L"], mu=L["mu"])
    zj = jerk_z(rec, fs, cfg)
    k = min(N, len(zj))
    zj = np.pad(zj[:k], (0, N - k), mode="edge")
    hr, t = track_hr(e0, fs, motion=refs)
    e1, runs, _ = clean(e0, refs, fs, zj, cfg, "both", hr=hr, t_hr=t)
    for x in rec["_events"]:
        if x["kind"] == "coupled":
            a, b = int((x["t"] - .75) * fs), int((x["t"] + 2.25) * fs)
            r0 = np.sqrt(np.mean((e0[a:b] - s[a:b]) ** 2))
            r1 = np.sqrt(np.mean((e1[a:b] - s[a:b]) ** 2))
            cases.append((r1 / r0, seed, x["t"], s, e0, e1, runs))
cases.sort(key=lambda c: c[0])
ratio, seed, t0, s, e0, e1, runs = cases[len(cases) // 2]
print(f"median event: seed {seed} t={t0:.1f}s error ratio {ratio:.2f}, of {len(cases)} running events")

rows = list(csv.DictReader(open("results/test.csv")))
base = {(r["seed"], r["act"]): float(r["rmse_art"]) for r in rows if r["method"] == "ANC"}
rng = np.random.default_rng(0)
names = {"ANC+soft": "Soft threshold, no gate", "WQN-imu": "WQN, IMU gate only", "WQN-both": "WQN, IMU + residual gate",
         "freeze+WQN": "Freeze + WQN", "freeze": "Freeze weights only", "WQN-res": "WQN, residual gate only",
         "WQN-oracle": "WQN, perfect detection"}
stats = []
for m, label in names.items():
    per = {}
    for r in rows:
        if r["method"] == m:
            per.setdefault(r["seed"], []).append((float(r["rmse_art"]), base[(r["seed"], r["act"])]))
    arr = np.array([np.mean(v, 0) for v in per.values()])
    f = lambda i: 100 * (arr[i, 0].mean(-1) / arr[i, 1].mean(-1) - 1)
    boot = f(rng.integers(0, len(arr), (5000, len(arr))))
    stats.append((f(np.arange(len(arr))), *np.percentile(boot, [2.5, 97.5]), label, m))
stats.sort(key=lambda x: -x[0])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.52))
tt = np.arange(len(s)) / fs
sel = (tt > t0 - 5) & (tt < t0 + 8)
for a, b in runs:
    if a / fs < t0 + 8 and b / fs > t0 - 5:
        ax.axvspan(a / fs - t0, b / fs - t0, color="#eeedE8", lw=0, zorder=0)
ax.plot(tt[sel] - t0, e0[sel], color=ORANGE, lw=2, label="ANC output", zorder=2)
ax.plot(tt[sel] - t0, e1[sel], color=BLUE, lw=2, label="After gated WQN", zorder=3)
ax.plot(tt[sel] - t0, s[sel], color=INK, lw=1.4, ls=(0, (3, 2)), label="True pulse", zorder=4)
ax.set_xlabel("Time from jerk (s)")
ax.set_ylabel("PPG (z-score units)")
ax.set_title("A median running event, with the corrected interval shaded", loc="left", fontsize=12, color=INK, pad=12)
ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=10)
ymax = np.abs(np.r_[e0[sel], s[sel]]).max()
ax.set_ylim(-1.15 * ymax, 1.45 * ymax)
ax.grid(axis="y", color=GRID, lw=.8)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)

for i, (v, lo, hi, label, m) in enumerate(stats):
    c = BLUE if m == "WQN-both" else GRAY
    bx.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
    bx.scatter([v], [i], s=70, color=c, zorder=3, edgecolor=SURF, linewidth=2)
    bx.text(v, i + .3, f"{v:+.0f}%", ha="center", va="bottom", fontsize=10, color=INK)
bx.axvline(0, color=MUTED, lw=1)
bx.set_yticks(range(len(stats)), [s_[3] for s_ in stats], fontsize=10.5)
bx.tick_params(axis="y", length=0)
bx.set_ylim(-.6, len(stats) - .2)
bx.set_xlim(-65, 20)
bx.set_xlabel("Change in error inside transients vs ANC alone")
bx.set_title("Error change, 120 test records (95% CI)", loc="left", fontsize=12, color=INK, pad=12)
bx.grid(axis="x", color=GRID, lw=.8)
bx.set_axisbelow(True)
for sp in ("top", "right", "left"):
    bx.spines[sp].set_visible(False)
fig.savefig("results/option3_simulation.png", dpi=200, bbox_inches="tight", facecolor=SURF)
for x in stats:
    print(f"{x[3]:30s} {x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]")
