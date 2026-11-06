"""Figure 1 for the report: spectral overlap of motion and pulse in a real running record.

    python -m figures.fig1_overlap                   # s3_run (default)
    python -m figures.fig1_overlap --record s4_run

Saves results/fig1_overlap.png and .pdf.
(a) spectrogram of the band-passed raw PPG, with the ECG heart rate drawn on top;
(b) PPG and IMU spectra averaged over the record, with the ECG heart-rate range shaded.
ECG is used only to draw the reference line/band (this is a figure, not a method).
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.load import load_record
from src.preprocess import preprocess
from src.hr import window_starts
from src.score import ecg_hr_at

p = argparse.ArgumentParser()
p.add_argument("--record", default="s3_run")
p.add_argument("--data", default=os.environ.get("DATA_DIR", "data"))
a = p.parse_args()

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PPG_C, IMU_C = "#2a78d6", "#eb6834"          # categorical slots 1, 2
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})

rec = load_record(os.path.join(a.data, a.record))
d, refs, fs = preprocess(rec, 64)
starts, win = window_starts(len(d), fs)
t = (starts + win / 2) / fs
nfft = 2048
f = np.fft.rfftfreq(nfft, 1 / fs)
bpm = f * 60
h = np.hanning(win)


def spec(x):
    seg = np.stack([x[s:s + win] - x[s:s + win].mean(0) for s in starts])
    return np.abs(np.fft.rfft(seg * (h if x.ndim == 1 else h[:, None]), nfft, axis=1)) ** 2


P = spec(d)                                               # (W, F)
P /= P.max(1, keepdims=True)
M = spec(refs)                                            # (W, F, 8)
M = (M / M.max(1, keepdims=True)).mean(2)
ecg = ecg_hr_at(t, rec["rpeaks_s"])
show = (bpm >= 30) & (bpm <= 240)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})

# (a) spectrogram
db = 10 * np.log10(P[:, show].T + 1e-6)
im = ax1.pcolormesh(t, bpm[show], db, shading="auto", cmap="Blues", vmin=-25, vmax=0, rasterized=True)
ax1.plot(t, ecg, color=INK, lw=1.2, ls="--", label="ECG heart rate")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("frequency (BPM)")
ax1.set_ylim(30, 240)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=7)
cb = fig.colorbar(im, ax=ax1, pad=0.02, fraction=0.05)
cb.set_label("PPG power (dB, per window)", fontsize=7)
cb.outline.set_visible(False)
ax1.set_title("(a) Band-passed PPG, " + a.record, fontsize=8, color=INK, loc="left")

# (b) averaged spectra
lo, hi = np.nanmin(ecg), np.nanmax(ecg)
ax2.axvspan(lo, hi, color=GRID, zorder=0)
ax2.text(hi + 3, 1.08, "ECG HR range", ha="left", va="bottom", fontsize=7, color=INK2)
pm, mm = P.mean(0), M.mean(0)
ax2.plot(bpm[show], pm[show] / pm[show].max(), color=PPG_C, lw=1.5, label="PPG")
ax2.plot(bpm[show], mm[show] / mm[show].max(), color=IMU_C, lw=1.5, label="IMU (8 references)")
ax2.set_xlim(30, 240)
ax2.set_ylim(0, 1.45)
ax2.set_xlabel("frequency (BPM)")
ax2.set_ylabel("mean spectrum (normalised)")
ax2.legend(loc="upper left", frameon=False, fontsize=7, ncol=2, handlelength=1.2, columnspacing=0.8)
ax2.grid(axis="y", color=GRID, lw=0.6)
ax2.set_title("(b) Averaged over the record", fontsize=8, color=INK, loc="left")

fig.tight_layout()
os.makedirs("results", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"results/fig1_overlap.{ext}", dpi=300)
print("saved results/fig1_overlap.png/.pdf")
print(f"ECG HR {lo:.0f}-{hi:.0f} BPM; strongest averaged PPG peak at {bpm[show][np.argmax(pm[show])]:.0f} BPM; "
      f"strongest IMU peak at {bpm[show][np.argmax(mm[show])]:.0f} BPM")
