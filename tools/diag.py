"""Diagnose HR errors on one record. Saves results/diag_<record>.png and prints a summary.

  python -m tools.diag data/s3_walk

Plot: spectrogram of raw PPG, NLMS-1 output and NLMS-all output, with
  white  = ECG HR (ground truth)
  red    = tracked HR
  cyan   = strongest peak per window with NO tracking (free peak)
  yellow = dominant IMU frequency (forearm axis)
If red sits on yellow -> tracker locked onto motion. If cyan follows white -> tracker is the problem.
If white looks wrong (jumpy / implausible) -> ground truth problem.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.load import load_record
from src.preprocess import preprocess, forearm_axis
from src.filters import nlms
from src.hr import track_hr
from src.score import ecg_hr_at, window_spectra, peak_freq
from tools.demo import synthetic

rec = load_record(sys.argv[1]) if len(sys.argv) > 1 else synthetic()
ax = forearm_axis(rec["acc"])
d, refs, fs = preprocess(rec, 64)
signals = [("raw PPG", d),
           ("NLMS-1 output", nlms(d, refs[:, ax], L=16, mu=0.01)[0]),
           ("NLMS-all output", nlms(d, refs, L=16, mu=0.01)[0])]

_, t = track_hr(d, fs)
ecg = ecg_hr_at(t, rec["rpeaks_s"])
f, S_imu = window_spectra(refs[:, ax], fs, t)
imu = peak_freq(f, S_imu) * 60
mae = lambda h: np.nanmean(np.abs(h - ecg))

rows = []
fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for a, (label, x) in zip(axs, signals):
    tracked, _ = track_hr(x, fs, motion=refs)
    _, S = window_spectra(x, fs, t)
    free = peak_freq(f, S) * 60
    m = f <= 4
    a.pcolormesh(t, f[m] * 60, 10 * np.log10(S[:, m].T + 1e-12), shading="auto", cmap="magma")
    a.plot(t, ecg, "w", lw=2, label="ECG HR")
    a.plot(t, tracked, "r", lw=1.5, label="tracked")
    a.plot(t, free, "c.", ms=3, label="free peak")
    a.plot(t, imu, "y.", ms=3, label="IMU dominant")
    a.set_ylabel("BPM"); a.set_title(label); a.set_ylim(30, 240)
    rows.append((label, mae(tracked), mae(free), free))
axs[0].legend(loc="upper right", fontsize=8); axs[2].set_xlabel("time (s)")
os.makedirs("results", exist_ok=True)
out = f"results/diag_{rec['name']}.png"
plt.tight_layout(); plt.savefig(out, dpi=110)

print(f"{rec['name']}  forearm axis={ax}  windows={len(t)}  ECG-HR NaN windows={np.isnan(ecg).sum()}")
print(f"ECG HR   : min {np.nanmin(ecg):.0f}  median {np.nanmedian(ecg):.0f}  max {np.nanmax(ecg):.0f} BPM")
print(f"IMU freq : median {np.median(imu):.0f} BPM")
for label, m_tr, m_free, _ in rows:
    print(f"{label:14s} MAE tracked {m_tr:5.1f} | MAE free peak {m_free:5.1f}")
print("first 5 windows  ECG:", np.round(ecg[:5]), " raw free peak:", np.round(rows[0][3][:5]))
print("saved", out)
