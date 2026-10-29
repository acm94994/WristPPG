"""Diagnose HR errors on one record. Saves results/diag_<record>.png and prints a summary.

  python diag.py data/s3_walk

Plot: spectrogram of raw PPG (top) and NLMS-1 output (bottom), with
  white  = ECG HR (ground truth)
  red    = tracked HR
  cyan   = strongest peak per window with NO tracking (free peak)
  yellow = dominant IMU frequency (forearm axis)
If red sits on yellow -> tracker locked onto motion. If cyan follows white -> tracker is the problem.
If white looks wrong (jumpy / implausible) -> ground truth problem.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.load import load_record
from src.preprocess import preprocess, forearm_axis
from src.filters import nlms
from src.hr import track_hr, window_starts
from demo import ecg_hr_quick, synthetic

NFFT, BAND = 2048, (0.5, 3.5)


def window_spectra(x, fs):
    starts, win = window_starts(len(x), fs)
    f = np.fft.rfftfreq(NFFT, 1 / fs)
    h = np.hanning(win)
    S = np.array([np.abs(np.fft.rfft((x[s:s+win] - x[s:s+win].mean()) * h, NFFT)) ** 2 for s in starts])
    return f, S, starts, win


def free_peak(f, S):
    m = (f >= BAND[0]) & (f <= BAND[1])
    return f[m][np.argmax(S[:, m], axis=1)] * 60


rec = load_record(sys.argv[1]) if len(sys.argv) > 1 else synthetic()
ax = forearm_axis(rec["acc"])
d, refs, fs = preprocess(rec, 64)
e, _ = nlms(d, refs[:, ax], L=16, mu=0.01)
e_all, _ = nlms(d, refs, L=16, mu=0.01)

starts, win = window_starts(len(d), fs)
t = (starts + win / 2) / fs
ecg = ecg_hr_quick(rec["rpeaks_s"], starts, win, fs)
f, S_raw, _, _ = window_spectra(d, fs)
_, S_e, _, _ = window_spectra(e, fs)
_, S_all, _, _ = window_spectra(e_all, fs)
_, S_imu, _, _ = window_spectra(refs[:, ax], fs)
imu = free_peak(f, S_imu)

rows = []
fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for a, S, x, label in [(axs[0], S_raw, d, "raw PPG"), (axs[1], S_e, e, "NLMS-1 output"), (axs[2], S_all, e_all, "NLMS-all output")]:
    tracked, _ = track_hr(x, fs, motion=refs)
    free = free_peak(f, S)
    m = f <= 4
    a.pcolormesh(t, f[m] * 60, 10 * np.log10(S[:, m].T + 1e-12), shading="auto", cmap="magma")
    a.plot(t, ecg, "w", lw=2, label="ECG HR")
    a.plot(t, tracked, "r", lw=1.5, label="tracked")
    a.plot(t, free, "c.", ms=3, label="free peak")
    a.plot(t, imu, "y.", ms=3, label="IMU dominant")
    a.set_ylabel("BPM"); a.set_title(label); a.set_ylim(30, 240)
    mae = lambda h: np.nanmean(np.abs(h - ecg))
    rows.append((label, mae(tracked), mae(free)))
axs[0].legend(loc="upper right", fontsize=8); axs[2].set_xlabel("time (s)")
os.makedirs("results", exist_ok=True)
out = f"results/diag_{rec['name']}.png"
plt.tight_layout(); plt.savefig(out, dpi=110)

print(f"{rec['name']}  forearm axis={ax}  windows={len(t)}  ECG-HR NaN windows={np.isnan(ecg).sum()}")
print(f"ECG HR   : min {np.nanmin(ecg):.0f}  median {np.nanmedian(ecg):.0f}  max {np.nanmax(ecg):.0f} BPM")
print(f"IMU freq : median {np.median(imu):.0f} BPM")
for label, m_tr, m_free in rows:
    print(f"{label:14s} MAE tracked {m_tr:5.1f} | MAE free peak {m_free:5.1f}")
print("first 5 windows  ECG:", np.round(ecg[:5]), " raw free peak:", np.round(free_peak(f, S_raw)[:5]))
print("saved", out)
