"""Semi-synthetic records with a known clean pulse (DECISIONS E13), for NMSE, correlation and
convergence time, which need ground truth that real PPG cannot give.

ppg = pulse + artifact + noise
  pulse    : 2-harmonic waveform whose phase advances 2*pi per real ECG RR interval (+ 0.25 s
             transit delay), so HR follows the subject's true HR
  artifact : sum of the record's real accel/gyro axes (0.4-4 Hz, z-scored), each through a random
             decaying 0.1 s FIR, scaled to `sir_db` relative to the pulse
  noise    : white, `snr_db` below the pulse
IMU channels are left untouched. The artifact is linear in the IMU, so this is a best case for
linear cancellers; say so in the report.

ECG note: R peaks are used here only to *build* ground truth. The returned record still goes
through evaluate.py, which strips rpeaks_s before any method sees it.
"""
import numpy as np
from src.preprocess import _resample, _bandpass


def semi_synthetic(rec, seed=0, sir_db=-6.0, snr_db=20.0, ptt_s=0.25):
    """Returns (new_rec, clean) with clean the pulse at rec['fs']."""
    rng = np.random.default_rng(seed)
    fs, n = rec["fs"], len(rec["ppg"])
    t = np.arange(n) / fs
    r = rec["rpeaks_s"] + ptt_s
    phase = 2 * np.pi * np.interp(t, r, np.arange(len(r)))   # flat (no pulse) outside the R-peak span
    pulse = np.cos(phase) + 0.4 * np.cos(2 * phase - 0.6)
    pulse /= pulse.std()

    imu = np.column_stack([rec["acc"], rec["gyro"]])
    imu = _bandpass(imu, fs)
    imu = (imu - imu.mean(0)) / (imu.std(0) + 1e-12)
    taps = int(0.1 * fs)
    art = np.zeros(n)
    for j in range(imu.shape[1]):
        h = rng.standard_normal(taps) * np.exp(-np.arange(taps) / (taps / 3))
        art += rng.uniform(0.2, 1.0) * np.convolve(imu[:, j], h)[:n]
    art *= 10 ** (-sir_db / 20) / art.std()

    noise = rng.standard_normal(n) * 10 ** (-snr_db / 20)
    return {**rec, "name": rec["name"] + "_syn", "ppg": pulse + art + noise}, pulse


def clean_like_d(ppg, clean, fs_in, fs_out):
    """The clean pulse in the same units as preprocess()'s d: same resample, band-pass and
    the z-score scale of the full PPG."""
    xp = _bandpass(_resample(ppg, fs_in, fs_out), fs_out)
    xc = _bandpass(_resample(clean, fs_in, fs_out), fs_out)
    return (xc - xc.mean()) / (xp.std() + 1e-12)
