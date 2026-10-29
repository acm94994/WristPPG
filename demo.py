"""First end-to-end check: NLMS-1 on one record (or synthetic data if no record given).

  python demo.py data/s3_run      # real record
  python demo.py                  # synthetic sanity test, no data needed

The ECG comparison here is a quick sanity print only. Real scoring lives in score.py (B).
"""
import sys
import numpy as np
from src.preprocess import preprocess, forearm_axis, REF_NAMES
from src.filters import nlms
from src.hr import track_hr, window_starts


def ecg_hr_quick(rpeaks_s, starts, win, fs):
    out = []
    for s in starts:
        t0, t1 = s / fs, (s + win) / fs
        r = rpeaks_s[(rpeaks_s >= t0) & (rpeaks_s < t1)]
        out.append(60 / np.mean(np.diff(r)) if len(r) > 1 else np.nan)
    return np.array(out)


def synthetic(fs=256, T=120, hr_hz=1.5, cad_hz=2.6):
    rng = np.random.default_rng(0)
    t = np.arange(int(T * fs)) / fs
    acc = np.column_stack([np.sin(2 * np.pi * cad_hz * t + p) for p in (0, 1, 2)]) * 9.8 + [9.8, 0, 0]
    gyro = np.column_stack([np.sin(2 * np.pi * cad_hz * t + p) for p in (0.5, 1.5, 2.5)]) * 100
    m = 3 * np.roll(acc[:, 1], int(0.05 * fs)) / 9.8            # delayed, scaled artifact
    ppg = np.sin(2 * np.pi * hr_hz * t) + m + 0.1 * rng.standard_normal(len(t))
    rpeaks = np.arange(0, T, 1 / hr_hz)
    return {"name": "synthetic", "subject": "s0", "activity": "run", "fs": fs,
            "ppg": ppg, "acc": acc, "gyro": gyro, "rpeaks_s": rpeaks}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from src.load import load_record
        rec = load_record(sys.argv[1])
    else:
        rec = synthetic()

    ax = forearm_axis(rec["acc"])
    d, refs, fs = preprocess(rec, fs_out=64)
    print(f"{rec['name']}: {len(d)/fs:.0f} s at {fs} Hz, forearm axis = {REF_NAMES[ax]}")

    e, err2 = nlms(d, refs[:, ax], L=16, mu=0.1)

    hr_raw, _ = track_hr(d, fs)
    hr_nlms, _ = track_hr(e, fs)
    starts, win = window_starts(len(d), fs)
    hr_ecg = ecg_hr_quick(rec["rpeaks_s"], starts, win, fs)

    mae = lambda h: np.nanmean(np.abs(h - hr_ecg))
    print(f"MAE raw PPG : {mae(hr_raw):5.1f} BPM")
    print(f"MAE NLMS-1  : {mae(hr_nlms):5.1f} BPM")
