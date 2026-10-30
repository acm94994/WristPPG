"""Smoke test: raw PPG vs NLMS-1 vs NLMS-all on one record (or synthetic data if none given).

  python demo.py data/s3_run      # real record
  python demo.py                  # synthetic sanity test, no data needed
"""
import sys
import numpy as np
from src.preprocess import preprocess, forearm_axis, REF_NAMES
from src.filters import nlms
from src.hr import track_hr
from src.score import ecg_hr_at


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

    outputs = {"raw PPG": d,
               "NLMS-1": nlms(d, refs[:, ax], L=16, mu=0.01)[0],
               "NLMS-all": nlms(d, refs, L=16, mu=0.01)[0]}
    for label, e in outputs.items():
        hr, t = track_hr(e, fs, motion=refs)
        print(f"MAE {label:8s}: {np.nanmean(np.abs(hr - ecg_hr_at(t, rec['rpeaks_s']))):5.1f} BPM")
