"""Scoring against ECG (the ONLY module that uses ECG). Minimal version: B extends it.

Windows are matched by TIME, so methods running at 20/35/64 Hz are scored identically.
"""
import numpy as np


def ecg_hr_at(t_center, rpeaks_s, win_s=8):
    """ECG HR (BPM) = 60 / mean RR of R peaks inside each [t-4 s, t+4 s] window."""
    out = []
    for tc in t_center:
        r = rpeaks_s[(rpeaks_s >= tc - win_s / 2) & (rpeaks_s < tc + win_s / 2)]
        out.append(60 / np.mean(np.diff(r)) if len(r) > 1 else np.nan)
    return np.array(out)


def score(out, rec):
    """out = run_method(...) result. Returns per-window table dict and MAE."""
    ecg = ecg_hr_at(out["t"], rec["rpeaks_s"])
    err = np.abs(out["hr"] - ecg)
    return {"t": out["t"], "hr": out["hr"], "ecg_hr": ecg, "abs_err": err}, float(np.nanmean(err))
