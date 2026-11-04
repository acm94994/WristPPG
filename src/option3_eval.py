"""Scoring for Option 3. ECG / ground truth is used HERE only, never in gate.py, wqn.py or option3.py."""
import numpy as np
from scipy.signal import find_peaks


def ref_hr(rpeaks_s, t_center, win_s=8):
    out = []
    for tc in t_center:
        r = rpeaks_s[(rpeaks_s >= tc - win_s / 2) & (rpeaks_s < tc + win_s / 2)]
        out.append(60 / np.mean(np.diff(r)) if len(r) > 1 else np.nan)
    return np.array(out)


def hr_mae(hr, t, rpeaks_s, keep=None):
    h = ref_hr(rpeaks_s, t)
    err = np.abs(hr - h)
    return np.nanmean(err if keep is None else err[keep]) if (keep is None or keep.any()) else np.nan


def beats(x, fs, hr, t):
    """Pulse peaks, with minimum spacing set by the tracked HR (never by ECG)."""
    p, _ = find_peaks(x, distance=int(0.6 * 60 / max(np.median(hr), 30) * fs))
    return p / fs


def match_lag(est, ref, max_lag=0.6):
    """Median est - ref lag from nearest pairs; ECG-to-pulse delay when ref is ECG."""
    if len(est) == 0 or len(ref) == 0:
        return 0.0
    idx = np.clip(np.searchsorted(ref, est), 1, len(ref) - 1)
    near = np.where(est - ref[idx - 1] < ref[idx] - est, idx - 1, idx)
    d = est - ref[near]
    d = d[np.abs(d) < max_lag]
    return float(np.median(d)) if len(d) else 0.0


def beat_f1(est, ref, tol=0.1):
    if len(ref) == 0:
        return np.nan
    if len(est) == 0:
        return 0.0
    used = np.zeros(len(ref), bool)
    tp = 0
    for b in est:
        j = np.argmin(np.abs(ref - b))
        if abs(ref[j] - b) <= tol and not used[j]:
            used[j] = True
            tp += 1
    return 2 * tp / (len(est) + len(ref))


def windows_mask(n, fs, centres, before, after):
    m = np.zeros(n, bool)
    for c in centres:
        m[max(0, int((c - before) * fs)):min(n, int((c + after) * fs))] = True
    return m
