"""Magnitude-squared coherence (MSC) weights for CW-NLMS (DECISIONS M4-M7).

Per 8 s window (2 s step): MSC between each IMU reference and the raw PPG, averaged over
0.5-4 Hz; values below tau set to 0. Uses PPG + IMU only, never ECG.
"""
import numpy as np
from scipy.signal import coherence
from src.hr import window_starts


def msc_weights(d, refs, fs, tau=0.3, win_s=8, step_s=2, band=(0.5, 4.0)):
    """d (N,), refs (N, C) -> W (n_windows, C), starts (n_windows,), win (samples)"""
    starts, win = window_starts(len(d), fs, win_s, step_s)
    nper = int(win_s * fs / 2)                       # 4 s segments, 75 % overlap -> 5 segments
    W = np.zeros((len(starts), refs.shape[1]))
    for i, s in enumerate(starts):
        for j in range(refs.shape[1]):
            f, C = coherence(refs[s:s + win, j], d[s:s + win], fs=fs,
                             nperseg=nper, noverlap=int(0.75 * nper))
            c = C[(f >= band[0]) & (f <= band[1])].mean()
            W[i, j] = c if c >= tau else 0.0
    return W, starts, win


def per_sample(W, starts, win, n_samples, step):
    """Expand window weights to samples: window i sets the weights for its last 2 s
    (samples [start+win-step, start+win)), so a sample's weight only uses data up to it.
    Samples before the first window's end use window 0."""
    out = np.repeat(W[:1], n_samples, axis=0)
    for i, s in enumerate(starts):
        out[s + win - step: s + win] = W[i]
    if len(starts):
        out[starts[-1] + win:] = W[-1]
    return out
