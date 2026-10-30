"""Heart-rate tracker on the cleaned PPG (DECISIONS E1, E2). Uses PPG + IMU only, never ECG.

Per 8 s window (2 s step):
  1. PPG power spectrum (Hann, zero-padded FFT), normalised per window. Several outputs
     (NLMS-FV) -> spectra averaged.
  2. Harmonic sum: score(f) = P(f) + harm_weight * P(2f), so the pulse fundamental beats
     its own 2nd harmonic.
  3. Motion harmonics: if `motion` refs are given, find the dominant IMU frequency f_m per
     window and multiply bins within +/- motion_tol_bpm of {1/2, 1, 3/2, 2, 3} x f_m by
     motion_atten (down-weighted, not removed, so cadence ~ HR is still possible). 3/2 catches
     the 3rd arm-swing harmonic when the IMU peak is the step frequency. Follows the
     F/2F/3F/4F analysis of Lee et al. 2022.
  4. mode="viterbi" (default): best HR path over the whole record, with a Gaussian
     penalty on window-to-window jumps (sigma_bpm) and a hard limit (max_jump_bpm).
     Offline, like Temko 2017 (WFPV + Viterbi). mode="greedy": window-by-window peak
     search within +/- max_jump_bpm of the previous estimate (online, kept for comparison).
"""
import numpy as np


def window_starts(n_samples, fs, win_s=8, step_s=2):
    win, step = int(win_s * fs), int(step_s * fs)
    return np.arange(0, n_samples - win + 1, step), win


def _spectra(x, starts, win, nfft):
    hann = np.hanning(win)[:, None]
    out = []
    for s in starts:
        seg = x[s:s + win] - x[s:s + win].mean(0)
        out.append(np.abs(np.fft.rfft(seg * hann, nfft, axis=0)) ** 2)
    return np.array(out)                                     # (W, F, K)


def hr_scores(e, fs, motion=None, win_s=8, step_s=2, nfft=2048, band=(0.5, 3.5),
              harm_weight=0.5, motion_tol_bpm=10, motion_atten=0.1,
              motion_harmonics=(0.5, 1, 1.5, 2, 3)):
    """Per-window score over candidate HR bins. Returns S (W, B), bpm (B,), t_center (W,)."""
    e = np.asarray(e, float)
    if e.ndim == 1:
        e = e[:, None]
    starts, win = window_starts(len(e), fs, win_s, step_s)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    bpm_all = freqs * 60

    P = _spectra(e, starts, win, nfft).mean(2)
    P /= P.max(1, keepdims=True) + 1e-12
    H = P.copy()                                             # harmonic sum
    idx2 = 2 * np.arange(P.shape[1])
    ok = idx2 < P.shape[1]
    H[:, ok] += harm_weight * P[:, idx2[ok]]

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if motion is not None:
        M = _spectra(np.asarray(motion, float).reshape(len(e), -1), starts, win, nfft)
        M = (M / (M.max(1, keepdims=True) + 1e-12)).sum(2)
        imu_band = (freqs >= band[0]) & (freqs <= 4.0)
        for w in range(len(starts)):
            fm = bpm_all[imu_band][np.argmax(M[w, imu_band])]
            for k in motion_harmonics:
                H[w, np.abs(bpm_all - k * fm) <= motion_tol_bpm] *= motion_atten
    return H[:, in_band], bpm_all[in_band], (starts + win / 2) / fs


def _viterbi(S, bpm, sigma_bpm, max_jump_bpm):
    W, B = S.shape
    logS = np.log(S + 1e-6)
    db = bpm[1] - bpm[0]
    kmax = int(np.floor(max_jump_bpm / db))
    shifts = np.arange(-kmax, kmax + 1)
    trans = -0.5 * (shifts * db / sigma_bpm) ** 2
    score = logS[0].copy()
    back = np.zeros((W, B), dtype=int)
    for w in range(1, W):
        best = np.full(B, -np.inf)
        arg = np.zeros(B, dtype=int)
        for s, tc in zip(shifts, trans):                     # state j comes from j - s
            src = np.arange(B) - s
            valid = (src >= 0) & (src < B)
            cand = np.full(B, -np.inf)
            cand[valid] = score[src[valid]] + tc
            better = cand > best
            best[better], arg[better] = cand[better], src[better]
        score = best + logS[w]
        back[w] = arg
    path = np.zeros(W, dtype=int)
    path[-1] = np.argmax(score)
    for w in range(W - 1, 0, -1):
        path[w - 1] = back[w, path[w]]
    return bpm[path]


def _greedy(S, bpm, max_jump_bpm):
    hr, prev = [], None
    for w in range(S.shape[0]):
        cand = np.ones(len(bpm), bool) if prev is None else np.abs(bpm - prev) <= max_jump_bpm
        prev = bpm[cand][np.argmax(S[w, cand])]
        hr.append(prev)
    return np.array(hr)


def track_hr(e, fs, motion=None, mode="viterbi", sigma_bpm=5, max_jump_bpm=15, **kw):
    """e      : (N,) cleaned PPG, or (N, K) outputs whose spectra are averaged
    motion : optional (N, C) IMU references used to down-weight motion harmonics
    returns: hr_bpm (n_windows,), t_center_s (n_windows,)"""
    S, bpm, t = hr_scores(e, fs, motion=motion, **kw)
    hr = _viterbi(S, bpm, sigma_bpm, max_jump_bpm) if mode == "viterbi" else _greedy(S, bpm, max_jump_bpm)
    return hr, t
