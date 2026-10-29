"""Heart-rate tracker on the cleaned PPG (DECISIONS E1, E2). Uses PPG only, never ECG."""
import numpy as np


def window_starts(n_samples, fs, win_s=8, step_s=2):
    win, step = int(win_s * fs), int(step_s * fs)
    return np.arange(0, n_samples - win + 1, step), win


def track_hr(e, fs, win_s=8, step_s=2, nfft=2048, band=(0.5, 3.5), search_bpm=15):
    """e: (N,) cleaned PPG, or (N, K) several outputs whose power spectra are averaged (NLMS-FV).
    Returns hr_bpm (n_windows,), t_center_s (n_windows,)."""
    e = np.asarray(e, float)
    if e.ndim == 1:
        e = e[:, None]
    starts, win = window_starts(len(e), fs, win_s, step_s)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    hann = np.hanning(win)[:, None]

    hr, prev = [], None
    for s in starts:
        seg = e[s:s + win] - e[s:s + win].mean(0)
        P = (np.abs(np.fft.rfft(seg * hann, nfft, axis=0)) ** 2).mean(1)
        cand = in_band.copy()
        if prev is not None:                                   # +/- search_bpm around previous
            near = cand & (np.abs(freqs * 60 - prev) <= search_bpm)
            if P[near].size and P[near].max() > 0:
                cand = near
        f_peak = freqs[cand][np.argmax(P[cand])]
        prev = f_peak * 60
        hr.append(prev)
    return np.array(hr), (starts + win / 2) / fs
