"""Scoring (DECISIONS E3, E5-E7, E11-E14). The ONLY module that reads ECG (rec['rpeaks_s']).

Windows are matched by centre time (track_hr's t_center), not sample index, so methods that run
at 20, 35 or 64 Hz are scored on the same windows.
"""
import numpy as np


def true_hr(rpeaks_s, t_center, win_s=8):
    """E3: 60 / mean RR of the R peaks inside each window. NaN if fewer than 2 peaks."""
    out = np.full(len(t_center), np.nan)
    for i, t in enumerate(t_center):
        r = rpeaks_s[(rpeaks_s >= t - win_s / 2) & (rpeaks_s < t + win_s / 2)]
        if len(r) > 1:
            out[i] = 60 / np.mean(np.diff(r))
    return out


def window_spectra(x, fs, t_center, win_s=8, nfft=2048):
    """Hann-windowed power spectrum of each window, same windows as track_hr.
    x: (N,) or (N, K); K spectra are averaged (NLMS-FV). Returns freqs (F,), P (n_windows, F)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    win = int(win_s * fs)
    hann = np.hanning(win)[:, None]
    P = []
    for t in t_center:
        s = int(round(t * fs - win / 2))
        seg = x[s:s + win] - x[s:s + win].mean(0)
        P.append((np.abs(np.fft.rfft(seg * hann, nfft, axis=0)) ** 2).mean(1))
    return np.fft.rfftfreq(nfft, 1 / fs), np.array(P)


def peak_freq(freqs, P, band=(0.5, 3.5)):
    """Largest spectral peak in band per window, no tracking (Hz)."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    return freqs[m][np.argmax(P[:, m], axis=1)]


def _band_power(freqs, P, centres_hz, half_width=0.1):
    """Power within +/- half_width Hz of a per-window centre frequency (or list of them)."""
    out = np.zeros(len(P))
    for c in np.atleast_2d(centres_hz):
        out += (P * (np.abs(freqs[None, :] - c[:, None]) <= half_width)).sum(1)
    return out


def snr_db(freqs, P, f_hr, band=(0.5, 4.0), half_width=0.1):
    """E6: power within +/- 0.1 Hz of HR and 2*HR over the rest of the 0.5-4 Hz band (dB)."""
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    sig = _band_power(freqs, P * in_band, [f_hr, 2 * f_hr], half_width)
    noise = (P * in_band).sum(1) - sig
    return 10 * np.log10(sig / noise)


def pulse_ratio_db(freqs, P_out, P_in, f_hr, half_width=0.1):
    """E6: power within +/- 0.1 Hz of true HR, after / before filtering (dB).
    Near 0 dB = pulse preserved; strongly negative = canceller removed the pulse."""
    return 10 * np.log10(_band_power(freqs, P_out, f_hr, half_width) /
                         _band_power(freqs, P_in, f_hr, half_width))


def window_metrics(rec, out):
    """Per-window scores for one method output (methods.run dict). Returns a dict of (n_windows,) arrays."""
    t, fs = out["t"], out["fs"]
    hr_true = true_hr(rec["rpeaks_s"], t)
    f_hr = hr_true / 60
    ok = np.isfinite(f_hr)
    f_safe = np.where(ok, f_hr, 1.0)

    freqs, P_e = window_spectra(out["e"], fs, t)
    _, P_d = window_spectra(out["d"], fs, t)
    snr_in, snr_out = snr_db(freqs, P_d, f_safe), snr_db(freqs, P_e, f_safe)
    nan = lambda v: np.where(ok, v, np.nan)
    return {"t": t, "hr_est": out["hr"], "hr_true": hr_true,
            "err": out["hr"] - hr_true,
            "abs_err": np.abs(out["hr"] - hr_true),
            "peak_err": nan(np.abs(peak_freq(freqs, P_e) * 60 - hr_true)),
            "snr_in": nan(snr_in), "snr_out": nan(snr_out), "snr_gain": nan(snr_out - snr_in),
            "pulse_ratio": nan(pulse_ratio_db(freqs, P_e, P_d, f_safe))}


def failure_label(win, err_bpm=10, tol_hz=0.1, msc_tau=None):
    """E7 per window: 'ok', 'cadence≈HR' (IMU dominant frequency within tol_hz of true HR or
    true HR / 2), 'low MSC' (needs an 'msc' column from CW-NLMS, else skipped), 'other'.
    'no ECG' where ground truth is missing. win: DataFrame from evaluate()."""
    f_hr = win["hr_true"] / 60
    cad = (np.abs(win["imu_freq"] - f_hr) <= tol_hz) | (np.abs(win["imu_freq"] - f_hr / 2) <= tol_hz)
    lab = np.where(win["abs_err"] <= err_bpm, "ok", np.where(cad, "cadence≈HR", "other"))
    if msc_tau is not None and "msc" in win:
        lab = np.where((lab == "other") & (win["msc"] < msc_tau), "low MSC", lab)
    return np.where(win["hr_true"].isna(), "no ECG", lab)


def convergence_time(err2, fs, smooth_s=2, tol_db=1.0):
    """E12: first time (s) the smoothed learning curve comes within tol_db of its steady-state level
    (median over the second half of the record). err2: (N,) squared error."""
    k = max(1, int(smooth_s * fs))
    curve = 10 * np.log10(np.convolve(err2, np.ones(k) / k, mode="valid") + 1e-12)
    steady = np.median(curve[len(curve) // 2:])
    return np.argmax(curve <= steady + tol_db) / fs


def nmse_db(e, clean):
    """E13 (semi-synthetic only): ||e - s||^2 / ||s||^2 in dB."""
    return 10 * np.log10(np.sum((e - clean) ** 2) / np.sum(clean ** 2))


def corr(e, clean):
    """E13 (semi-synthetic only): Pearson correlation of cleaned PPG with the known pulse."""
    return np.corrcoef(e, clean)[0, 1]
