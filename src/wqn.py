"""Wavelet quantile normalisation on an undecimated wavelet transform (Option 3).

Follows Dora, Jaffard & Holcman: for each scale j,
    T_j(x)  = F_ref^-1( F_art(x) )                 transport of coefficient amplitudes
    c_corr  = sign(c) * min(|c|, T_j(|c|))          magnitude can never grow
PPG-specific changes:
  * SWT instead of DWT, so a short jerk still has coefficients at the HR-band scales.
    Neighbouring SWT coefficients are correlated, so this adds resolution, not data.
  * The gain is tapered at the interval edges, so no step appears at the boundary.
At fs = 64 Hz: D3 4-8, D4 2-4, D5 1-2, D6 0.5-1 Hz. The 0.4-4 Hz band-pass leaves D1-D2 empty.
"""
import numpy as np
import pywt


def swt(x, wavelet="sym5", level=6):
    """Returns (coeffs, n, pad). coeffs = [cA_L, cD_L, ..., cD_1]."""
    n = len(x)
    block = 2 ** level
    pad = (block - n % block) % block
    xp = np.pad(x, (0, pad), mode="reflect")
    return pywt.swt(xp, wavelet, level=level, norm=True, trim_approx=True), n, pad


def iswt(coeffs, n, wavelet="sym5"):
    return pywt.iswt(coeffs, wavelet, norm=True)[:n]


def transport_gain(c_art, c_ref):
    """Per-coefficient gain in [0, 1]: lambda(c) / c."""
    a = np.abs(c_art)
    ranks = (np.argsort(np.argsort(a)) + 0.5) / len(a)      # empirical F_art
    T = np.quantile(np.abs(c_ref), ranks)                   # F_ref^-1
    return np.minimum(1.0, T / (a + 1e-12))


def _taper(n, m):
    """Window that is 1 in the middle and falls to 0 over m samples at each end."""
    w = np.ones(n)
    m = min(m, n // 2)
    if m > 0:
        ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(m) + 0.5) / m))
        w[:m], w[-m:] = ramp, ramp[::-1]
    return w


def wqn(x, intervals, fs, wavelet="sym5", level=6, taper_s=0.25):
    """x: signal. intervals: list of (art_start, art_stop, ref_start, ref_stop) in samples.
    Details are corrected, the approximation is left alone. Returns the corrected signal."""
    if not intervals:
        return np.array(x, float, copy=True)
    coeffs, n, _ = swt(np.asarray(x, float), wavelet, level)
    out = [coeffs[0]] + [c.copy() for c in coeffs[1:]]
    m = int(taper_s * fs)
    for a0, a1, r0, r1 in intervals:
        w = _taper(a1 - a0, m)
        for j in range(1, len(coeffs)):
            g = transport_gain(coeffs[j][a0:a1], coeffs[j][r0:r1])
            out[j][a0:a1] = coeffs[j][a0:a1] * (1 - w * (1 - g))
    return iswt(out, n, wavelet)


def universal_soft(x, wavelet="sym5", level=6):
    """Baseline: Donoho-Johnstone universal soft threshold on all details, everywhere."""
    coeffs, n, _ = swt(np.asarray(x, float), wavelet, level)
    out = [coeffs[0]]
    for c in coeffs[1:]:
        sigma = np.median(np.abs(c)) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(c)))
        out.append(pywt.threshold(c, thr, mode="soft"))
    return iswt(out, n, wavelet)
