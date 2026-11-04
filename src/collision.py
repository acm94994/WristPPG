"""Collision-aware adaptation: stop NLMS from notching the pulse when HR nears a motion harmonic.

Near a motion tone, NLMS acts as a notch with half-width of about 4.77 * mu * fs BPM. Holding the
weights removes that notch: the pulse passes at gain 1 and the cost is a leftover motion term
(w* - w)^T x. Weights are averaged before the hold, because w has already absorbed some pulse.

Two passes, so this version is offline:
  pass 1  NLMS + tracker -> HR and IMU frequency -> hold mask
  pass 2  NLMS with weights held inside the mask
ECG is never used here.
"""
from dataclasses import dataclass
import numpy as np
from src.filters import _regressors, nlms
from scipy.signal import find_peaks
from src.hr import _viterbi, hr_scores, track_hr, window_starts


@dataclass
class CollisionConfig:
    gate_bpm: float = 15.0            # hold when HR is this close to a cancelled line; must exceed
                                      # the tracker's +/-10 BPM down-weighting, which repels HR
    line_rho: float = 0.25            # keep cancelled lines above this fraction of the strongest
    max_lines: int = 3
    line_mu: float = 0.01             # lines come from a slow pilot NLMS: a fast filter tracks the
                                      # pulse too, so its output would report HR as a motion line
    avg_s: float = 4.0                # weight averaging, at least one beat period at the gate edge
    min_gap_s: float = 2.0            # join holds separated by less than this
    max_hold_s: float = 60.0          # a hold longer than this is released at its end
    beta: float = 0.1                 # step scale for the "mu" mode


def cancelled_lines(y, fs, cfg, nfft=2048):
    """Per window, the frequencies (BPM) NLMS is removing, read from its output y = d - e.
    This shows where the notches really are, whichever harmonic dominates the IMU."""
    starts, win = window_starts(len(y), fs)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    band = (freqs >= 0.5) & (freqs <= 4.0)
    hann = np.hanning(win)
    out = []
    for s in starts:
        P = np.abs(np.fft.rfft((y[s:s + win] - y[s:s + win].mean()) * hann, nfft)) ** 2
        P = P * band
        pk, _ = find_peaks(P, height=cfg.line_rho * P.max())
        pk = pk[np.argsort(P[pk])[::-1][:cfg.max_lines]]
        out.append(60 * freqs[pk])
    return out


def pilot_lines(d, refs, fs, L, mu, cfg, e=None):
    """Cancelled lines read from a slow pilot filter. Reuses e when mu already equals line_mu."""
    if e is None or mu != cfg.line_mu:
        e = nlms(d, refs, L=L, mu=cfg.line_mu)[0]
    return cancelled_lines(d - e, fs, cfg)


def window_gap(hr, lines):
    return np.array([np.min(np.abs(h - l)) if len(l) else np.inf for h, l in zip(hr, lines)])


def hold_mask(hr, t, lines, n, fs, cfg):
    """Sample-level hold mask from the per-window gap (2 s window step)."""
    hit = window_gap(hr, lines) < cfg.gate_bpm
    mask = np.zeros(n, bool)
    for tc, h in zip(t, hit):
        if h:
            mask[max(0, int((tc - 1) * fs)):min(n, int((tc + 1) * fs))] = True
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    a, b = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = list(zip(a, b))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < cfg.min_gap_s * fs:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    out = np.zeros(n, bool)
    for x, y in merged:
        out[x:min(y, x + int(cfg.max_hold_s * fs))] = True
    return out


def nlms_hold(d, refs, hold, fs, L=16, mu=0.01, mode="avg", cfg=None, delta=1e-6):
    """mode: 'avg' hold averaged weights, 'last' hold latest weights, 'mu' scale the step by beta."""
    cfg = cfg or CollisionConfig()
    X = _regressors(np.asarray(refs, float), L)
    N = X.shape[0]
    w = np.zeros(X.shape[1])
    e = np.empty(N)
    k = int(cfg.avg_s * fs)
    buf = np.zeros((k, len(w)))
    total = np.zeros(len(w))
    filled, pos, held, was = 0, 0, None, False
    for n in range(N):
        x = X[n]
        now = hold[n]
        if now and not was and mode != "mu":
            held = total / filled if (mode == "avg" and filled) else w.copy()
        if now and mode != "mu":
            e[n] = d[n] - held @ x
        else:
            if was and not now and mode != "mu":
                w = held.copy()
            e[n] = d[n] - w @ x
            step = mu * (cfg.beta if now else 1.0) / (delta + x @ x)
            w += step * e[n] * x
            total += w - buf[pos]
            buf[pos] = w
            pos = (pos + 1) % k
            filled = min(filled + 1, k)
        was = now
    return e


def track_aware(e, fs, refs, hold_win):
    """Tracker that stops down-weighting motion bins inside a hold. Elsewhere it is unchanged."""
    S, bpm, t = hr_scores(e, fs, motion=refs)
    S2 = hr_scores(e, fs, motion=refs, motion_atten=1.0)[0]
    return _viterbi(np.where(np.asarray(hold_win)[:, None], S2, S), bpm, 5, 15), t


def mask_to_windows(mask, t, fs):
    return np.array([mask[max(0, int((c - 1) * fs)):int((c + 1) * fs)].any() for c in t])


def run_collision(d, refs, fs, L=16, mu=0.01, cfg=None, mode="avg", passes=1, aware=True, true_hr=None):
    """Returns dict(e, hr, t, hold). true_hr replaces the tracked HR, for oracle tests only."""
    cfg = cfg or CollisionConfig()
    e, _ = nlms(d, refs, L=L, mu=mu)
    hr, t = track_hr(e, fs, motion=refs)
    lines = pilot_lines(d, refs, fs, L, mu, cfg, e)
    hold = np.zeros(len(d), bool)
    for _ in range(passes):
        hold = hold_mask(hr if true_hr is None else true_hr, t, lines, len(d), fs, cfg)
        e = nlms_hold(d, refs, hold, fs, L, mu, mode, cfg)
        win = mask_to_windows(hold, t, fs)
        hr = track_aware(e, fs, refs, win)[0] if aware else track_hr(e, fs, motion=refs)[0]
    return dict(e=e, hr=hr, t=t, hold=hold)
