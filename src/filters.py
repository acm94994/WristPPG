"""Adaptive noise cancellers: d(n) = pulse + artifact, refs = motion references.
Output e(n) = d(n) - w^T x(n) is the cleaned PPG.

x(n) stacks the last L samples of every reference channel: length C*L.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _regressors(refs, L):
    """(N, C) -> (N, C*L); row n = [ch0(n..n-L+1), ch1(n..n-L+1), ...], zero-padded at start."""
    N, C = refs.shape
    padded = np.vstack([np.zeros((L - 1, C)), refs])
    win = sliding_window_view(padded, L, axis=0)        # (N, C, L), oldest first
    return win[:, :, ::-1].reshape(N, C * L)            # newest first


def _expand(v, N, C):
    """Per-channel weights: None, (C,) or (N, C) -> (N, C)."""
    if v is None:
        return None
    v = np.asarray(v, float)
    return np.broadcast_to(v, (N, C)) if v.ndim == 1 else v


def adaptive_filter(d, refs, L=16, mu=0.1, delta=1e-6, weights=None, normalized=True):
    """LMS (normalized=False) or NLMS (normalized=True).

    d       : (N,) PPG
    refs    : (N, C) references (or (N,) for one channel)
    L       : taps per channel
    mu      : scalar, or (N,) array for a variable step size (M7)
    weights : None, (C,) or (N, C) per-channel scaling of the references (CW-NLMS)
    returns : e (N,) cleaned PPG, err2 (N,) squared error for learning curves
    """
    refs = np.asarray(refs, float)
    if refs.ndim == 1:
        refs = refs[:, None]
    N, C = refs.shape
    W = _expand(weights, N, C)
    if W is not None:
        refs = refs * W
    X = _regressors(refs, L)
    mu = np.broadcast_to(np.asarray(mu, float), (N,))

    w = np.zeros(C * L)
    e = np.empty(N)
    for n in range(N):
        x = X[n]
        e[n] = d[n] - w @ x
        step = mu[n] / (delta + x @ x) if normalized else mu[n]
        w += step * e[n] * x
    return e, e ** 2


def lms(d, refs, L=16, mu=0.01, **kw):
    return adaptive_filter(d, refs, L=L, mu=mu, normalized=False, **kw)


def nlms(d, refs, L=16, mu=0.1, **kw):
    return adaptive_filter(d, refs, L=L, mu=mu, normalized=True, **kw)


def nlms_bank(d, refs, L=70, mu=0.1, delta=1e-6):
    """NLMS-FV: one single-reference NLMS per channel. Returns E (N, C), one cleaned PPG per channel.
    Feed E straight into track_hr, which averages the spectra."""
    refs = np.asarray(refs, float)
    outs = [nlms(d, refs[:, j], L=L, mu=mu, delta=delta)[0] for j in range(refs.shape[1])]
    return np.column_stack(outs)
