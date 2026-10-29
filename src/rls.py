"""RLS-all baseline (DECISIONS M3, M9): exponentially weighted RLS on the same stacked regressor as filters.py.

Band-passed (0.4-4 Hz), delayed IMU channels are strongly correlated, so most directions of the
M = C*L regressor are barely excited. Plain RLS then "winds up": P grows as lam^-n along those
directions and the output blows up (|e| ~ 1e8 seen on real records). Fix: regularized RLS. Every
sample also adds rho * u u^T to R = P^-1 with u cycling through the unit vectors, so each
coordinate gets rho once every M samples and R stays above roughly floor * I, floor = beta / (1 - lam).
Since refs are z-scored, 1 / (1 - lam) is the steady-state diagonal of R, so beta is relative.

Cost is O(M^2) per sample (two rank-one updates), vs O(M) for (N)LMS.
"""
import numpy as np
from src.filters import _regressors


def rls(d, refs, L=16, lam=0.999, delta=0.01, beta=1e-2):
    """d (N,), refs (N, C) or (N,). P(0) = I / delta. beta = 0 gives plain RLS.
    returns: e (N,) a-priori error = cleaned PPG, err2 (N,) for learning curves."""
    refs = np.asarray(refs, float)
    if refs.ndim == 1:
        refs = refs[:, None]
    X = _regressors(refs, L)
    N, M = X.shape
    rho = beta / (1 - lam) * (1 - lam ** M)     # per-visit increment that holds the floor

    w = np.zeros(M)
    P = np.eye(M) / delta
    e = np.empty(N)
    for n in range(N):
        x = X[n]
        Px = P @ x
        k = Px / (lam + x @ Px)
        e[n] = d[n] - w @ x
        w += k * e[n]
        P -= np.outer(k, Px)
        P /= lam
        if beta:                               # Sherman-Morrison for R + rho * e_i e_i^T
            i = n % M
            Pi = P[:, i].copy()
            P -= np.outer(Pi, Pi) * (rho / (1 + rho * Pi[i]))
        if n % 64 == 0:
            P = (P + P.T) / 2                  # round-off makes P drift from symmetric
    return e, e ** 2
