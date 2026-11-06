"""Run: python -m tests.test_filters  (from the repo root).

Tests for the main cancellers, the coherence weights and the cost model."""
import numpy as np
from scipy.signal import lfilter

from src.coherence import msc_weights
from src.configs import flops
from src.filters import lms, nlms, rls
from src.preprocess import _bandpass

fs = 64
rng = np.random.default_rng(0)


def _system(n=64 * 120, C=2, L=8):
    """Refs (n, C) of white noise and d = sum of known FIRs of them: an exact ANC problem."""
    refs = rng.standard_normal((n, C))
    h = rng.standard_normal((C, L)) * np.exp(-np.arange(L) / 3)
    d = sum(lfilter(h[j], 1, refs[:, j]) for j in range(C))
    return d, refs, h


def test_lms_and_nlms_cancel_a_known_system():
    d, refs, _ = _system()
    for f, mu in ((lms, 0.005), (nlms, 0.1)):
        e, err2 = f(d, refs, L=8, mu=mu)
        assert np.allclose(err2, e ** 2)
        assert np.mean(e[-640:] ** 2) < 1e-3 * np.mean(d ** 2), f.__name__


def test_rls_cancels_a_known_system():
    d, refs, _ = _system(n=64 * 30)
    e, _ = rls(d, refs, L=8, lam=0.999)
    assert np.mean(e[-640:] ** 2) < 1e-2 * np.mean(d ** 2)


def test_regularised_rls_stays_bounded_on_correlated_band_limited_refs():
    """The failure that led to the regularisation: band-passed, strongly correlated references."""
    n = 64 * 300
    base = _bandpass(rng.standard_normal(n), fs)
    refs = np.column_stack([base + 0.05 * _bandpass(rng.standard_normal(n), fs) for _ in range(4)])
    refs = (refs - refs.mean(0)) / refs.std(0)
    d = refs[:, 0] + 0.5 * _bandpass(rng.standard_normal(n), fs)
    d = (d - d.mean()) / d.std()
    e, _ = rls(d, refs, L=8, lam=0.998)
    assert np.all(np.isfinite(e)) and np.max(np.abs(e)) < 50


def test_zero_weights_pass_the_ppg_through():
    """CW-NLMS with every reference below tau: output equals input, nothing is adapted."""
    d, refs, _ = _system(n=2000)
    e, _ = nlms(d, refs, L=8, mu=0.0, weights=np.zeros(refs.shape[1]))
    assert np.array_equal(e, d)


def test_coherence_weights():
    n = 64 * 60
    x = _bandpass(rng.standard_normal(n), fs)
    noise = _bandpass(rng.standard_normal(n), fs)
    refs = np.column_stack([x, noise])
    W, starts, win = msc_weights(x + 0.01 * _bandpass(rng.standard_normal(n), fs), refs, fs, tau=0.0)
    assert W.shape == (len(starts), 2)
    assert np.all(W[:, 0] > 0.9)          # reference that predicts the PPG
    # An independent reference is NOT near zero: 5 overlapping 4 s segments give a chance level of about
    # 0.33 for band-passed noise, so tau = 0.3 does not separate it from an uninformative reference.
    assert 0.2 < W[:, 1].mean() < 0.45
    assert W[:, 1].max() < 0.6


def test_coherence_band_edge_does_not_change_weights():
    """0.4-4 Hz and 0.5-4 Hz pick the same 0.25 Hz bins at 64 Hz (DECISIONS M4)."""
    n = 64 * 60
    refs = rng.standard_normal((n, 3))
    d = refs[:, 0] + rng.standard_normal(n)
    a = msc_weights(d, refs, fs, tau=0.0, band=(0.4, 4.0))[0]
    b = msc_weights(d, refs, fs, tau=0.0, band=(0.5, 4.0))[0]
    assert np.array_equal(a, b)


def test_cost_model_matches_report():
    """Multiplications per sample as reported in the results table."""
    assert flops("LMS-HK", {}) == 13
    assert flops("NLMS-FV", {}) == 636
    assert flops("NLMS-1", {"L": 32}) == 98
    assert flops("NLMS-all", {"L": 16}) == 386
    assert flops("RLS-all", {"L": 8}) == 16640
    assert flops("CW-NLMS", {"L": 16}) == 394


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
