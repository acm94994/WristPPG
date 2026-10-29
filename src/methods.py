"""Method registry: every method has the same signature, so tune.py, run_all.py and robustness.py
treat them identically.

    run(name, rec, **params) -> dict
        fs    : sampling rate the method ran at
        d     : (N,) preprocessed PPG fed to the canceller
        e     : (N,) cleaned PPG, or (N, K) for several outputs (spectra get averaged)
        err2  : (N,) squared error (learning curve)
        hr, t : track_hr output
        flops : multiplies per sample (E14)

rec must be IMU-only: evaluate.py removes 'rpeaks_s' before calling run().
`drop` (multi-reference methods): REF_NAMES to remove, for the channel ablation.

A: replace the three _pending(...) entries with the real functions (same signature) and set
their grids in GRIDS. Nothing else needs to change.
"""
import numpy as np
from src.preprocess import preprocess, forearm_axis, REF_NAMES
from src.filters import nlms
from src.rls import rls
from src.hr import track_hr

MU = (0.01, 0.05, 0.1, 0.5)          # E4
ORDERS = (8, 16, 32)                 # M8
LAMS = (0.99, 0.995, 0.999)          # M9


def _flops(kind, M):
    """Multiplies per sample for an M-tap filter (E14)."""
    return {"lms": 2 * M + 1, "nlms": 3 * M + 2, "rls": 4 * M ** 2 + 4 * M}[kind]


def _keep(refs, drop):
    keep = [i for i, n in enumerate(REF_NAMES) if n not in drop]
    return refs[:, keep]


def _out(fs, d, e, err2, flops):
    hr, t = track_hr(e, fs)
    return {"fs": fs, "d": d, "e": e, "err2": err2, "hr": hr, "t": t, "flops": flops}


def raw(rec):
    """No cancellation: HR straight from the band-passed PPG (reference row in every table)."""
    d, _, fs = preprocess(rec)
    return _out(fs, d, d, d ** 2, 0)


def nlms_1(rec, mu=0.1, L=16):
    """NLMS on the forearm accel axis only (M1, R1b)."""
    d, refs, fs = preprocess(rec)
    e, err2 = nlms(d, refs[:, forearm_axis(rec["acc"])], L=L, mu=mu)
    return _out(fs, d, e, err2, _flops("nlms", L))


def nlms_all(rec, mu=0.1, L=16, drop=()):
    """NLMS on all 8 references, no selection or weighting."""
    d, refs, fs = preprocess(rec)
    refs = _keep(refs, drop)
    e, err2 = nlms(d, refs, L=L, mu=mu)
    return _out(fs, d, e, err2, _flops("nlms", refs.shape[1] * L))


def rls_all(rec, lam=0.999, L=16, drop=()):
    """Regularized RLS on all 8 references (stronger baseline, M3, M9)."""
    d, refs, fs = preprocess(rec)
    refs = _keep(refs, drop)
    e, err2 = rls(d, refs, L=L, lam=lam)
    return _out(fs, d, e, err2, _flops("rls", refs.shape[1] * L))


def _pending(note):
    def f(rec, **params):
        raise NotImplementedError(note)
    f.pending = True
    return f


METHODS = {
    "raw": raw,
    "LMS-HK": _pending("A: Han & Kim reproduction (R1, R1b), 20 Hz"),
    "NLMS-FV": _pending("A: Fallet & Vesin reproduction (R2), 35 Hz, nlms_bank"),
    "NLMS-1": nlms_1,
    "NLMS-all": nlms_all,
    "RLS-all": rls_all,
    "CW-NLMS": _pending("A: coherence-weighted NLMS (M4-M7), needs src/coherence.py"),
}

GRIDS = {                            # LOSO grid per method (E4); one entry = fixed settings
    "raw": [{}],
    "LMS-HK": [{}],                  # paper settings, not tuned (R1)
    "NLMS-FV": [{}],                 # paper settings, not tuned (R2)
    "NLMS-1": [dict(mu=mu, L=L) for mu in MU for L in ORDERS],
    "NLMS-all": [dict(mu=mu, L=L) for mu in MU for L in ORDERS],
    "RLS-all": [dict(lam=lam, L=L) for lam in LAMS for L in ORDERS],
    "CW-NLMS": [{}],                 # A: mu x L x tau (M5, M8)
}

MULTI_REF = {"NLMS-all", "RLS-all", "CW-NLMS"}   # accept drop= (channel ablation)


def available():
    return [n for n, f in METHODS.items() if not getattr(f, "pending", False)]


def run(name, rec, **params):
    return METHODS[name](rec, **params)


def params_key(params):
    """Stable string for a params dict, e.g. 'L=16,mu=0.1' ('-' for none)."""
    return ",".join(f"{k}={params[k]}" for k in sorted(params)) or "-"


def parse_key(key):
    if key == "-":
        return {}
    out = {}
    for kv in key.split(","):
        k, v = kv.split("=")
        out[k] = int(v) if k == "L" else float(v)
    return out
