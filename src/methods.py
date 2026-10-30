"""One entry point for every method in the report.

    out = run_method("CW-NLMS", rec, mu=0.05, L=16, tau=0.3)
    out["hr"], out["t"]   -> per-window HR estimate (BPM) and window centre times (s)
    out["e"], out["err2"] -> cleaned PPG and squared error (learning curves); out["fs"]

Methods (names fixed, see DECISIONS.md):
  LMS-HK   Han & Kim reproduction: LMS, forearm axis + radial, 3 taps, mu 0.009, 20 Hz   (R1)
  NLMS-FV  Fallet & Vesin reproduction: 3 parallel NLMS, 70 taps, mu 0.1, 35 Hz          (R2)
  NLMS-1   NLMS, forearm axis only                                                      (M1)
  NLMS-all NLMS, all 8 references
  RLS-all  RLS, all 8 references                                                        (M3)
  CW-NLMS  NLMS, all 8 references scaled by MSC weights, variable step size              (M4-M7)
  RAW      no filtering (reference row for the tables)
Tunable parameters per method are in TUNABLE; the two reproductions keep their papers' settings.
ECG is never used here.
"""
import numpy as np
from src.preprocess import preprocess, forearm_axis, _resample, _bandpass
from src.filters import lms, nlms, nlms_bank, rls
from src.coherence import msc_weights, per_sample
from src.hr import track_hr

METHODS = ["RAW", "LMS-HK", "NLMS-FV", "NLMS-1", "NLMS-all", "RLS-all", "CW-NLMS"]
TUNABLE = {"NLMS-1": ["mu", "L"], "NLMS-all": ["mu", "L"], "RLS-all": ["L", "lam"],
           "CW-NLMS": ["mu", "L", "tau"]}


def _zs(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def _hk_refs(rec, ax, fs_out):
    """Han & Kim references: forearm axis and radial = |other two axes|, filtered like the rest."""
    acc = rec["acc"]
    others = [i for i in range(3) if i != ax]
    raw = np.column_stack([acc[:, ax], np.linalg.norm(acc[:, others], axis=1)])
    return _zs(_bandpass(_resample(raw, rec["fs"], fs_out), fs_out))


def run_method(name, rec, mu=0.01, L=16, tau=0.3, vss=True, lam=0.999):
    ax = forearm_axis(rec["acc"])

    if name == "LMS-HK":
        fs = 20
        d, refs, _ = preprocess(rec, fs_out=fs)
        e, err2 = lms(d, _hk_refs(rec, ax, fs), L=3, mu=0.009)
    elif name == "NLMS-FV":
        fs = 35
        d, refs, _ = preprocess(rec, fs_out=fs)
        E = nlms_bank(d, refs[:, :3], L=70, mu=0.1)
        hr, t = track_hr(E, fs, motion=refs)
        e = E.mean(1)
        return {"hr": hr, "t": t, "e": e, "err2": (E ** 2).mean(1), "fs": fs}
    else:
        fs = 64
        d, refs, _ = preprocess(rec, fs_out=fs)
        if name == "RAW":
            e, err2 = d, d ** 2
        elif name == "NLMS-1":
            e, err2 = nlms(d, refs[:, ax], L=L, mu=mu)
        elif name == "NLMS-all":
            e, err2 = nlms(d, refs, L=L, mu=mu)
        elif name == "RLS-all":
            e, err2 = rls(d, refs, L=L, lam=lam)
        elif name == "CW-NLMS":
            Wwin, starts, win = msc_weights(d, refs, fs, tau=tau)
            W = per_sample(Wwin, starts, win, len(d), step=2 * fs)
            mu_n = mu * W.mean(1) if vss else mu          # M7: adapt less when nothing is coherent
            e, err2 = nlms(d, refs, L=L, mu=mu_n, weights=W)
            hr, t = track_hr(e, fs, motion=refs)
            return {"hr": hr, "t": t, "e": e, "err2": err2, "fs": fs, "msc": Wwin}
        else:
            raise ValueError(f"unknown method {name}; choose from {METHODS}")

    hr, t = track_hr(e, fs, motion=refs)
    return {"hr": hr, "t": t, "e": e, "err2": err2, "fs": fs}
