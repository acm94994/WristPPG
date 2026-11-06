"""IMU-gated WQN on the ANC residual.

  rec -> preprocess -> NLMS -> jerk gate + residual check -> reference selection -> SWT-WQN

ECG is never used here. `hr` comes from the PPG + IMU tracker.
"""
import numpy as np
from src.filters import nlms
from src.gate import (GateConfig, choose_reference, find_artifacts, jerk_envelope,
                      reference_table, robust_z)
from src.hr import track_hr
from src.wqn import wqn


def jerk_z(rec, fs, cfg):
    env = jerk_envelope(rec["acc"], rec["gyro"], rec["fs"], fs, cfg.env_s)
    return robust_z(env, fs, cfg.base_s)


def clean(e, refs, fs, zj, cfg, mode="both", runs=None, hr=None, t_hr=None, wqn_kw=None):
    """One WQN pass over residual e. `runs` overrides detection (used for oracle tests)."""
    if hr is None:
        hr, t_hr = track_hr(e, fs, motion=refs)
    if runs is None:
        zj_use = zj[:len(e)]
        runs, imu, res = find_artifacts(e, zj_use, fs, cfg, mode)
    else:
        imu = res = np.zeros(len(e), bool)
    flagged = imu | res
    for a, b in runs:
        flagged[a:b] = True
    table = reference_table(e, refs, fs, hr, t_hr, flagged, cfg)
    intervals, report = [], []
    for a, b in runs:
        ref = choose_reference(table, a, b, fs, cfg)
        if ref is None:
            report.append(dict(t0=a / fs, t1=b / fs, status="no_reference"))
            continue
        intervals.append((a, b, int(ref["start"]), int(ref["stop"])))
        report.append(dict(t0=a / fs, t1=b / fs, status="corrected",
                           ref_t0=ref["start"] / fs, q=float(ref["q"])))
    return wqn(e, intervals, fs, **(wqn_kw or {})), runs, report


def run_wqn(rec, d, refs, fs, L=16, mu=0.01, cfg=None, mode="both", freeze=False):
    """Returns dict(e_anc, e, runs, report). freeze=True re-runs NLMS with mu=0 on artifacts."""
    cfg = cfg or GateConfig()
    e_anc, _ = nlms(d, refs, L=L, mu=mu)
    zj = jerk_z(rec, fs, cfg)
    n = min(len(e_anc), len(zj))
    zj = np.pad(zj[:n], (0, len(e_anc) - n), mode="edge")
    e_in = e_anc
    if freeze:
        runs0 = find_artifacts(e_anc, zj, fs, cfg, mode)[0]
        m = np.full(len(d), float(mu))
        for a, b in runs0:
            m[a:b] = 0.0
        e_in, _ = nlms(d, refs, L=L, mu=m)
    e, runs, report = clean(e_in, refs, fs, zj, cfg, mode)
    return dict(e_anc=e_anc, e_in=e_in, e=e, runs=runs, report=report, zj=zj)
