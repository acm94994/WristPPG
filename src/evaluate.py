"""Run one method on one record and score it. The only place a method's output meets ECG ground truth.

evaluate(name, rec, params) -> (windows DataFrame, record-level dict)
"""
import time
import numpy as np
import pandas as pd
from src import score
from src.configs import flops, params_key
from src.methods import run_method
from src.preprocess import preprocess
from src.synth import clean_like_d


def run(name, rec, **params):
    """run_method plus what scoring needs: d (the canceller input at the method's rate) and flops."""
    out = run_method(name, rec, **params)
    out["d"] = preprocess(rec, fs_out=out["fs"])[0]
    out["flops"] = flops(name, params)
    return out


def imu_features(rec, t_center, win_s=8):
    """Per window, IMU only: dominant accel frequency (E7) and motion level (E15).
    imu_freq: peak of the summed power spectra of the 3 band-passed accel axes, 0.4-4 Hz (Hz).
    motion  : std of raw |a| in the window (m/s^2), pre-filtering, so it is comparable across records."""
    _, refs, fs = preprocess(rec)
    freqs, P = score.window_spectra(refs[:, :3], fs, t_center, win_s)
    imu_freq = score.peak_freq(freqs, P, band=(0.4, 4.0))
    amag, fs0 = np.linalg.norm(rec["acc"], axis=1), rec["fs"]
    motion = [amag[int((t - win_s / 2) * fs0):int((t + win_s / 2) * fs0)].std() for t in t_center]
    return imu_freq, np.array(motion)


def evaluate(name, rec, params, clean=None, keep_curve=False, **tags):
    """clean: known pulse at rec['fs'] (semi-synthetic records), adds NMSE / correlation.
    tags: extra constant columns (split, condition, ...).
    Returns windows (DataFrame), record (dict). If keep_curve: record['curve'] = err2, plus
    record['curve_clean'] = (e - s)^2 when clean is given."""
    imu_rec = {k: v for k, v in rec.items() if k != "rpeaks_s"}     # ECG never reaches a method
    t0 = time.perf_counter()
    out = run(name, imu_rec, **params)
    runtime = time.perf_counter() - t0

    base = {"method": name, "params": params_key(params), "record": rec["name"],
            "subject": rec["subject"], "activity": rec["activity"], **tags}
    win = pd.DataFrame(score.window_metrics(rec, out))
    win["imu_freq"], win["motion"] = imu_features(imu_rec, out["t"])
    if "msc" in out:                     # CW-NLMS: strongest thresholded MSC weight (0 = no ref coherent)
        win["msc"] = out["msc"].max(1)
    win = win.assign(**base)

    dur = len(out["d"]) / out["fs"]
    record = {**base, "duration_s": dur, "runtime_s": runtime, "rt_factor": runtime / dur,
              "flops": out["flops"], "conv_time_s": score.convergence_time(out["err2"], out["fs"]),
              "mae": win["abs_err"].mean(), "peak_err": win["peak_err"].mean(),
              "snr_gain": win["snr_gain"].mean(), "pulse_ratio": win["pulse_ratio"].mean()}
    if clean is not None:
        s = clean_like_d(rec["ppg"], clean, rec["fs"], out["fs"])[:len(out["d"])]
        e = out["e"].mean(1) if out["e"].ndim == 2 else out["e"]
        record.update(nmse_db=score.nmse_db(e, s), corr=score.corr(e, s),
                      nmse_in_db=score.nmse_db(out["d"], s),
                      conv_time_clean_s=score.convergence_time((e - s) ** 2, out["fs"]))
    if keep_curve:
        record["curve"] = out["err2"]
        record["fs"] = out["fs"]
        if clean is not None:
            record["curve_clean"] = (e - s) ** 2
    return win, record
