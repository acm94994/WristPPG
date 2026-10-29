"""Preprocessing (DECISIONS D3-D5, R1b).

Reference channel order (8): acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, |a|, |w|
"""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

REF_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "acc_mag", "gyro_mag"]


def _resample(x, fs_in, fs_out):
    r = Fraction(fs_out, fs_in).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator, axis=0)


def _bandpass(x, fs, lo=0.4, hi=4.0, order=4):
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=0)


def forearm_axis(acc_raw):
    """R1b: accel axis with largest mean |value| BEFORE band-pass (gravity ~ along forearm).
    Returns 0, 1 or 2. Check visually on one record."""
    return int(np.argmax(np.mean(np.abs(acc_raw), axis=0)))


def preprocess(rec, fs_out=64):
    """Returns d (N,), refs (N, 8), fs_out.
    fs_out=64 for main methods; 20 for LMS-HK, 35 for NLMS-FV (reproductions)."""
    fs = rec["fs"]
    acc, gyro = rec["acc"], rec["gyro"]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)    # magnitudes before filtering
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)

    raw = np.column_stack([rec["ppg"], acc, gyro, acc_mag, gyro_mag])
    x = _resample(raw, fs, fs_out)
    x = _bandpass(x, fs_out)

    d = x[:, 0]
    refs = x[:, 1:]
    refs = (refs - refs.mean(0)) / (refs.std(0) + 1e-12)     # z-score each reference
    d = (d - d.mean()) / (d.std() + 1e-12)
    return d, refs, fs_out


def radial_ref(rec, axis):
    """Han & Kim 'radial' reference: magnitude of the two non-forearm accel axes (raw).
    Pass it through preprocess-style filtering yourself if you use it (see LMS-HK)."""
    others = [i for i in range(3) if i != axis]
    return np.linalg.norm(rec["acc"][:, others], axis=1)
