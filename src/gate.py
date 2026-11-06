"""IMU-gated WQN gates: which samples are artifact, and which stretch is a clean reference.
PPG + IMU only. ECG is never used here.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import hilbert, resample_poly


@dataclass
class GateConfig:
    env_s: float = 1.0        # jerk RMS window, at least one step so footstrikes average out
    base_s: float = 30.0      # running baseline length
    k_imu: float = 6.0        # robust z threshold on jerk envelope
    k_res: float = 5.0        # robust z threshold on residual envelope
    res_smooth_s: float = 0.5
    join_s: float = 0.5       # reach between an IMU flag and the residual excursion
    pad_s: float = 0.5        # margin around each artifact interval
    max_art_s: float = 6.0    # longer than this is not a transient: leave it alone
    ref_s: float = 16.0       # reference length
    ref_span_s: float = 30.0  # how far from the artifact a reference may start or end
    ref_hop_s: float = 2.0
    q_min: float = 0.35       # minimum fraction of band power near HR and 2*HR
    collision_bpm: float = 6.0


def robust_z(x, fs, base_s):
    """(x - running median) / (1.4826 * running MAD), computed at about 4 Hz for speed."""
    q = max(1, int(fs // 4))
    xs = x[::q]
    size = max(3, int(base_s * fs / q) | 1)
    med = median_filter(xs, size=size, mode="nearest")
    mad = median_filter(np.abs(xs - med), size=size, mode="nearest")
    floor = 0.05 * np.median(xs) + 1e-12
    z = (xs - med) / (1.4826 * mad + floor)
    return np.interp(np.arange(len(x)), np.arange(len(xs)) * q, z)


def jerk_envelope(acc, gyro, fs_raw, fs_out, env_s=1.0):
    """Unfiltered IMU at the raw rate -> jerk RMS envelope at fs_out.
    The band-pass would hide jerks, so this must use the raw channels."""
    ja = fs_raw * np.linalg.norm(np.diff(acc, axis=0, prepend=acc[:1]), axis=1)
    jg = fs_raw * np.linalg.norm(np.diff(gyro, axis=0, prepend=gyro[:1]), axis=1)
    scale = lambda v: max(np.median(v), 0.1 * np.mean(v)) + 1e-12      # quantised IMU can have median 0
    j = ja / scale(ja) + jg / scale(jg)
    env = np.sqrt(uniform_filter1d(j ** 2, max(1, int(env_s * fs_raw)), mode="nearest"))
    q = int(round(fs_raw / fs_out))
    return env[::q] if q * fs_out == fs_raw else resample_poly(env, fs_out, fs_raw)


def residual_envelope(e, fs, smooth_s=0.5):
    return uniform_filter1d(np.abs(hilbert(e)), max(1, int(smooth_s * fs)), mode="nearest")


def _runs(mask):
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def _dilate(mask, n):
    return uniform_filter1d(mask.astype(float), 2 * n + 1, mode="constant") > 1e-9


def find_artifacts(e, zj, fs, cfg, mode="both"):
    """Returns (runs, imu_flag, res_flag). mode: 'both', 'imu' or 'res'."""
    ze = robust_z(residual_envelope(e, fs, cfg.res_smooth_s), fs, cfg.base_s)
    imu = zj > cfg.k_imu
    res = ze > cfg.k_res
    if mode == "imu":
        keep = imu
    elif mode == "res":
        keep = res
    else:
        # keep residual excursions that an IMU flag reaches; hysteresis gives the interval edges
        near = _dilate(imu, int(cfg.join_s * fs))
        soft = ze > 0.5 * cfg.k_res
        keep = np.zeros(len(e), bool)
        for a, b in _runs(soft):
            if (res[a:b] & near[a:b]).any():
                keep[a:b] = True
    pad = int(cfg.pad_s * fs)
    runs = [(a, b) for a, b in _runs(_dilate(keep, pad)) if (b - a) <= cfg.max_art_s * fs]
    return runs, imu, res


def dominant_motion_bpm(refs, fs, starts, win, band=(0.5, 4.0), nfft=2048):
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    idx = (freqs >= band[0]) & (freqs <= band[1])
    hann = np.hanning(win)[:, None]
    out = []
    for s in starts:
        P = np.abs(np.fft.rfft(refs[s:s + win] * hann, nfft, axis=0)) ** 2
        P = (P / (P.max(0, keepdims=True) + 1e-12)).sum(1)
        out.append(60 * freqs[idx][np.argmax(P[idx])])
    return np.array(out)


def reference_table(e, refs, fs, hr, t_hr, flagged, cfg, nfft=4096):
    """Every candidate reference window with its quality and why it may be rejected."""
    win = int(cfg.ref_s * fs)
    hop = int(cfg.ref_hop_s * fs)
    starts = np.arange(0, len(e) - win + 1, hop)
    fm = dominant_motion_bpm(refs, fs, starts, win)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    bpm = freqs * 60
    band = (freqs >= 0.5) & (freqs <= 4.0)
    hann = np.hanning(win)
    bad = np.cumsum(np.concatenate([[0], flagged.astype(int)]))
    rows = []
    for i, s in enumerate(starts):
        tc = (s + win / 2) / fs
        sel = np.abs(t_hr - tc) <= cfg.ref_s / 2
        h = np.median(hr[sel]) if sel.any() else np.nan
        P = np.abs(np.fft.rfft(e[s:s + win] * hann, nfft)) ** 2
        near = (np.abs(bpm - h) <= 9) | (np.abs(bpm - 2 * h) <= 9)
        q = P[band & near].sum() / (P[band].sum() + 1e-12)
        coll = any(abs(h - k * fm[i]) < cfg.collision_bpm for k in (0.5, 1, 1.5, 2))
        rows.append((s, s + win, q, bad[s + win] > bad[s], coll))
    return np.array(rows, dtype=[("start", int), ("stop", int), ("q", float), ("flagged", bool), ("collision", bool)])


def choose_reference(table, a0, a1, fs, cfg):
    """Best qualifying window near the artifact, or None (leave that artifact untouched)."""
    span = int(cfg.ref_span_s * fs)
    ok = (~table["flagged"]) & (~table["collision"]) & (table["q"] >= cfg.q_min) \
        & (table["stop"] > a0 - span) & (table["start"] < a1 + span) \
        & ((table["stop"] <= a0) | (table["start"] >= a1))
    if not ok.any():
        return None
    rows = table[ok]
    return rows[np.argmax(rows["q"])]
