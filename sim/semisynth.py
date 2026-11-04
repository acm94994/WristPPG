"""Synthetic wrist-PPG records with known ground truth, shaped like load_record() output.

This tests mechanics only. Results here say nothing about real wrists. The scenario mix is
hand-written, including traps that favour an IMU gate, so a win here is conditional on it.

Transient kinds:
  coupled    PPG transient plus IMU burst      should be cleaned
  ppg_only   PPG transient, IMU silent         gates requiring the IMU will miss it
  imu_only   IMU burst, PPG unaffected         a pure IMU gate will falsely fire
  surge      real pulse amplitude rise         a pure residual gate will falsely fire
"""
import numpy as np

FS = 256
ACT = {  # arm-swing Hz, HR start/end BPM, PPG motion harmonics, relative to pulse amplitude 1
    "walk": dict(F=(0.80, 1.00), hr=(85, 115), C=(1.2, 2.5, 0.5)),
    "run": dict(F=(1.20, 1.50), hr=(105, 165), C=(1.5, 3.0, 1.2)),
    "bike": dict(F=(0.90, 1.30), hr=(90, 140), C=(0.3, 0.5, 0.1)),
}
KINDS = ("coupled", "ppg_only", "imu_only", "surge")
PROBS = (0.5, 0.15, 0.2, 0.15)


def ou(rng, n, fs, tau, sd=1.0):
    """Ornstein-Uhlenbeck noise, generated at 8 Hz and interpolated."""
    q = max(1, fs // 8)
    m = n // q + 2
    a = np.exp(-q / (fs * tau))
    x = np.zeros(m)
    noise = rng.standard_normal(m) * sd * np.sqrt(1 - a * a)
    for i in range(1, m):
        x[i] = a * x[i - 1] + noise[i]
    return np.interp(np.arange(n), np.arange(m) * q, x)


def _burst(rng, t, t0):
    """Decaying oscillation plus a slow lurch: one IMU channel's response to a jerk."""
    tau, f = rng.uniform(0.1, 0.3), rng.uniform(8, 14)
    u = np.clip(t - t0, 0, None)
    osc = np.exp(-u / tau) * np.sin(2 * np.pi * f * u) * (t >= t0)
    sig = rng.uniform(0.1, 0.3)
    return osc + 0.5 * rng.choice([-1, 1]) * np.exp(-0.5 * ((t - t0 - 0.2) / sig) ** 2)


def _ppg_transient(rng, t, t0, amp):
    """Biphasic optical transient. Its shape is unrelated to the IMU, so ANC cannot predict it."""
    sig = rng.uniform(0.1, 0.4)
    lag = rng.uniform(0.0, 0.15)
    g = lambda c, s: np.exp(-0.5 * ((t - c) / s) ** 2)
    return amp * rng.choice([-1, 1]) * (g(t0 + lag, sig) - 0.6 * g(t0 + lag + 2 * sig, 1.5 * sig))


def make_record(seed, activity="run", T=240, jerks_per_min=2.0, kinds=None):
    rng = np.random.default_rng(seed)
    a = ACT[activity]
    n = T * FS
    t = np.arange(n) / FS

    hr0, hr1 = a["hr"]
    hr = hr0 + (hr1 - hr0) * (1 - np.exp(-t / rng.uniform(40, 90))) + ou(rng, n, FS, 25, 3)
    hr = hr + 1.5 * np.sin(2 * np.pi * 0.25 * t)
    phi = 2 * np.pi * np.cumsum(hr / 60) / FS
    amp = (1 + 0.1 * np.sin(2 * np.pi * 0.25 * t + 1)) * (1 + 0.15 * np.tanh(ou(rng, n, FS, 15)))

    F0 = rng.uniform(*a["F"])
    F = F0 * (1 + 0.02 * ou(rng, n, FS, 20))
    phim = 2 * np.pi * np.cumsum(F) / FS

    beta = rng.uniform(0, 2 * np.pi, 3)
    m = np.zeros(n)
    for k, c in enumerate(a["C"], 1):
        m += c * (1 + 0.25 * np.tanh(ou(rng, n, FS, 20))) * np.cos(k * phim + beta[k - 1])

    G = rng.uniform(0.3, 1.0, (3, 3))
    acc = np.zeros((n, 3))
    gyro = np.zeros((n, 3))
    for i in range(3):
        for k in (1, 2, 3):
            ga = 1 + 0.1 * np.tanh(ou(rng, n, FS, 20))
            gg = 1 + 0.1 * np.tanh(ou(rng, n, FS, 20))
            acc[:, i] += G[i, k - 1] * ga * np.cos(k * phim + rng.uniform(0, 2 * np.pi))
            gyro[:, i] += 40 * G[i, k - 1] * gg * np.cos(k * phim + rng.uniform(0, 2 * np.pi))
    acc[:, 2] += 1.0
    # footstrikes: a short high-frequency ring at each step, rhythmic so it must not trigger
    steps = np.where(np.diff(np.floor(2 * phim / (2 * np.pi))) > 0)[0]
    ring = np.exp(-np.arange(int(0.08 * FS)) / (0.02 * FS)) * np.sin(2 * np.pi * 20 * np.arange(int(0.08 * FS)) / FS)
    strike = np.zeros(n)
    strike[steps] = rng.uniform(0.7, 1.3, len(steps))
    strike = np.convolve(strike, ring)[:n]
    level = {"walk": 1.0, "run": 2.0, "bike": 0.2}[activity]
    acc += level * strike[:, None] * rng.uniform(0.5, 1, 3)

    events, jump = [], np.zeros(n)
    tt = rng.uniform(25, 40)
    while jerks_per_min > 0.001 and tt < T - 25:
        kind = kinds[rng.integers(len(kinds))] if kinds else rng.choice(KINDS, p=PROBS)
        size = rng.uniform(3, 10)
        if kind in ("coupled", "imu_only"):
            for i in range(3):
                acc[:, i] += level * rng.uniform(4, 9) * _burst(rng, t, tt)
                gyro[:, i] += 40 * rng.uniform(1.5, 3) * _burst(rng, t, tt)
        if kind in ("coupled", "ppg_only"):
            jump += _ppg_transient(rng, t, tt, size)
        if kind == "surge":
            d = rng.uniform(4, 8)
            amp *= 1 + rng.uniform(0.8, 1.5) * np.exp(-0.5 * ((t - tt - d / 2) / (d / 4)) ** 2)
        events.append(dict(t=float(tt), kind=str(kind), size=float(size)))
        tt += max(12.0, rng.exponential(60 / max(jerks_per_min, 1e-6)))

    s = amp * (np.cos(phi) + 0.45 * np.cos(2 * phi + 0.8) + 0.15 * np.cos(3 * phi + 1.6))
    ppg = s + m + jump + 0.08 * rng.standard_normal(n)
    acc += 0.02 * rng.standard_normal((n, 3))
    gyro += 0.5 * rng.standard_normal((n, 3))

    beats = np.where(np.diff(np.floor(phi / (2 * np.pi))) > 0)[0] / FS
    return {"name": f"sim{seed}_{activity}", "subject": f"sim{seed}", "activity": activity,
            "fs": FS, "ppg": ppg, "acc": acc, "gyro": gyro,
            "rpeaks_s": beats - 0.25, "_s": s, "_beats_s": beats, "_events": events, "_hr": hr}
