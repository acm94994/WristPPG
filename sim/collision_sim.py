"""Synthetic records with a controlled collision: HR ramps onto twice the arm-swing rate (the step
rate, usually the strongest artifact), dwells, then leaves.
Checks mechanics only, with the same limits as semisynth.py.
"""
import numpy as np
from sim.semisynth import ACT, FS, ou


def make_collision(seed, activity="run", T=240, dwell=20.0, lock=False, drift=0.25, none=False, offset=0.0, leak=0.0):
    """dwell: seconds near the collision. lock: HR exactly follows the step rate during the dwell
    (cardiac-locomotor coupling). none: control record that never collides. offset: dwell this many
    BPM away from the step rate, a near miss that punishes any pull of the tracker towards cadence.
    leak: extra artifact at the step rate, as a fraction of its main line, with fast phase wander so the
    references cannot predict it. This keeps a real motion line in the ANC output, as on real wrists."""
    rng = np.random.default_rng(seed)
    a = ACT[activity]
    n = T * FS
    t = np.arange(n) / FS
    F0 = rng.uniform(*a["F"])
    F = F0 * (1 + 0.01 * ou(rng, n, FS, 30))
    target = 2 * F * 60
    t1, t2 = 60.0, 90.0
    t3, t4 = t2 + dwell, t2 + dwell + 30
    g = np.interp(t, [0, t1, t2, t3, t4, T], [-35, -35, offset, offset, 30, 30])
    if none:
        g = np.full(n, -35.0)
    wobble = ou(rng, n, FS, 15, 1.0) + 1.0 * np.sin(2 * np.pi * 0.25 * t)
    inside = (t >= t2) & (t <= t3) & (not none)
    hr = target + g + wobble
    if lock:
        hr = np.where(inside, target, hr)
    phi = 2 * np.pi * np.cumsum(hr / 60) / FS
    amp = (1 + 0.1 * np.sin(2 * np.pi * 0.25 * t + 1)) * (1 + 0.15 * np.tanh(ou(rng, n, FS, 15)))
    phim = 2 * np.pi * np.cumsum(F) / FS

    beta = rng.uniform(0, 2 * np.pi, 3)
    m = np.zeros(n)
    for k, c in enumerate(a["C"], 1):
        m += c * (1 + drift * np.tanh(ou(rng, n, FS, 20))) * np.cos(k * phim + beta[k - 1])
    if leak > 0:
        m += leak * a["C"][1] * np.cos(2 * phim + ou(rng, n, FS, 2, 1.5))
    G = rng.uniform(0.3, 1.0, (3, 3))
    acc, gyro = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(3):
        for k in (1, 2, 3):
            acc[:, i] += G[i, k - 1] * np.cos(k * phim + rng.uniform(0, 2 * np.pi))
            gyro[:, i] += 40 * G[i, k - 1] * np.cos(k * phim + rng.uniform(0, 2 * np.pi))
    acc[:, 2] += 1.0
    s = amp * (np.cos(phi) + 0.45 * np.cos(2 * phi + 0.8) + 0.15 * np.cos(3 * phi + 1.6))
    ppg = s + m + 0.08 * rng.standard_normal(n)
    acc += 0.02 * rng.standard_normal((n, 3))
    gyro += 0.5 * rng.standard_normal((n, 3))
    beats = np.where(np.diff(np.floor(phi / (2 * np.pi))) > 0)[0] / FS
    return {"name": f"col{seed}_{activity}", "subject": f"col{seed}", "activity": activity, "fs": FS,
            "ppg": ppg, "acc": acc, "gyro": gyro, "rpeaks_s": beats - 0.25, "_s": s, "_beats_s": beats,
            "_hr": hr, "_step_bpm": target, "_events": []}
