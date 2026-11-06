"""Run: python -m tests.test_extensions  (from the repo root)."""
import numpy as np
from sim.semisynth import make_record
from src.gate import GateConfig, jerk_envelope, robust_z, find_artifacts
from src.wqn_pipeline import run_wqn
from src.preprocess import _bandpass, preprocess
from src.wqn import iswt, swt, transport_gain, wqn

fs = 64
rng = np.random.default_rng(0)


def test_swt_perfect_reconstruction():
    for n in (6400, 6401, 6463):
        x = rng.standard_normal(n)
        c, m, _ = swt(x)
        assert np.max(np.abs(iswt(c, m) - x)) < 1e-9


def test_gain_never_amplifies():
    g = transport_gain(rng.standard_normal(300) * 5, rng.standard_normal(1000))
    assert g.max() <= 1 and g.min() >= 0


def test_same_distribution_is_near_identity():
    x = rng.standard_normal(200000)
    g = transport_gain(x[:100000], x[100000:])
    assert np.median(g) > 0.98


def test_no_intervals_is_identity():
    x = rng.standard_normal(4096)
    assert np.array_equal(wqn(x, [], fs), x)


def test_changes_stay_near_interval():
    # sym5 at level 6 is long, so corrections spread about 4 s; beyond that they must vanish
    t = np.arange(240 * fs) / fs
    x = np.sin(2 * np.pi * 1.7 * t)
    j = _bandpass(8 * np.exp(-0.5 * ((t - 101.5) / 0.2) ** 2), fs)
    a, b = 99 * fs, 104 * fs
    y = x + j
    out = wqn(y, [(a, b, 60 * fs, 76 * fs)], fs)
    far = np.r_[0:a - 4 * fs, b + 4 * fs:len(y)]
    assert np.max(np.abs(out[far] - y[far])) < 1e-4
    assert np.std(out[a:b] - x[a:b]) < 0.5 * np.std(y[a:b] - x[a:b])


def test_rhythmic_footstrikes_do_not_trigger():
    # no transients at all: the jerk gate should stay quiet on steady running
    fired = 0
    cfg = GateConfig()
    for seed in range(5):
        r = make_record(seed, "run", T=120, jerks_per_min=0)
        env = jerk_envelope(r["acc"], r["gyro"], r["fs"], fs, cfg.env_s)
        fired += (robust_z(env, fs, cfg.base_s) > cfg.k_imu).mean()
    assert fired / 5 < 0.002


def test_ecg_not_used():
    import pathlib
    for f in ("gate.py", "wqn.py", "wqn_pipeline.py"):
        src = pathlib.Path("src", f).read_text()
        assert "rpeaks" not in src, f
    r = make_record(3, "walk", T=120)
    d, refs, _ = preprocess(r)
    a = run_wqn(r, d, refs, fs)["e"]
    r["rpeaks_s"] = r["rpeaks_s"] + 7.0
    assert np.array_equal(a, run_wqn(r, d, refs, fs)["e"])


def test_collision_no_hold_equals_nlms():
    from src.collision import nlms_hold
    from src.filters import nlms
    r = make_record(1, "run", T=60)
    d, refs, _ = preprocess(r)
    a = nlms_hold(d, refs, np.zeros(len(d), bool), fs, 16, 0.01)
    assert np.max(np.abs(a - nlms(d, refs, L=16, mu=0.01)[0])) < 1e-9


def test_collision_hold_stops_notching_pulse():
    from src.collision import CollisionConfig, nlms_hold
    from src.preprocess import _bandpass
    t = np.arange(120 * fs) / fs
    x = np.sin(2 * np.pi * 2.0 * t)[:, None]
    refs = np.column_stack([x, np.cos(2 * np.pi * 2.0 * t)])
    d = 5 * np.sin(2 * np.pi * 2.0 * t - 0.9) + np.sin(2 * np.pi * 2.03 * t + 0.7)
    hold = t > 60
    held = nlms_hold(d, refs, hold, fs, 8, 0.05, "avg", CollisionConfig())
    free = nlms_hold(d, refs, np.zeros_like(hold), fs, 8, 0.05, "avg", CollisionConfig())
    z = np.exp(-2j * np.pi * 2.03 * t) * np.exp(-0.7j)
    gain = lambda e: abs((e[80 * fs:] * z[80 * fs:]).mean()) * 2
    assert gain(held) > 0.9 and gain(free) < 0.5


def test_ecg_not_used_by_collision():
    import pathlib
    assert "rpeaks" not in pathlib.Path("src", "collision.py").read_text()


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
