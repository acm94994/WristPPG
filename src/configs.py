"""Tuning grids and bookkeeping around A's src/methods.py (run_method, METHODS, TUNABLE).
B's file: tune.py, run_all.py and evaluate.py import from here, so methods.py stays A's alone.

GRIDS[m] lists the configs tried in LOSO (E4); one empty entry = fixed settings.
"""
from src.methods import METHODS, TUNABLE

MU = (0.01, 0.05, 0.1, 0.5)          # E4
ORDERS = (8, 16, 32)                 # M8
LAMS = (0.998, 0.999)                # M3
TAUS = (0.0, 0.2, 0.3, 0.5)          # M5
N_REFS = 8

GRIDS = {
    "RAW": [{}],
    "LMS-HK": [{}],                  # paper settings, not tuned (R1, M10)
    "NLMS-FV": [{}],                 # paper settings, not tuned (R2, M10)
    "NLMS-1": [dict(mu=mu, L=L) for mu in MU for L in ORDERS],
    "NLMS-all": [dict(mu=mu, L=L) for mu in MU for L in ORDERS],
    "RLS-all": [dict(lam=lam, L=L) for lam in LAMS for L in ORDERS],
    "CW-NLMS": [dict(mu=mu, L=L, tau=tau) for mu in MU for L in ORDERS for tau in TAUS],
}
assert set(GRIDS) == set(METHODS), "every method in methods.METHODS needs a grid"
assert all(set(g) <= set(TUNABLE.get(m, [])) for m in GRIDS for g in GRIDS[m])


def flops(name, params):
    """E14: multiplies per sample (LMS 2M+1, NLMS 3M+2, RLS 4M^2+4M, M = taps in the regressor).
    CW-NLMS adds C for scaling the references; its MSC estimate (once per 2 s) is not counted."""
    L = params.get("L", 16)
    nlms = lambda M: 3 * M + 2
    return {"RAW": 0,
            "LMS-HK": 2 * (2 * 3) + 1,                   # 2 references x 3 taps
            "NLMS-FV": 3 * nlms(70),                     # 3 parallel single-reference NLMS
            "NLMS-1": nlms(L),
            "NLMS-all": nlms(N_REFS * L),
            "RLS-all": 4 * (N_REFS * L) ** 2 + 4 * N_REFS * L,
            "CW-NLMS": nlms(N_REFS * L) + N_REFS}[name]


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
