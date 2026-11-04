"""Paired statistics for results/collision_test.csv. The unit is the seed, not the window."""
import csv
import numpy as np

rows = list(csv.DictReader(open("results/collision_test.csv")))
for r in rows:
    for k in ("seed", "mae", "mae_zone", "rmse", "rmse_zone", "f1_zone", "hold_frac"):
        r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
METHODS = ("aware-only", "hold", "hold-last+aware", "mu+aware", "hold+aware", "hold+aware-oracle")
GROUPS = {"collisions": ("crossing", "dwell20", "dwell40"), "phase-locked": ("lock20", "lock40"),
          "near misses": ("near+8", "near-8", "near+12", "near-12"), "control": ("control",)}
key = lambda r: (r["seed"], r["act"], r["scen"])
base = {key(r): r for r in rows if r["method"] == "ANC"}
rng = np.random.default_rng(0)


def stat(m, metric, scens, acts=("walk", "run", "bike")):
    per = {}
    for r in rows:
        if r["method"] == m and r["scen"] in scens and r["act"] in acts:
            a, b = r[metric], base[key(r)][metric]
            if np.isfinite(a - b):
                per.setdefault(r["seed"], []).append(a - b)
    x = np.array([np.mean(v) for v in per.values()])
    idx = rng.integers(0, len(x), (5000, len(x)))
    return x.mean(), *np.percentile(x[idx].mean(1), [2.5, 97.5])


def level(m, metric, scens, acts=("walk", "run", "bike")):
    return np.nanmean([r[metric] for r in rows if r["method"] == m and r["scen"] in scens and r["act"] in acts])


for metric in ("mae", "mae_zone", "rmse_zone", "f1_zone"):
    print(f"\n=== {metric}: ANC level, then change vs ANC [95% CI] ===")
    for g, sc in GROUPS.items():
        if metric != "mae" and g == "control":
            continue
        print(f"{g:13s} ANC {level('ANC', metric, sc):6.3f}")
        for m in METHODS:
            v = stat(m, metric, sc)
            print(f"   {m:18s} {v[0]:+7.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
print("\n=== by activity, hold+aware vs ANC, mae_zone in collisions and phase-locked ===")
sc = GROUPS["collisions"] + GROUPS["phase-locked"]
for a in ("walk", "run", "bike"):
    v = stat("hold+aware", "mae_zone", sc, (a,))
    v2 = stat("aware-only", "mae_zone", sc, (a,))
    print(f"{a:5s} ANC {level('ANC','mae_zone',sc,(a,)):5.2f}  hold+aware {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]  aware-only {v2[0]:+.3f} [{v2[1]:+.3f}, {v2[2]:+.3f}]")
print("\n=== by scenario, mae (whole record) and mae_zone ===")
for s in ("crossing", "dwell20", "dwell40", "lock20", "lock40", "near+8", "near-8", "near+12", "near-12", "control"):
    q = lambda m, k: level(m, k, (s,))
    print(f"{s:9s} " + " ".join(f"{m}: {q(m,'mae'):.2f}/{q(m,'mae_zone'):.2f}" for m in ("ANC", "aware-only", "hold", "hold+aware")) + f"  hold% {100*q('hold+aware','hold_frac'):.1f}")
print("\nworst records, hold+aware minus ANC whole-record MAE")
d = sorted(((r["mae"] - base[key(r)]["mae"], r["scen"], r["act"], int(r["seed"])) for r in rows if r["method"] == "hold+aware"), reverse=True)
print([ (round(x[0], 1), *x[1:]) for x in d[:6]], "share worse by >1 BPM:", round(np.mean([x[0] > 1 for x in d]), 3), "better by >1:", round(np.mean([x[0] < -1 for x in d]), 3))
print("records", len(base))
