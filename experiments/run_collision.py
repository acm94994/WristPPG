"""Collision-aware adaptation on the real Wrist PPG During Exercise records.

  python -m experiments.run_collision --data data --out results/collision --loso results/loso_params.csv
  python -m experiments.run_collision --data data --out results/collision_oracle --loso results/loso_params.csv --gate-hr ecg

Writes
  census.csv    how much time each record spends in a collision, and how long collisions last
  scores.csv    HR MAE overall and in collision zones, beat F1 in zones, by method
  holdtest.csv  how fast held weights go stale, away from collisions

ECG reaches only the census and scoring here, through src/extension_score.py. It never enters the
gate, the filter or the tracker, except with --gate-hr ecg: that gives the gate the true HR, as an
upper bound on what gating could do. It is not a usable method.
"""
import argparse, csv, json, os
import numpy as np
from src.collision import (CollisionConfig, hold_mask, mask_to_windows, nlms_hold, pilot_lines,
                           run_collision, track_aware, window_gap)
from src.filters import nlms
from src.hr import track_hr
from src.load import list_records, load_record
from src.loso import loso_params
from src.extension_score import beat_f1, beats, hr_mae, match_lag, ref_hr
from src.preprocess import preprocess

fs = 64
ZONE_BPM = 15


def line_power(x, lines_bpm, fs):
    """Power of x near the cancelled lines, 8 s Hann window, within 3 BPM of each line."""
    nfft = 2048
    P = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), nfft)) ** 2
    bpm = np.fft.rfftfreq(nfft, 1 / fs) * 60
    sel = np.zeros(len(P), bool)
    for l in lines_bpm:
        sel |= np.abs(bpm - l) <= 3
    return P[sel].sum()


def census(rec, t, true, lines, hr):
    gap = window_gap(np.nan_to_num(true, nan=1e6), lines)
    z = gap < ZONE_BPM
    near = gap < 6
    edges = np.diff(np.concatenate([[0], near.astype(int), [0]]))
    a, b = np.where(edges == 1)[0], np.where(edges == -1)[0]
    durs = (b - a) * 2.0
    return dict(record=rec["name"], activity=rec["activity"], minutes=len(t) * 2 / 60,
                frac_within_15=float(z.mean()), frac_within_6=float(near.mean()), n_within_6=len(durs),
                median_dur_s=float(np.median(durs)) if len(durs) else 0.0,
                max_dur_s=float(durs.max()) if len(durs) else 0.0,
                tracked_hr_zone=float(np.nanmean(hr[z])) if z.any() else np.nan,
                true_hr_zone=float(np.nanmean(true[z])) if z.any() else np.nan)


def holdtest(d, refs, e0, lines, t, gap, L, mu, cfg, holds=(2, 5, 10, 20, 40)):
    """Freeze at collision-free moments and watch leftover motion power grow with hold time.
    Output: leftover power on cancelled lines after holding, divided by adaptive NLMS and by input."""
    rows, N = [], len(d)
    win = 8 * fs
    starts = [i for i in range(0, len(t), 15) if gap[i] > 25 and t[i] > 30]
    for h in holds:
        ratio_a, ratio_d = [], []
        for i in starts:
            start = int(t[i] * fs)
            stop = start + (h + 8) * fs
            if stop >= N:
                continue
            hold = np.zeros(N, bool)
            hold[start:stop] = True
            e = nlms_hold(d[:stop], refs[:stop], hold[:stop], fs, L, mu, "avg", cfg)
            a = stop - win
            j = int(np.clip(np.searchsorted(t, (a + win / 2) / fs), 0, len(lines) - 1))
            if len(lines[j]) == 0:
                continue
            p = line_power(e[a:stop], lines[j], fs)
            ratio_a.append(p / (line_power(e0[a:stop], lines[j], fs) + 1e-12))
            ratio_d.append(p / (line_power(d[a:stop], lines[j], fs) + 1e-12))
        if ratio_d:
            rows.append(dict(hold_s=h, n=len(ratio_d), left_vs_adaptive=float(np.median(ratio_a)),
                             left_vs_input=float(np.median(ratio_d)), epsilon=float(np.sqrt(np.median(ratio_d)))))
    return rows


def write(rows, path):
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results/collision")
    p.add_argument("--records", default="")
    p.add_argument("--loso", default="", help="use each subject's held-out NLMS-all settings")
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--gate", type=float, default=12)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--gate-hr", choices=("tracked", "ecg"), default="tracked",
                   help="ecg: gate on the true HR, an upper bound only")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = CollisionConfig(gate_bpm=a.gate)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    names = a.records.split(",") if a.records else list_records(a.data)
    cen, sco, hol = [], [], []
    tuned = loso_params(a.loso) if a.loso else {}
    for name in names:
        rec = load_record(os.path.join(a.data, name))
        a.L, a.mu = tuned.get(rec["subject"], (a.L, a.mu))
        d, refs, _ = preprocess(rec)
        e0, _ = nlms(d, refs, L=a.L, mu=a.mu)
        h0, t = track_hr(e0, fs, motion=refs)
        lines = pilot_lines(d, refs, fs, a.L, a.mu, cfg, e0)[:len(t)]
        true = ref_hr(rec["rpeaks_s"], t)
        cen.append(census(rec, t, true, lines, h0))
        gap = window_gap(np.nan_to_num(true, nan=1e6), lines)
        zone = gap < ZONE_BPM
        oracle = np.nan_to_num(true, nan=1e6) if a.gate_hr == "ecg" else None
        hold0 = hold_mask(h0 if oracle is None else oracle, t, lines, len(d), fs, cfg)
        outs = {"ANC": (e0, h0, np.zeros(len(d), bool)),
                "aware-only": (e0, track_aware(e0, fs, refs, mask_to_windows(hold0, t, fs))[0], hold0)}
        e1 = nlms_hold(d, refs, hold0, fs, a.L, a.mu, "avg", cfg)
        outs["hold"] = (e1, track_hr(e1, fs, motion=refs)[0], hold0)
        o = run_collision(d, refs, fs, a.L, a.mu, cfg, "avg", a.passes, true_hr=oracle)
        outs["hold+aware"] = (o["e"], o["hr"], o["hold"])
        o2 = run_collision(d, refs, fs, a.L, a.mu, cfg, "mu", a.passes, true_hr=oracle)
        e2 = nlms_hold(d, refs, hold0, fs, a.L, a.mu, "mu", cfg)
        outs["mu"] = (e2, track_hr(e2, fs, motion=refs)[0], hold0)
        outs["mu+aware"] = (o2["e"], o2["hr"], o2["hold"])
        zn = np.zeros(len(d), bool)
        for c, z in zip(t, zone):
            if z:
                zn[max(0, int((c - 1) * fs)):int((c + 1) * fs)] = True
        for m, (e, hr, hold) in outs.items():
            est, ref = beats(e, fs, hr, t), rec["rpeaks_s"]
            est = est - match_lag(est, ref)
            sel = lambda b: b[zn[np.clip((b * fs).astype(int), 0, len(d) - 1)]]
            sco.append(dict(record=rec["name"], subject=rec["subject"], activity=rec["activity"], method=m, mu=a.mu,
                            mae=hr_mae(hr, t, ref), mae_zone=hr_mae(hr, t, ref, zone) if zone.any() else np.nan,
                            f1_zone=beat_f1(sel(est), sel(ref)) if zone.any() else np.nan, hold_frac=float(hold.mean())))
        if oracle is None:
            hol += [dict(record=rec["name"], activity=rec["activity"], **r) for r in holdtest(d, refs, e0, lines, t, gap, a.L, a.mu, cfg)]
        print(name, f"within 15 BPM: {100 * cen[-1]['frac_within_15']:.0f}% of time", flush=True)
    write(cen, os.path.join(a.out, "census.csv"))
    write(sco, os.path.join(a.out, "scores.csv"))
    write(hol, os.path.join(a.out, "holdtest.csv"))
    print(f"\n{'method':12s}{'mae':>8s}{'mae_zone':>10s}{'f1_zone':>9s}{'hold%':>8s}")
    for m in ("ANC", "aware-only", "hold", "mu", "hold+aware", "mu+aware"):
        q = [r for r in sco if r["method"] == m]
        print(f"{m:12s}{np.nanmean([r['mae'] for r in q]):8.2f}{np.nanmean([r['mae_zone'] for r in q]):10.2f}"
              f"{np.nanmean([r['f1_zone'] for r in q]):9.3f}{100 * np.mean([r['hold_frac'] for r in q]):8.1f}")
    print("\nheld weights, median over records")
    for h in sorted({r["hold_s"] for r in hol}):
        q = [r for r in hol if r["hold_s"] == h]
        print(f"  hold {h:2d} s: leftover vs adaptive {np.median([r['left_vs_adaptive'] for r in q]):6.2f}   epsilon {np.median([r['epsilon'] for r in q]):.2f}")


if __name__ == "__main__":
    main()
