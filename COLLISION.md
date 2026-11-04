# Collision-aware adaptation

Problem: near a motion line, NLMS acts as a notch about 4.77 * mu * fs BPM wide (half-width), so it
removes pulse when HR is close to cadence. Holding the weights, or cutting mu, removes that notch.

    pass 1:  NLMS -> tracked HR, and cancelled lines from y = d - e
             hold when |HR - line| < 12 BPM
    pass 2:  NLMS with mu * 0.1 (or weights held) inside the hold
    tracker: optional, stop down-weighting motion bins inside the hold  <- risky, see below

Files: `src/collision.py`, `src/loso.py`, `sim/collision_sim.py`, `run_collision_sim.py`, `analyze_collision.py`,
`run_collision.py`, `make_collision_figure.py`. It is offline because it uses two passes.

Why the gate reads y and not the IMU peak: the dominant IMU frequency was sometimes the wrong
harmonic, while y shows where the notches really are. The gate must also be wider than the tracker's
+/-10 BPM down-weighting, which pushes the tracked HR away from cadence. At 20 BPM it did harm.

## Prior work (quick search, not a novelty check)
No PPG paper or patent found that gates ANC adaptation on the gap between HR and motion frequency.
- Yang et al., ASNC: targets close frequencies, using one spectral fit with no gap rule.
- US 10772567: adaptive motion cancellation with an L1 penalty against over-cancelling, no gating.
- US 10631740: crossover window around step rate, but it revises the HR value afterwards.
- Echo cancellation freezes adaptation when the wanted signal is present, so the principle is old.
Not read in full: Fallet & Vesin 2017, the IEEE paper on periodic artifacts (10.1109/TBME.2016.2553060)
and the variable-step-size PPG papers. Read those before claiming anything is new.

## Simulation (1200 test records, 40 seeds x 3 activities x 10 scenarios, settings locked on dev seeds)
Change vs NLMS, mean [95% CI over seeds].

| Method | Collision zone HR (BPM) | Phase-locked zone HR | Near-miss zone HR | Beat F1, collision | Beat F1, phase-locked |
|---|---|---|---|---|---|
| NLMS level | 3.19 | 4.59 | 1.99 | 0.955 | 0.890 |
| Tracker change only | -2.03 [-2.21, -1.84] | -2.20 [-2.50, -1.89] | -1.04 [-1.14, -0.93] | 0.000 | 0.000 |
| Hold only | -0.87 [-1.10, -0.65] | -2.14 [-2.36, -1.92] | -0.06 [-0.07, -0.05] | +0.029 | +0.090 |
| Mu cut + tracker | -2.40 [-2.68, -2.14] | -3.57 [-3.89, -3.24] | -1.05 [-1.15, -0.94] | +0.033 | +0.096 |
| Hold + tracker | -2.39 [-2.68, -2.12] | -3.51 [-4.00, -2.92] | -0.99 [-1.11, -0.85] | +0.029 | +0.098 |
| Hold + tracker, true HR in gate | -2.59 [-2.89, -2.31] | -3.99 [-4.43, -3.55] | -1.34 [-1.46, -1.22] | +0.031 | +0.102 |

Control records: +0.007 BPM [0.000, +0.022], with holds on 0.1% of the time.
Averaged weights helped waveforms on near misses: error rose +0.012 versus +0.030 with latest weights.

## Stress tests that decide the recommendation
With motion the references cannot predict (20 seeds, walk + run, change vs NLMS):

| Unpredictable motion | Zone, hold + tracker | Near miss whole record, hold + tracker | Near miss whole record, hold only |
|---|---|---|---|
| 0 | -2.63 [-3.28, -2.11] | -0.16 [-0.47, +0.40] | +0.07 [-0.02, +0.23] |
| 0.5 | -5.00 [-5.66, -4.34] | +0.01 [-0.49, +0.63] | +0.16 [-0.25, +0.73] |
| 1 | -3.92 [-4.73, -3.16] | +1.48 [+0.45, +2.62] | +0.21 [-0.12, +0.58] |
| 2 | -3.78 [-4.42, -3.16] | +1.33 [+0.35, +2.31] | +0.10 [-0.35, +0.59] |

At 1 or above, the zone looks better but near-miss records get worse overall, and collision records
show no whole-record gain. NLMS error here is about 10-21 BPM, which is closer to real running.

Coupling drift, waveform error in the zone, NLMS -> hold + tracker:
0% 0.147 -> 0.071, 25% 0.172 -> 0.165, 50% 0.223 -> 0.295.

Failure seen: a wrong early HR tripped the gate, and the tracker then sat on cadence (103 vs true 70 BPM).
Of 1200 test records, 0.7% got more than 1 BPM worse and 13.2% got more than 1 BPM better.

Short records that start near cadence are also unsafe, because weights have not converged.

## Real data result (19 records, 8 subjects, held-out NLMS-all settings, gate locked in simulation)
Verdict: fails the decision rule. Keep the census as a finding, not the method.

Census: 19% of time within 6 BPM of a motion line, 42% within 15 BPM, concentrated in s9_walk 75%,
s4_run 73%, s2_walk 67%, s6_walk 52%, s6_low 34%, s3_low 29%. Longest run 58 s. Walk 33%, run 18%,
low bike 14%, high bike 4%. Across records it does not explain error: Spearman rho 0.14, p = 0.57.

Change vs NLMS-all, 8 subjects, mean [95% bootstrap CI]:
| Method | Whole-record MAE (23.90) | Zone MAE (26.08) | Zone beat F1 (0.332) |
|---|---|---|---|
| Mu cut | +0.09 [-0.57, +0.76] | +0.21 [-0.49, +0.89] | +0.005 [-0.014, +0.022] |
| Hold | +0.46 [-0.55, +1.62] | +0.58 [-0.63, +1.86] | -0.001 [-0.020, +0.017] |
| Tracker change only | -0.25 [-0.82, +0.32] | -0.48 [-1.63, +0.53] | +0.001 [0.000, +0.001] |
| Mu cut + tracker | +0.04 [-0.78, +0.88] | -0.27 [-1.43, +0.89] | +0.009 [-0.007, +0.024] |
| Hold + tracker | -4.84 [-14.49, +0.43] | -5.18 [-14.80, +0.55] | +0.008 [-0.012, +0.024] |

Hold + tracker is one record: s9_walk 96.4 -> 58.6. Without s9_walk every method is within about 0.2 BPM
(NLMS 19.88, hold + tracker 19.77). It helped 4 of 8 subjects. It is not a finding, only a lead on s9_walk.

Why it fails: held weights go stale at once. Epsilon is about 0.55 at the tuned settings and 0.33-0.48 at
mu = 0.01, against 0.05-0.2 in simulation. The slowly drifting coupling the method assumes is not there.
Exploratory fixed mu = 0.01: no method beat NLMS either (18.83; mu cut 18.67, hold + tracker 19.29).
That same run has NLMS at 18.83 against 23.90 with tuned mu. It was not tuned by LOSO, so it is not a
result, but the tuned setting is worth a second look.

Design fix found on real data, before scoring: lines come from a slow pilot NLMS (mu = 0.01). At mu = 0.5
the filter output contains the pulse, so s1_high_resistance_bike showed 57% collision time against 1% by IMU.

## Recommendation
Report the census and this negative result. Do not add collision gating to the main method.

## Run on real data
    python run_collision.py --data data --out results/collision --loso results/loso_params.csv

1. `census.csv`: how much time is within 6 and 15 BPM of a cancelled line, and for how long.
2. `holdtest.csv`: epsilon against hold time. Near a collision you need epsilon * M well below A.
3. `scores.csv`: compare methods. Judge the tracker change on whole-record MAE, not zone MAE.

Decision rule, fixed before real data (proposed thresholds):
- Keep mu reduction if beat F1 in zones rises with a subject-level CI above 0, and no activity's
  whole-record MAE worsens by more than 0.5 BPM.
- Keep the tracker change only if whole-record MAE improves with a subject-level CI below 0, and no
  record gets more than 2 BPM worse.
- If collisions take under 5% of the time, stop this line: it is not where the real error comes from.
