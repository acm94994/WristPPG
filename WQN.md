# IMU-gated WQN on the ANC residual

Target: short jerks that survive the canceller. It is not meant to solve cadence = HR.

    raw IMU -> jerk envelope -> robust z ----------.
                                                    v
    PPG + refs -> preprocess -> NLMS -> e[n] -> artifact gate -> reference selector -> SWT-WQN -> e_clean
                                          |                         ^
                                          '-> HR tracker + motion --'

| Stage | File | Rule |
|---|---|---|
| Jerk gate | `src/gate.py` | Raw acc + gyro, 1 s RMS so footstrikes average out, 30 s running median/MAD, z > k_imu |
| Residual check | `src/gate.py` | Hilbert envelope of e[n], z > k_res, kept only if an IMU flag is within 0.5 s |
| Reference | `src/gate.py` | 16 s of e[n] within 30 s, no flags, no cadence = HR, power concentrated near HR and 2HR |
| Fallback | `src/wqn_pipeline.py` | No valid reference: leave that interval untouched and log it |
| Correction | `src/wqn.py` | sym5 SWT level 6, per-scale quantile transport, magnitude never grows, tapered edges |
| Scoring | `src/extension_score.py` | The only new file that touches ECG |

## Run on real data
From the repo root:

    python -m tests.test_extensions
    python -m experiments.run_wqn --data data --out results/wqn --loso results/loso_params.csv --inject
    python -m experiments.run_wqn --data data --out results/wqn --loso results/loso_params.csv --grid

Outputs: `scores.csv` (HR, beats vs ECG), `events.csv` (each interval, reference, status, anc_gain),
`injected.csv` (known-truth error), `loso_picks.csv` (subject-independent k_imu, k_res).
`--loso` uses each subject's held-out NLMS-all settings.

## Action plan
1. Run the tests and the first command above on all 19 records.
2. Gate audit: open `events.csv`. Record events per minute by activity, share with `no_reference`,
   and hand-check about 20 intervals against the PPG and IMU.
3. Set k_imu and k_res with `--grid`. Do not hand-tune on real results.
4. Score the injections and ECG beats by subject.
5. Test the canceller finding: in intervals with anc_gain > 1, does ANC add energy at a jerk?
6. Apply the decision rule below.

## Decision rule (fixed before seeing real data)
Keep the method if all hold:
- injected-transient error falls by at least 25%, with a subject-level 95% CI below 0
- error outside injections rises by less than 5%
- beat F1 near events rises, with a subject-level CI above 0
- no activity's HR MAE worsens by more than 0.5 BPM

Stop and move to cadence = HR work if real events are rare (under 0.5 per minute),
or more than half of them have no valid reference.
Do not claim an HR improvement unless real ECG scoring shows one.

## Real data result (19 records, 8 subjects, held-out NLMS-all settings, gate locked in simulation)
Verdict: fails the decision rule, with a small real effect on injected transients.
- Injected error fell 9.2% [14.1, 4.7] by subject, better in 8 of 8, against the 25% needed.
- Error outside injections rose about 10% (0.115 to 0.127), against the 5% allowed.
- Only 55 of 127 injected transients were detected.
- Beat F1 near events +0.006 [-0.018, +0.038]. HR MAE +0.99 [-0.18, +3.18], so no HR benefit.
- 71 events in 111 min (0.64 per minute); 39% had no valid reference.
- ANC output exceeded its input in 15% of intervals, so the ANC-made-artifact finding is weak here.
- Freezing weights during jerks made injected error worse: +43% [+18, +68].

## Simulation result (120 synthetic test records, 40 seeds x 3 activities, locked on separate seeds)
Change vs NLMS alone, mean [95% CI over seeds]:

| Method | Error in transients | Beat F1 | HR MAE (BPM) |
|---|---|---|---|
| WQN, IMU + residual gate | -40.0% [-43.0, -36.8] | +0.022 [+0.015, +0.029] | -0.007 [-0.016, 0.000] |
| WQN, IMU gate | -37.0% [-39.9, -34.1] | +0.021 [+0.013, +0.032] | -0.005 [-0.015, +0.004] |
| WQN, residual gate | -53.6% [-55.0, -52.0] | +0.031 [+0.022, +0.041] | +0.001 [-0.015, +0.019] |
| WQN, perfect detection | -48.1% [-49.1, -47.1] | +0.028 [+0.021, +0.035] | -0.008 [-0.016, -0.002] |
| Freeze weights only | +7.4% [+6.2, +8.8] | -0.003 [-0.009, +0.003] | -0.005 [-0.023, +0.011] |
| Soft threshold, no gate | -15.7% [-17.2, -14.3] | -0.703 [-0.727, -0.677] | +94.4 [+93.3, +95.4] |

The residual gate wins here partly by also firing on real pulse rises: pulse gain fell by 0.23 there,
versus no change with the IMU + residual gate. The IMU + residual gate caught 97.8% of coupled jerks
and 0% of PPG-only transients.

## Limits of the simulation
- It is hand-written and includes traps that favour an IMU gate. It checks mechanics, not real performance.
- NLMS HR error is about 2.4 BPM here, far below about 20 BPM on real running, so it lacks real failure modes.
- Transients cost at most about 0.1 BPM here, so HR had little room to improve.
- The reference quality threshold never changed a result on dev seeds, so rejection is untested.
- k_imu barely mattered because simulated IMU bursts are very large.
- WQN keeps the artifact's phase. It shrinks transients but does not restore the beats underneath.
