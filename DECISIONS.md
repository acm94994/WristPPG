# Project 03 – Decisions log

Every choice that affects results goes here, the moment it's made.
Section 5 of the report (Method and experimental setup) is written from this file.

Status: `proposed` = default not yet confirmed, `decided` = locked in, `changed` = revised (keep the old value in Notes).
Only A changes status to `decided`. Anyone can add a row.

## Data and preprocessing

| # | Decision | Value | Status | Decided by / date | Notes |
|---|---|---|---|---|---|
| D1 | Dataset | Wrist PPG During Exercise (PhysioNet), all 19 records, 8 subjects | decided | A | Only allowed dataset with accel + gyro + ECG R peaks |
| D2 | Accelerometer | ±16 g wide-range | decided | A, 28 Oct | ±2 g can saturate when running |
| D3 | Sampling rate | Downsample 256 → 64 Hz (`resample_poly(x, 1, 4)`) | decided | A, 28 Oct | HR band < 4 Hz |
| D4 | Pre-filter | 4th-order Butterworth band-pass **0.4–4 Hz**, zero-phase `filtfilt`, on PPG and all references | decided | A, 28 Oct | Same filter type and band as Lee et al., CMPB 2022 (cite it). Offline (non-causal); state this in the report |
| D5 | Reference channels | 8: acc x/y/z, gyro x/y/z, ‖a‖, ‖ω‖; each z-scored per record | decided | A, 28 Oct | |
| D6 | NaN handling | Trim trailing NaN padding before anything else | proposed | | Dataset pads Shimmer signals to ECG length |

## Methods

| # | Decision | Value | Status | Decided by / date | Notes |
|---|---|---|---|---|---|
| M1 | Axis for NLMS-1 | Same forearm (longitudinal) axis as R1b | proposed | | Keeps NLMS-1 comparable to Han & Kim's main reference; IMU only |
| M2 | NLMS regularisation δ | 1e-6 | proposed | | |
| M3 | RLS forgetting factor λ, P init | λ = 0.999 (memory ≈ 1/(1−λ) = 1000 samples ≈ 16 s at 64 Hz), P = I / 0.01; tune λ ∈ {0.998, 0.999} | changed | A, 29 Oct | Was λ = 0.99 (1.6 s memory): on the walking simulation it cancelled the pulse (MAE 28 BPM vs 1.4 at 0.999), same mechanism as a too-large μ |
| M4 | MSC estimation | `scipy.signal.coherence`, 8 s window, nperseg = 256, noverlap = 192, mean over 0.5–4 Hz | proposed | | Computed vs raw PPG, never ECG |
| M5 | MSC threshold τ | Tuned over {0, 0.2, 0.3, 0.5} on training subjects | proposed | | |
| M6 | Weight update rate | MSC weights recomputed every 2 s step, held constant within it | proposed | | All 8 channels always in the regressor |
| M7 | Variable step size | μ_window = μ × mean(MSC weights of that window) | proposed | | Brief's "step-size control"; keep only if it doesn't hurt on training subjects |
| M8 | Filter order L | Tuned over {8, 16, 32}; fixed per held-out subject | proposed | | Order is tuned, not adaptive; say so in the report |
| M9 | RLS regularization | `filters.rls` adds ρ·u uᵀ to R = P⁻¹ each sample, u cycling through the unit vectors, so R stays above ≈ β/(1−λ)·I; β = 1e-2. λ ∈ {0.998, 0.999} and L ∈ {8, 16, 32} tuned by LOSO (M3, M8) | proposed | B, 30 Oct | Plain RLS winds up on band-passed, correlated delayed refs: s1_walk |e| up to 2.5e5 at all 6 settings, blow-up after ≈ 200 s at λ = 0.998 and ≈ 400 s at 0.999 (P grows as λ⁻ⁿ in unexcited directions). s3_run (307 s) only survives λ = 0.999 because it ends first. Regularized: stable everywhere, s1_walk 2.2 BPM, s3_run 1.7–1.8 BPM; same as plain RLS where plain is stable. Cost unchanged, O(M²) |

## Reproduction (section 6.1)

| # | Decision | Value | Status | Decided by / date | Notes |
|---|---|---|---|---|---|
| R1 | Han & Kim LMS settings ("LMS-HK") | Paper: LMS, 2 references (longitudinal axis + radial = √(a_y² + a_z²)), third-order filter, μ = 0.009, finger sensor, 20 Hz. **Ours:** resample to 20 Hz for this run only, 2 references as in the paper with the wrist's forearm axis as "longitudinal", L = 3 taps, μ = 0.009 on z-scored references | proposed | A, 28 Oct | Differences to state: wrist not finger; "third-order" read as 3 taps; μ not directly transferable because signal scaling differs |
| R1b | Which axis is "longitudinal" (forearm axis) | Accel axis with the largest mean absolute value **before** band-pass filtering (gravity lies mostly along the forearm when the arm hangs) | proposed | | IMU-only rule; check visually on one record |
| R2 | Fallet & Vesin NLMS settings ("NLMS-FV") | Paper: NLMS, 70 taps, μ = 0.1, γ small (value not reported), signals resampled to 35 Hz, one NLMS per (PPG channel, accel axis) pair. **Ours:** resample to 35 Hz for this run only, 3 parallel NLMS (one per accel axis), L = 70, μ = 0.1, γ = 1e-6. HR from the average of the 3 outputs' power spectra, then our standard tracker | proposed | A, 28 Oct | Differences to state: one PPG channel instead of two; γ not reported; their adaptive band-pass frequency tracker replaced by spectrum averaging + our tracker |

## HR estimation and evaluation

| # | Decision | Value | Status | Decided by / date | Notes |
|---|---|---|---|---|---|
| E1 | Windows | 8 s, 2 s step | decided | A, 28 Oct | Standard; cite Fallet & Vesin 2015/2017 (8 s, 6 s overlap) and Lee et al. 2022. Same for every method |
| E2 | HR tracker | Per 8 s window: normalised PPG power spectrum (Hann, nfft 2048); harmonic sum P(f) + 0.5·P(2f); bins within ±10 BPM of {½, 1, 3/2, 2, 3} × the dominant IMU frequency multiplied by 0.1 (down-weighted, not removed). HR path chosen by **Viterbi** over the whole record: Gaussian jump penalty σ = 5 BPM per 2 s step, hard limit ±15 BPM. Same tracker for every method | changed | A, 29 Oct | v1 (plain greedy peak tracking) locked onto the 2F walking artifact (~172 BPM) on s3_walk. v2 (greedy + motion-peak exclusion) fixed s3_run (raw 12 → 3 BPM) but still got stuck on s3_walk even though NLMS-all exposed the pulse at ~98. Viterbi follows Temko 2017 (WFPV + Viterbi decoding); motion harmonics follow Lee 2022 (F, 2F, 3F, 4F). Offline (non-causal), state in report. PPG + IMU only, never ECG. `mode="greedy"` kept for comparison |
| E3 | Ground-truth HR | 60 / mean RR of R peaks inside each window | proposed | | Only in score.py |
| E4 | Tuning protocol | Leave-one-subject-out: grid μ ∈ {0.01, 0.05, 0.1, 0.5} × L ∈ {8, 16, 32} × τ on 7 subjects, score held-out subject once | decided | A, 28 Oct | Splits by subject only, never by window |
| E5 | Main metric | Mean absolute error (BPM) per activity, pooled over held-out subjects | proposed | | |
| E6 | Waveform metric | Power within ±0.1 Hz of true HR after / before filtering | proposed | | |
| E7 | Failure cases | Window error > 10 BPM; "cadence ≈ HR" if IMU dominant frequency within 0.1 Hz of true HR or its half; else "low MSC" or "other" | proposed | | |
| E8 | Robustness | 10 random rotations of accel and gyro triplets; one accel axis zeroed; gyro + ‖ω‖ removed | proposed | | Reduce to 3 rotations if short on time; don't drop |

## Observations

- 29 Oct, s3_run / s3_walk: NLMS-1 with μ = 0.1 made HR worse (run 12 → 49 BPM). On a walking-like simulation μ = 0.1 cancels the pulse when HR is near the arm-swing frequency; μ ≤ 0.03 fixes it. demo/diag now use μ = 0.01; LOSO tunes μ properly.
- 29 Oct: walking PPG's strongest peak is the 2F bounce artifact, which a single forearm-axis reference cannot model; NLMS-all (which includes ‖a‖) removes it in simulation. Supports the multi-reference argument; check on real data.

- 29 Oct: all 7 methods run end-to-end through src/methods.py:run_method on a walking simulation (MAE 0.9–1.6 BPM except RLS at λ = 0.99). CW-NLMS ≈ 1 s per 5-min record, RLS ≈ 1 s.
- 29 Oct, simulations with buried pulse: Viterbi rescues NLMS-all / CW-NLMS on walk-like data (≈95 → 0.5 BPM); raw and NLMS-1 still fail there because the single forearm axis cannot model the 2F bounce, so the pulse stays buried. Good illustration for the report.

## Cut order if behind schedule

1. Fewer random rotations (10 → 3)
2. Skip tuning τ (fix at 0.3)
3. Skip variable step size (report fixed μ, say so)
