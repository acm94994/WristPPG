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
| M3 | RLS forgetting factor λ, P init | λ = 0.99, P = I / 0.01 | proposed | | |
| M4 | MSC estimation | `scipy.signal.coherence`, 8 s window, nperseg = 256, noverlap = 192, mean over 0.5–4 Hz | proposed | | Computed vs raw PPG, never ECG |
| M5 | MSC threshold τ | Tuned over {0, 0.2, 0.3, 0.5} on training subjects | proposed | | |
| M6 | Weight update rate | MSC weights recomputed every 2 s step, held constant within it | proposed | | All 8 channels always in the regressor |
| M7 | Variable step size | μ_window = μ × mean(MSC weights of that window) | proposed | | Brief's "step-size control"; keep only if it doesn't hurt on training subjects |
| M8 | Filter order L | Tuned over {8, 16, 32}; fixed per held-out subject | proposed | | Order is tuned, not adaptive; say so in the report |
| M9 | RLS-all | **Regularized** RLS on all 8 refs: P(0) = I / 0.01, each sample also adds ρ·u uᵀ to R with u cycling through the unit vectors, floor β/(1−λ), β = 1e-2. LOSO grid λ ∈ {0.99, 0.995, 0.999} × L ∈ {8, 16, 32} | proposed | B, 29 Oct | Plain RLS winds up on band-passed, correlated delayed refs (|e| up to 1e8 on s1_walk / s3_run). Replaces M3's fixed λ = 0.99 with tuning. Cost 4M² + 4M multiplies/sample |
| M10 | Reproductions in LOSO | LMS-HK and NLMS-FV run with fixed paper settings, not tuned | proposed | B, 29 Oct | They are reproductions; tuning them would no longer be the paper's method |
| M11 | "raw" row | HR from the band-passed PPG with no canceller, in every table | proposed | B, 29 Oct | Shows what each canceller adds |

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
| E2 | HR tracker | Hann window, FFT zero-padded to 2048; first window = largest peak in 0.5–3.5 Hz; then ±15 BPM around previous, fallback to global peak | decided | A, 28 Oct | Spectral-peak tracking around the previous estimate is standard (e.g. TROIKA, Zhang 2015); the ±15 BPM range is our own choice, state it as such. Starts from PPG, never from ECG |
| E3 | Ground-truth HR | 60 / mean RR of R peaks inside each window | proposed | | Only in score.py. Windows matched by centre time, so 20/35/64 Hz methods use the same windows |
| E4 | Tuning protocol | Leave-one-subject-out: grid μ ∈ {0.01, 0.05, 0.1, 0.5} × L ∈ {8, 16, 32} × τ on 7 subjects, score held-out subject once | decided | A, 28 Oct | Splits by subject only, never by window |
| E5 | Main metric | Mean absolute error (BPM) per activity, pooled over held-out subjects | proposed | | |
| E6 | Waveform metric | Two per-window metrics. **Pulse ratio**: power within ±0.1 Hz of true HR after / before filtering (dB; ≈ 0 = pulse kept). **SNR gain**: SNR after − before, SNR = power within ±0.1 Hz of HR and 2·HR / rest of 0.5–4 Hz (dB) | proposed | B, 29 Oct | The pulse ratio answers the brief's "without attenuating the pulsatile component"; SNR gain is the brief's "PPG SNR gain" |
| E7 | Failure cases | Window error > 10 BPM; "cadence ≈ HR" if IMU dominant frequency within 0.1 Hz of true HR or its half; else "low MSC" or "other" | proposed | | IMU dominant frequency = peak of the summed spectra of the 3 band-passed accel axes, 0.4–4 Hz (B). "low MSC" needs an `msc` column from CW-NLMS; until then those windows count as "other" |
| E8 | Robustness | 10 random rotations of accel and gyro triplets; one accel axis zeroed; gyro + ‖ω‖ removed | proposed | | Reduce to 3 rotations if short on time; don't drop. B: same rotation for accel and gyro (sensor re-mounted), applied to the raw record so ‖a‖, ‖ω‖ and the forearm axis are recomputed; split=subject LOSO configs, no retuning |
| E9 | LOSO objective | Config with the lowest mean over training subjects of each subject's window MAE | proposed | B, 29 Oct | Subject-averaged so long records don't dominate |
| E10 | Activity-specific split | Second LOSO: tune on the other subjects' records of the same activity only (`split = subject+activity`) | proposed | B, 29 Oct | Brief WP5 "subject-independent and activity-specific splits". Each activity has ≥ 3 subjects |
| E11 | Spectral-peak error | |largest peak of the cleaned PPG in 0.5–3.5 Hz, no tracking − true HR| (BPM) | proposed | B, 29 Oct | Separates canceller quality from tracker behaviour |
| E12 | Convergence time | First time the 2 s moving average of the squared error (dB) is within 1 dB of its steady level (median over the second half) | proposed | B, 29 Oct | On real data e² includes the pulse, so often ≈ 0 s; the semi-synthetic version uses (e − s)² and is the one to report |
| E13 | Waveform ground truth (NMSE, correlation) | Semi-synthetic records: 2-harmonic pulse following the real ECG RR intervals (+0.25 s delay) + real accel/gyro axes through random decaying 0.1 s FIRs at SIR −6 dB + white noise at 20 dB | proposed | B, 29 Oct | Real PPG has no clean reference. Artifact is linear in the IMU = best case for linear cancellers; state as a limitation. R peaks only build the ground truth, the methods never see them |
| E14 | Computational cost | Multiplies per sample (LMS 2M+1, NLMS 3M+2, RLS 4M²+4M, M = C·L) + measured runtime per second of signal | proposed | B, 29 Oct | |
| E15 | Severe motion | Windows in the top 10 % of std(‖a‖) (raw, m/s²), pooled over all records | proposed | B, 29 Oct | Brief WP6. Also shown as MAE per motion decile |
| E16 | Channel ablation | Leave one of the 8 references out, for multi-reference methods, split=subject LOSO configs | proposed | B, 29 Oct | Component ablation (MSC weighting, variable μ) needs CW-NLMS |

## Cut order if behind schedule

1. Fewer random rotations (10 → 3)
2. Skip tuning τ (fix at 0.3)
3. Skip variable step size (report fixed μ, say so)
