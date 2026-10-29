## T1. HR MAE (BPM) per activity, held-out subjects, windows pooled

| split | activity | raw | NLMS-1 | NLMS-all | RLS-all |
|---|---|---|---|---|---|
| subject | high_resistance_bike | 45.7 | 38.9 | 29.8 | 32.9 |
| subject | low_resistance_bike | 30.5 | 35.0 | 18.2 | 30.0 |
| subject | run | 39.6 | 63.5 | 51.3 | 51.0 |
| subject | walk | 41.2 | 42.2 | 48.9 | 39.0 |
| subject | all (pooled) | 38.9 | 44.4 | 37.7 | 38.1 |
| subject+activity | high_resistance_bike | 45.7 | 28.3 | 8.9 | 22.9 |
| subject+activity | low_resistance_bike | 30.5 | 18.3 | 30.6 | 16.6 |
| subject+activity | run | 39.6 | 43.5 | 25.5 | 22.9 |
| subject+activity | walk | 41.2 | 69.1 | 51.7 | 28.5 |
| subject+activity | all (pooled) | 38.9 | 42.4 | 32.3 | 23.1 |

## T2. HR MAE (BPM) per held-out subject (split=subject)

| subject | raw | NLMS-1 | NLMS-all | RLS-all |
|---|---|---|---|---|
| s1 | 14.1 | 26.6 | 19.9 | 21.3 |
| s2 | 51.2 | 59.5 | 42.5 | 43.3 |
| s3 | 41.2 | 36.5 | 38.9 | 39.6 |
| s4 | 85.6 | 64.8 | 32.5 | 53.1 |
| s5 | 51.4 | 38.6 | 40.6 | 54.3 |
| s6 | 22.7 | 36.2 | 34.1 | 29.8 |
| s8 | 80.3 | 69.1 | 88.9 | 44.5 |
| s9 | 4.4 | 94.3 | 4.0 | 92.2 |

## T3. Real-data metrics (split=subject)

| method | HR MAE (BPM) | HR MAE, mean of subjects | spectral-peak err (BPM) | windows within 10 BPM (%) | SNR gain (dB) | pulse ratio (dB) | convergence (s) | multiplies / sample | runtime / signal-s (ms) |
|---|---|---|---|---|---|---|---|---|---|
| raw | 38.90 | 43.86 | 44.38 | 27.29 | 0.00 | 0.00 | 1.19 | 0.00 | 0.38 |
| NLMS-1 | 44.36 | 53.22 | 49.00 | 16.75 | -0.95 | -9.07 | 3.53 | 98.00 | 0.73 |
| NLMS-all | 37.74 | 37.67 | 28.44 | 34.70 | 2.67 | -2.32 | 6.06 | 194.00 | 0.76 |
| RLS-all | 38.15 | 47.25 | 22.22 | 38.55 | 2.97 | -3.40 | 0.00 | 66048.00 | 7.38 |

## T4. Semi-synthetic benchmark (known pulse, E13)

| method | NMSE before (dB) | NMSE after (dB) | correlation | HR MAE (BPM) | SNR gain (dB) | pulse ratio (dB) | convergence (s) |
|---|---|---|---|---|---|---|---|
| raw | 6.11 | 6.11 | 0.45 | 36.75 | 0.00 | 0.00 | 0.00 |
| NLMS-1 | 6.11 | 2.97 | 0.22 | 50.58 | -3.14 | -11.20 | 0.00 |
| NLMS-all | 6.11 | -5.02 | 0.84 | 5.00 | 7.51 | -1.96 | 32.92 |
| RLS-all | 6.11 | -7.85 | 0.89 | 2.86 | 8.14 | -2.80 | 10.06 |

## T5. Robustness: HR MAE (BPM), LOSO configs, no retuning

| condition | NLMS-1 | NLMS-all | RLS-all |
|---|---|---|---|
| baseline | 44.4 | 37.7 | 38.1 |
| rotation, mean of 10 | 46.9 | 30.1 | 36.2 |
| rotation, min | 38.3 | 27.3 | 33.2 |
| rotation, max | 57.9 | 36.4 | 39.4 |
| zero_acc_x | 42.6 | 31.8 | 36.1 |
| zero_acc_y | 49.7 | 40.9 | 29.0 |
| zero_acc_z | 40.5 | 26.3 | 36.1 |
| no_gyro | 44.4 | 32.6 | 39.4 |

## T6. Channel ablation: change in HR MAE when one reference is dropped

| dropped channel (dMAE, BPM) | NLMS-all | RLS-all |
|---|---|---|
| acc_mag | -4.4 | -6.1 |
| acc_x | -6.4 | -4.3 |
| acc_y | -8.0 | -5.8 |
| acc_z | -12.0 | -0.4 |
| gyro_mag | -3.8 | -5.6 |
| gyro_x | -6.3 | -10.8 |
| gyro_y | -1.4 | -8.2 |
| gyro_z | +0.8 | -2.6 |

## T7. Failure cases (E7)

| % of windows | ok | cadence≈HR | other |
|---|---|---|---|
| raw | 27.3 | 16.2 | 56.5 |
| NLMS-1 | 16.7 | 24.4 | 58.9 |
| NLMS-all | 34.7 | 19.5 | 45.8 |
| RLS-all | 38.6 | 22.9 | 38.6 |

## T8. Severe motion (E15)

| motion >= 2.73 m/s^2 (top 10 %) | raw | NLMS-1 | NLMS-all | RLS-all |
|---|---|---|---|---|
| MAE, severe motion | 48.6 | 60.9 | 44.9 | 41.5 |
| MAE, other windows | 37.8 | 42.5 | 36.9 | 37.8 |

## T9. LOSO-chosen config per held-out subject (split=subject)

| held_out | raw | NLMS-1 | NLMS-all | RLS-all |
|---|---|---|---|---|
| s1 | - | L=32,mu=0.5 | L=8,mu=0.01 | L=16,lam=0.999 |
| s2 | - | L=8,mu=0.01 | L=8,mu=0.01 | L=16,lam=0.999 |
| s3 | - | L=32,mu=0.5 | L=8,mu=0.01 | L=16,lam=0.999 |
| s4 | - | L=32,mu=0.5 | L=8,mu=0.01 | L=32,lam=0.999 |
| s5 | - | L=32,mu=0.5 | L=8,mu=0.01 | L=16,lam=0.999 |
| s6 | - | L=32,mu=0.5 | L=8,mu=0.01 | L=16,lam=0.999 |
| s8 | - | L=32,mu=0.5 | L=32,mu=0.01 | L=32,lam=0.999 |
| s9 | - | L=16,mu=0.1 | L=8,mu=0.01 | L=8,lam=0.995 |
