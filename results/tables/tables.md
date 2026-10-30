## T1. HR MAE (BPM) per activity, held-out subjects, windows pooled

| split | activity | RAW | LMS-HK | NLMS-FV | NLMS-1 | NLMS-all | RLS-all | CW-NLMS |
|---|---|---|---|---|---|---|---|---|
| subject | high_resistance_bike | 30.3 | 10.5 | 6.8 | 4.4 | 5.8 | 3.4 | 4.4 |
| subject | low_resistance_bike | 13.1 | 30.8 | 13.4 | 11.8 | 20.2 | 17.5 | 12.7 |
| subject | run | 52.4 | 45.3 | 37.3 | 70.4 | 21.5 | 22.8 | 21.9 |
| subject | walk | 46.7 | 21.0 | 38.1 | 21.9 | 28.4 | 14.2 | 27.0 |
| subject | all (pooled) | 36.1 | 26.9 | 25.5 | 26.6 | 20.5 | 14.9 | 17.8 |
| subject+activity | high_resistance_bike | 30.3 | 10.5 | 6.8 | 3.9 | 3.3 | 3.6 | 3.3 |
| subject+activity | low_resistance_bike | 13.1 | 30.8 | 13.4 | 6.3 | 20.7 | 14.4 | 13.3 |
| subject+activity | run | 52.4 | 45.3 | 37.3 | 66.4 | 12.6 | 22.8 | 22.0 |
| subject+activity | walk | 46.7 | 21.0 | 38.1 | 20.3 | 28.4 | 14.0 | 39.3 |
| subject+activity | all (pooled) | 36.1 | 26.9 | 25.5 | 23.6 | 18.2 | 14.1 | 21.9 |

## T2. HR MAE (BPM) per held-out subject (split=subject)

| subject | RAW | LMS-HK | NLMS-FV | NLMS-1 | NLMS-all | RLS-all | CW-NLMS |
|---|---|---|---|---|---|---|---|
| s1 | 6.0 | 3.1 | 6.1 | 2.7 | 7.0 | 2.7 | 3.8 |
| s2 | 34.9 | 42.3 | 10.1 | 11.2 | 11.1 | 8.8 | 15.7 |
| s3 | 27.3 | 7.2 | 6.5 | 26.0 | 12.5 | 7.1 | 7.3 |
| s4 | 85.4 | 86.6 | 76.4 | 93.7 | 25.8 | 64.0 | 24.5 |
| s5 | 52.7 | 43.3 | 30.3 | 44.3 | 46.9 | 45.5 | 30.2 |
| s6 | 16.5 | 15.6 | 14.5 | 20.6 | 22.9 | 14.5 | 22.4 |
| s8 | 102.7 | 50.2 | 99.7 | 58.0 | 32.4 | 25.5 | 30.5 |
| s9 | 95.1 | 94.9 | 98.5 | 94.9 | 96.4 | 8.8 | 98.4 |

## T3. Real-data metrics (split=subject)

| method | HR MAE (BPM) | HR MAE, mean of subjects | spectral-peak err (BPM) | windows within 10 BPM (%) | SNR gain (dB) | pulse ratio (dB) | convergence (s) | multiplies / sample | runtime / signal-s (ms) |
|---|---|---|---|---|---|---|---|---|---|
| RAW | 36.07 | 52.58 | 44.38 | 39.40 | 0.00 | 0.00 | 1.19 | 0.00 | 1.60 |
| LMS-HK | 26.89 | 42.88 | 38.88 | 50.21 | 0.80 | -0.57 | 5.70 | 13.00 | 2.43 |
| NLMS-FV | 25.51 | 42.75 | 36.69 | 53.83 | 2.06 | -3.06 | 18.94 | 636.00 | 1.51 |
| NLMS-1 | 26.58 | 43.92 | 37.88 | 50.99 | 1.04 | -2.89 | 3.67 | 98.00 | 1.14 |
| NLMS-all | 20.45 | 31.87 | 37.53 | 45.18 | 0.78 | -10.82 | 4.34 | 386.00 | 1.24 |
| RLS-all | 14.89 | 22.11 | 21.90 | 62.38 | 3.01 | -2.90 | 0.00 | 16640.00 | 3.89 |
| CW-NLMS | 17.83 | 29.11 | 34.75 | 53.34 | 1.53 | -7.74 | 7.88 | 394.00 | 8.61 |

## T4. Semi-synthetic benchmark (known pulse, E13)

| method | NMSE before (dB) | NMSE after (dB) | correlation | HR MAE (BPM) | SNR gain (dB) | pulse ratio (dB) | convergence (s) |
|---|---|---|---|---|---|---|---|
| RAW | 6.11 | 6.11 | 0.45 | 15.86 | 0.00 | 0.00 | 0.00 |
| LMS-HK | 6.12 | 4.30 | 0.52 | 10.70 | 2.71 | -1.66 | 2.10 |
| NLMS-FV | 6.11 | -0.02 | 0.61 | 11.70 | 4.20 | -2.99 | 1.23 |
| NLMS-1 | 6.11 | 2.71 | 0.46 | 18.17 | 1.40 | -3.61 | 0.23 |
| NLMS-all | 6.11 | -1.06 | 0.40 | 22.29 | 2.22 | -12.14 | 0.00 |
| RLS-all | 6.11 | -9.28 | 0.91 | 7.57 | 8.55 | -2.40 | 9.75 |
| CW-NLMS | 6.11 | -1.52 | 0.51 | 21.78 | 3.55 | -8.91 | 0.38 |

## T5. Robustness: HR MAE (BPM) per record, LOSO configs, no retuning

| condition | CW-NLMS, mean | NLMS-all, mean | CW-NLMS, median | NLMS-all, median |
|---|---|---|---|---|
| original | 21.1 | 23.9 | 18.1 | 23.5 |
| rotated | 21.9 | 23.5 | 19.8 | 21.8 |
| forearm_zeroed | 34.2 | 36.4 | 20.9 | 23.2 |
| no_gyro | 20.9 | 26.3 | 14.2 | 25.3 |

## T7. Failure cases (E7)

| % of windows | ok | cadence≈HR | low MSC | other |
|---|---|---|---|---|
| RAW | 39.4 | 23.1 | 0.0 | 37.5 |
| LMS-HK | 50.2 | 20.2 | 0.0 | 29.6 |
| NLMS-FV | 53.8 | 20.1 | 0.0 | 26.1 |
| NLMS-1 | 51.0 | 21.8 | 0.0 | 27.2 |
| NLMS-all | 45.2 | 25.1 | 0.0 | 29.8 |
| RLS-all | 62.4 | 17.6 | 0.0 | 20.0 |
| CW-NLMS | 53.3 | 22.2 | 0.1 | 24.4 |

## T8. Severe motion (E15)

| motion >= 2.73 m/s^2 (top 10 %) | RAW | LMS-HK | NLMS-FV | NLMS-1 | NLMS-all | RLS-all | CW-NLMS |
|---|---|---|---|---|---|---|---|
| MAE, severe motion | 47.7 | 46.0 | 38.8 | 62.9 | 22.0 | 33.6 | 20.0 |
| MAE, other windows | 34.8 | 24.8 | 24.0 | 22.6 | 20.3 | 12.8 | 17.6 |

## T9. LOSO-chosen config per held-out subject (split=subject)

| held_out | RAW | LMS-HK | NLMS-FV | NLMS-1 | NLMS-all | RLS-all | CW-NLMS |
|---|---|---|---|---|---|---|---|
| s1 | - | - | - | L=32,mu=0.05 | L=16,mu=0.5 | L=8,lam=0.999 | L=16,mu=0.5,tau=0.0 |
| s2 | - | - | - | L=32,mu=0.05 | L=16,mu=0.5 | L=8,lam=0.999 | L=16,mu=0.5,tau=0.3 |
| s3 | - | - | - | L=32,mu=0.05 | L=16,mu=0.5 | L=8,lam=0.999 | L=16,mu=0.5,tau=0.2 |
| s4 | - | - | - | L=16,mu=0.01 | L=16,mu=0.5 | L=8,lam=0.998 | L=16,mu=0.5,tau=0.0 |
| s5 | - | - | - | L=32,mu=0.05 | L=16,mu=0.05 | L=8,lam=0.999 | L=32,mu=0.05,tau=0.0 |
| s6 | - | - | - | L=32,mu=0.05 | L=16,mu=0.5 | L=8,lam=0.999 | L=16,mu=0.5,tau=0.0 |
| s8 | - | - | - | L=32,mu=0.05 | L=16,mu=0.01 | L=8,lam=0.999 | L=8,mu=0.01,tau=0.3 |
| s9 | - | - | - | L=32,mu=0.05 | L=16,mu=0.5 | L=8,lam=0.999 | L=16,mu=0.5,tau=0.0 |
