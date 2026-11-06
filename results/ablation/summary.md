## Held-out results (split=subject)

| variant | pooled MAE | mean of subjects | <=10 BPM (%) | pulse ratio (dB) |
| --- | --- | --- | --- | --- |
| NLMS-all | 20.45 | 31.87 | 45.18 | -10.82 |
| NLMS-all, CW-matched mu | 18.26 | 30.84 | 54.37 | -7.82 |
| NLMS-all, dense mu grid | 22.17 | 33.94 | 45.3 | -9.86 |
| CW-NLMS | 17.83 | 29.11 | 53.34 | -7.74 |
| CW: weights only | 23.24 | 34.39 | 41.3 | -9.52 |
| CW: step only | 17.49 | 29.85 | 53.55 | -7.25 |
| CW-NLMS, tau > 0 | 18.93 | 30.72 | 53.34 | -7.7 |

## Per-subject MAE (BPM)

| variant | s1 | s2 | s3 | s4 | s5 | s6 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CW-NLMS | 3.8 | 15.7 | 7.3 | 24.5 | 30.2 | 22.4 | 30.5 | 98.4 |
| CW-NLMS, tau > 0 | 3.8 | 15.7 | 7.3 | 24.5 | 43.0 | 22.4 | 30.5 | 98.4 |
| CW: step only | 3.3 | 10.6 | 8.0 | 28.6 | 31.8 | 21.8 | 32.4 | 102.3 |
| CW: weights only | 9.9 | 10.2 | 20.2 | 28.1 | 48.1 | 33.4 | 23.1 | 102.1 |
| NLMS-all | 7.0 | 11.1 | 12.5 | 25.8 | 46.9 | 22.9 | 32.4 | 96.4 |
| NLMS-all, CW-matched mu | 3.9 | 10.4 | 5.7 | 26.7 | 46.6 | 22.4 | 30.6 | 100.3 |
| NLMS-all, dense mu grid | 7.0 | 19.0 | 12.5 | 29.3 | 46.9 | 23.5 | 32.4 | 100.9 |

## Subjects with lower MAE than CW-NLMS

| variant | subjects below CW-NLMS |
| --- | --- |
| CW-NLMS | 0 |
| CW-NLMS, tau > 0 | 1 |
| CW: step only | 3 |
| CW: weights only | 2 |
| NLMS-all | 2 |
| NLMS-all, CW-matched mu | 2 |
| NLMS-all, dense mu grid | 0 |

## LOSO choices (setting: folds)

- NLMS-all: {'L=16,mu=0.01': 1, 'L=16,mu=0.05': 1, 'L=16,mu=0.5': 6}
- NLMS-all, dense mu grid: {'L=16,mu=0.01': 1, 'L=16,mu=0.05': 1, 'L=16,mu=0.3': 1, 'L=16,mu=0.5': 2, 'L=32,mu=0.3': 1, 'L=8,mu=0.3': 2}
- CW-NLMS: {'L=16,mu=0.5,tau=0.0': 4, 'L=16,mu=0.5,tau=0.2': 1, 'L=16,mu=0.5,tau=0.3': 1, 'L=32,mu=0.05,tau=0.0': 1, 'L=8,mu=0.01,tau=0.3': 1}
- CW: weights only: {'L=32,mu=0.05,tau=0.5,vss=False': 1, 'L=32,mu=0.1,tau=0.0,vss=False': 3, 'L=32,mu=0.5,tau=0.2,vss=False': 3, 'L=32,mu=0.5,tau=0.3,vss=False': 1}
- CW: step only: {'L=16,mu=0.05,tau=0.5,weight_refs=False': 1, 'L=16,mu=0.5,tau=0.3,weight_refs=False': 1, 'L=32,mu=0.5,tau=0.3,weight_refs=False': 4, 'L=8,mu=0.01,tau=0.0,weight_refs=False': 1, 'L=8,mu=0.5,tau=0.2,weight_refs=False': 1}
- CW-NLMS, tau > 0: {'L=16,mu=0.5,tau=0.2': 5, 'L=16,mu=0.5,tau=0.3': 1, 'L=32,mu=0.1,tau=0.5': 1, 'L=8,mu=0.01,tau=0.3': 1}

## CW-matched constant mu per record

| record | L | mu |
| --- | --- | --- |
| s1_high_resistance_bike | 16.0 | 0.224136 |
| s1_low_resistance_bike | 16.0 | 0.218823 |
| s1_walk | 16.0 | 0.303447 |
| s2_high_resistance_bike | 16.0 | 0.227587 |
| s2_low_resistance_bike | 16.0 | 0.19271 |
| s2_walk | 16.0 | 0.264306 |
| s3_high_resistance_bike | 16.0 | 0.200511 |
| s3_low_resistance_bike | 16.0 | 0.206341 |
| s3_run | 16.0 | 0.312472 |
| s3_walk | 16.0 | 0.254189 |
| s4_run | 16.0 | 0.243674 |
| s5_low_resistance_bike | 32.0 | 0.01982 |
| s5_run | 32.0 | 0.026528 |
| s6_low_resistance_bike | 16.0 | 0.224889 |
| s6_run | 16.0 | 0.246073 |
| s6_walk | 16.0 | 0.233801 |
| s8_run | 8.0 | 0.005383 |
| s8_walk | 8.0 | 0.004755 |
| s9_walk | 16.0 | 0.282794 |

## Channel ablation, pooled MAE (BPM)

| dropped | NLMS-all | CW-NLMS |
| --- | --- | --- |
| none | 20.45 | 17.83 |
| acc_x | 24.01 | 21.32 |
| acc_y | 21.38 | 17.3 |
| acc_z | 24.09 | 17.9 |
| gyro_x | 22.61 | 17.38 |
| gyro_y | 23.21 | 18.36 |
| gyro_z | 23.37 | 18.52 |
| acc_mag | 21.92 | 16.95 |
| gyro_mag | 22.49 | 17.61 |

## Channel ablation, mean of subjects (BPM)

| dropped | CW-NLMS | NLMS-all |
| --- | --- | --- |
| acc_mag | 28.62 | 33.13 |
| acc_x | 32.96 | 34.56 |
| acc_y | 29.65 | 32.93 |
| acc_z | 29.86 | 35.2 |
| gyro_mag | 29.01 | 33.03 |
| gyro_x | 29.82 | 33.78 |
| gyro_y | 30.15 | 34.0 |
| gyro_z | 30.16 | 34.35 |
| none | 29.11 | 31.87 |
