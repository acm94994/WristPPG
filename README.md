# Project 03 – Multi-reference adaptive motion-artifact cancellation (wrist PPG)

## Setup
```
pip install -r requirements.txt
```
Download *Wrist PPG During Exercise* into `data/` (not committed):
```
wget -r -N -c -np -nH --cut-dirs=3 -P data https://physionet.org/files/wrist/1.0.0/
```
Check it works: `python demo.py data/s3_run` (or `python demo.py` for a synthetic test).

## Reproducing the results
```
python tune.py          # LOSO grid search  -> results/grid/, results/loso_params.csv
python run_all.py       # held-out runs     -> results/main_*.csv*, synth_records.csv, curves.npz
python robustness.py    # E8                -> results/robustness.csv
python tables.py        # T1-T9             -> results/tables/
python plots.py         # figures           -> results/fig_*.png/.pdf
python diag.py data/s3_walk   # spectrogram diagnostics for one record
```
Post-submission extensions (heart rate near cadence), after `tune.py` has written `results/loso_params.csv`:
```
python run_collision.py --data data --out results/collision --loso results/loso_params.csv
python run_collision.py --data data --out results/collision_oracle --loso results/loso_params.csv --gate-hr ecg
python run_option3.py --data data --out results/option3 --loso results/loso_params.csv --inject
python analyze_real.py        # summary of the three runs above
```
See COLLISION.md and OPTION3.md. `--gate-hr ecg` is an upper bound that gives the gate the true HR; it is
not a usable method.
Each step only reads what the previous ones wrote, so any step can be rerun on its own.
`tune.py` and `run_all.py` take `--methods` to (re)run a subset.

## Interfaces (don't change signatures without telling the group)
| Function | File | Returns |
|---|---|---|
| `load_record(path)` | src/load.py | dict: name, subject, activity, fs, ppg, acc (N,3), gyro (N,3), rpeaks_s |
| `list_records(data_dir)` | src/load.py | list of record names |
| `preprocess(rec, fs_out=64)` | src/preprocess.py | d (N,), refs (N,8), fs. Order in `REF_NAMES` |
| `forearm_axis(acc_raw)` | src/preprocess.py | 0/1/2, the accel axis used by NLMS-1 and LMS-HK |
| `lms / nlms(d, refs, L, mu, weights=None)` | src/filters.py | e (cleaned PPG), err2 (learning curve) |
| `nlms_bank(d, refs, L, mu)` | src/filters.py | (N, C) outputs for NLMS-FV |
| `rls(d, refs, L, lam)` | src/filters.py | e, err2 (regularized RLS, M9) |
| `track_hr(e, fs, motion=refs)` | src/hr.py | hr_bpm per window, window centre times |
| **`run_method(name, rec, mu, L, tau, lam)`** | src/methods.py | dict: hr, t, e, err2, fs (+ msc for CW-NLMS). Names in `METHODS`, tunable params in `TUNABLE` |
| `score(out, rec)` | src/score.py | per-window table, MAE. Matches windows by centre time, so 20/35/64 Hz methods score alike |
| `evaluate(name, rec, params)` | src/evaluate.py | per-window DataFrame and per-record dict with the metrics of DECISIONS E5–E7 and E11–E15 |
| `semi_synthetic(rec, seed)` | src/synth.py | (record with known pulse, clean pulse), E13 |

`mu` can be an (N,) array (variable step size); `weights` can be (C,) or (N, C) (CW-NLMS).

## Rules
- Only `score.py` uses ECG (`rpeaks_s`). Never pass it to filters or `track_hr`.
- One owner per file. `git pull` before work; push small and often.
- Every setting choice goes in `DECISIONS.md`.
