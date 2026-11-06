# Multi-reference adaptive motion-artifact cancellation for wrist PPG

ED5017 Digital Signal Processing research project. Wrist PPG is cleaned with adaptive filters that use the
accelerometer and gyroscope as motion references, and heart rate is tracked from the cleaned signal. The
proposed method, coherence-weighted NLMS (CW-NLMS), scales each IMU reference by its coherence with the PPG.
It is compared with LMS, NLMS and RLS baselines on the *Wrist PPG During Exercise* database
(19 records, 8 subjects), with all settings chosen by leave-one-subject-out validation.

## Setup
Python 3.11.
```
pip install -r requirements.txt
```
Download *Wrist PPG During Exercise* into `data/` (not committed):
```
wget -r -N -c -np -nH --cut-dirs=3 -P data https://physionet.org/files/wrist/1.0.0/
```
Check it works: `python demo.py data/s3_run` (or `python demo.py` for a synthetic test).

## Layout
| Path | Contents |
|---|---|
| `src/` | Library: loading, preprocessing, filters, coherence weights, HR tracker, scoring, extensions |
| `sim/` | Synthetic and semi-synthetic record generators |
| `tests/` | Unit tests: `python -m tests.test_filters` (cancellers, coherence, cost) and `python -m tests.test_option3` (extensions) |
| `results/` | Committed outputs of every script below, so tables and figures can be checked without rerunning |
| `DECISIONS.md` | Every setting that affects results, with its reason |
| `COLLISION.md`, `OPTION3.md` | Design and results of the two extensions |

## Reproducing the main results
```
python tune.py          # LOSO grid search  -> results/grid/, results/loso_params.csv
python run_all.py       # held-out runs     -> results/main_*.csv*, synth_records.csv, curves.npz
python robustness.py    # robustness tests  -> results/robustness.csv
python tables.py        # tables T1-T9      -> results/tables/
python plots.py         # residual-power and robustness figures -> results/fig_*.png/.pdf
python fig1_overlap.py  # report Fig. 1, motion/pulse overlap on s3_run -> results/fig1_overlap.png/.pdf
python diag.py data/s3_walk   # spectrogram diagnostics for one record
```
Each step only reads what the previous ones wrote, so any step can be rerun on its own.
`tune.py` and `run_all.py` take `--methods` to (re)run a subset.

## Extensions: heart rate near the cadence
Two extensions, each with a decision rule fixed before scoring. Neither met its rule; see COLLISION.md
and OPTION3.md. Settings are locked on synthetic data first, then applied once to the real records.

IMU-gated WQN for transient jerks (OPTION3.md):
```
python run_sim.py dev         # grid on synthetic dev seeds        -> results/dev.csv
python run_sim.py test        # locked settings, held-out seeds    -> results/test.csv
python analyze.py             # paired statistics + stress test    -> printed (saved as results/analysis.txt)
python make_figure.py         # verification figure                -> results/option3_simulation.png
python run_option3.py --data data --out results/option3 --loso results/loso_params.csv --inject
```
Collision-aware adaptation (COLLISION.md):
```
python run_collision_sim.py dev     # grid on synthetic dev seeds        -> results/collision_dev.csv
python run_collision_sim.py test    # locked settings, held-out seeds    -> results/collision_test.csv
python run_collision_sim.py leak    # uncancellable-motion sweep -> results/collision_leak*.csv (log: collision_leak.log)
python run_collision_sim.py drift   # coupling-drift sensitivity -> printed (saved as results/collision_drift.log)
python analyze_collision.py         # -> printed (saved as results/collision_analysis.txt)
python make_collision_figure.py     # -> results/collision_simulation.png
python run_collision.py --data data --out results/collision --loso results/loso_params.csv
python run_collision.py --data data --out results/collision_oracle --loso results/loso_params.csv --gate-hr ecg
```
After each `dev` run the chosen settings were written by hand to `results/locked.json` and
`results/collision_locked.json`, before the `test` run and before any real record was scored.
`--gate-hr ecg` is an upper bound that gives the gate the true HR; it is not a usable method.
The real-data runs need `results/loso_params.csv` from `tune.py`. `python analyze_real.py` summarises them.

## Main functions
| Function | File | Returns |
|---|---|---|
| `load_record(path)` | src/load.py | dict: name, subject, activity, fs, ppg, acc (N,3), gyro (N,3), rpeaks_s |
| `list_records(data_dir)` | src/load.py | list of record names |
| `preprocess(rec, fs_out=64)` | src/preprocess.py | d (N,), refs (N,8), fs. Order in `REF_NAMES` |
| `forearm_axis(acc_raw)` | src/preprocess.py | 0/1/2, the accel axis used by NLMS-1 and LMS-HK |
| `lms / nlms(d, refs, L, mu, weights=None)` | src/filters.py | e (cleaned PPG), err2 (squared output) |
| `nlms_bank(d, refs, L, mu)` | src/filters.py | (N, C) outputs for NLMS-FV |
| `rls(d, refs, L, lam)` | src/filters.py | e, err2 (regularized RLS, DECISIONS M9) |
| `msc_weights(d, refs, fs, tau)` | src/coherence.py | per-window coherence weights for CW-NLMS |
| `track_hr(e, fs, motion=refs)` | src/hr.py | hr_bpm per window, window centre times |
| **`run_method(name, rec, mu, L, tau, lam)`** | src/methods.py | dict: hr, t, e, err2, fs (+ msc for CW-NLMS). Names in `METHODS`, tunable params in `TUNABLE` |
| `score(out, rec)` | src/score.py | per-window table, MAE. Matches windows by centre time, so 20/35/64 Hz methods score alike |
| `evaluate(name, rec, params)` | src/evaluate.py | per-window DataFrame and per-record dict with the metrics of DECISIONS E5–E7 and E11–E15 |
| `semi_synthetic(rec, seed)` | src/synth.py | (record with known pulse, clean pulse), DECISIONS E13 |

`mu` can be an (N,) array (variable step size); `weights` can be (C,) or (N, C) (CW-NLMS).

## Ground truth
The ECG R peaks (`rpeaks_s`) are used only for scoring, in `src/score.py` and the extension scorers. They
never reach the filters, the coherence weights or `track_hr`; `tests/test_option3.py` checks this for the
extensions. The one exception is the labelled upper bound `--gate-hr ecg`.

## License
MIT, see LICENSE.
