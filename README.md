# Multi-reference adaptive motion-artifact cancellation for wrist PPG

Wrist PPG is cleaned with adaptive filters that use the accelerometer and gyroscope as motion references,
and heart rate is tracked from the cleaned signal. The proposed method, coherence-weighted NLMS (CW-NLMS),
scales each IMU reference by its coherence with the PPG. It is compared with LMS, NLMS and RLS baselines on
the *Wrist PPG During Exercise* database (19 records, 8 subjects), with all settings chosen by
leave-one-subject-out validation. Results and discussion are in the accompanying report.

## Setup
Python 3.11.
```
pip install -r requirements.txt
```
Download *Wrist PPG During Exercise* into `data/` (not committed):
```
wget -r -N -c -np -nH --cut-dirs=3 -P data https://physionet.org/files/wrist/1.0.0/
```
Run every command below from the repository root. Check the setup with `python -m tools.demo data/s3_run`
(or `python -m tools.demo` for a synthetic test that needs no data).

## Layout
| Path | Contents |
|---|---|
| `src/` | Library: loading, preprocessing, filters, coherence weights, HR tracker, scoring, extensions |
| `sim/` | Synthetic and semi-synthetic record generators |
| `experiments/` | Runs: tuning, held-out evaluation, robustness, and the two extensions |
| `analysis/` | Tables and paired statistics from the stored runs |
| `figures/` | Figure scripts |
| `tools/` | Quick checks on a single record: `demo`, `diag` |
| `tests/` | `python -m tests.test_filters` (cancellers, coherence, cost), `python -m tests.test_extensions` (extensions) |
| `results/` | Committed outputs of every script, so tables and figures can be checked without rerunning |
| `DECISIONS.md` | Every setting that affects results, with its reason |
| `COLLISION.md`, `WQN.md` | Design of the two extensions |

## Reproducing the main results
```
python -m experiments.tune         # LOSO grid search  -> results/grid/, results/loso_params.csv
python -m experiments.run_all      # held-out runs     -> results/main_*.csv*, synth_records.csv, curves.npz
python -m experiments.robustness   # robustness tests  -> results/robustness.csv
python -m analysis.tables          # tables T1-T9      -> results/tables/
python -m figures.plots            # residual-power and robustness figures -> results/fig_*.png/.pdf
python -m figures.fig1_overlap     # motion/pulse overlap on s3_run        -> results/fig1_overlap.png/.pdf
python -m tools.diag data/s3_walk  # spectrogram diagnostics for one record
```
Each step only reads what the previous ones wrote, so any step can be rerun on its own.
`experiments.tune` and `experiments.run_all` take `--methods` to (re)run a subset.

## Extensions: heart rate near the cadence
Two extensions, each with a decision rule fixed before scoring; see COLLISION.md and WQN.md. Settings are
locked on synthetic data first, then applied once to the real records.

IMU-gated WQN for transient jerks (WQN.md):
```
python -m experiments.run_wqn_sim dev   # grid on synthetic dev seeds        -> results/dev.csv
python -m experiments.run_wqn_sim test  # locked settings, held-out seeds    -> results/test.csv
python -m analysis.analyze_wqn          # paired statistics + stress test    -> printed (saved as results/analysis.txt)
python -m figures.wqn_figure            # verification figure                -> results/wqn_simulation.png
python -m experiments.run_wqn --data data --out results/wqn --loso results/loso_params.csv --inject
```
Collision-aware adaptation (COLLISION.md):
```
python -m experiments.run_collision_sim dev    # grid on synthetic dev seeds     -> results/collision_dev.csv
python -m experiments.run_collision_sim test   # locked settings, held-out seeds -> results/collision_test.csv
python -m experiments.run_collision_sim leak   # uncancellable-motion sweep      -> results/collision_leak*.csv (log: collision_leak.log)
python -m experiments.run_collision_sim drift  # coupling-drift sensitivity      -> printed (saved as results/collision_drift.log)
python -m analysis.analyze_collision           # -> printed (saved as results/collision_analysis.txt)
python -m figures.collision_figure             # -> results/collision_simulation.png
python -m experiments.run_collision --data data --out results/collision --loso results/loso_params.csv
python -m experiments.run_collision --data data --out results/collision_oracle --loso results/loso_params.csv --gate-hr ecg
```
After each `dev` run the chosen settings were written by hand to `results/locked.json` and
`results/collision_locked.json`, before the `test` run and before any real record was scored.
`--gate-hr ecg` is an upper bound that gives the gate the true HR; it is not a usable method.
The real-data runs need `results/loso_params.csv` from `experiments.tune`. `python -m analysis.analyze_real`
summarises them.

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
never reach the filters, the coherence weights or `track_hr`; `tests/test_extensions.py` checks this for the
extensions. The one exception is the labelled upper bound `--gate-hr ecg`.

## License
MIT, see LICENSE.
