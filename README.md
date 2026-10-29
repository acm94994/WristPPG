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

## Interfaces (don't change signatures without telling the group)
| Function | File | Returns |
|---|---|---|
| `load_record(path)` | src/load.py | dict: name, subject, activity, fs, ppg, acc (N,3), gyro (N,3), rpeaks_s |
| `list_records(data_dir)` | src/load.py | list of record names |
| `preprocess(rec, fs_out=64)` | src/preprocess.py | d (N,), refs (N,8), fs. Order in `REF_NAMES` |
| `forearm_axis(acc_raw)` | src/preprocess.py | 0/1/2, the accel axis used by NLMS-1 and LMS-HK |
| `lms / nlms(d, refs, L, mu, weights=None)` | src/filters.py | e (cleaned PPG), err2 (learning curve) |
| `nlms_bank(d, refs, L, mu)` | src/filters.py | (N, C) outputs for NLMS-FV |
| `track_hr(e, fs)` | src/hr.py | hr_bpm per window, window centre times |
| `window_starts(N, fs)` | src/hr.py | window start indices, window length (use in score.py so windows match) |

| `rls(d, refs, L, lam)` | src/rls.py | e, err2 (regularized RLS, M9) |
| `run(name, rec, **params)` | src/methods.py | dict: fs, d, e, err2, hr, t, flops. `METHODS`, `GRIDS` |
| `evaluate(name, rec, params)` | src/evaluate.py | per-window DataFrame, per-record dict |
| `window_metrics`, `true_hr`, `failure_label`, ... | src/score.py | the only ECG reader |
| `semi_synthetic(rec, seed)` | src/synth.py | (record with known pulse, clean pulse), E13 |
| `rotate`, `zero_acc_axis`, `no_gyro` | src/perturb.py | perturbed record, E8 |

`mu` can be an (N,) array (variable step size); `weights` can be (C,) or (N, C) (CW-NLMS).

**Adding a method (A):** replace its `_pending(...)` entry in `METHODS` (src/methods.py) with a
function `f(rec, **params) -> _out(...)`, put its grid in `GRIDS`, then
`python tune.py --methods CW-NLMS && python run_all.py --methods CW-NLMS && python robustness.py --methods CW-NLMS`.
Results for other methods are kept. CW-NLMS is already in `MULTI_REF`, so it must accept `drop=` (channel ablation).
If CW-NLMS adds an `msc` column to its windows, `failure_label` can use the "low MSC" label.

## Pipeline (B, D)
```
python tune.py         # LOSO grid, ~1 min on 16 cores  -> results/loso_params.csv, results/grid/
python run_all.py      # held-out runs + semi-synthetic -> results/main_*.csv*, synth_records.csv, curves.npz
python robustness.py   # rotation / missing axes / channel ablation, ~2 min -> results/robustness_*
python tables.py       # -> results/tables/T1..T9.csv, tables.md
python plots.py        # -> results/figures/F1..F6.png
```
All scripts take `--methods ...` and `--jobs N`.

## Rules
- Only `score.py` uses ECG (`rpeaks_s`). Never pass it to filters or `track_hr`.
  `evaluate.py` strips `rpeaks_s` before any method runs (`synth.py` uses R peaks only to build ground truth).
- One owner per file. `git pull` before work; push small and often.
- Every setting choice goes in `DECISIONS.md`.

## Still to write
- `src/coherence.py` (A), CW-NLMS + LMS-HK / NLMS-FV configs (A): stubs in src/methods.py
- Component ablation of CW-NLMS (MSC weighting on/off, variable μ on/off): add as extra configs once CW-NLMS exists
