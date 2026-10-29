# Project 03 – Multi-reference adaptive motion-artifact cancellation (wrist PPG)

## Setup
```
pip install numpy scipy wfdb
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
| `track_hr(e, fs, motion=refs)` | src/hr.py | hr_bpm per window, window centre times |
| **`run_method(name, rec, mu, L, tau, lam)`** | src/methods.py | dict: hr, t, e, err2, fs (+ msc for CW-NLMS). Names in `METHODS`, tunable params in `TUNABLE` |
| `score(out, rec)` | src/score.py | per-window table, MAE. Matches windows by time, so 20/35/64 Hz methods score alike |
| `window_starts(N, fs)` | src/hr.py | window start indices, window length (use in score.py so windows match) |

`mu` can be an (N,) array (variable step size); `weights` can be (C,) or (N, C) (CW-NLMS).

## Rules
- Only `score.py` uses ECG (`rpeaks_s`). Never pass it to filters or `track_hr`.
- One owner per file. `git pull` before work; push small and often.
- Every setting choice goes in `DECISIONS.md`.

## For B: the loop
```python
from src.load import load_record, list_records
from src.methods import run_method, METHODS
from src.score import score
for r in list_records('data'):
    rec = load_record('data/' + r)
    for m in METHODS:
        table, mae = score(run_method(m, rec), rec)
```
LOSO: for each held-out subject, grid-search `TUNABLE[m]` on the other 7 subjects' records, then score the held-out subject once. LMS-HK, NLMS-FV and RAW have no tuning.

## Still to write
- LOSO tuning, `run_all.py`, Tables 1–3, extend `score.py` (pulse preservation, failure labels) (B)
- `robustness.py`, `plots.py` (D)
