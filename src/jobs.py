"""Parallel evaluate() jobs for tune.py and run_all.py.

A job is (method, record_name, params, opts):
  opts["synth_seed"]  : run on semi_synthetic(rec, seed) instead (adds NMSE / correlation)
  opts["tags"]        : extra constant columns
  opts["keep_curve"]  : also return the learning curve
Records are loaded once per worker process.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")            # one core per worker
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from src.load import load_record
from src.evaluate import evaluate
from src.synth import semi_synthetic

DATA = "data"


@lru_cache(maxsize=None)
def record(name):
    return load_record(os.path.join(DATA, name))


def _job(job):
    name, rec_name, params, opts = job
    rec, clean = record(rec_name), None
    if "synth_seed" in opts:
        rec, clean = semi_synthetic(rec, seed=opts["synth_seed"])
    return evaluate(name, rec, params, clean=clean, keep_curve=opts.get("keep_curve", False),
                    **opts.get("tags", {}))


def run_jobs(jobs, n_workers=None):
    """Returns [(windows DataFrame, record dict), ...] in job order."""
    with ProcessPoolExecutor(n_workers or os.cpu_count()) as ex:
        return list(ex.map(_job, jobs, chunksize=1))
