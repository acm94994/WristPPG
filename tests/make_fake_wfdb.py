"""Writes simulator records as WFDB files with the real dataset's channel names, to test the runner."""
import os, sys
import numpy as np
import wfdb
from sim.semisynth import make_record
from src.load import ACC, GYRO

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
names = []
for i, act in enumerate(("run", "walk", "low_resistance_bike", "high_resistance_bike")):
    r = make_record(i, {"run": "run", "walk": "walk"}.get(act, "bike"), T=120)
    name = f"s{i + 1}_{act}"
    sig = np.column_stack([r["ppg"], r["acc"], r["gyro"]])
    sig = np.vstack([sig, np.full((256, sig.shape[1]), np.nan)])      # NaN padding like the real files
    labels = ["wrist_ppg"] + ACC + GYRO
    wfdb.wrsamp(name, fs=256, units=["V"] * len(labels), sig_name=labels, p_signal=sig,
                fmt=["32"] * len(labels), adc_gain=[1000.0] * len(labels), baseline=[0] * len(labels),
                write_dir=out)
    samples = (r["rpeaks_s"][r["rpeaks_s"] > 0] * 256).astype(int)
    wfdb.wrann(name, "atr", samples, symbol=["N"] * len(samples), write_dir=out)
    names.append(name)
open(os.path.join(out, "RECORDS"), "w").write("\n".join(names) + "\n")
print(names)
