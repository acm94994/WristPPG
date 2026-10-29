"""Load one record of 'Wrist PPG During Exercise' (PhysioNet, wrist/1.0.0).

Each record (e.g. data/s3_run) has .hea/.dat (15 signals at 256 Hz) and .atr
(hand-labelled ECG R peaks). Channel names come from the .hea files.

ECG note: `rpeaks_s` is returned for scoring only. Filter/HR code must never use it.
"""
import os
import numpy as np
import wfdb

ACC = ["wrist_wide_range_accelerometer_x",   # ±16 g (DECISIONS D2)
       "wrist_wide_range_accelerometer_y",
       "wrist_wide_range_accelerometer_z"]
GYRO = ["wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z"]


def load_record(path):
    """path: record path without extension, e.g. 'data/s3_run'."""
    sig, f = wfdb.rdsamp(path)
    names = f["sig_name"]
    col = lambda n: sig[:, names.index(n)]

    ppg = col("wrist_ppg")
    acc = np.column_stack([col(n) for n in ACC])
    gyro = np.column_stack([col(n) for n in GYRO])

    # D6: wrist signals are NaN-padded at the end to match the ECG length -> trim
    wrist = np.column_stack([ppg, acc, gyro])
    bad = np.where(np.isnan(wrist).any(axis=1))[0]
    n = bad[0] if len(bad) else len(ppg)
    fs = f["fs"]

    ann = wfdb.rdann(path, "atr")
    rpeaks_s = ann.sample / fs
    rpeaks_s = rpeaks_s[rpeaks_s < n / fs]

    rec_name = os.path.basename(path)                  # 's3_run', 's1_low_resistance_bike'
    subject, activity = rec_name.split("_", 1)
    return {"name": rec_name, "subject": subject, "activity": activity, "fs": fs,
            "ppg": ppg[:n], "acc": acc[:n], "gyro": gyro[:n], "rpeaks_s": rpeaks_s}


def list_records(data_dir="data"):
    """All record names in data_dir (uses the RECORDS file if present)."""
    rec_file = os.path.join(data_dir, "RECORDS")
    if os.path.exists(rec_file):
        return [l.strip() for l in open(rec_file) if l.strip()]
    return sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea"))
