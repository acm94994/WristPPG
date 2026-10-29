"""Record perturbations for the robustness tests (DECISIONS E8). Applied to the raw record,
before preprocess(), so |a|, |w| and the forearm axis are recomputed from the perturbed IMU."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotate(rec, seed):
    """Same random 3-D rotation of the accel and gyro triplets (sensor re-mounted on the wrist).
    |a| and |w| are unchanged by construction."""
    R = Rotation.random(random_state=seed).as_matrix()
    return {**rec, "acc": rec["acc"] @ R.T, "gyro": rec["gyro"] @ R.T}


def zero_acc_axis(rec, axis):
    """One accel axis dead (reads 0). |a| then comes from the remaining two axes."""
    acc = rec["acc"].copy()
    acc[:, axis] = 0
    return {**rec, "acc": acc}


def no_gyro(rec, _=None):
    """Gyroscope missing: gyro x/y/z and |w| all zero (an accel-only wearable)."""
    return {**rec, "gyro": np.zeros_like(rec["gyro"])}


TRANSFORMS = {"rotate": rotate, "zero_acc_axis": zero_acc_axis, "no_gyro": no_gyro}
