"""Per-subject NLMS-all settings from the repo's LOSO results, so no subject is tuned on itself."""
import csv


def loso_params(path="results/loso_params.csv", method="NLMS-all"):
    """Returns {subject: (L, mu)} for split=subject."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["method"] == method and r["split"] == "subject":
            p = dict(kv.split("=") for kv in r["params"].split(","))
            out[r["held_out"]] = (int(p["L"]), float(p["mu"]))
    return out
