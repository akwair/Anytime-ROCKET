"""Read-only data/reference preflight; writes one audit JSON inside the workspace."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import statistics, save_json


def main():
    stats = statistics()
    names = stats[(stats.channel_num == 1) & (~stats.has_nan)].dataset_name.tolist()
    report = {"datasets": len(names), "expected_runs": len(names) * 5,
              "checked_runs": 0, "missing_fit_classes": [], "errors": []}
    for name in names:
        try:
            with np.load(ROOT / "data" / "UCR" / name / "arrays.npz") as data:
                xtr, xte, ytr, yte = [data[k] for k in ("Xtr", "Xte", "ytr", "yte")]
                assert xtr.ndim == xte.ndim == 2 and xtr.shape[1] == xte.shape[1]
                assert np.isfinite(xtr).all() and np.isfinite(xte).all()
                assert len(xtr) == len(ytr) and len(xte) == len(yte)
                y = np.concatenate([ytr, yte])
                for seed in range(5):
                    result_dir = ROOT / "results" / "v2"
                    stem = f"{name}_s{seed}"
                    result = json.loads((result_dir / f"{stem}.json").read_text())
                    idx = np.random.default_rng(seed).permutation(len(y))
                    nfit, ncal = len(xtr), len(xte) // 2
                    fit, test = idx[:nfit], idx[nfit + ncal:]
                    classes = np.unique(y[fit])
                    assert result["T"] == xtr.shape[1]
                    assert (result["n_fit"], result["n_cal"], result["n_test"]) == (nfit, ncal, len(test))
                    with np.load(result_dir / f"{stem}.npz") as pred:
                        assert np.array_equal(pred["y_test"], y[test])
                        full_pred = classes[pred["strat_imp_Pt"][:, -1]]
                        assert abs(float(np.mean(full_pred == y[test])) - result["orders"]["strat_imp"]["full_acc"]) < 1e-12
                    if len(classes) < len(np.unique(y)):
                        report["missing_fit_classes"].append(stem)
                    report["checked_runs"] += 1
        except Exception as exc:
            report["errors"].append({"dataset": name, "error": repr(exc)})
    destination = ROOT / "notes" / "baseline_input_audit.json"
    save_json(report, str(destination))
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]) or report["checked_runs"] != report["expected_runs"])


if __name__ == "__main__":
    raise SystemExit(main())
