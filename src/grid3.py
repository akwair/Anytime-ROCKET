"""Grid v3: cross-fitted calibration for small calibration sets.

The deployed model and its calib/test outputs are those of grid2 (read from results/v2/*.npz). In
addition, K-fold cross-fitting on the fit set gives out-of-fold (OOF) losses: for each fold, an anytime
model of the same order and the same checkpoint unit counts is refit on the other folds, and its
disagreement with its own full head is recorded on the held-out fold. These OOF losses are pooled with
the unlabeled calibration losses (n = n_fit + n_cal) before LTT / CRC.

Caveat (stated in the paper): OOF models are not the deployed model, so the pooled sample is not
exchangeable with test and the finite-sample guarantee becomes approximate (as for CV+/jackknife-type
calibration, cf. CV-CRC, Cohen et al., arXiv 2401.11974). Validity is checked empirically on the untouched
test split. MiniRocket biases are kept from the full fit set (heads and scaler are refit per fold).

Only run where it matters: datasets with n_cal < N_MAX (others are certifiable from the calibration set).
Output results/v3/<ds>_s<seed>.json (skipped if present).
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as at  # noqa: E402
from data import load_ucr, save_json  # noqa: E402
from grid2 import CALIBS, EPS, FRACTIONS, RULES, select  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "results", "v2")
OUT = os.path.join(ROOT, "results", "v3")
ORDERS = ["random", "strat_imp"]
K = 5
N_MAX = 460  # PAC sample size for eps = 0.5%, delta = 0.1


def oof_outputs(Ffit, yfit, U, T, order, ckpt_units, seed):
    """Margins / predictions of fold models on their held-out folds, [n_fit, K_ckpt]."""
    n = len(yfit)
    _, cnt = np.unique(yfit, return_counts=True)
    if cnt.min() >= K:
        split = StratifiedKFold(K, shuffle=True, random_state=seed).split(Ffit, yfit)
    else:
        split = KFold(K, shuffle=True, random_state=seed).split(Ffit)
    M = np.zeros((n, len(ckpt_units)), np.float32)
    P = np.zeros((n, len(ckpt_units)), np.int64)
    for f, (tr, ho) in enumerate(split):
        m = at.build_anytime(Ffit[tr], yfit[tr], U, T, FRACTIONS, order=order,
                             rng=np.random.default_rng(seed + 2000 + f), ckpt_units=ckpt_units)
        M[ho], P[ho] = at.head_outputs(m, Ffit[ho])
    return M, P


def losses_static(P):
    full = P[:, -1]
    return np.stack([(P[:, k] != full).astype(int) for k in range(P.shape[1] - 1, -1, -1)])


def losses_adaptive(M, P, rule):
    full = P[:, -1]
    out = []
    for lam in at.LAM_GRID:
        e = at.exit_index(M, P, lam, rule)
        out.append((P[np.arange(len(e)), e] != full).astype(int))
    return np.stack(out)


def run(name, seed):
    path = os.path.join(OUT, f"{name}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(OUT, exist_ok=True)
    ref = json.load(open(os.path.join(V2, f"{name}_s{seed}.json")))
    res = dict(dataset=name, seed=seed, n_fit=ref["n_fit"], n_cal=ref["n_cal"], n_test=ref["n_test"],
               K=K, skipped=ref["n_cal"] >= N_MAX, orders={})
    if res["skipped"]:
        save_json(res, path)
        return res
    A = np.load(os.path.join(V2, f"{name}_s{seed}.npz"))
    Xtr, ytr, Xte, yte = load_ucr(name)
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    T = X.shape[1]
    idx = np.random.default_rng(seed).permutation(len(X))
    i_fit = idx[:len(Xtr)]
    params = at.fit_params(X[i_fit], seed=seed)
    Ffit = at.transform(X[i_fit], params)
    U = at.units_of(params, T)
    yt = A["y_test"]
    for order in ORDERS:
        t0 = time.time()
        o = ref["orders"][order]
        Mf, Pf = oof_outputs(Ffit, y[i_fit], U, T, order, o["ckpt_units"], seed)
        Mc, Pc, Mt, Pt = (A[f"{order}_{k}"].astype(np.float64 if k[0] == "M" else np.int64)
                          for k in ("Mc", "Pc", "Mt", "Pt"))
        model = dict(conv=A[f"{order}_conv"], head_cost=A[f"{order}_head"])
        model["full_cost"] = model["conv"][-1] + model["head_cost"][-1]
        cls = np.unique(y[i_fit])  # = RidgeClassifierCV.classes_ of the deployed heads
        full_t = Pt[:, -1]
        assert abs(np.mean(cls[full_t] == yt) - o["full_acc"]) < 1e-9
        r = dict(cv_s=time.time() - t0, oof_ckpt_disagree=[float(np.mean(Pf[:, k] != Pf[:, -1]))
                                                            for k in range(Pf.shape[1])])
        Ls = np.concatenate([losses_static(Pf), losses_static(Pc)], 1)
        La = {rule: np.concatenate([losses_adaptive(Mf, Pf, rule), losses_adaptive(Mc, Pc, rule)], 1)
              for rule in RULES}
        for calib in CALIBS:
            r[calib] = {}
            for eps in EPS:
                j = select(Ls, eps, calib)
                k = Pt.shape[1] - 1 - max(j, 0)
                er = dict(static=dict(k=int(k), cost=float((model["conv"][k] + model["head_cost"][k]) / model["full_cost"]),
                                      disagree=float(np.mean(Pt[:, k] != full_t)),
                                      acc=float(np.mean(cls[Pt[:, k]] == yt))))
                for rule in RULES:
                    lam = at.LAM_GRID[max(select(La[rule], eps, calib), 0)]
                    e = at.exit_index(Mt, Pt, lam, rule)
                    pe = Pt[np.arange(len(e)), e]
                    cost = 1.0 if np.isinf(lam) else float(np.mean(at.anytime_cost(model, e)))
                    er[rule] = dict(lam=float(lam), cost=cost, disagree=float(np.mean(pe != full_t)),
                                    acc=float(np.mean(cls[pe] == yt)))
                r[calib][str(eps)] = er
        r["full_acc"] = o["full_acc"]
        res["orders"][order] = r
    save_json(res, path)
    return res


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    for nm in names:
        for s in seeds:
            t0 = time.time()
            r = run(nm, s)
            if r["skipped"]:
                print(f"{nm} s{s} skipped (n_cal={r['n_cal']})", flush=True)
                continue
            o = r["orders"]["strat_imp"]
            print(f"{nm} s{s} n={r['n_fit']}+{r['n_cal']} strat_imp eps=1%: "
                  f"ltt {o['ltt']['0.01']['static']['cost']:.3f}/{o['ltt']['0.01']['margin']['cost']:.3f} "
                  f"crc {o['crc']['0.01']['static']['cost']:.3f}/{o['crc']['0.01']['margin']['cost']:.3f} "
                  f"dis {o['ltt']['0.01']['margin']['disagree']:.4f} ({time.time() - t0:.0f}s)", flush=True)
