"""Kill pilot: certified adaptive exit vs certified static prefix on MiniRocket, label-free.

Protocol per dataset and seed: merge the official train/test, re-split into fit (original train
size), unlabeled calibration and test (remainder halved). Calibrate on disagreement with the
full model. Results go to results/pilot/<dataset>_s<seed>.json (skipped if present).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as at  # noqa: E402
from data import load_ucr, save_json  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "pilot")
FRACTIONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
EPS = [0.005, 0.01, 0.02, 0.05]
DELTA = 0.1
RULES = ["margin", "consist"]


def certify_static(model, Pcal, eps):
    """Smallest prefix whose disagreement with the full model is certified <= eps (fixed sequence
    from the full model downwards)."""
    full = Pcal[:, -1]
    K = Pcal.shape[1]
    losses = np.stack([(Pcal[:, k] != full).astype(int) for k in range(K - 1, -1, -1)])
    j = at.ltt_fixed_sequence(losses, eps, DELTA)
    return K - 1 - max(j, 0) if j >= 0 else K - 1


def certify_adaptive(Mcal, Pcal, eps, rule):
    full = Pcal[:, -1]
    losses = []
    for lam in at.LAM_GRID:
        e = at.exit_index(Mcal, Pcal, lam, rule)
        losses.append((Pcal[np.arange(len(e)), e] != full).astype(int))
    j = at.ltt_fixed_sequence(np.stack(losses), eps, DELTA)
    return at.LAM_GRID[max(j, 0)]


def run(name, seed):
    path = os.path.join(OUT, f"{name}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    Xtr, ytr, Xte, yte = load_ucr(name)
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    T = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_fit = len(Xtr)
    n_cal = (len(X) - n_fit) // 2
    i_fit, i_cal, i_te = idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]

    t0 = time.time()
    params = at.fit_params(X[i_fit], seed=seed)
    F = at.transform(X, params)
    t_tr = time.time() - t0
    U = at.units_of(params, T)
    res = dict(dataset=name, seed=seed, T=T, n_fit=len(i_fit), n_cal=len(i_cal), n_test=len(i_te),
               n_classes=int(len(np.unique(y))), n_features=int(U["p"]), transform_s=t_tr, orders={})
    for order in ["importance", "random"]:
        t0 = time.time()
        model = at.build_anytime(F[i_fit], y[i_fit], U, T, FRACTIONS, order=order,
                                 rng=np.random.default_rng(seed + 1000))
        Mc, Pc = at.head_outputs(model, F[i_cal])
        Mt, Pt = at.head_outputs(model, F[i_te])
        cls = model["classes"]
        yt = y[i_te]
        full_t = Pt[:, -1]
        r = dict(fit_s=time.time() - t0, ckpt_units=model["ckpt_units"],
                 ckpt_cost=(model["conv"] + model["head_cost"]).tolist(),
                 ckpt_rel_cost=((model["conv"] + model["head_cost"]) / model["full_cost"]).tolist(),
                 ckpt_acc=[float(np.mean(cls[Pt[:, k]] == yt)) for k in range(Pt.shape[1])],
                 ckpt_disagree=[float(np.mean(Pt[:, k] != full_t)) for k in range(Pt.shape[1])],
                 full_acc=float(np.mean(cls[full_t] == yt)), eps={})
        for eps in EPS:
            k = certify_static(model, Pc, eps)
            st_cost = (model["conv"][k] + model["head_cost"][k]) / model["full_cost"]
            er = dict(static=dict(k=int(k), cost=float(st_cost),
                                  disagree=float(np.mean(Pt[:, k] != full_t)),
                                  acc=float(np.mean(cls[Pt[:, k]] == yt))))
            for rule in RULES:
                lam = certify_adaptive(Mc, Pc, eps, rule)
                e = at.exit_index(Mt, Pt, lam, rule)
                pe = Pt[np.arange(len(e)), e]
                # nothing certified -> deploy the plain full model (no intermediate heads)
                cost = 1.0 if np.isinf(lam) else float(np.mean(at.anytime_cost(model, e)))
                er[rule] = dict(lam=float(lam), cost=cost,
                                disagree=float(np.mean(pe != full_t)), acc=float(np.mean(cls[pe] == yt)),
                                exit_hist=np.bincount(e, minlength=Pt.shape[1]).tolist())
            r["eps"][str(eps)] = er
        res["orders"][order] = r
    save_json(res, path)
    return res


PILOT = ["FordA", "FaceAll", "Phoneme", "InsectWingbeatSound", "MixedShapesRegularTrain",
         "FreezerRegularTrain", "Yoga", "UWaveGestureLibraryAll", "ChlorineConcentration",
         "TwoPatterns", "ECG5000", "Wafer", "ElectricDevices", "StarLightCurves", "Crop"]

if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else PILOT
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    for nm in names:
        for s in seeds:
            t0 = time.time()
            r = run(nm, s)
            o = r["orders"]["importance"]["eps"]["0.01"]
            print(f"{nm} s{s} ncal={r['n_cal']} static={o['static']['cost']:.3f} "
                  f"margin={o['margin']['cost']:.3f} consist={o['consist']['cost']:.3f} "
                  f"dis={o['margin']['disagree']:.4f} acc_full={r['orders']['importance']['full_acc']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
