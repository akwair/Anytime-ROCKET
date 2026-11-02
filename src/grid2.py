"""Grid v2: same split and model as pilot.py, plus
  * unit orders: importance, random, strat (dilation round-robin, random), strat_imp (round-robin by importance);
  * calibration: 'ltt' (PAC: P(risk <= eps) >= 1-delta, fixed-sequence binomial) and 'crc' (conformal
    risk control: E[risk] <= eps), both label-free (loss = disagreement with the full model).
Per-checkpoint margins/predictions on calib and test are stored in an npz so that other calibration
rules can be evaluated offline. Output results/v2/<ds>_s<seed>.{json,npz} (skipped if present).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as at  # noqa: E402
from data import load_ucr, save_json  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2")
FRACTIONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
EPS = [0.005, 0.01, 0.02, 0.05]
DELTA = 0.1
RULES = ["margin", "consist"]
ORDERS = ["importance", "random", "strat", "strat_imp"]
CALIBS = ["ltt", "crc"]


def select(losses, eps, calib):
    if calib == "ltt":
        return at.ltt_fixed_sequence(losses, eps, DELTA)
    return at.crc_select(losses, eps)


def certify_static(Pcal, eps, calib):
    full = Pcal[:, -1]
    K = Pcal.shape[1]
    losses = np.stack([(Pcal[:, k] != full).astype(int) for k in range(K - 1, -1, -1)])
    j = select(losses, eps, calib)
    return K - 1 - max(j, 0)


def certify_adaptive(Mcal, Pcal, eps, rule, calib):
    full = Pcal[:, -1]
    losses = []
    for lam in at.LAM_GRID:
        e = at.exit_index(Mcal, Pcal, lam, rule)
        losses.append((Pcal[np.arange(len(e)), e] != full).astype(int))
    j = select(np.stack(losses), eps, calib)
    return at.LAM_GRID[max(j, 0)]


def evaluate(model, Mc, Pc, Mt, Pt, yt):
    cls = model["classes"]
    full_t = Pt[:, -1]
    out = {}
    for calib in CALIBS:
        out[calib] = {}
        for eps in EPS:
            k = certify_static(Pc, eps, calib)
            er = dict(static=dict(k=int(k), cost=float((model["conv"][k] + model["head_cost"][k]) / model["full_cost"]),
                                  disagree=float(np.mean(Pt[:, k] != full_t)),
                                  acc=float(np.mean(cls[Pt[:, k]] == yt))))
            for rule in RULES:
                lam = certify_adaptive(Mc, Pc, eps, rule, calib)
                e = at.exit_index(Mt, Pt, lam, rule)
                pe = Pt[np.arange(len(e)), e]
                # nothing certified -> deploy the plain full model (no intermediate heads)
                cost = 1.0 if np.isinf(lam) else float(np.mean(at.anytime_cost(model, e)))
                er[rule] = dict(lam=float(lam), cost=cost, disagree=float(np.mean(pe != full_t)),
                                acc=float(np.mean(cls[pe] == yt)),
                                exit_hist=np.bincount(e, minlength=Pt.shape[1]).tolist())
            out[calib][str(eps)] = er
    return out


def run(name, seed):
    path = os.path.join(OUT, f"{name}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(OUT, exist_ok=True)
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
    arrays = dict(y_test=y[i_te])
    for order in ORDERS:
        t0 = time.time()
        model = at.build_anytime(F[i_fit], y[i_fit], U, T, FRACTIONS, order=order,
                                 rng=np.random.default_rng(seed + 1000))
        Mc, Pc = at.head_outputs(model, F[i_cal])
        Mt, Pt = at.head_outputs(model, F[i_te])
        cls = model["classes"]
        yt = y[i_te]
        full_t = Pt[:, -1]
        r = dict(fit_s=time.time() - t0, ckpt_units=model["ckpt_units"],
                 ckpt_rel_cost=((model["conv"] + model["head_cost"]) / model["full_cost"]).tolist(),
                 ckpt_acc=[float(np.mean(cls[Pt[:, k]] == yt)) for k in range(Pt.shape[1])],
                 ckpt_disagree=[float(np.mean(Pt[:, k] != full_t)) for k in range(Pt.shape[1])],
                 full_acc=float(np.mean(cls[full_t] == yt)))
        r.update(evaluate(model, Mc, Pc, Mt, Pt, yt))
        res["orders"][order] = r
        arrays.update({f"{order}_Mc": Mc.astype(np.float32), f"{order}_Pc": Pc.astype(np.int16),
                       f"{order}_Mt": Mt.astype(np.float32), f"{order}_Pt": Pt.astype(np.int16),
                       f"{order}_conv": model["conv"], f"{order}_head": model["head_cost"]})
    tmp = os.path.join(OUT, f"{name}_s{seed}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, os.path.join(OUT, f"{name}_s{seed}.npz"))
    save_json(res, path)  # json last: its presence marks the job as done
    return res


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    for nm in names:
        for s in seeds:
            t0 = time.time()
            r = run(nm, s)
            line = [f"{nm} s{s} ncal={r['n_cal']}"]
            for od in ORDERS:
                o = r["orders"][od]
                line.append(f"{od}: ltt {o['ltt']['0.01']['static']['cost']:.3f}/{o['ltt']['0.01']['margin']['cost']:.3f}"
                            f" crc {o['crc']['0.01']['static']['cost']:.3f}/{o['crc']['0.01']['margin']['cost']:.3f}")
            print(" | ".join(line) + f" ({time.time() - t0:.0f}s)", flush=True)
