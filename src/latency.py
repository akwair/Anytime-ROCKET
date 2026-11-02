"""Batch-1 latency of the certified anytime model vs the full model (same compiled kernel, and
aeon's MiniRocket transform + ridge as an external reference). Single thread.
Writes results/latency/<ds>_s<seed>.json and checks agreement with the offline simulation."""
import json
import os
import sys
import time

import numpy as np
from numba import set_num_threads

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anytime as at  # noqa: E402
import fastexit as fx  # noqa: E402
import pilot  # noqa: E402
from data import load_ucr, save_json  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "latency")


def timeit(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(name, seed=0, eps=0.01, n_max=500):
    path = os.path.join(OUT, f"{name}_s{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    set_num_threads(1)
    Xtr, ytr, Xte, yte = load_ucr(name)
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    T = X.shape[1]
    idx = np.random.default_rng(seed).permutation(len(X))
    n_fit = len(Xtr)
    n_cal = (len(X) - n_fit) // 2
    i_fit, i_cal, i_te = idx[:n_fit], idx[n_fit:n_fit + n_cal], idx[n_fit + n_cal:]
    params = at.fit_params(X[i_fit], seed=seed)
    F = at.transform(X, params)
    U = at.units_of(params, T)
    model = at.build_anytime(F[i_fit], y[i_fit], U, T, pilot.FRACTIONS, order="importance")
    Mc, Pc = at.head_outputs(model, F[i_cal])
    Mt, Pt = at.head_outputs(model, F[i_te])
    res = dict(dataset=name, seed=seed, eps=eps, T=T)
    Xt = X[i_te][:n_max]
    for rule in pilot.RULES:
        lam = pilot.certify_adaptive(Mc, Pc, eps, rule)
        pk = fx.pack(model, U, params, rule)
        fx.run_batch1(pk, Xt[:2], lam)  # compile
        pred, ex = fx.run_batch1(pk, Xt, lam)
        e_sim = at.exit_index(Mt[:n_max], Pt[:n_max], lam, rule)
        p_sim = Pt[np.arange(len(e_sim)), e_sim]
        t_any = timeit(lambda: fx.run_batch1(pk, Xt, lam))
        res[rule] = dict(lam=float(lam), sec_per_series=t_any / len(Xt), match_pred=float(np.mean(pred == p_sim)),
                         match_exit=float(np.mean(ex == e_sim)), rel_ops=float(np.mean(at.anytime_cost(model, e_sim))))
    pk = fx.pack(model, U, params, "margin")
    fx.run_batch1(pk, Xt[:2], 0, full_only=True)
    pf, _ = fx.run_batch1(pk, Xt, 0, full_only=True)
    res["full_same_kernel"] = dict(sec_per_series=timeit(lambda: fx.run_batch1(pk, Xt, 0, full_only=True)) / len(Xt),
                                   match_pred=float(np.mean(pf == Pt[:n_max, -1])))
    full = model["heads"][-1].clf
    sc = model["scaler"]
    Xt32 = Xt.astype(np.float32)
    at.transform(Xt32[:1], params)

    def aeon_b1():
        for i in range(len(Xt32)):
            full.decision_function(sc.transform(at.transform(Xt32[i:i + 1], params)))

    res["full_aeon"] = dict(sec_per_series=timeit(aeon_b1) / len(Xt))
    save_json(res, path)
    return res


if __name__ == "__main__":
    for nm in sys.argv[1].split(","):
        r = run(nm)
        f = r["full_same_kernel"]["sec_per_series"]
        print(f"{nm}: full {f * 1e3:.3f} ms (aeon {r['full_aeon']['sec_per_series'] * 1e3:.3f}), "
              + ", ".join(f"{k} {r[k]['sec_per_series'] * 1e3:.3f} ms x{f / r[k]['sec_per_series']:.1f} "
                          f"ops {r[k]['rel_ops']:.3f} match {r[k]['match_pred']:.3f}/{r[k]['match_exit']:.3f}"
                          for k in pilot.RULES), flush=True)
