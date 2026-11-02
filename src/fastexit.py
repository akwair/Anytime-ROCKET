"""Batch-1 compiled anytime MiniRocket: units are computed in the learned order, per-dilation
convolution sums are built lazily on first use, and the exit test runs inside the compiled loop.
Used only for wall-clock latency; predictions must match the offline simulation in anytime.py.
"""
import numpy as np
from numba import njit

from anytime import feature_index

INDICES = np.array([c for c in __import__("itertools").combinations(range(9), 3)], dtype=np.int32)


def pack(model, U, params, rule="margin"):
    """Flatten a model from anytime.build_anytime into arrays for the compiled kernel. Head
    weights are re-indexed to the perm-order feature buffer and the scaler is folded in."""
    dilations, _, biases = params
    perm = model["perm"]
    buf_idx = feature_index(U, perm)  # aeon feature index of each buffer slot
    pos = np.empty(U["p"], np.int64)
    pos[buf_idx] = np.arange(U["p"])
    mu, sd = model["scaler"].mean_, model["scaler"].scale_
    W, B, woff, nfeat = [], [], [0], []
    for h, fi in zip(model["heads"], model["feats"]):
        coef = np.atleast_2d(h.clf.coef_)
        icpt = np.atleast_1d(h.clf.intercept_)
        if coef.shape[0] == 1:
            coef = np.vstack([-coef, coef])
            icpt = np.array([-icpt[0], icpt[0]])
        Wk = coef / sd[fi]
        bk = icpt - Wk @ mu[fi]
        order = np.argsort(pos[fi])  # head features sorted by buffer slot = buffer prefix
        assert np.array_equal(pos[fi][order], np.arange(len(fi)))
        W.append(Wk[:, order].ravel())
        B.append(bk)
        woff.append(woff[-1] + Wk.size)
        nfeat.append(len(fi))
    return dict(
        dilations=dilations.astype(np.int32), biases=biases.astype(np.float32),
        u_dil=U["dil"][perm].astype(np.int32), u_ker=U["ker"][perm].astype(np.int32),
        u_start=U["start"][perm].astype(np.int64), u_nf=U["nf"][perm].astype(np.int32),
        ck_units=np.array(model["ckpt_units"], np.int32), ck_nfeat=np.array(nfeat, np.int64),
        W=np.concatenate(W).astype(np.float32), B=np.stack(B).astype(np.float32),
        woff=np.array(woff, np.int64), consist=rule == "consist")


@njit(fastmath=True, cache=True)
def predict_one(x, dilations, biases, indices, u_dil, u_ker, u_start, u_nf, ck_units, ck_nfeat,
                W, B, woff, lam, consist, first_ck):
    T = x.shape[0]
    n_dil = dilations.shape[0]
    n_cls = B.shape[1]
    A = -x
    G = 3 * x
    have = np.zeros(n_dil, np.bool_)
    Calpha = np.empty((n_dil, T), np.float32)
    Cgamma = np.empty((n_dil, 9, T), np.float32)
    buf = np.empty(ck_nfeat[-1], np.float32)
    C = np.empty(T, np.float32)
    s = np.zeros(n_cls, np.float32)
    f = 0
    u = 0
    prev = -1
    K = ck_units.shape[0]
    for k in range(K):
        while u < ck_units[k]:
            di = u_dil[u]
            d = dilations[di]
            pad = ((9 - 1) * d) // 2
            if not have[di]:
                Calpha[di, :] = A
                Cgamma[di, :, :] = 0
                Cgamma[di, 4, :] = G
                start = d
                end = T - pad
                for g in range(4):
                    for t in range(end):
                        Calpha[di, T - end + t] += A[t]
                        Cgamma[di, g, T - end + t] = G[t]
                    end += d
                for g in range(5, 9):
                    for t in range(T - start):
                        Calpha[di, t] += A[start + t]
                        Cgamma[di, g, t] = G[start + t]
                    start += d
                have[di] = True
            ki = u_ker[u]
            ca = Calpha[di]
            g0 = Cgamma[di, indices[ki, 0]]
            g1 = Cgamma[di, indices[ki, 1]]
            g2 = Cgamma[di, indices[ki, 2]]
            for t in range(T):
                C[t] = ca[t] + g0[t] + g1[t] + g2[t]
            lo, hi = 0, T
            if (di % 2 + ki) % 2 != 0:
                lo, hi = pad, T - pad
            Cs = C[lo:hi]
            inv = np.float32(1) / np.float32(hi - lo)
            for j in range(u_nf[u]):
                b = biases[u_start[u] + j]
                c = 0
                for t in range(Cs.shape[0]):
                    c += Cs[t] > b
                buf[f] = c * inv
                f += 1
            u += 1
        if k < first_ck:
            continue
        nf = ck_nfeat[k]
        o = woff[k]
        for c in range(n_cls):
            acc = B[k, c]
            for j in range(nf):
                acc += W[o + c * nf + j] * buf[j]
            s[c] = acc
        best, second, arg = -1e30, -1e30, 0
        for c in range(n_cls):
            if s[c] > best:
                second = best
                best = s[c]
                arg = c
            elif s[c] > second:
                second = s[c]
        ok = best - second >= lam
        if consist and arg != prev:
            ok = False
        prev = arg
        if ok or k == K - 1:
            return arg, k
    return arg, K - 1


def run_batch1(pk, X, lam, full_only=False):
    X = X.astype(np.float32)
    first = len(pk["ck_units"]) - 1 if full_only else 0
    pred = np.empty(len(X), np.int64)
    ex = np.empty(len(X), np.int64)
    for i in range(len(X)):
        pred[i], ex[i] = predict_one(X[i], pk["dilations"], pk["biases"], INDICES, pk["u_dil"], pk["u_ker"],
                                     pk["u_start"], pk["u_nf"], pk["ck_units"], pk["ck_nfeat"], pk["W"], pk["B"],
                                     pk["woff"], np.float32(lam), pk["consist"], first)
    return pred, ex
