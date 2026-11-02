"""Anytime MiniRocket: importance-ordered (kernel, dilation) units, independent ridge heads at
geometric checkpoints, and a label-free Learn-then-Test (LTT) calibrated early-exit rule.

Everything here is offline simulation on precomputed features. Costs are a shared structural
operation proxy (shared dilation/unit work plus PPV/head terms), not exact arithmetic counts
or measured time. Latency is measured separately with a batch-1 kernel.
"""
import numpy as np
from scipy.stats import binom
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.convolution_based._minirocket import _fit as mr_fit
from aeon.transformations.collection.convolution_based._minirocket import _transform as mr_transform

ALPHAS = np.logspace(-3, 3, 10)
N_KERNELS = 84


def fit_params(X, num_features=10_000, seed=0):
    return mr_fit(X.astype(np.float32), num_features, 32, np.int32(seed))


def transform(X, params):
    return mr_transform(X.astype(np.float32), params)


def units_of(params, T):
    """Feature units in aeon's layout (dilation-major, then kernel). Returns per-unit arrays."""
    dilations, nfpd, _ = params
    dil, ker, start, nf, teff = [], [], [], [], []
    f = 0
    for di, d in enumerate(dilations):
        pad = ((9 - 1) * int(d)) // 2
        for k in range(N_KERNELS):
            dil.append(di)
            ker.append(k)
            start.append(f)
            nf.append(int(nfpd[di]))
            teff.append(T if (di % 2 + k) % 2 == 0 else T - 2 * pad)
            f += int(nfpd[di])
    return dict(dil=np.array(dil), ker=np.array(ker), start=np.array(start), nf=np.array(nf),
                teff=np.array(teff), n_dil=len(dilations), p=f)


def conv_cost(U, unit_ids, T):
    """Ops of computing the given units for one series: 2T for A,G; 16T per touched dilation
    (C_alpha adds + C_gamma copies); 3T per unit for C; T_eff compares per feature (PPV)."""
    unit_ids = np.asarray(unit_ids)
    n_dil = len(np.unique(U["dil"][unit_ids]))
    return 2 * T + 16 * T * n_dil + 3 * T * len(unit_ids) + float(np.sum(U["nf"][unit_ids] * U["teff"][unit_ids]))


def feature_index(U, unit_ids):
    return np.concatenate([np.arange(U["start"][u], U["start"][u] + U["nf"][u]) for u in unit_ids])


class Head:
    def __init__(self, Z, y):
        self.clf = RidgeClassifierCV(alphas=ALPHAS).fit(Z, y)

    def scores(self, Z):
        d = self.clf.decision_function(Z)
        if d.ndim == 1:
            d = np.stack([-d, d], 1)
        return d


def margin_pred(S):
    s = np.sort(S, 1)
    return s[:, -1] - s[:, -2], S.argmax(1)


def unit_order(U, full, order, rng):
    """'importance': units by squared full-head weight; 'random': permutation; 'strat' / 'strat_imp':
    round-robin over dilations (one unit per dilation per round), units within a dilation and the
    dilation order taken at random ('strat') or by importance ('strat_imp')."""
    n_units = len(U["nf"])
    coef = np.atleast_2d(full.clf.coef_)
    w2 = np.sum(coef ** 2, 0)
    imp = np.array([w2[U["start"][u]:U["start"][u] + U["nf"][u]].sum() for u in range(n_units)])
    if order == "importance":
        return np.argsort(-imp, kind="stable")
    if order == "random":
        return rng.permutation(n_units)
    if order == "strat":
        key, dkey = rng.random(n_units), rng.random(U["n_dil"])
    elif order == "strat_imp":
        key, dkey = imp, np.bincount(U["dil"], weights=imp, minlength=U["n_dil"])
    else:
        raise ValueError(order)
    rank = np.empty(n_units)
    for di in range(U["n_dil"]):
        ids = np.flatnonzero(U["dil"] == di)
        rank[ids[np.argsort(-key[ids], kind="stable")]] = np.arange(len(ids))
    drank = np.empty(U["n_dil"])
    drank[np.argsort(-dkey, kind="stable")] = np.arange(U["n_dil"])
    return np.lexsort((drank[U["dil"]], rank))


def build_anytime(Ffit, yfit, U, T, fractions, order="importance", rng=None, ckpt_units=None):
    """Fit scaler, the full head, a unit order and one independent head per checkpoint prefix.
    ckpt_units (optional) fixes the checkpoint unit counts, e.g. to match a reference model."""
    scaler = StandardScaler().fit(Ffit)
    Z = scaler.transform(Ffit)
    full = Head(Z, yfit)
    n_units = len(U["nf"])
    perm = unit_order(U, full, order, rng)
    if ckpt_units is None:
        cum = np.cumsum(U["nf"][perm])
        ckpt_units = sorted({int(np.searchsorted(cum, fr * U["p"]) + 1) for fr in fractions})
        ckpt_units = [min(c, n_units) for c in ckpt_units]
        if ckpt_units[-1] != n_units:
            ckpt_units.append(n_units)
        ckpt_units = sorted(set(ckpt_units))
    n_cls = len(full.clf.classes_)
    c_eff = 1 if n_cls == 2 else n_cls
    heads, feats, conv, head_cost = [], [], [], []
    for cu in ckpt_units:
        ids = perm[:cu]
        fi = np.arange(U["p"]) if cu == n_units else feature_index(U, ids)
        feats.append(fi)
        heads.append(full if cu == n_units else Head(Z[:, fi], yfit))
        conv.append(conv_cost(U, ids, T))
        head_cost.append(len(fi) * (c_eff + 1))
    full_cost = conv[-1] + head_cost[-1]
    return dict(scaler=scaler, heads=heads, feats=feats, ckpt_units=ckpt_units, perm=perm,
                conv=np.array(conv), head_cost=np.array(head_cost), full_cost=full_cost,
                classes=full.clf.classes_, p=U["p"])


def head_outputs(model, F):
    Z = model["scaler"].transform(F)
    M, P = [], []
    for h, fi in zip(model["heads"], model["feats"]):
        m, p = margin_pred(h.scores(Z[:, fi]))
        M.append(m)
        P.append(p)
    return np.stack(M, 1), np.stack(P, 1)  # [n, K]


def exit_index(M, P, lam, rule="margin"):
    """First checkpoint whose margin >= lam (rule 'margin'), additionally agreeing with the
    previous checkpoint for rule 'consist'. The last checkpoint always exits."""
    ok = M >= lam
    if rule == "consist":
        agree = np.zeros_like(ok)
        agree[:, 1:] = P[:, 1:] == P[:, :-1]
        ok = ok & agree
    ok[:, -1] = True
    return ok.argmax(1)


def anytime_cost(model, e):
    """Per-instance ops when exiting at checkpoint e: conv up to e plus every head evaluated."""
    cum_head = np.cumsum(model["head_cost"])
    return (model["conv"][e] + cum_head[e]) / model["full_cost"]


def binom_pvalue(n_err, n, eps):
    """p-value of H0: risk > eps given n_err disagreements out of n (exact binomial tail)."""
    return binom.cdf(n_err, n, eps)


def ltt_fixed_sequence(losses, eps, delta):
    """losses: [L, n] 0/1 losses ordered from most to least conservative. Fixed-sequence test:
    return the last index j such that hypotheses 0..j are all rejected, or -1 if none."""
    n = losses.shape[1]
    j_best = -1
    for j in range(losses.shape[0]):
        if binom_pvalue(int(losses[j].sum()), n, eps) <= delta:
            j_best = j
        else:
            break
    return j_best


def crc_select(losses, eps):
    """Conformal risk control (Angelopoulos et al., ICLR 2024) on [L, n] 0/1 losses ordered from
    most to least conservative, row 0 having zero loss. Losses are made monotone per instance by a
    running max (an upper bound, so validity holds); return the last j with (sum + 1)/(n + 1) <= eps,
    giving E[loss] <= eps. Needs only n >= 1/eps - 1 instead of ln(1/delta)/eps."""
    n = losses.shape[1]
    risk = (np.maximum.accumulate(losses, 0).sum(1) + 1) / (n + 1)
    ok = np.flatnonzero(risk <= eps)
    return int(ok[-1]) if len(ok) else -1


LAM_GRID = np.concatenate([[np.inf], np.linspace(4.0, 0.0, 201)])
