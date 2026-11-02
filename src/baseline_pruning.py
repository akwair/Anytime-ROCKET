"""Fit-only MiniRocket pruning, adapted from the pinned official SFD/POCKET cores.

No calibration/test arguments are accepted by selectors. These are standardized
comparators, not reproductions of published train/test protocols or scores.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold

POCKET_K = [10., 1., .1, .01, 100., 1000.]
POCKET_EPOCHS = 50
DETACH_DROP = .05


def center_l2(X):
    Z = np.array(X, dtype=np.float64, copy=True)
    mean = Z.mean(axis=0)
    Z -= mean
    norm = np.linalg.norm(Z, axis=0)
    norm[norm == 0] = 1.
    Z /= norm
    return Z, mean, norm


class PocketSolve:
    """Action of (k I + X.T X)^-1, with a min(n,p) Cholesky factor.

    The n<p branch uses Woodbury without forming a dense p-by-p inverse.
    The p<=n branch is the equivalent primal solve. Reused across retention budgets.
    """
    def __init__(self, X, k, gram=None):
        self.X, self.k = X, float(k)
        self.dual = len(X) < X.shape[1]
        G = (X @ X.T if self.dual else X.T @ X) if gram is None else gram.copy()
        G.flat[::len(G) + 1] += k
        self.factor = cho_factor(G, lower=True, overwrite_a=True, check_finite=False)

    def __call__(self, rhs):
        if self.dual:
            return (rhs - self.X.T @ cho_solve(self.factor, self.X @ rhs, check_finite=False)) / self.k
        return cho_solve(self.factor, rhs, check_finite=False)


def pocket_core(X, y, remain, k, epochs=POCKET_EPOCHS, solver=None):
    """Official MiniROCKET one-dual updates (reweight=False, epsilon=None).

    Matches W/Theta and the threshold-based feature mask; robust 0/0 shrinkage
    maps zero rows to zero. Noncontiguous labels are encoded using fit classes.
    """
    X = np.asarray(X, dtype=np.float64)
    classes, yi = np.unique(y, return_inverse=True)
    Y = -np.ones((len(y), len(classes)))
    Y[np.arange(len(y)), yi] = 1.
    mean = Y.mean(0)
    Y -= mean
    p = X.shape[1]
    remain = min(max(int(remain), 1), p - 1)
    solve = solver if solver is not None else PocketSolve(X, k)
    theta, dual = np.zeros((p, len(classes))), np.zeros((p, len(classes)))
    XtY = X.T @ Y
    for _ in range(epochs):
        W = solve(k * (theta + dual) + XtY)
        W_tilde = W - dual
        norm = np.linalg.norm(W_tilde, axis=1)
        threshold = np.partition(norm, p - remain - 1)[p - remain - 1]
        shrink = np.zeros(p)
        positive = norm > 0
        shrink[positive] = np.maximum(0., 1. - threshold / norm[positive])
        theta = shrink[:, None] * W_tilde
        dual += theta - W
    if not np.isfinite(W).all():
        raise FloatingPointError('Nonfinite POCKET solution')
    mask = np.flatnonzero(shrink > 0)
    sparse = np.zeros_like(W)
    sparse[mask] = W[mask]
    return dict(W=W, theta=theta, sparse_W=sparse, mask=mask, classes=classes, mean=mean)


def pocket_masks(Ffit, yfit, targets, seed=0):
    """Tune k once at 32% retention, then reuse k for all target masks.

    Three stratified fit-only folds (reduced for class counts of two); singleton
    classes skip CV and use the official default k=1. CV scores the official
    threshold-based pruner, then deployment refits a common ridge separately.
    """
    _, counts = np.unique(yfit, return_counts=True)
    nfold = min(3, int(counts.min())) if len(counts) > 1 else 0
    p = Ffit.shape[1]
    tuning_target = min(p - 1, max(1, int(.32 * p)))
    scores = np.zeros((len(POCKET_K), nfold)) if nfold >= 2 else None
    if nfold >= 2:
        cv = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
        for fold, (train, val) in enumerate(cv.split(Ffit, yfit)):
            X, mean, norm = center_l2(Ffit[train])
            Xv = (Ffit[val] - mean) / norm
            gram = X @ X.T if len(X) < p else X.T @ X
            for i, k in enumerate(POCKET_K):
                result = pocket_core(X, yfit[train], tuning_target, k,
                                     solver=PocketSolve(X, k, gram))
                pred = result['classes'][(Xv @ result['sparse_W'] + result['mean']).argmax(1)]
                scores[i, fold] = np.mean(pred == yfit[val])
        k = POCKET_K[int(scores.mean(1).argmax())]
        cv_status = 'fit_only_stratified_cv'
    else:
        k, cv_status = 1., 'default_k_singleton_or_single_class_no_cv'
    X, _, _ = center_l2(Ffit)
    solver = PocketSolve(X, k)
    masks = {int(target): pocket_core(X, yfit, int(target), k, solver=solver)['mask']
             for target in targets}
    metadata = dict(k=k, cv_status=cv_status, cv_folds=nfold if nfold >= 2 else 0,
                    cv_scores=None if scores is None else scores.tolist(), k_grid=POCKET_K,
                    epochs=POCKET_EPOCHS, tuning_target=tuning_target,
                    adaptation='k tuned once at 32% retention; reused across all budgets; common ridge refit',
                    mask_rule='official final positive shrinkage; threshold ties may retain fewer features')
    return masks, metadata


def detach_masks(Zfit, yfit, targets, alpha, drop_ratio=DETACH_DROP):
    """Official geometric SFD schedule, fixed alpha, max absolute coefficient.

    Return the first schedule mask at or below each target count. Only currently
    active features compete, so discarded zero-weight features cannot reappear.
    Stop at the smallest requested target instead of scanning all 150 steps.
    """
    # The official equations are unchanged; double precision avoids unstable
    # float32 dual ridge solves at the small fit-selected alpha values.
    Zfit = np.asarray(Zfit, dtype=np.float64)
    p = Zfit.shape[1]
    targets = sorted(set(int(t) for t in targets), reverse=True)
    active = np.arange(p)
    clf = RidgeClassifier(alpha=float(alpha)).fit(Zfit, yfit)
    importance = np.max(np.abs(np.atleast_2d(clf.coef_)), axis=0)
    output, step, previous = {}, 0, p + 1
    while len(output) < len(targets):
        count = max(1, int(p * (1. - drop_ratio) ** step))
        step += 1
        if count == previous:
            continue
        previous = count
        active = np.sort(active[np.argsort(importance)[-count:]])
        clf.fit(Zfit[:, active], yfit)
        importance = np.max(np.abs(np.atleast_2d(clf.coef_)), axis=0)
        for target in targets:
            if target not in output and count <= target:
                output[target] = active.copy()
    return output
