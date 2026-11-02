"""Paired split-calibrated MiniRocket baselines; python src/runall.py ALL 0,1,2,3,4 8 baselines.

All selectors, transformations, normalization and hyperparameters use fit rows
only. Calibration sees predictions only. Test labels enter reporting only.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anytime as at
import grid2
from baseline_pruning import detach_masks, pocket_masks, POCKET_K, POCKET_EPOCHS
from data import load_ucr, save_json
import data as data_module

ROOT = Path(__file__).resolve().parents[1]
OUT = os.environ.get('BASELINE_OUT', str(ROOT / 'results/baselines'))
REFERENCE_DIR = os.environ.get('BASELINE_REFERENCE_DIR', str(ROOT / 'results/v2')) or None
NUM_FEATURES = 10000
FRACTIONS = [.01, .02, .04, .08, .16, .32, .64]
SCHEMA = 'split-baselines-v1'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def settings():
    sources = ['src/baselines.py', 'src/baseline_pruning.py', 'src/anytime.py', 'src/grid2.py',
               'src/data.py', 'third_party/detach_rocket/detach_rocket/sfd.py',
               'third_party/pocket/MiniROCKET/PROCKET_pruner.py']
    return dict(schema=SCHEMA, num_features=NUM_FEATURES, fractions=FRACTIONS,
                eps=grid2.EPS, delta=grid2.DELTA, calibrations=grid2.CALIBS,
                pocket_k=POCKET_K, pocket_epochs=POCKET_EPOCHS, pocket_tune_fraction=.32,
                pocket_cv_max_folds=3, detach_drop=.05, order='strat_imp', rule='margin',
                ridge_alphas=at.ALPHAS.tolist(), split='v2 seeded permutation; fit=official train size',
                calibration='exact split; no cross-fitting',
                provenance=dict(detach_commit='26749ba88c9c7644f4be7d081eef46a86406d8d3',
                                pocket_commit='510a530d527aa6abfe4caae91a6ea7867e5b348a',
                                protocol='standardized adaptations, not published-score reproductions'),
                source_hashes={s: file_hash(ROOT / s) for s in sources})


def protocol_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def split_indices(n, n_fit, seed):
    idx = np.random.default_rng(seed).permutation(n)
    n_cal = (n - n_fit) // 2
    return idx[:n_fit], idx[n_fit:n_fit+n_cal], idx[n_fit+n_cal:]


def subset_conv_cost(U, features, T):
    """Ideal pruned-transform operations, counting selected PPVs and shared work.

    This is NOT a measured latency or the cost of slicing a full feature matrix.
    """
    fi = np.unique(np.asarray(features, dtype=int))
    if not len(fi):
        return 0.
    if fi.min() < 0 or fi.max() >= U['p']:
        raise ValueError('Feature outside full layout')
    feature_units = np.repeat(np.arange(len(U['nf'])), U['nf'])
    unit_ids, counts = np.unique(feature_units[fi], return_counts=True)
    return float(2*T + 16*T*len(np.unique(U['dil'][unit_ids])) + 3*T*len(unit_ids)
                 + np.sum(counts * U['teff'][unit_ids]))


class ConstantHead:
    """Defined fit-only fallback for one class or an empty POCKET mask."""
    def __init__(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.winner = int(counts.argmax())
        self.clf = SimpleNamespace(classes_=classes, coef_=np.zeros((max(1, len(classes)), X.shape[1])), alpha_=1.)

    def scores(self, X):
        scores = np.zeros((len(X), max(2, len(self.clf.classes_))))
        scores[:, self.winner] = 1.
        return scores


def fit_head(Zfit, yfit):
    if len(np.unique(yfit)) == 1 or Zfit.shape[1] == 0:
        return ConstantHead(Zfit, yfit)
    return at.Head(Zfit, yfit)


def head_predict(head, Z):
    # margin_pred returns positions in the FIT classes, never dataset labels.
    return head.clf.classes_[at.margin_pred(head.scores(Z))[1]]


def build_reference(Ffit, yfit, U, T, seed):
    if len(np.unique(yfit)) > 1:
        return at.build_anytime(Ffit, yfit, U, T, FRACTIONS + [1.], order='strat_imp',
                                rng=np.random.default_rng(seed + 1000))
    scaler = StandardScaler().fit(Ffit)
    Z = scaler.transform(Ffit)
    full = fit_head(Z, yfit)
    perm = at.unit_order(U, full, 'strat_imp', np.random.default_rng(seed + 1000))
    cum = np.cumsum(U['nf'][perm])
    counts = sorted({min(len(perm), int(np.searchsorted(cum, fr*U['p'])+1))
                     for fr in FRACTIONS + [1.]})
    feats = [np.arange(U['p']) if c == len(perm) else at.feature_index(U, perm[:c]) for c in counts]
    heads = [full if len(f) == U['p'] else fit_head(Z[:, f], yfit) for f in feats]
    conv = np.array([at.conv_cost(U, perm[:c], T) for c in counts])
    head_cost = np.array([2*len(f) for f in feats])
    return dict(scaler=scaler, heads=heads, feats=feats, ckpt_units=counts, perm=perm,
                conv=conv, head_cost=head_cost, full_cost=conv[-1]+head_cost[-1],
                classes=np.unique(yfit), p=U['p'])


def choose_candidate(Pcal, eps, calib):
    losses = (Pcal != Pcal[:, [0]]).T.astype(np.uint8)
    j = grid2.select(losses, eps, calib)
    return dict(index=max(0, int(j)), certified=bool(j >= 0),
                status=('certified' if j >= 0 else 'known_safe_full_fallback'))


def candidate(mask, pc, pt, ideal, actual, **metadata):
    return dict(mask=np.asarray(mask, dtype=np.int32), pc=pc, pt=pt,
                ideal=float(ideal), actual=float(actual), **metadata)


def pack_family(name, candidates, full, ytest, arrays, adaptive=False):
    """Anchor full first; fixed order determined without calibration/test data."""
    omitted = []
    if not adaptive:
        kept = []
        for c in candidates:
            if c['ideal'] < full['ideal']:
                kept.append(c)
            else:
                omitted.append({k: v for k, v in c.items() if k not in ('mask', 'pc', 'pt')})
        candidates = sorted(kept, key=lambda c: -c['ideal'])
    candidates = [full] + candidates
    Pc = np.stack([c['pc'] for c in candidates], axis=1)
    Pt = np.stack([c['pt'] for c in candidates], axis=1)
    ideal = np.array([c['ideal'] / full['ideal'] for c in candidates])
    actual = np.array([c['actual'] / full['ideal'] for c in candidates])
    arrays.update({name+'_Pcal': Pc, name+'_Ptest': Pt,
                   name+'_ideal_cost': ideal, name+'_full_transform_cost': actual})
    descriptors = []
    for i, c in enumerate(candidates):
        arrays[f'{name}_mask_{i}'] = c['mask']
        descriptors.append({k: v for k, v in c.items() if k not in ('mask', 'pc', 'pt')})
    result = dict(candidates=descriptors, omitted_not_cheaper_than_full=omitted,
                  cost_kind='modeled operation proxy: ideal selective transform; full_transform_cost models full transform plus heads',
                  candidate_disagreement=np.mean(Pt != Pt[:, [0]], axis=0).tolist(),
                  candidate_accuracy=np.mean(Pt == ytest[:, None], axis=0).tolist())
    for calib in grid2.CALIBS:
        result[calib] = {}
        for eps in grid2.EPS:
            selected = choose_candidate(Pc, eps, calib)
            j = selected['index']
            selected.update(cost=float(ideal[j]), full_transform_cost=float(actual[j]),
                            disagree=float(np.mean(Pt[:, j] != Pt[:, 0])),
                            acc=float(np.mean(Pt[:, j] == ytest)),
                            cal_disagreements=int(np.sum(Pc[:, j] != Pc[:, 0])),
                            test_disagreements=int(np.sum(Pt[:, j] != Pt[:, 0])))
            result[calib][str(eps)] = selected
    return result


def check_v2(name, seed, model, Mc, Pc, Mt, Pt, ytest):
    if not REFERENCE_DIR:
        return dict(status='regenerated_no_external_reference')
    path = Path(REFERENCE_DIR) / f'{name}_s{seed}.npz'
    if not path.exists():
        return dict(status='regenerated_external_npz_absent')
    # v2 saves margins as float32 and predictions as class POSITIONS.
    expected = {'strat_imp_Mc': Mc.astype(np.float32), 'strat_imp_Mt': Mt.astype(np.float32),
                'strat_imp_Pc': Pc, 'strat_imp_Pt': Pt, 'strat_imp_conv': model['conv'],
                'strat_imp_head': model['head_cost'], 'y_test': ytest}
    with np.load(path, allow_pickle=False) as z:
        for key, value in expected.items():
            if key not in z or not np.array_equal(z[key], value):
                raise ValueError(f'Paired v2 equality failed: {name} seed={seed}, {key}')
    return dict(status='exact_v2_arrays_verified', sha256=file_hash(path), keys=list(expected))


def reusable(path, npz_path, config_hash, name, seed):
    if not path.exists() or not npz_path.exists():
        return None
    try:
        result = json.loads(path.read_text())
        if (result.get('schema') != SCHEMA or result.get('protocol_hash') != config_hash
                or result.get('dataset') != name or result.get('seed') != seed
                or result.get('npz_sha256') != file_hash(npz_path)):
            return None
        data_path = Path(data_module.ROOT) / name / 'arrays.npz'
        if data_path.exists() and result.get('input_sha256') != file_hash(data_path):
            return None
        with np.load(npz_path, allow_pickle=False) as arrays:
            if str(arrays['protocol_hash']) != config_hash:
                return None
            if len(arrays['full_Pcal']) != result['n_cal']:
                return None
        # Enabling a new external reference must not silently reuse an unverified result.
        ref = Path(REFERENCE_DIR) / f'{name}_s{seed}.npz' if REFERENCE_DIR else None
        if ref and ref.exists() and result['reference_check'].get('sha256') != file_hash(ref):
            return None
        return result
    except (ValueError, KeyError, OSError):
        return None


def run(name, seed):
    config = settings()
    config_hash = protocol_hash(config)
    out = Path(OUT)
    path, npz_path = out / f'{name}_s{seed}.json', out / f'{name}_s{seed}.npz'
    cached = reusable(path, npz_path, config_hash, name, seed)
    if cached is not None:
        return cached
    started = time.perf_counter()
    Xtr, ytr, Xte, yte = load_ucr(name)
    X, y = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    i_fit, i_cal, i_test = split_indices(len(X), len(Xtr), seed)
    if not len(i_cal) or not len(i_test):
        raise ValueError('Need nonempty calibration and test splits')
    T = X.shape[1]
    Xfit, yfit, ytest = X[i_fit], y[i_fit], y[i_test]
    params = at.fit_params(Xfit, num_features=NUM_FEATURES, seed=seed)
    F = at.transform(X, params)
    U = at.units_of(params, T)
    Ffit, Fcal, Ftest = F[i_fit], F[i_cal], F[i_test]
    del F
    model = build_reference(Ffit, yfit, U, T, seed)
    Mc, Pc = at.head_outputs(model, Fcal)
    Mt, Pt = at.head_outputs(model, Ftest)
    refcheck = check_v2(name, seed, model, Mc, Pc, Mt, Pt, ytest)
    classes = model['classes']
    pc, pt = classes[Pc[:, -1]], classes[Pt[:, -1]]
    fullconv, fullcost = float(model['conv'][-1]), float(model['full_cost'])
    c_eff = 1 if len(classes) <= 2 else len(classes)
    full = candidate(np.arange(U['p']), pc, pt, fullcost, fullcost,
                     kind='identical_full_reference', feature_count=int(U['p']))
    arrays = dict(protocol_hash=np.array(config_hash), fit_indices=i_fit, cal_indices=i_cal,
                  test_indices=i_test, y_test=ytest, classes=classes, full_Pcal=pc, full_Ptest=pt,
                  full_dilations=params[0], full_nfpd=params[1], full_biases=params[2],
                  strat_imp_Mc=Mc.astype(np.float32), strat_imp_Mt=Mt.astype(np.float32),
                  strat_imp_Pc=Pc, strat_imp_Pt=Pt, strat_imp_perm=model['perm'],
                  strat_imp_conv=model['conv'], strat_imp_head=model['head_cost'])
    result = dict(schema=SCHEMA, protocol_hash=config_hash, settings=config, dataset=name, seed=seed,
                  T=T, n_fit=len(i_fit), n_cal=len(i_cal), n_test=len(i_test),
                  n_features=int(U['p']), classes=classes.tolist(), n_classes_total=int(len(np.unique(y))),
                  full_acc=float(np.mean(pt == ytest)), full_cost=fullcost,
                  reference_check=refcheck, families={}, timings={},
                  packages={p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','aeon','numba')})
    data_path = Path(data_module.ROOT) / name / 'arrays.npz'
    result['input_sha256'] = file_hash(data_path) if data_path.exists() else None
    result['timings']['reference_s'] = time.perf_counter() - started
    Zfit = model['scaler'].transform(Ffit)
    Zcal = model['scaler'].transform(Fcal)
    Ztest = model['scaler'].transform(Ftest)

    def masked(mask, **meta):
        head = fit_head(Zfit[:, mask], yfit)
        ideal = subset_conv_cost(U, mask, T) + len(mask)*(c_eff+1)
        actual = fullconv + len(mask)*(c_eff+1)
        return candidate(mask, head_predict(head, Zcal[:, mask]), head_predict(head, Ztest[:, mask]),
                         ideal, actual, feature_count=len(mask), alpha=float(head.clf.alpha_), **meta)

    t0 = time.perf_counter()
    small = []
    for fraction in FRACTIONS:
        requested = max(84, int(NUM_FEATURES * fraction))
        par = at.fit_params(Xfit, num_features=requested, seed=seed)
        Fs = at.transform(X, par)
        scaler = StandardScaler().fit(Fs[i_fit])
        head = fit_head(scaler.transform(Fs[i_fit]), yfit)
        Us = at.units_of(par, T)
        cost = at.conv_cost(Us, np.arange(len(Us['nf'])), T) + Us['p']*(c_eff+1)
        small.append(candidate(np.arange(Us['p']), head_predict(head, scaler.transform(Fs[i_cal])),
                               head_predict(head, scaler.transform(Fs[i_test])), cost, cost,
                               fraction=fraction, feature_count=int(Us['p']), requested_features=requested,
                               alpha=float(head.clf.alpha_), kind='independent_small_refit'))
        key = str(fraction)
        arrays[f'small_{key}_dilations'], arrays[f'small_{key}_nfpd'], arrays[f'small_{key}_biases'] = par
    del Fs
    result['families']['small_minirocket'] = pack_family('small_minirocket', small, full, ytest, arrays)
    result['timings']['small_refits_s'] = time.perf_counter() - t0

    targets = [max(1, int(fr * U['p'])) for fr in FRACTIONS]
    t0 = time.perf_counter()
    prefix = [masked(np.arange(target), fraction=fr, kind='natural_layout_prefix')
              for fr, target in zip(FRACTIONS, targets)]
    result['families']['ridge_prefix'] = pack_family('ridge_prefix', prefix, full, ytest, arrays)
    result['timings']['prefix_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    alpha = float(model['heads'][-1].clf.alpha_)
    detach = detach_masks(Zfit, yfit, targets, alpha=alpha)
    candidates = [masked(detach[target], fraction=fr, target=target, kind='detach_sfd')
                  for fr, target in zip(FRACTIONS, targets)]
    result['families']['detach_minirocket'] = pack_family('detach_minirocket', candidates, full, ytest, arrays)
    result['families']['detach_minirocket']['selector'] = dict(alpha=alpha, drop_ratio=.05,
        importance='max absolute class coefficient', schedule='first geometric step at or below budget',
        adaptation='float64 SFD; stop at smallest target; no zero-weight resurrection; common ridge CV refit')
    result['timings']['detach_s'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pocket, metadata = pocket_masks(Ffit, yfit, targets, seed)
    candidates = [masked(pocket[target], fraction=fr, target=target, kind='pocket_pruning')
                  for fr, target in zip(FRACTIONS, targets)]
    result['families']['pocket_minirocket'] = pack_family('pocket_minirocket', candidates, full, ytest, arrays)
    result['families']['pocket_minirocket']['selector'] = metadata
    result['timings']['pocket_s'] = time.perf_counter() - t0

    static = [candidate(fi, classes[Pc[:, j]], classes[Pt[:, j]],
                        model['conv'][j]+model['head_cost'][j], fullconv+model['head_cost'][j],
                        feature_count=len(fi), checkpoint=j, kind='strat_imp_static')
              for j, fi in enumerate(model['feats'][:-1])]
    result['families']['strat_imp_static'] = pack_family('strat_imp_static', static, full, ytest, arrays)

    # Fixed margin grid from v2, with the identical plain full anchor at infinity.
    # Costs below are test-average modeled operations, never used to order/select.
    adaptive = []
    for lam in at.LAM_GRID[1:]:
        ec = at.exit_index(Mc, Pc, lam, 'margin')
        et = at.exit_index(Mt, Pt, lam, 'margin')
        ideal = np.mean(at.anytime_cost(model, et)) * fullcost
        actual = fullconv + np.mean(np.cumsum(model['head_cost'])[et])
        adaptive.append(candidate([], classes[Pc[np.arange(len(ec)), ec]],
                                  classes[Pt[np.arange(len(et)), et]], ideal, actual,
                                  lam=float(lam), kind='strat_imp_adaptive'))
    result['families']['strat_imp_adaptive'] = pack_family('strat_imp_adaptive', adaptive, full, ytest, arrays, adaptive=True)
    result['families']['strat_imp_adaptive']['order'] = 'fixed descending margin thresholds, as v2; not ordered by observed test cost'
    # Existing grid2 API independently checks the paired evaluation for both rules.
    existing = grid2.evaluate(model, Mc, Pc, Mt, Pt, ytest)
    for calib in grid2.CALIBS:
        for eps in grid2.EPS:
            key = str(eps)
            for family, old in [('strat_imp_static','static'), ('strat_imp_adaptive','margin')]:
                new = result['families'][family][calib][key]
                for metric in ('cost', 'acc', 'disagree'):
                    if not np.isclose(new[metric], existing[calib][key][old][metric], rtol=1e-12, atol=1e-12):
                        raise ValueError(f'grid2 comparator mismatch: {family}/{calib}/{eps}/{metric}')
    result['timings']['total_s'] = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f'{name}_s{seed}.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, npz_path)
    result['npz_sha256'] = file_hash(npz_path)
    save_json(result, str(path))  # completion marker LAST
    return result


if __name__ == '__main__':
    for name in sys.argv[1].split(','):
        for seed in [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else '0').split(',')]:
            r = run(name, seed)
            print(f"{name} s{seed}: {r['reference_check']['status']}; {r['timings']['total_s']:.1f}s", flush=True)
