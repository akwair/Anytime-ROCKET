"""Summarize paired baseline runs, keeping dataset and run denominators explicit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from scipy.stats import wilcoxon


def verify_predictions(path, result):
    """Reject damaged, mismatched or structurally inconsistent raw predictions."""
    npz_path = path.with_suffix('.npz')
    if not npz_path.exists():
        raise ValueError(f'Missing predictions: {path}')
    digest = hashlib.sha256()
    with npz_path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    if result.get('npz_sha256') != digest.hexdigest():
        raise ValueError(f'NPZ hash mismatch: {npz_path}')
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            if str(arrays['protocol_hash']) != result['protocol_hash']:
                raise ValueError(f'NPZ internal protocol mismatch: {npz_path}')
            classes = arrays['classes']
            if classes.ndim != 1 or not len(classes) or not np.array_equal(classes, result['classes']):
                raise ValueError(f'Fitted class mismatch: {npz_path}')
            full = {}
            for split, size in [('cal', result['n_cal']), ('test', result['n_test'])]:
                full[split] = arrays['full_P' + split]
                if size <= 0 or full[split].shape != (size,) or not np.isin(full[split], classes).all():
                    raise ValueError(f'Invalid full prediction dimensions/classes: {npz_path}')
            if arrays['y_test'].shape != (result['n_test'],):
                raise ValueError(f'Test label dimensions mismatch: {npz_path}')
            for family, values in result['families'].items():
                n_candidates = len(values['candidates'])
                if not n_candidates:
                    raise ValueError(f'Empty candidate family: {family} in {npz_path}')
                for split in ('cal', 'test'):
                    predictions = arrays[family + '_P' + split]
                    if predictions.shape != (len(full[split]), n_candidates):
                        raise ValueError(f'Candidate prediction dimensions mismatch: {family} in {npz_path}')
                    if not np.array_equal(predictions[:, 0], full[split]):
                        raise ValueError(f'Candidate full anchor mismatch: {family} in {npz_path}')
                    if not np.isin(predictions, classes).all():
                        raise ValueError(f'Candidate prediction classes mismatch: {family} in {npz_path}')
                for cost in ('ideal_cost', 'full_transform_cost'):
                    costs = arrays[family + '_' + cost]
                    if (costs.shape != (n_candidates,) or not np.isfinite(costs).all()
                            or np.any(costs < 0) or costs[0] != 1.):
                        raise ValueError(f'Invalid candidate costs: {family} in {npz_path}')
    except (KeyError, OSError, EOFError, BadZipFile) as exc:
        raise ValueError(f'Invalid raw predictions: {npz_path}: {exc}') from exc


def load_verified_runs(directory, seeds):
    """Primary summaries require exact paired v2 verification and one environment."""
    runs = []
    seen = set()
    package_versions = None
    for path in sorted(Path(directory).glob('*.json')):
        r = json.loads(path.read_text())
        if r.get('schema', '').startswith('split-baselines-') and r['seed'] in seeds:
            pair = (r['dataset'], r['seed'])
            if pair in seen:
                raise ValueError(f'Duplicate dataset/seed pair: {pair} in {path}')
            seen.add(pair)
            if r.get('reference_check', {}).get('status') != 'exact_v2_arrays_verified':
                raise ValueError(f'Primary summary requires exact v2 reference verification: {path}')
            packages = r.get('packages')
            if not isinstance(packages, dict) or not packages:
                raise ValueError(f'Missing package versions: {path}')
            if package_versions is None:
                package_versions = packages
            elif packages != package_versions:
                raise ValueError(f'Mixed package versions: summarize environments separately ({path})')
            verify_predictions(path, r)
            runs.append(r)
    if not runs:
        raise ValueError(f'No baseline result pairs in {directory}')
    return runs


def summarize(directory, seeds=(0, 1, 2, 3, 4)):
    runs = load_verified_runs(directory, seeds)
    hashes = {r['protocol_hash'] for r in runs}
    if len(hashes) != 1:
        raise ValueError('Mixed protocol/source hashes: summarize separately or regenerate stale runs')
    grouped = defaultdict(list)
    for r in runs:
        grouped[r['dataset']].append(r)
    complete = {name: values for name, values in grouped.items() if {v['seed'] for v in values} == set(seeds)}
    output = dict(protocol_hash=next(iter(hashes)), n_runs=len(runs), n_datasets=len(grouped),
                  packages=runs[0]['packages'], integrity='NPZ hashes, internal protocol, shapes/classes/full anchors verified; exact v2 references required',
                  expected_datasets=113, expected_runs=113*len(seeds), seeds=list(seeds),
                  datasets_with_all_seeds=len(complete),
                  incomplete_datasets={name: sorted(set(seeds)-{v['seed'] for v in values})
                                       for name, values in grouped.items() if name not in complete},
                  scope='same v2 split, exact split calibration, standardized baseline adaptations',
                  cost_warning='both costs are modeled operation proxies, not measured speedups; full_transform_cost models full transform plus heads',
                  tables={}, paired_complete_datasets={})
    families = list(runs[0]['families'])
    for calib in ('ltt', 'crc'):
        for eps in runs[0]['settings']['eps']:
            key = f'{calib}_{eps}'
            output['tables'][key] = {}
            output['paired_complete_datasets'][key] = {}
            for family in families:
                metrics = [r['families'][family][calib][str(eps)] for r in runs]
                # Macro summaries average seeds within each dataset first.
                per_dataset = {name: {m: float(np.mean([r['families'][family][calib][str(eps)][m]
                                                       for r in values]))
                                     for m in ('cost','full_transform_cost','acc','disagree')}
                               for name, values in grouped.items()}
                table = {m: float(np.mean([v[m] for v in per_dataset.values()]))
                         for m in ('cost','full_transform_cost','acc','disagree')}
                table.update(certified_runs=sum(m['certified'] for m in metrics),
                             cheaper_selected_runs=sum(m['cost'] < 1.-1e-12 for m in metrics),
                             known_safe_full_fallbacks=sum(not m['certified'] for m in metrics),
                             pooled_test_disagreement=sum(m['test_disagreements'] for m in metrics) /
                                                      sum(r['n_test'] for r in runs),
                             delta_acc_pp=100 * float(np.mean([
                                 per_dataset[name]['acc'] - np.mean([r['full_acc'] for r in values])
                                 for name, values in grouped.items()])))
                output['tables'][key][family] = table
                if family == 'strat_imp_adaptive' or not complete:
                    continue
                base, adaptive = [], []
                for values in complete.values():
                    base.append(np.mean([r['families'][family][calib][str(eps)]['cost'] for r in values]))
                    adaptive.append(np.mean([r['families']['strat_imp_adaptive'][calib][str(eps)]['cost'] for r in values]))
                base, adaptive = np.array(base), np.array(adaptive)
                difference = base - adaptive
                tie = np.isclose(difference, 0, atol=1e-12, rtol=0)
                nonzero = difference[~tie]
                # No inference with fewer than two nonzero paired datasets.
                p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else None
                positive = (base > 0) & (adaptive > 0)
                ratio = float(np.exp(np.mean(np.log(base[positive]/adaptive[positive])))) if positive.any() else None
                output['paired_complete_datasets'][key][family] = dict(
                    n=len(base), adaptive_wins=int(np.sum((difference > 0) & ~tie)),
                    ties=int(tie.sum()), adaptive_losses=int(np.sum((difference < 0) & ~tie)),
                    geomean_baseline_over_adaptive=ratio, wilcoxon_two_sided_p=p)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default=str(Path(__file__).resolve().parents[1] / 'results/baselines'))
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--json', dest='json_path')
    args = parser.parse_args()
    result = summarize(args.directory, tuple(int(v) for v in args.seeds.split(',')))
    print(f"Runs {result['n_runs']}/{result['expected_runs']}; datasets {result['n_datasets']}/113; "
          f"complete across requested seeds: {result['datasets_with_all_seeds']}")
    print(result['cost_warning'])
    print('cost/fullmask=relative operations; risk=macro test disagreement; dAcc=percentage points')
    for setting, rows in result['tables'].items():
        print('\n' + setting)
        for family, values in rows.items():
            print(f"  {family:23s} cost {values['cost']:.4f} fullmask {values['full_transform_cost']:.4f} "
                  f"risk {100*values['disagree']:.3f}% dAcc {values['delta_acc_pp']:+.3f} "
                  f"cert {values['certified_runs']} cheaper {values['cheaper_selected_runs']}")
        for family, paired in result['paired_complete_datasets'][setting].items():
            print(f"  paired adaptive vs {family}: W/T/L {paired['adaptive_wins']}/{paired['ties']}/{paired['adaptive_losses']}; "
                  f"ratio {paired['geomean_baseline_over_adaptive']}; p={paired['wilcoxon_two_sided_p']}")
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
