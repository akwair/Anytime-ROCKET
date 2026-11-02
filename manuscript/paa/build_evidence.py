"""Generate manuscript tables from frozen results without changing experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SUMMARY = ROOT / 'results/summary_baselines.json'
s = json.loads(SUMMARY.read_text())
assert (s['n_runs'], s['n_datasets']) == (565, 113)
assert not s['incomplete_datasets']
v2 = [json.loads(p.read_text()) for p in sorted((ROOT/'results/server/results/v2').glob('*.json'))]
assert len(v2) == 565
by_name = {}
for r in v2:
    by_name.setdefault(r['dataset'], []).append(r)
assert len(by_name) == 113
assert all(sorted(r['seed'] for r in rs) == list(range(5)) for rs in by_name.values())
names = {
    'small_minirocket': 'Small MiniRocket',
    'ridge_prefix': 'Natural prefix',
    'detach_minirocket': 'SFD adaptation',
    'pocket_minirocket': 'POCKET adaptation',
    'strat_imp_static': 'Stratified static',
    'strat_imp_adaptive': 'Anytime-ROCKET',
}

def table(caption, label, columns, header, rows, bold_rows=()):
    cols = '|' + '|'.join(columns) + '|'
    header_bold = r' & '.join(r'\textbf{' + c.strip() + r'}' for c in header.split('&'))
    out = [
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{' + caption + r'}\label{' + label + '}',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{' + cols + '}',
        r'\hline\hline', header_bold + r' \\', r'\hline',
    ]
    for row in rows:
        cells = [r'\textbf{' + c + r'}' for c in row] if row[0] in bold_rows else row
        out.append(' & '.join(cells) + r' \\')
        out.append(r'\hline')
    out += [r'\hline\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)

tables = {}
rows = []
for key, name in names.items():
    row = [name]
    for c in ['crc', 'ltt']:
        r = s['tables'][f'{c}_0.01'][key]
        row += [f"{r['cost']:.4f}", f"{100*r['disagree']:.3f}", f"{r['delta_acc_pp']:+.3f}"]
    rows.append(row)
tables['MAIN'] = table(
    r'Comparison at $\epsilon=1\%$. Each metric first averages five seeds within a dataset, then weights all 113 datasets equally. Cost is a structural operation proxy relative to the shared full model; disagreement (Dis.) is in percent and accuracy change ($\Delta$Acc.) in percentage points. LTT uses $\delta=0.1$. The two pruning baselines are standardized MiniRocket adaptations.',
    'tab:main', 'lrrrrrr',
    r'Method & CRC cost & Dis. & $\Delta$Acc. & LTT cost & Dis. & $\Delta$Acc.', rows,
    bold_rows=('Anytime-ROCKET',))

rows = []
for key, name in list(names.items())[:-1]:
    row = [name]
    for c in ['crc', 'ltt']:
        r = s['paired_complete_datasets'][f'{c}_0.01'][key]
        row += [f"{r['adaptive_wins']}/{r['ties']}/{r['adaptive_losses']}"]
    rows.append(row)
tables['PAIRED'] = table(
    r'Dataset-level modeled-cost comparisons at $\epsilon=1\%$. W/T/L counts datasets on which Anytime-ROCKET has lower/equal/higher mean cost than the comparator. Ties use absolute tolerance $10^{-12}$; each dataset contributes one five-seed mean.',
    'tab:paired', 'lrr', 'Comparator & CRC W/T/L & LTT W/T/L', rows)

rows = []
budget = {}
for c in ['ltt', 'crc']:
    for eps in ['0.005', '0.01', '0.02', '0.05']:
        r = s['tables'][f'{c}_{eps}']['strat_imp_adaptive']
        finite = sum(np.isfinite(x['orders']['strat_imp'][c][eps]['margin']['lam']) for x in v2)
        budget[f'{c}_{eps}'] = dict(r, finite_threshold_runs=int(finite))
        rows.append([c.upper(), f'{100*float(eps):g}', f"{r['cost']:.4f}",
                     f"{100*r['disagree']:.3f}", f"{100*r['pooled_test_disagreement']:.3f}",
                     f"{r['delta_acc_pp']:+.3f}", f'{finite}/565'])
tables['BUDGET'] = table(
    r'Risk-budget sensitivity of the stratified-importance margin policy. Cost is a structural operation proxy relative to the shared full model; accuracy change ($\Delta$Acc.) is in percentage points. Macro and pooled disagreement are percentages. Pooled values weight test-instance appearances across runs; repeated seeds reuse observations. The Active column counts runs selecting a finite exit threshold.',
    'tab:budget', 'lrrrrrr',
    r'Rule & $\epsilon$ (\%) & Cost & Macro dis. & Pooled dis. & $\Delta$Acc. & Active', rows)

rows = []
ordering = {}
for order, name in [('importance', 'Importance'), ('random', 'Random'), ('strat', 'Stratified random'), ('strat_imp', 'Stratified importance')]:
    row = [name]
    for c in ['crc', 'ltt']:
        metrics = []
        for rs in by_name.values():
            entries = [r['orders'][order][c]['0.01']['margin'] for r in rs]
            full = np.mean([r['orders'][order]['full_acc'] for r in rs])
            metrics.append([np.mean([e['cost'] for e in entries]),
                            100*np.mean([e['disagree'] for e in entries]),
                            100*(np.mean([e['acc'] for e in entries])-full)])
        result = np.mean(metrics, axis=0).tolist()
        ordering[f'{order}_{c}'] = result
        row += [f'{result[0]:.4f}', f'{result[1]:.3f}', f'{result[2]:+.3f}']
    rows.append(row)
tables['ORDER'] = table(
    r'Ordering ablation at $\epsilon=1\%$ with the margin rule. All 113 datasets and full-model fallbacks are included. Units match Table~\ref{tab:main}. Configuration selection followed exploratory benchmark evaluation.',
    'tab:order', 'lrrrrrr',
    r'Order & CRC cost & Dis. & $\Delta$Acc. & LTT cost & Dis. & $\Delta$Acc.', rows)

rows = []
crossfit = {}
for c in ['ltt', 'crc']:
    for version in ['v2', 'v3']:
        vals = []
        active = 0
        for r in v2:
            o = r['orders']['strat_imp']
            if version == 'v3':
                p = ROOT/f"results/server/results/v3/{r['dataset']}_s{r['seed']}.json"
                candidate = json.loads(p.read_text())
                if not candidate['skipped']:
                    o = candidate['orders']['strat_imp']
            e = o[c]['0.01']['margin']
            active += int(np.isfinite(e['lam']))
            vals.append([e['cost'], 100*e['disagree'], 100*(e['acc']-o['full_acc'])])
        avg = np.mean(vals, axis=0).tolist()  # five runs per dataset, equal to dataset macro
        crossfit[f'{c}_{version}'] = dict(cost=avg[0], disagreement_percent=avg[1], delta_acc_pp=avg[2], active_runs=active)
        label = 'Independent split' if version == 'v2' else 'Pooled cross-fit heuristic'
        rows.append([c.upper(), label, f'{avg[0]:.4f}', f'{avg[1]:.3f}', f'{avg[2]:+.3f}', f'{active}/565'])
tables['CROSSFIT'] = table(
    r'Exploratory pooled cross-fitting at $\epsilon=1\%$. Datasets with fewer than 460 calibration examples use fold-derived losses; other results remain unchanged. Cost is a structural operation proxy relative to the shared full model; accuracy change ($\Delta$Acc.) is in percentage points. The Active column counts runs selecting a finite exit threshold. The pooled procedure is defined in Appendix~\ref{app:crossfit}.',
    'tab:crossfit', 'llrrrr', r'Selector & Calibration & Cost & Dis. (\%) & $\Delta$Acc. & Active', rows)

rows = []
for key, name in names.items():
    r = s['tables']['crc_0.01'][key]
    rows.append([name, f"{r['cost']:.4f}", f"{r['full_transform_cost']:.4f}"])
tables['EXECUTION'] = table(
    r'Two structural cost views at CRC $\epsilon=1\%$. Selective counts required transform work; full-transform charges all features before selection, as in the feature-masking implementation. Both include the applicable classifier costs.',
    'tab:execution', 'lrr', 'Method & Selective & Full-transform', rows,
    bold_rows=('Anytime-ROCKET',))

dataset_manifest = [dict(dataset=n, seeds=list(range(5)), n_fit=rs[0]['n_fit'],
                         n_cal=rs[0]['n_cal'], n_test=rs[0]['n_test'],
                         length=rs[0]['T'], classes=rs[0]['n_classes'])
                    for n, rs in sorted(by_name.items())]
payload = dict(checked_on='2026-11-02', protocol_hash=s['protocol_hash'],
               source_summary_sha256=hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
               n_datasets=113, n_runs=565, tables=s['tables'],
               paired=s['paired_complete_datasets'], budget=budget,
               ordering=ordering, crossfit=crossfit)
(BASE/'evidence.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
(BASE/'datasets.json').write_text(json.dumps(dataset_manifest, indent=2), encoding='utf-8')
source = (BASE/'draft.tex.in').read_text(encoding='utf-8')
for key, value in tables.items():
    assert source.count('@@'+key+'@@') == 1
    source = source.replace('@@'+key+'@@', value)
assert '@@' not in source
(BASE/'main.tex').write_text(source, encoding='utf-8')
print(json.dumps(dict(n_datasets=113, n_runs=565, ordering=ordering, crossfit=crossfit), indent=2))
