# Exact-split baseline comparison protocol

This stage compares four standardized MiniRocket baseline families with paired
strat_imp static/adaptive models. It does **not** reproduce the published
Detach-ROCKET or POCKET scores. Scope: the same 113 UCR datasets, seeds 0–4,
official-train-size fit set, and the identical v2 seeded permutation and half/half
calibration/test split. Cross-fitting is excluded from this comparison.

- **Full anchor:** one aeon MiniRocket fit, StandardScaler, and RidgeClassifierCV
  with the existing ten alphas. All families use its identical calibration/test
  predictions at candidate 0. Regenerated strat_imp margins, class-index
  predictions, costs, and test labels must equal a supplied v2 NPZ exactly.
  Predictions saved for baseline candidates are actual fitted class labels;
  missing fit classes and noncontiguous labels do not change this mapping.
- **Small MiniRocket:** independent fits at requested fractions
  .01/.02/.04/.08/.16/.32/.64 of 10,000 features, same seed, at least 84 requested
  features. Actual feature counts follow the MiniRocket multiple-of-84 layout.
  The full candidate is the shared anchor, not another random fit.
- **Natural ridge prefix:** the first floor(fraction × actual full feature count)
  features in the original dilation/kernel/bias layout, with a ridge refit.
- **Detach-MiniRocket:** pinned official SFD geometry (p=.05), fixed alpha chosen
  by the full fit-only ridge CV, max absolute class coefficient, float64 solves.
  Select the first geometric step at or below each requested feature budget;
  stop at the smallest budget (about 90 steps). Discarded features never reenter,
  including zero-coefficient ties. Refit the common ridge CV on each selected
  mask. This last refit and the bounded scan are protocol adaptations.
- **POCKET-MiniRocket:** pinned official MiniROCKET one-dual equations,
  50 epochs, fit-only centering and column L2 normalization, no reweighting.
  Woodbury/Cholesky actions factor min(n,p) and avoid the official dense p×p
  inverse and diagonal shrinkage matrix. Zero-norm rows shrink to zero. The
  final positive-shrinkage mask matches the official threshold rule, so ties
  can retain fewer features than requested. Tune the official six k values
  [10,1,.1,.01,100,1000] at .32 retention once, then reuse k across budgets.
  CV has three stratified fit-only folds, reduced to two when necessary;
  singleton-class or single-class fits use default k=1 without CV. Normalization
  is refit inside each fold. Score the official thresholded predictor during
  CV, then refit common RidgeClassifierCV for deployment. The three-fold,
  single-budget tuning and common ridge are explicit protocol adaptations.

All feature selectors, transform parameters, normalizers, class encodings,
hyperparameters and candidate ordering use fit data only. Static candidates are
ordered by decreasing modeled cost; candidates costing at least the full model
are omitted and recorded. Calibration uses only prediction disagreement with
the shared full anchor, through existing LTT fixed sequence (delta=.1) and CRC
running-maximum selectors, at eps=.005/.01/.02/.05. The original margin grid
orders adaptive candidates, independently of observed test costs. Its overhead
can exceed the full model; results retain that cost rather than clipping it.

Certification is per family and setting, not a simultaneous guarantee across
all reported families/eps. CRC controls expected disagreement; LTT is the
split-sample PAC statement. If the selector returns -1, the recorded status is
`known_safe_full_fallback` and `certified=false`: full disagreement is identically
zero without asserting a calibration certificate. Test labels only report
accuracy. A missing-fit-class test item remains an ordinary error.

Two structural operation proxies are saved. `ideal_cost` charges 2T preprocessing, 16T for each
touched dilation, 3T for each touched kernel/dilation unit, and T_eff for each
selected PPV comparison, plus selected-feature scaling and head costs. It is
not the retained feature fraction or an exact arithmetic count. Selecting every feature exactly recovers
the existing full-transform proxy. `full_transform_cost` instead charges the
entire transform plus the smaller head and retained-feature scaling (all
evaluated heads for adaptive). Feature masking currently requires a full
transform; ideal-operation reductions are **not measured CPU speedups**. These
proxies exclude memory allocation and indexing, simplify padding/copy/PPV arithmetic, and follow the existing v2
operation convention; no dedicated pruned transform is implemented here.

`BASELINE_OUT` selects the output directory; `BASELINE_REFERENCE_DIR` selects
the v2 directory (empty disables external verification). Each atomic NPZ saves
raw per-candidate calibration/test predictions, feature masks, both cost views,
split indices, classes, transform parameters, and comparator margins. JSON is
written last and records settings/source hashes, pinned source commits, package
versions, source data/NPZ hashes, verification and timings. Resume requires a
matching schema/protocol hash, data hash, NPZ hash and external-reference hash
when present. Source edits invalidate old runs. No full feature matrices persist.

Run with one numerical thread per worker:

```text
python -m unittest discover -s tests -v
python src/baselines.py Coffee 0
python src/runall.py ALL 0,1,2,3,4 8 baselines
python src/summarize_baselines.py results/baselines --json results/summary_baselines.json
```

The primary summary requires `exact_v2_arrays_verified` for every included run,
matching package versions and protocol hashes, unique dataset/seed pairs, valid
NPZ hashes/internal protocol, and consistent prediction shapes, classes and full
anchors. It rejects damaged or unverified results rather than aggregating them.
The summary makes missing seeds explicit; pooled risk uses test-instance counts,
macro metrics first average seeds within each dataset, and paired comparisons
use only datasets with every requested seed. The driver catches individual
exceptions, so a zero driver exit code alone does not establish completeness.
Official source attribution and byte hashes are in `notes/baseline_sources.json`.
