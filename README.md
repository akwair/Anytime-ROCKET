# Anytime-ROCKET

Label-free calibrated early exits for MiniRocket time series classification.

Anytime-ROCKET turns MiniRocket's fixed feature budget into an input-adaptive one.
Kernel--dilation units are ordered by dilation-stratified importance, each prefix
is fitted with its own ridge head, and computation stops once a prefix margin reaches
a threshold calibrated against the full model. Calibration uses only disagreement
with full-model predictions, so no additional class labels are needed. Two
high-probability and expected-risk controls are supported:

- **Learn then Test (LTT)** -- finite-sample high-probability control.
- **Conformal Risk Control (CRC)** -- expected-risk control, applied to a monotone
  upper envelope that restores the monotone-loss condition across the nonmonotone
  disagreement sequence.

## Results

Evaluated on 113 univariate, equal-length UCR datasets, five random splits per
dataset, against four static baseline families (independent small-MiniRocket
refits, natural ridge prefixes, an SFD/Detach-style adaptation, and a POCKET-style
adaptation) plus a same-ordering static comparator. At a 1% CRC target the
adaptive model uses 0.5569 of the full model's structural operation proxy (static
families: 0.8001--0.9222), with 0.447% macro test disagreement and a 0.118
percentage-point mean accuracy change.

Costs are structural operation proxies that account for shared convolutional
responses and per-prefix head overhead; they are not wall-clock measurements.

## Repository layout

```
src/            Method and experiment drivers (anytime.py, grid2/3.py, baselines, ...)
tests/          Regression and integrity tests
notes/          Baseline protocol, stage report, frozen-audit JSON
results/        Frozen aggregate summaries (summary_*.json/.txt)
manuscript/paa/ LaTeX source (draft.tex.in -> build_evidence.py -> main.tex), figures, evidence
output/pdf/     Compiled manuscript and review record
third_party/    Pinned Detach-ROCKET and POCKET reference implementations
```

## Reproduce

```bash
python -m pip install -r requirements.txt
# Build manuscript tables from frozen summaries
python manuscript/paa/build_evidence.py
# Compile manuscript (pdflatex + bibtex over manuscript/paa/main.tex)
```

## Citation

The manuscript is under review. A citation will be added here once the paper is accepted.

## License

Code in `src/` and `tools/` is released for reproducibility. Third-party
directories retain their original licenses.
