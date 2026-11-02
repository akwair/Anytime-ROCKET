#!/usr/bin/env bash
# Only the handoff-authorized server workspace is used. Restart to resume.
set -euo pipefail
WS=/data1/B24041423/ky/cpu_need
cd "$WS"
test "$(pwd -P)" = "$WS"
export TMPDIR="$WS/tmp" NUMBA_CACHE_DIR="$WS/cache/numba" XDG_CACHE_HOME="$WS/cache"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BASELINE_OUT="$WS/results/baselines" BASELINE_REFERENCE_DIR="$WS/results/v2"
mkdir -p logs tmp cache/numba results/baselines
exec 9>logs/baselines.lock
if ! flock -n 9; then
  echo 'A baseline run is already active.'
  exit 0
fi
echo $$ > logs/baselines.pid
trap 'rm -f "$WS/logs/baselines.pid"' EXIT
DATASETS=${DATASETS:-ALL}
SEEDS=${SEEDS:-0,1,2,3,4}
WORKERS=${WORKERS:-12}
export DATASETS SEEDS
echo "Started $(date -Is): datasets=$DATASETS seeds=$SEEDS workers=$WORKERS"
.venv/bin/python -u src/runall.py "$DATASETS" "$SEEDS" "$WORKERS" baselines
# runall.py logs job errors but exits 0; independently check every expected artifact.
.venv/bin/python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, 'src')
import baselines as b
from data import statistics
s = statistics()
names = s[(s.channel_num == 1) & (~s.has_nan)].dataset_name.tolist() if os.environ['DATASETS'] == 'ALL' else os.environ['DATASETS'].split(',')
seeds = [int(x) for x in os.environ['SEEDS'].split(',')]
signature = b.protocol_hash(b.settings())
invalid = []
for name in names:
    for seed in seeds:
        stem = Path(b.OUT) / f'{name}_s{seed}'
        result = b.reusable(stem.with_suffix('.json'), stem.with_suffix('.npz'), signature, name, seed)
        if result is None or result['reference_check']['status'] != 'exact_v2_arrays_verified':
            invalid.append(f'{name}_s{seed}')
print(f'Final integrity check: {len(names)*len(seeds)-len(invalid)}/{len(names)*len(seeds)} valid; invalid={invalid}', flush=True)
raise SystemExit(bool(invalid))
PY
echo "Finished $(date -Is)"
