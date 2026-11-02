#!/bin/bash
# Resumable full run on the lab server. Everything (data, caches, temp, logs, results) stays in the
# workspace. Safe to start again after a power cut: finished (dataset, seed) jobs are skipped.
WS=/data1/B24041423/ky/cpu_need
cd "$WS" || exit 1
export TMPDIR=$WS/tmp NUMBA_CACHE_DIR=$WS/cache/numba XDG_CACHE_HOME=$WS/cache
mkdir -p logs tmp cache
if [ -f logs/v2.pid ] && ps -p "$(cat logs/v2.pid)" -o args= 2>/dev/null | grep -q run_v2; then
  echo "already running (pid $(cat logs/v2.pid))"; exit 0
fi
echo $$ > logs/v2.pid
SEEDS=${SEEDS:-0,1,2,3,4}
WORKERS=${WORKERS:-16}
# download first, one dataset at a time, so workers never race on the mirror
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from data import statistics, load_ucr
s = statistics(); names = s[(s.channel_num == 1) & (~s.has_nan)].dataset_name.tolist()
for n in names:
    try: load_ucr(n)
    except Exception as e: print('download failed', n, e, flush=True)
print('downloads done', len(names), flush=True)
" >> logs/download.log 2>&1
.venv/bin/python -u src/runall.py ALL "$SEEDS" "$WORKERS" grid2 >> logs/v2.log 2>&1
echo "finished $(date)" >> logs/v2.log
rm -f logs/v2.pid
