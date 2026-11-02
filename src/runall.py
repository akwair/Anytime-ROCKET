"""Parallel, resumable driver: runs pilot.run(dataset, seed) for every job with a process pool.
Each worker uses one thread. Finished jobs are skipped (result file exists), so after a power cut
just start the same command again.

usage: python src/runall.py <datasets comma list | ALL> <seeds> <workers> [module, default pilot]
"""
import os
import sys
import time
import traceback

for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def job(name, seed, module):
    import importlib
    t0 = time.time()
    try:
        importlib.import_module(module).run(name, seed)
        return name, seed, "ok", time.time() - t0
    except Exception:
        return name, seed, "ERR " + traceback.format_exc().strip().splitlines()[-1], time.time() - t0


def main():
    from data import statistics
    names = sys.argv[1]
    if names == "ALL":
        s = statistics()
        names = s[(s.channel_num == 1) & (~s.has_nan)].dataset_name.tolist()
    else:
        names = names.split(",")
    seeds = [int(x) for x in sys.argv[2].split(",")]
    workers = int(sys.argv[3])
    module = sys.argv[4] if len(sys.argv) > 4 else "pilot"
    jobs = [(n, s) for s in seeds for n in names]
    print(f"{len(jobs)} jobs, {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(job, n, s, module) for n, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            n, s, st, dt = f.result()
            print(f"[{i}/{len(jobs)}] {n} s{s} {st} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
