"""Compare calibration on the calibration set alone (v2) with cross-fitted pooling (v3, fit OOF + calib).
For skipped runs (n_cal >= N_MAX) v3 equals v2. Per dataset, mean over seeds."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
V2, V3 = defaultdict(dict), defaultdict(dict)
for f in glob.glob(os.path.join(base, "v2", "*.json")):
    r = json.load(open(f))
    V2[r["dataset"]][r["seed"]] = r
for f in glob.glob(os.path.join(base, "v3", "*.json")):
    r = json.load(open(f))
    V3[r["dataset"]][r["seed"]] = r


def entry(name, seed, order, calib, eps, src):
    r3 = V3[name].get(seed)
    if src == "v3" and r3 is not None and not r3["skipped"]:
        o = r3["orders"][order]
        return o[calib][eps], o["full_acc"]
    o = V2[name][seed]["orders"][order]
    return o[calib][eps], o["full_acc"]


names = sorted(n for n in V2 if len(V3[n]) == len(V2[n]))
print(f"{len(names)} datasets with complete v3 runs")
for order in ["strat_imp", "random"]:
    for calib in ["ltt", "crc"]:
        for eps in ["0.005", "0.01", "0.02", "0.05"]:
            for rule in ["margin", "consist"]:
                for src in ["v2", "v3"]:
                    w = t = l = cert = 0
                    ratio, cost, viol, dis, dacc, nrun = [], [], 0, [], [], 0
                    for n in names:
                        es = [entry(n, s, order, calib, eps, src) for s in sorted(V2[n])]
                        st = np.mean([e["static"]["cost"] for e, _ in es])
                        ad = np.mean([e[rule]["cost"] for e, _ in es])
                        w += ad < st - 1e-3
                        l += ad > st + 1e-3
                        t += abs(ad - st) <= 1e-3
                        cert += np.mean([np.isfinite(e[rule]["lam"]) for e, _ in es]) > 0.5
                        ratio.append(st / ad)
                        cost.append(ad)
                        for e, fa in es:
                            viol += e[rule]["disagree"] > float(eps)
                            dis.append(e[rule]["disagree"])
                            dacc.append(e[rule]["acc"] - fa)
                            nrun += 1
                    print(f"{order:9s} {calib} eps={eps} {rule:7s} {src} W/T/L={w}/{t}/{l} cert={cert} "
                          f"geo={np.exp(np.mean(np.log(ratio))):.2f} cost={np.mean(cost):.3f} "
                          f"viol={viol}/{nrun} meandis={np.mean(dis) * 100:.2f}% dAcc={np.mean(dacc) * 100:+.2f}pp")

# violations only over the runs that v3 actually changed (cross-fitted, n_cal < N_MAX)
print("\n== runs with cross-fitting only (n_cal < N_MAX), strat_imp margin ==")
for calib in ["ltt", "crc"]:
    for eps in ["0.005", "0.01", "0.02", "0.05"]:
        ds, nv, nc, nr = [], 0, 0, 0
        for n in names:
            for s, r3 in V3[n].items():
                if r3["skipped"]:
                    continue
                e = r3["orders"]["strat_imp"][calib][eps]["margin"]
                nr += 1
                if np.isfinite(e["lam"]):
                    nc += 1
                    nv += e["disagree"] > float(eps)
                    ds.append(e["disagree"])
        print(f"{calib} eps={eps}: runs={nr} certified={nc} viol={nv}/{nc} "
              f"meandis(certified)={np.mean(ds) * 100 if ds else float('nan'):.2f}%")
