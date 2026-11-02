"""Aggregate results/v2/*.json. Per dataset (mean over seeds): W/T/L of adaptive vs certified static,
geomean static/adaptive cost, mean adaptive cost, number of datasets where adaptive is certified
(lam < inf) in a majority of seeds, guarantee check on test (ltt: fraction of runs with test
disagreement > eps; crc: mean test disagreement), accuracy change vs the full model."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v2")
R = defaultdict(list)
for f in glob.glob(os.path.join(d, "*.json")):
    r = json.load(open(f))
    R[r["dataset"]].append(r)
print(f"{len(R)} datasets, {sum(len(v) for v in R.values())} runs")
ORDERS = ["importance", "random", "strat", "strat_imp"]
for min_cal in [0, 100, 200]:
    names = sorted(n for n in R if R[n][0]["n_cal"] >= min_cal)
    print(f"\n== n_cal >= {min_cal}: {len(names)} datasets ==")
    for calib in ["ltt", "crc"]:
        for eps in ["0.005", "0.01", "0.02", "0.05"]:
            for rule in ["margin", "consist"]:
                for od in ORDERS:
                    w = t = l = cert = 0
                    ratio, ad_all, viol, dis, nrun, dacc = [], [], 0, [], 0, []
                    for n in names:
                        es = [r["orders"][od][calib][eps] for r in R[n]]
                        st = np.mean([e["static"]["cost"] for e in es])
                        ad = np.mean([e[rule]["cost"] for e in es])
                        w += ad < st - 1e-3
                        l += ad > st + 1e-3
                        t += abs(ad - st) <= 1e-3
                        cert += np.mean([np.isfinite(e[rule]["lam"]) for e in es]) > 0.5
                        ratio.append(st / ad)
                        ad_all.append(ad)
                        for r, e in zip(R[n], es):
                            viol += e[rule]["disagree"] > float(eps)
                            dis.append(e[rule]["disagree"])
                            dacc.append(e[rule]["acc"] - r["orders"][od]["full_acc"])
                            nrun += 1
                    print(f"{calib} eps={eps} {rule:7s} {od:10s} W/T/L={w}/{t}/{l} cert={cert} "
                          f"geo={np.exp(np.mean(np.log(ratio))):.2f} cost={np.mean(ad_all):.3f} "
                          f"viol={viol}/{nrun} meandis={np.mean(dis) * 100:.2f}% dAcc={np.mean(dacc) * 100:+.2f}pp")

# ordering: per-checkpoint disagreement with the full model on test at matched relative cost
print("\n== orders: median over runs of test disagreement / rel. cost per checkpoint ==")
for od in ORDERS:
    dis = [r["orders"][od]["ckpt_disagree"] for v in R.values() for r in v if len(r["orders"][od]["ckpt_disagree"]) == 8]
    rc = [r["orders"][od]["ckpt_rel_cost"] for v in R.values() for r in v if len(r["orders"][od]["ckpt_rel_cost"]) == 8]
    print(f"{od:10s} dis {np.round(np.median(dis, 0), 3)}  cost {np.round(np.median(rc, 0), 3)}")
