"""Aggregate results/pilot/*.json: per dataset (mean over seeds) relative op cost of certified
static vs adaptive exit, win/tie/loss, violation frequency of the eps guarantee on test."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "pilot")
R = defaultdict(list)
for f in glob.glob(os.path.join(d, "*.json")):
    r = json.load(open(f))
    # runs written before the fallback fix in pilot.py: uncertified (lam=inf) -> full model, cost 1
    for o in r["orders"].values():
        for e in o["eps"].values():
            for rule in ("margin", "consist"):
                if e[rule]["lam"] == float("inf"):
                    e[rule]["cost"] = 1.0
    R[r["dataset"]].append(r)
print(f"{len(R)} datasets, {sum(len(v) for v in R.values())} runs")
for min_cal in [0, 200]:
    names = [n for n in R if R[n][0]["n_cal"] >= min_cal]
    print(f"\n== n_cal >= {min_cal}: {len(names)} datasets ==")
    for eps in ["0.005", "0.01", "0.02", "0.05"]:
        for od in ["importance", "random"]:
            for rule in ["margin", "consist"]:
                w = t = l = 0
                ratio, viol, nrun, dacc = [], 0, 0, []
                for n in names:
                    st = np.mean([r["orders"][od]["eps"][eps]["static"]["cost"] for r in R[n]])
                    ad = np.mean([r["orders"][od]["eps"][eps][rule]["cost"] for r in R[n]])
                    w += ad < st - 1e-3
                    l += ad > st + 1e-3
                    t += abs(ad - st) <= 1e-3
                    ratio.append(st / ad)
                    for r in R[n]:
                        viol += r["orders"][od]["eps"][eps][rule]["disagree"] > float(eps)
                        nrun += 1
                        dacc.append(r["orders"][od]["eps"][eps][rule]["acc"] - r["orders"][od]["full_acc"])
                print(f"eps={eps} {od:10s} {rule:7s} W/T/L={w}/{t}/{l} geomean static/adapt="
                      f"{np.exp(np.mean(np.log(ratio))):.2f} mean adapt cost="
                      f"{np.mean([np.mean([r['orders'][od]['eps'][eps][rule]['cost'] for r in R[n]]) for n in names]):.3f}"
                      f" viol={viol}/{nrun} dAcc={np.mean(dacc) * 100:+.2f}pp")
