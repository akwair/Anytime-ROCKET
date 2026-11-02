"""UCR 2018 loader backed by the hf-mirror copy of ChengsenWang/UCR-2018 (parquet).

Values match the official Zenodo .ts files (checked on ECG5000: max abs diff 2.4e-7, labels
remapped one-to-one to 0..k-1). Arrays are cached as .npz next to the parquet files.
"""
import os
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "UCR")
BASE = "https://hf-mirror.com/datasets/ChengsenWang/UCR-2018/resolve/main"


def _download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    subprocess.run(["curl", "-s", "--ssl-no-revoke", "-L", "--fail", "--retry", "5", "--retry-all-errors",
                    "--retry-delay", "3", "-m", "1800", "-o", tmp, url], check=True)
    os.replace(tmp, path)


def _to_array(col):
    rows = [np.asarray(np.stack(v), dtype=np.float64)[:, 0] for v in col]
    lens = {len(r) for r in rows}
    if len(lens) != 1:
        raise ValueError(f"variable-length series ({min(lens)}-{max(lens)}) not supported yet")
    return np.stack(rows)


def load_ucr(name):
    """Return X_train, y_train, X_test, y_test (float64 [n, T], int64 [n])."""
    d = os.path.join(ROOT, name)
    cache = os.path.join(d, "arrays.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    out = {}
    for split in ("train", "test"):
        p = os.path.join(d, f"{split}.parquet")
        if not os.path.exists(p):
            _download(f"{BASE}/{name}/{split}.parquet", p)
        df = pd.read_parquet(p)
        out[split] = (_to_array(df["data"]), df["label"].to_numpy(np.int64))
    (Xtr, ytr), (Xte, yte) = out["train"], out["test"]
    tmp = os.path.join(d, "arrays.tmp.npz")
    np.savez(tmp, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    os.replace(tmp, cache)
    return Xtr, ytr, Xte, yte


def save_json(obj, path):
    """Write via a temp file and atomic rename, so a power cut never leaves a truncated result."""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def statistics():
    p = os.path.join(ROOT, "statistics.csv")
    if not os.path.exists(p):
        _download(f"{BASE}/statistics.csv", p)
    return pd.read_csv(p)
