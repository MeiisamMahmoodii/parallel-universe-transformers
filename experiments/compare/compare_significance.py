"""Compare two methods across multiple seeds with a simple bootstrap CI.

This is useful once you have per-seed JSON outputs from:
  python -m experiments.compare.run_protocol --seeds 42 43 44 ...

It reads method JSON files (ours and baseline) for each seed, computes paired
differences for a chosen metric (e.g. delta_correlation), and reports a 95% CI.

Usage:
  python -m experiments.compare.compare_significance \\
    --results-dir results \\
    --metric delta_correlation \\
    --a-prefix method_ours_seed \\
    --b-prefix method_baseline_seed
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _load_metric(path: Path, metric: str):
    with open(path, "r") as f:
        data = json.load(f)
    v = data.get(metric)
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def bootstrap_ci(diffs, n_boot: int = 10000, seed: int = 0):
    """Return (mean, lo, hi, p_two_sided) via bootstrap on paired diffs."""
    diffs = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    if n == 0:
        return None
    mean = float(diffs.mean())
    if n == 1:
        return mean, mean, mean, 1.0
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    # Two-sided p-value estimate: proportion of bootstrap means crossing 0 (x2)
    p = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
    p = min(max(p, 0.0), 1.0)
    return mean, lo, hi, p


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CI for paired metric differences across seeds")
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--metric", type=str, default="delta_correlation")
    ap.add_argument("--a-prefix", type=str, default="method_ours_seed")
    ap.add_argument("--b-prefix", type=str, default="method_baseline_seed")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.exists():
        raise FileNotFoundError(f"results-dir not found: {rdir}")

    # Find seeds available based on filenames
    a_files = sorted(rdir.glob(f"{args.a_prefix}*.json"))
    b_files = sorted(rdir.glob(f"{args.b_prefix}*.json"))
    if not a_files or not b_files:
        raise FileNotFoundError("No matching method JSON files found in results-dir")

    # Build mapping seed->file by extracting trailing digits before .json
    def seed_map(files):
        m = {}
        for p in files:
            stem = p.stem
            # Expect ...seed<digits>
            digits = "".join(ch for ch in stem[::-1] if ch.isdigit())[::-1]
            if digits:
                m[int(digits)] = p
        return m

    a_map = seed_map(a_files)
    b_map = seed_map(b_files)
    common_seeds = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_seeds:
        raise ValueError("No common seeds found between A and B files")

    diffs = []
    used = []
    for s in common_seeds:
        va = _load_metric(a_map[s], args.metric)
        vb = _load_metric(b_map[s], args.metric)
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
        used.append(s)

    if not diffs:
        raise ValueError(f"No usable paired values for metric={args.metric}")

    mean, lo, hi, p = bootstrap_ci(diffs, n_boot=args.n_boot, seed=args.seed)
    print(f"Metric: {args.metric}")
    print(f"Seeds used: {used}")
    print(f"Paired diff (A - B): mean={mean:.6f}, 95% CI=({lo:.6f}, {hi:.6f}), p~{p:.4f}")


if __name__ == "__main__":
    main()

