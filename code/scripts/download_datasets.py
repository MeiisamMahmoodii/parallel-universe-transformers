#!/usr/bin/env python3
"""Download datasets needed for benchmarks (IHDP, optional ACIC).

IHDP: Fetches CEVAE-style CSVs from GitHub (no header); adds header and saves to data/.
ACIC: If causallib is installed, generates a small sample and saves to data/acic_sample.csv.

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --data-dir data --ihdp-count 3
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# CEVAE IHDP: no header, 30 columns = treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
IHDP_HEADER = "treatment,y_factual,y_cfactual,mu0,mu1," + ",".join(f"x{i}" for i in range(1, 26))
CEVAE_BASE = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv"


def download_ihdp(data_dir: Path, count: int = 3) -> list:
    """Download first `count` IHDP replications from CEVAE; add header; save to data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for i in range(1, count + 1):
        name = f"ihdp_npci_{i}.csv"
        url = f"{CEVAE_BASE}/{name}"
        out_path = data_dir / name
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            # CEVAE files have no header; prepend one so our loader works
            with open(out_path, "w") as f:
                f.write(IHDP_HEADER + "\n")
                f.write(body)
            downloaded.append(str(out_path))
            print(f"  Downloaded {name} -> {out_path}")
        except Exception as e:
            print(f"  Failed {name}: {e}", file=sys.stderr)
    return downloaded


def generate_synthetic_acic(data_dir: Path, n_samples: int = 4802, n_covariates: int = 58, seed: int = 42) -> bool:
    """Generate ACIC-style synthetic data when causallib is unavailable.
    ACIC 2016: 58 covariates, binary treatment, continuous outcome, mu0/mu1 for PEHE.
    """
    import numpy as np
    import pandas as pd

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_covariates).astype(np.float32)
    z = (rng.rand(n_samples) < 0.5).astype(np.float32)
    beta0 = rng.randn(n_covariates).astype(np.float32) * 0.2
    beta1 = rng.randn(n_covariates).astype(np.float32) * 0.2
    offset = 0.5
    mu0 = X @ beta0
    mu1 = X @ beta1 + offset
    eps = rng.randn(n_samples).astype(np.float32) * 0.5
    y0 = mu0 + eps
    y1 = mu1 + eps
    y = (1 - z) * y0 + z * y1
    cols = [f"x{i}" for i in range(1, n_covariates + 1)]
    df = pd.DataFrame(X, columns=cols)
    df["z"] = z
    df["y"] = y
    df["mu0"] = mu0
    df["mu1"] = mu1
    out_path = data_dir / "acic_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"  ACIC sample (synthetic) -> {out_path}")
    return True


def try_download_acic(data_dir: Path) -> bool:
    """If causallib is available, generate one ACIC instance. Else generate synthetic ACIC-style data."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        from causallib.datasets import load_acic16
    except ImportError:
        print("  ACIC: causallib not installed. Generating synthetic ACIC-style data.")
        return generate_synthetic_acic(data_dir)
    try:
        data = load_acic16(instance=1)
        import pandas as pd
        # causallib may return dict with X, z, y, mu0, mu1 etc.
        if isinstance(data, dict):
            X = data.get("X") if "X" in data else data.get("x")
            z = data.get("z") if "z" in data else data.get("treatment")
            y = data.get("y")
            mu0 = data.get("mu0")
            mu1 = data.get("mu1")
            if X is not None and z is not None and y is not None:
                df = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df["z"] = z
                df["y"] = y
                if mu0 is not None:
                    df["mu0"] = mu0
                if mu1 is not None:
                    df["mu1"] = mu1
                out_path = data_dir / "acic_sample.csv"
                df.to_csv(out_path, index=False)
                print(f"  ACIC sample (causallib) -> {out_path}")
                return True
        # Fallback: try to get a DataFrame
        if hasattr(data, "to_csv"):
            out_path = data_dir / "acic_sample.csv"
            data.to_csv(out_path, index=False)
            print(f"  ACIC sample (causallib) -> {out_path}")
            return True
    except Exception as e:
        print(f"  ACIC: causallib failed ({e}), generating synthetic ACIC-style data.", file=sys.stderr)
        return generate_synthetic_acic(data_dir)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download IHDP and optional ACIC datasets")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to save files")
    parser.add_argument("--ihdp-count", type=int, default=3, help="Number of IHDP replications (1-10)")
    parser.add_argument("--no-acic", action="store_true", help="Skip ACIC (even if causallib installed)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    print("Downloading IHDP (CEVAE)...")
    downloaded = download_ihdp(data_dir, min(max(1, args.ihdp_count), 10))
    print(f"IHDP: {len(downloaded)} file(s) in {data_dir}")
    if not args.no_acic:
        print("ACIC (optional)...")
        try_download_acic(data_dir)
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(main())
