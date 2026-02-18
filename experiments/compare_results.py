#!/usr/bin/env python3
"""Aggregate experiment results and print comparison table.

Scans experiments/results/ for subfolders, reads metrics.json from each,
and prints a table: Experiment | IHDP PEHE | IHDP ATE | ACIC PEHE | ACIC ATE | Twins PEHE | Twins ATE.

Optionally writes results/comparison_table.json and results/comparison_table.csv.

Usage:
  PYTHONPATH=code uv run python experiments/compare_results.py
  PYTHONPATH=code uv run python experiments/compare_results.py --output-json --output-csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory (default: experiments/results)")
    parser.add_argument("--output-json", action="store_true", help="Write comparison_table.json")
    parser.add_argument("--output-csv", action="store_true", help="Write comparison_table.csv")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else exp_dir / "results"

    if not results_dir.exists():
        print(f"No results directory: {results_dir}")
        return 0

    rows = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        row = {"Experiment": subdir.name}
        ds_labels = {"ihdp": "IHDP", "acic": "ACIC", "twins": "Twins"}
        for ds in ["ihdp", "acic", "twins"]:
            pehe_mean = m.get(f"{ds}_PEHE_mean")
            pehe_std = m.get(f"{ds}_PEHE_std")
            ate_mean = m.get(f"{ds}_ATE_Err_mean")
            ate_std = m.get(f"{ds}_ATE_Err_std")
            label = ds_labels[ds]

            if pehe_mean is not None and pehe_std is not None:
                row[f"{label} PEHE"] = f"{pehe_mean:.4f} ± {pehe_std:.4f}"
            else:
                row[f"{label} PEHE"] = "N/A"

            if ate_mean is not None and ate_std is not None:
                row[f"{label} ATE"] = f"{ate_mean:.4f} ± {ate_std:.4f}"
            else:
                row[f"{label} ATE"] = "N/A"

        rows.append(row)

    if not rows:
        print("No experiment results found.")
        return 0

    # Print table
    cols = ["Experiment", "IHDP PEHE", "IHDP ATE", "ACIC PEHE", "ACIC ATE", "Twins PEHE", "Twins ATE"]
    widths = [max(len(str(r.get(c, ""))) for r in rows) for c in cols]
    widths = [max(w, len(c)) for w, c in zip(widths, cols)]

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))

    # Write outputs
    if args.output_json or args.output_csv:
        out_dir = results_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_json:
        out_path = results_dir / "comparison_table.json"
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {out_path}")

    if args.output_csv:
        out_path = results_dir / "comparison_table.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
