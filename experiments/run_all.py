#!/usr/bin/env python3
"""Run all experiments.

Lists all configs in experiments/configs/, runs run_experiment.py for each.
Optionally runs in parallel with --parallel N.

Usage:
  PYTHONPATH=code uv run python experiments/run_all.py
  PYTHONPATH=code uv run python experiments/run_all.py --parallel 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--parallel", type=int, default=1, help="Number of experiments to run in parallel (default: 1)")
    parser.add_argument("--device", type=str, default=None, help="Device for experiments")
    parser.add_argument("--skip-compare", action="store_true", help="Skip pretrained vs finetuned comparison")
    parser.add_argument("--from-exp", type=str, default=None, help="Start from experiment (e.g. --from-exp exp_06)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds (e.g. --seeds 42 for single seed)")
    args = parser.parse_args()

    exp_dir = Path(__file__).resolve().parent
    configs_dir = exp_dir / "configs"

    if not configs_dir.exists():
        print(f"No configs directory: {configs_dir}")
        return 1

    configs = sorted(configs_dir.glob("exp_*.json"))
    if not configs:
        print("No experiment configs found.")
        return 0

    exp_ids = [p.stem for p in configs]
    if args.from_exp:
        from_lower = args.from_exp.lower().replace("-", "_")
        filtered = []
        for eid in exp_ids:
            num = None
            parts = eid.split("_")
            if len(parts) >= 2 and parts[0] == "exp" and parts[1].isdigit():
                num = int(parts[1])
            from_num = None
            if from_lower.startswith("exp") and "_" in from_lower:
                fn = from_lower.split("_")[1]
                if fn.isdigit():
                    from_num = int(fn)
            if from_num is not None and num is not None and num >= from_num:
                filtered.append(eid)
            elif from_num is None and eid.startswith(from_lower):
                filtered.append(eid)
        exp_ids = filtered
        print(f"Starting from {args.from_exp}: {', '.join(exp_ids)}")
    print(f"Running {len(exp_ids)} experiments: {', '.join(exp_ids)}")

    extra = []
    if args.device:
        extra += ["--device", args.device]
    if args.skip_compare:
        extra += ["--skip-compare"]
    if args.seeds is not None:
        extra += ["--seeds"] + [str(s) for s in args.seeds]

    if args.parallel <= 1:
        failed = []
        for exp_id in exp_ids:
            cmd = [sys.executable, str(exp_dir / "run_experiment.py"), exp_id] + extra
            ret = subprocess.run(cmd, cwd=exp_dir.parents[0])
            if ret.returncode != 0:
                failed.append(exp_id)

        if failed:
            print(f"\nFailed experiments: {', '.join(failed)}")
            return 1
    else:
        import concurrent.futures

        def run_one(exp_id):
            cmd = [sys.executable, str(exp_dir / "run_experiment.py"), exp_id] + extra
            return exp_id, subprocess.run(cmd, cwd=exp_dir.parents[0]).returncode

        failed = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as ex:
            for exp_id, code in ex.map(run_one, exp_ids):
                if code != 0:
                    failed.append(exp_id)

        if failed:
            print(f"\nFailed experiments: {', '.join(failed)}")
            return 1

    print("\nAll experiments completed. Run compare_results.py for summary:")
    print("  PYTHONPATH=code uv run python experiments/compare_results.py --output-json --output-csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
