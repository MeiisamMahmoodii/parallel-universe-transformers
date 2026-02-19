#!/usr/bin/env python3
"""Run one experiment by name.

Loads config from experiments/configs/exp_XX_name.json, runs run_finetune_multi_seed.py
with the config values, then compare_pretrained_vs_finetuned.py. Saves metrics.json
and per_seed.json under experiments/results/<exp_name>/.

Usage:
  PYTHONPATH=code uv run python experiments/run_experiment.py exp_01
  PYTHONPATH=code uv run python experiments/run_experiment.py exp_01_baseline
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_configs_dir() -> Path:
    """Configs dir: prefer repo root (cwd) so all exp_*.json are found when run from repo."""
    script_dir = Path(__file__).resolve().parent
    script_configs = script_dir / "configs"
    repo_root = script_dir.parent
    for base in [Path.cwd(), repo_root]:
        d = base / "experiments" / "configs"
        if d.exists() and list(d.glob("exp_*.json")):
            return d
    return script_configs


def find_config(exp_id: str, configs_dir: Optional[Path] = None) -> Optional[Path]:
    """Find config by prefix (exp_01 or exp_01_baseline)."""
    if configs_dir is None:
        configs_dir = get_configs_dir()
    if not configs_dir.exists():
        return None
    exp_lower = exp_id.lower()
    for p in sorted(configs_dir.glob("exp_*.json")):
        name = p.stem
        if name == exp_lower:
            return p
        if name.startswith(exp_lower + "_"):
            return p
        parts = name.split("_")
        if len(parts) >= 2 and f"{parts[0]}_{parts[1]}" == exp_lower:
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Run one experiment by name")
    parser.add_argument("exp_id", type=str, help="Experiment ID (e.g. exp_01 or exp_01_baseline)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available else cpu)")
    parser.add_argument("--skip-compare", action="store_true", help="Skip pretrained vs finetuned comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds from config (e.g. --seeds 42 for single seed)")
    args = parser.parse_args()
    if args.device is None:
        try:
            args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    configs_dir = get_configs_dir()
    config_path = find_config(args.exp_id, configs_dir)

    if config_path is None:
        print(f"Error: No config found for '{args.exp_id}'")
        print("Searched in:", configs_dir)
        print("Available configs:", sorted(p.stem for p in configs_dir.glob("exp_*.json")))
        return 1

    with open(config_path) as f:
        cfg = json.load(f)

    exp_name = cfg.get("name", config_path.stem)
    finetune = cfg.get("finetune", {})
    eval_cfg = cfg.get("eval", {})

    results_dir = Path(__file__).resolve().parent / "results" / exp_name
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_from = finetune.get("resume_from", "checkpoints/checkpoint_step_40000.pt")
    resume_path = repo_root / resume_from
    if not resume_path.exists():
        print(f"Error: Checkpoint not found: {resume_path}")
        return 1

    # Build run_finetune_multi_seed cmd
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_finetune_multi_seed.py"),
        "--resume-from", str(resume_path),
        "--output-dir", str(results_dir),
        "--checkpoint-dir", str(ckpt_dir),
        "--device", args.device,
        "--max-steps", str(finetune.get("max_steps", 1000)),
        "--lr", str(finetune.get("lr", 5e-6)),
        "--weight-decay", str(finetune.get("weight_decay", 0.1)),
        "--datasets", *[str(d) for d in finetune.get("datasets", ["ihdp"])],
        "--max-samples", str(eval_cfg.get("max_samples", 500)),
    ]

    mix_ratio = finetune.get("mix_ratio")
    if mix_ratio:
        cmd += ["--mix-ratio"] + [str(r) for r in mix_ratio]

    acic_data = finetune.get("acic_data")
    if acic_data:
        acic_path = repo_root / acic_data
        if acic_path.exists():
            cmd += ["--acic-data", str(acic_path)]

    seeds = args.seeds if args.seeds is not None else eval_cfg.get("seeds", [42, 123, 456])
    cmd += ["--seeds"] + [str(s) for s in seeds]

    if finetune.get("lambda_delta") is not None:
        cmd += ["--lambda-delta", str(finetune["lambda_delta"])]

    if finetune.get("synthetic_mix_ratio", 0) > 0:
        cmd += ["--synthetic-mix-ratio", str(finetune["synthetic_mix_ratio"])]

    if finetune.get("scale_outcome", False):
        cmd += ["--scale-outcome"]

    env = {**__import__("os").environ, "PYTHONPATH": str(code_dir)}

    print(f"Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=repo_root, env=env)
    if ret.returncode != 0:
        print("run_finetune_multi_seed failed")
        return ret.returncode

    # Rename/copy summary to metrics.json
    summary_path = results_dir / "finetune_multi_seed_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        metrics = {k: v for k, v in summary.items() if k != "per_seed"}
        per_seed = summary.get("per_seed", {})
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(results_dir / "per_seed.json", "w") as f:
            json.dump(per_seed, f, indent=2)

    # Run compare_pretrained_vs_finetuned
    if not args.skip_compare and seeds:
        first_ckpt = ckpt_dir / f"finetuned_seed{seeds[0]}.pt"
        if first_ckpt.exists():
            compare_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "compare_pretrained_vs_finetuned.py"),
                "--pretrained", str(resume_path),
                "--finetuned", str(first_ckpt),
                "--seeds", *[str(s) for s in seeds],
                "--device", args.device,
                "--max-samples", str(eval_cfg.get("max_samples", 500)),
                "--output", str(results_dir / "comparison.json"),
            ]
            if acic_data:
                compare_cmd += ["--acic-data", str(repo_root / acic_data)]
            subprocess.run(compare_cmd, cwd=repo_root, env=env)

    print(f"\nResults saved to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
