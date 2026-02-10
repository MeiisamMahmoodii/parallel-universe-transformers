#!/usr/bin/env python3
"""Run full test and eval: unit tests, checkpoint sanity tests, eval matrix, comparison protocol.

Runs every test and evaluation for all checkpoints under a single output directory (GPU by default).
Results and a summary.json are written under --output-dir for later inspection.

Usage:
  python scripts/run_full_test_and_eval.py
  python scripts/run_full_test_and_eval.py --checkpoint-dir checkpoints --output-dir results/full_run
  python scripts/run_full_test_and_eval.py --quick
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _project_root():
    return Path(__file__).resolve().parents[1]


def _discover_checkpoints(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return []
    paths = sorted(checkpoint_dir.glob("*.pt"))
    return [str(p) for p in paths]


def _run(cmd, cwd, env=None, capture_output=True, log_path=None):
    env = env or os.environ
    if log_path:
        with open(log_path, "w") as f:
            r = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=True,
        )
    return r.returncode


def _run_and_capture(cmd, cwd, env=None):
    env = env or os.environ
    r = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return r.returncode, r.stdout, r.stderr


def _parse_significance_stdout(stdout: str):
    """Parse compare_significance printed output into a small dict."""
    out = {}
    m = re.search(r"Metric:\s*(\S+)", stdout)
    if m:
        out["metric"] = m.group(1)
    m = re.search(r"Seeds used:\s*\[([^\]]*)\]", stdout)
    if m:
        out["seeds_used"] = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    m = re.search(r"mean=([\d.-]+),\s*95% CI=\(([\d.-]+),\s*([\d.-]+)\),\s*p~([\d.]+)", stdout)
    if m:
        out["mean"] = float(m.group(1))
        out["ci_95_lo"] = float(m.group(2))
        out["ci_95_hi"] = float(m.group(3))
        out["p_value"] = float(m.group(4))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run full test and eval for all checkpoints (GPU, saved results)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to glob *.pt for all checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory (default: results/full_test_<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for eval/compare (default: cuda)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44],
        help="Seeds for eval and comparison (default: 42 43 44)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=200,
        help="Number of batches for eval and run_protocol (default: 200)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="*",
        default=["stage_1_basic"],
        help="Curriculum stage(s) for comparison (default: stage_1_basic). Use 'all' for all stages.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: one seed (42), 10 batches, one stage",
    )
    args = parser.parse_args()

    root = _project_root()
    if args.output_dir is None:
        args.output_dir = f"results/full_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.seeds = [42]
        args.num_batches = 10
        args.stages = ["stage_1_basic"]

    checkpoint_dir = root / args.checkpoint_dir
    checkpoints = _discover_checkpoints(checkpoint_dir)
    stage_list = args.stages
    if stage_list == ["all"]:
        sys.path.insert(0, str(root))
        from episodes.config import CurriculumConfig
        stage_list = [s.name for s in CurriculumConfig.get_default_curriculum()]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "args": {
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(out_dir),
            "device": args.device,
            "seeds": args.seeds,
            "num_batches": args.num_batches,
            "stages": stage_list,
        },
        "checkpoints_found": [str(p) for p in checkpoints],
        "unit_tests": {},
        "checkpoint_tests": {},
        "eval_matrix": {},
        "compare": {},
        "unit_tests_failed": False,
        "checkpoint_tests_failed": False,
        "failed_checkpoint_stems": [],
    }

    # Phase 1 — Unit tests
    unit_log = out_dir / "unit_tests.log"
    print(f"[Phase 1] Unit tests -> {unit_log}")
    code = _run(
        [sys.executable, "-m", "pytest", "tests/test_scm.py", "tests/test_model.py", "-v"],
        cwd=root,
        log_path=unit_log,
    )
    summary["unit_tests"]["exit_code"] = code
    summary["unit_tests"]["log"] = str(unit_log)
    if code != 0:
        summary["unit_tests_failed"] = True
    print(f"  exit_code={code}")

    # Phase 2 — Checkpoint sanity tests (per checkpoint)
    for ckpt_path in checkpoints:
        stem = Path(ckpt_path).stem
        log_path = out_dir / f"pytest_checkpoint_{stem}.log"
        print(f"[Phase 2] Checkpoint {stem} -> {log_path}")
        env = {**os.environ, "CHECKPOINT_PATH": ckpt_path}
        code = _run(
            [sys.executable, "-m", "pytest", "tests/test_checkpoint.py", "-v"],
            cwd=root,
            env=env,
            log_path=log_path,
        )
        summary["checkpoint_tests"][stem] = {"exit_code": code, "log": str(log_path)}
        if code != 0:
            summary["checkpoint_tests_failed"] = True
            summary["failed_checkpoint_stems"].append(stem)
        print(f"  exit_code={code}")

    # Phase 3 — Eval matrix (all checkpoints × difficulties)
    eval_matrix_dir = out_dir / "eval_matrix"
    eval_matrix_dir.mkdir(parents=True, exist_ok=True)
    summary["eval_matrix"]["dir"] = str(eval_matrix_dir)
    if checkpoints:
        print(f"[Phase 3] Eval matrix -> {eval_matrix_dir}")
        cmd = [
            sys.executable, "-m", "experiments.eval.run_all_checkpoints_matrix",
            "--checkpoint-dir", args.checkpoint_dir,
            "--output-dir", str(eval_matrix_dir),
            "--device", args.device,
            "--num-batches", str(args.num_batches),
        ]
        if len(args.seeds) > 0:
            cmd += ["--seeds"] + [str(s) for s in args.seeds]
        code = _run(cmd, cwd=root, capture_output=False, log_path=None)
        if code != 0:
            summary["eval_matrix"]["exit_code"] = code
        summary["eval_matrix"]["csv"] = str(eval_matrix_dir / "checkpoint_difficulty_matrix.csv")
        summary["eval_matrix"]["json"] = str(eval_matrix_dir / "checkpoint_difficulty_matrix.json")
        print(f"  done (exit_code={code})")
    else:
        summary["eval_matrix"]["skipped"] = "no checkpoints"
        print("[Phase 3] Eval matrix skipped (no checkpoints)")

    # Phase 4 — Comparison protocol (ours vs baseline) per checkpoint, per stage
    for ckpt_path in checkpoints:
        stem = Path(ckpt_path).stem
        for stage in stage_list:
            if len(stage_list) == 1:
                compare_dir = out_dir / "compare" / stem
            else:
                compare_dir = out_dir / "compare" / stem / stage
            compare_dir.mkdir(parents=True, exist_ok=True)
            key = f"{stem}/{stage}" if len(stage_list) > 1 else stem
            if key not in summary["compare"]:
                summary["compare"][key] = {}
            summary["compare"][key]["dir"] = str(compare_dir)

            print(f"[Phase 4] Compare {stem} stage={stage} -> {compare_dir}")
            cmd_protocol = [
                sys.executable, "-m", "experiments.compare.run_protocol",
                "--checkpoint", ckpt_path,
                "--stage", stage,
                "--output-dir", str(compare_dir),
                "--device", args.device,
                "--num-batches", str(args.num_batches),
            ]
            if args.seeds:
                cmd_protocol += ["--seeds"] + [str(s) for s in args.seeds]
            code = _run(cmd_protocol, cwd=root, capture_output=False, log_path=None)
            if code != 0:
                summary["compare"][key]["run_protocol_exit_code"] = code

            # Significance (bootstrap CI) on method_ours vs method_baseline
            code_sig, stdout, stderr = _run_and_capture(
                [
                    sys.executable, "-m", "experiments.compare.compare_significance",
                    "--results-dir", str(compare_dir),
                    "--metric", "delta_correlation",
                ],
                cwd=root,
            )
            sig_path = compare_dir / "significance.json"
            if code_sig == 0 and stdout:
                parsed = _parse_significance_stdout(stdout)
                if parsed:
                    with open(sig_path, "w") as f:
                        json.dump(parsed, f, indent=2)
                    summary["compare"][key]["significance"] = str(sig_path)
            else:
                with open(compare_dir / "significance.log", "w") as f:
                    f.write(stdout or "")
                    f.write(stderr or "")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return 0 if not summary["unit_tests_failed"] and not summary["checkpoint_tests_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
