#!/usr/bin/env python3
"""Run full test and comparison: unit tests, checkpoint tests, eval matrix, protocol, IHDP, ablations.

Runs:
  1. Unit tests (pytest)
  2. Checkpoint sanity tests per checkpoint (optional, --checkpoint-tests)
  3. Eval matrix (all checkpoints × stages, multi-seed)
  4. Comparison protocol (SCM) with methods: ours, mean_stub, outcome, dr, bart
  5. IHDP protocol (optional, --include-ihdp or when not --quick)
  6. Ablations: no-cross-world and single-world (optional, when not --quick)
  7. Writes full_comparison_summary.json and .md

Usage:
  python scripts/run_full_comparison.py
  python scripts/run_full_comparison.py --checkpoint-dir checkpoints --output-dir results/full_comp
  python scripts/run_full_comparison.py --quick
  python scripts/run_full_comparison.py --checkpoint-tests --stages all  # full eval like legacy run_full_test_and_eval
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
    return Path(__file__).resolve().parents[2]


def _discover_checkpoints(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return []
    return sorted([str(p) for p in checkpoint_dir.glob("*.pt")])


def _run_and_capture(cmd, cwd, env=None):
    env = env or os.environ
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
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


def _run(cmd, cwd, env=None, capture_output=True, log_path=None):
    env = env or os.environ
    if log_path:
        with open(log_path, "w") as f:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=capture_output, text=True)
    return r.returncode


def _collect_summary_rows(results_dir: Path, dataset_label: str):
    """Read method_*_summary.json from results_dir; return list of row dicts for table."""
    rows = []
    if not results_dir.exists():
        return rows
    for f in results_dir.glob("method_*_summary.json"):
        try:
            with open(f) as fp:
                s = json.load(fp)
        except Exception:
            continue
        method = s.get("method", f.stem.replace("method_", "").replace("_summary", ""))
        row = {
            "method": method,
            "dataset": dataset_label,
            "delta_corr": s.get("delta_correlation_mean"),
            "sqrt_pehe": s.get("delta_rmse_mean"),
            "pehe": s.get("pehe_mean"),
            "delta_slope": s.get("delta_slope_mean"),
            "sign_acc": s.get("sign_accuracy_mean"),
            "ate_mae": s.get("ate_mae_mean"),
            "calibration_ratio": s.get("calibration_ratio_mean"),
        }
        rows.append(row)
    return rows


def _rows_to_markdown_table(rows):
    if not rows:
        return "No data."
    headers = ["Method", "Dataset", "delta_corr", "sqrt(PEHE)", "delta_slope", "sign_acc", "ate_mae", "calibration_ratio"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        def fmt(v):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        cells = [r.get("method", ""), r.get("dataset", ""), fmt(r.get("delta_corr")), fmt(r.get("sqrt_pehe")),
                 fmt(r.get("delta_slope")), fmt(r.get("sign_acc")), fmt(r.get("ate_mae")), fmt(r.get("calibration_ratio"))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Full test and comparison (all methods, SCM + IHDP, ablations)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--stages", type=str, nargs="*", default=["stage_1_basic"],
                        help="Curriculum stage(s) for comparison. Use 'all' for all stages.")
    parser.add_argument("--ihdp-data", type=str, default=None, help="Path to IHDP CSV (optional).")
    parser.add_argument("--include-ihdp", action="store_true", help="Run IHDP protocol (default: True when not --quick)")
    parser.add_argument("--checkpoint-tests", action="store_true", help="Run pytest test_checkpoint per checkpoint")
    parser.add_argument("--compare-all-checkpoints", action="store_true",
                        help="Run protocol per checkpoint × per stage (with significance)")
    parser.add_argument("--quick", action="store_true", help="Fewer seeds/batches, skip IHDP/ablations")
    args = parser.parse_args()

    root = _project_root()
    args.output_dir = args.output_dir or f"results/full_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / args.checkpoint_dir
    checkpoints = _discover_checkpoints(checkpoint_dir)
    one_ckpt = checkpoints[0] if checkpoints else None

    # Resolve stages
    stage_list = args.stages
    if stage_list == ["all"]:
        sys.path.insert(0, str(root / "code"))
        from episodes.config import CurriculumConfig
        stage_list = [s.name for s in CurriculumConfig.get_default_curriculum()]

    if args.quick:
        args.seeds = [42]
        args.num_batches = 20

    run_ihdp = (not args.quick) or args.include_ihdp

    all_rows = []
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "checkpoints_found": checkpoints,
        "unit_tests": {},
        "checkpoint_tests": {},
        "eval_matrix": {},
        "protocol_scm": {},
        "protocol_ihdp": {},
        "ablations": {},
        "compare": {},
        "table_rows": [],
        "unit_tests_failed": False,
        "checkpoint_tests_failed": False,
    }

    # 1. Unit tests
    unit_log = out_dir / "unit_tests.log"
    print("[1/6] Unit tests")
    code = _run(
        [sys.executable, "-m", "pytest", "test/test_scm.py", "test/test_model.py", "-v"],
        cwd=root,
        log_path=unit_log,
    )
    summary_report["unit_tests"] = {"exit_code": code, "log": str(unit_log)}
    if code != 0:
        summary_report["unit_tests_failed"] = True

    # 2. Checkpoint sanity tests (optional)
    if args.checkpoint_tests and checkpoints:
        for ckpt_path in checkpoints:
            stem = Path(ckpt_path).stem
            log_path = out_dir / f"pytest_checkpoint_{stem}.log"
            print(f"[2a] Checkpoint {stem} -> {log_path}")
            env = {**os.environ, "CHECKPOINT_PATH": ckpt_path}
            code = _run(
                [sys.executable, "-m", "pytest", "test/test_checkpoint.py", "-v"],
                cwd=root, env=env, log_path=log_path,
            )
            summary_report["checkpoint_tests"][stem] = {"exit_code": code, "log": str(log_path)}
            if code != 0:
                summary_report["checkpoint_tests_failed"] = True

    # 3. Eval matrix
    eval_matrix_dir = out_dir / "eval_matrix"
    eval_matrix_dir.mkdir(parents=True, exist_ok=True)
    summary_report["eval_matrix"]["dir"] = str(eval_matrix_dir)
    print("[3/7] Eval matrix")
    if checkpoints:
        _run(
            [
                sys.executable, "-m", "experiments.eval.run_all_checkpoints_matrix",
                "--checkpoint-dir", args.checkpoint_dir,
                "--output-dir", str(eval_matrix_dir),
                "--device", args.device,
                "--num-batches", str(args.num_batches),
                "--seeds",
            ] + [str(s) for s in args.seeds],
            cwd=root,
        )
    else:
        summary_report["eval_matrix"]["skipped"] = "no checkpoints"

    # 4. Protocol on SCM with all methods
    protocol_dir = out_dir / "protocol_scm"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    summary_report["protocol_scm"]["dir"] = str(protocol_dir)
    print("[4/7] Protocol (SCM)")
    if one_ckpt:
        for stage in stage_list:
            stage_dir = protocol_dir / stage if len(stage_list) > 1 else protocol_dir
            if len(stage_list) > 1:
                stage_dir.mkdir(parents=True, exist_ok=True)
            code = _run(
                [
                    sys.executable, "-m", "experiments.compare.run_protocol",
                    "--checkpoint", one_ckpt,
                    "--methods", "ours,mean_stub,outcome,dr,bart",
                    "--stage", stage,
                    "--output-dir", str(stage_dir),
                    "--device", args.device,
                    "--num-batches", str(args.num_batches),
                    "--seeds",
                ] + [str(s) for s in args.seeds],
                cwd=root,
            )
            summary_report["protocol_scm"]["exit_code"] = code
            all_rows.extend(_collect_summary_rows(stage_dir, stage))
    else:
        summary_report["protocol_scm"]["skipped"] = "no checkpoint"

    # 5. IHDP protocol (same methods)
    ihdp_dir = out_dir / "protocol_ihdp"
    ihdp_dir.mkdir(parents=True, exist_ok=True)
    summary_report["protocol_ihdp"]["dir"] = str(ihdp_dir)
    print("[5/7] IHDP protocol")
    if one_ckpt and run_ihdp:
        cmd_ihdp = [
            sys.executable, "-m", "experiments.benchmarks.run_ihdp",
            "--checkpoint", one_ckpt,
            "--output-dir", str(ihdp_dir),
            "--methods", "ours,mean_stub,outcome,dr,bart",
            "--seeds",
        ] + [str(s) for s in args.seeds]
        if args.ihdp_data:
            cmd_ihdp += ["--data", args.ihdp_data]
        code = _run(cmd_ihdp, cwd=root)
        summary_report["protocol_ihdp"]["exit_code"] = code
        all_rows.extend(_collect_summary_rows(ihdp_dir, "IHDP"))
    else:
        summary_report["protocol_ihdp"]["skipped"] = "no checkpoint or IHDP disabled"

    # 6. Ablations: no-cross-world and single-world
    ablate_dir = out_dir / "ablations"
    ablate_dir.mkdir(parents=True, exist_ok=True)
    summary_report["ablations"]["dir"] = str(ablate_dir)
    print("[6/7] Ablations")
    if one_ckpt and not args.quick:
        # No-cross-world: run protocol with --ablate-no-cross-world
        ablate_nocw = ablate_dir / "no_cross_world"
        ablate_nocw.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable, "-m", "experiments.compare.run_protocol",
                "--checkpoint", one_ckpt,
                "--methods", "ours",
                "--stage", "stage_1_basic",
                "--output-dir", str(ablate_nocw),
                "--device", args.device,
                "--num-batches", str(args.num_batches),
                "--seeds",
            ] + [str(s) for s in args.seeds] + ["--ablate-no-cross-world"],
            cwd=root,
        )
        for p in ablate_nocw.glob("method_*_summary.json"):
            with open(p) as f:
                s = json.load(f)
            all_rows.append({
                "method": "ours (no cross-world)",
                "dataset": "stage_1_basic",
                "delta_corr": s.get("delta_correlation_mean"),
                "sqrt_pehe": s.get("delta_rmse_mean"),
                "pehe": s.get("pehe_mean"),
                "delta_slope": s.get("delta_slope_mean"),
                "sign_acc": s.get("sign_accuracy_mean"),
                "ate_mae": s.get("ate_mae_mean"),
                "calibration_ratio": s.get("calibration_ratio_mean"),
            })
        # Single-world: run_checkpoint_by_difficulty --single-world
        _run(
            [
                sys.executable, "-m", "experiments.eval.run_checkpoint_by_difficulty",
                "--checkpoint", one_ckpt,
                "--difficulties", "stage_1_basic",
                "--output-dir", str(ablate_dir),
                "--device", args.device,
                "--num-batches", str(args.num_batches),
                "--seeds",
            ] + [str(s) for s in args.seeds] + ["--single-world"],
            cwd=root,
        )
        ckpt_stem = Path(one_ckpt).stem
        sum_path = ablate_dir / f"{ckpt_stem}_stage_1_basic_single_world_summary.json"
        if sum_path.exists():
            with open(sum_path) as f:
                s = json.load(f)
            all_rows.append({
                "method": "ours (single-world)",
                "dataset": "stage_1_basic",
                "delta_corr": s.get("delta_correlation_mean"),
                "sqrt_pehe": s.get("delta_rmse_mean"),
                "pehe": s.get("pehe_mean"),
                "delta_slope": s.get("delta_slope_mean"),
                "sign_acc": s.get("sign_accuracy_mean"),
                "ate_mae": s.get("ate_mae_mean"),
                "calibration_ratio": s.get("calibration_ratio_mean"),
            })
    else:
        summary_report["ablations"]["skipped"] = "no checkpoint or quick run"

    # 7. Compare all checkpoints × stages with significance (optional)
    if args.compare_all_checkpoints and checkpoints:
        print("[7/7] Compare all checkpoints × stages")
        for ckpt_path in checkpoints:
            stem = Path(ckpt_path).stem
            for stage in stage_list:
                compare_dir = out_dir / "compare" / stem / stage if len(stage_list) > 1 else out_dir / "compare" / stem
                compare_dir.mkdir(parents=True, exist_ok=True)
                key = f"{stem}/{stage}" if len(stage_list) > 1 else stem
                summary_report["compare"][key] = {"dir": str(compare_dir)}
                cmd_protocol = [
                    sys.executable, "-m", "experiments.compare.run_protocol",
                    "--checkpoint", ckpt_path,
                    "--methods", "ours,mean_stub",
                    "--stage", stage,
                    "--output-dir", str(compare_dir),
                    "--device", args.device,
                    "--num-batches", str(args.num_batches),
                    "--seeds",
                ] + [str(s) for s in args.seeds]
                code = _run(cmd_protocol, cwd=root, capture_output=False)
                if code != 0:
                    summary_report["compare"][key]["run_protocol_exit_code"] = code
                code_sig, stdout, stderr = _run_and_capture(
                    [
                        sys.executable, "-m", "experiments.compare.compare_significance",
                        "--results-dir", str(compare_dir),
                        "--metric", "delta_correlation",
                        "--b-prefix", "method_mean_stub_seed",
                    ],
                    cwd=root,
                )
                sig_path = compare_dir / "significance.json"
                if code_sig == 0 and stdout:
                    parsed = _parse_significance_stdout(stdout)
                    if parsed:
                        with open(sig_path, "w") as f:
                            json.dump(parsed, f, indent=2)
                        summary_report["compare"][key]["significance"] = str(sig_path)
    else:
        print("[7/7] Summary")

    # Write summary JSON and Markdown table
    summary_report["table_rows"] = all_rows
    summary_path = out_dir / "full_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    md_path = out_dir / "full_comparison_summary.md"
    with open(md_path, "w") as f:
        f.write("# Full comparison summary\n\n")
        f.write(_rows_to_markdown_table(all_rows))
    print(f"Summary -> {summary_path} and {md_path}")
    print("\n" + _rows_to_markdown_table(all_rows))
    return 0 if not summary_report["unit_tests_failed"] and not summary_report["checkpoint_tests_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
