"""Run the frozen P162 maximum 64-source file-batch audit."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import psutil
from PIL import Image

if __package__:
    from scripts.audit_reference_match_file_memory_v1 import (
        _add_worktree,
        _atomic_write_json,
        _canonical_json,
        _generated_rgb,
        _kill_process_tree,
        _remove_worktree,
        _sha256_bytes,
        _sha256_file,
        normalize_report,
        preflight,
    )
else:
    from audit_reference_match_file_memory_v1 import (
        _add_worktree,
        _atomic_write_json,
        _canonical_json,
        _generated_rgb,
        _kill_process_tree,
        _remove_worktree,
        _sha256_bytes,
        _sha256_file,
        normalize_report,
        preflight,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reference_match_max_batch_scale_v1.json"


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("node") != "P162":
        raise ValueError("config must be the P162 max-batch contract")
    commit = payload.get("candidate_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("candidate_commit must be a lowercase full Git commit")
    image = payload.get("image")
    if (
        not isinstance(image, dict)
        or image.get("width") != 1000
        or image.get("height") != 1000
        or image.get("source_seed_stop_exclusive")
        - image.get("source_seed_start")
        != 64
    ):
        raise ValueError("image contract must contain 64 one-megapixel sources")
    if payload.get("file_match", {}).get("source_count") != 64:
        raise ValueError("source_count differs from the product maximum")
    if payload.get("repeat_count") != 2:
        raise ValueError("repeat_count differs from the frozen contract")
    return payload


def generate_inputs(config: dict[str, Any], input_dir: Path) -> dict[str, Any]:
    image = config["image"]
    input_dir.mkdir(parents=True, exist_ok=False)
    reference_path = input_dir / "reference.png"
    pixels = _generated_rgb(
        int(image["width"]),
        int(image["height"]),
        int(image["reference_seed"]),
    )
    Image.fromarray(pixels, mode="RGB").save(
        reference_path,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    del pixels
    source_hashes: list[str] = []
    for index, seed in enumerate(
        range(
            int(image["source_seed_start"]),
            int(image["source_seed_stop_exclusive"]),
        )
    ):
        path = input_dir / f"source-{index}.png"
        pixels = _generated_rgb(
            int(image["width"]),
            int(image["height"]),
            seed,
        )
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="PNG",
            compress_level=6,
            optimize=False,
        )
        source_hashes.append(_sha256_file(path))
        del pixels
        if index % 8 == 7:
            gc.collect()
    return {
        "reference": _sha256_file(reference_path),
        "sources": source_hashes,
    }


def worker(
    *,
    repo_root: Path,
    input_dir: Path,
    run_dir: Path,
    result_path: Path,
    output_bit_depth: int,
) -> None:
    repo_root = repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo_root))
    from src.color_match import match_reference_files

    run_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [input_dir / f"source-{index}.png" for index in range(64)]
    output_paths = [run_dir / f"output-{index}.png" for index in range(64)]
    recipe_path = run_dir / "recipe.json"
    report_path = run_dir / "report.json"
    started = perf_counter()
    result = match_reference_files(
        input_dir / "reference.png",
        source_paths,
        output_paths,
        recipe_path=recipe_path,
        report_path=report_path,
        output_bit_depth=output_bit_depth,
    )
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    normalized = normalize_report(report_payload)
    stage_temporaries = sorted(
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if (
            "reference-match-stage" in path.name
            or path.name.endswith(".reference-match-backup")
        )
    )
    payload = {
        "repo_head": subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "worker_wall_seconds": perf_counter() - started,
        "reference_file_sha256": result.reference_file_sha256,
        "source_file_sha256": [
            row.source_file_sha256 for row in result.outputs
        ],
        "output_sha256": [_sha256_file(path) for path in output_paths],
        "recipe_sha256": _sha256_file(recipe_path),
        "report_sha256": _sha256_file(report_path),
        "normalized_report_sha256": _sha256_bytes(_canonical_json(normalized)),
        "normalized_report": normalized,
        "safety_actions": [row.safety.action for row in result.outputs],
        "staging_temporaries": stage_temporaries,
    }
    _atomic_write_json(result_path, payload)


def launch_worker(
    *,
    repo_root: Path,
    input_dir: Path,
    run_dir: Path,
    result_path: Path,
    interval: float,
    timeout: float,
    output_bit_depth: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--repo-root",
        str(repo_root),
        "--input-dir",
        str(input_dir),
        "--run-dir",
        str(run_dir),
        "--worker-result",
        str(result_path),
        "--output-bit-depth",
        str(output_bit_depth),
    ]
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    root_process = psutil.Process(process.pid)
    observed_process_ids = {process.pid}
    peak_rss = 0
    timed_out = False
    while process.poll() is None:
        try:
            tree = [root_process, *root_process.children(recursive=True)]
            observed_process_ids.update(item.pid for item in tree)
            tree_rss = 0
            for item in tree:
                try:
                    tree_rss += int(item.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
            peak_rss = max(peak_rss, tree_rss)
        except psutil.NoSuchProcess:
            pass
        if perf_counter() - started > timeout:
            timed_out = True
            _kill_process_tree(root_process)
            break
        sleep(interval)
    stdout, stderr = process.communicate()
    existing = [
        psutil.Process(pid)
        for pid in observed_process_ids
        if psutil.pid_exists(pid)
    ]
    if existing:
        psutil.wait_procs(existing, timeout=2.0)
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "parent_wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "observed_process_ids": sorted(observed_process_ids),
        "orphan_worker_count": sum(
            int(psutil.pid_exists(pid)) for pid in observed_process_ids
        ),
        "stdout": stdout,
        "stderr": stderr,
        "result_exists": result_path.is_file(),
        "result_temporary_exists": result_path.with_name(
            result_path.name + ".tmp"
        ).exists(),
    }


def evaluate_runs(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    complete = bool(len(runs) == 2 and all(row["run_pass"] for row in runs))
    fields = (
        "source_file_sha256",
        "output_sha256",
        "recipe_sha256",
        "normalized_report_sha256",
    )
    parity = {
        field: bool(
            complete
            and len(
                {
                    json.dumps(
                        row["worker_result"][field],
                        separators=(",", ":"),
                    )
                    for row in runs
                }
            )
            == 1
        )
        for field in fields
    }
    expected_paths = [f"<INPUT>/source-{index}.png" for index in range(64)]
    ordered_binding_pass = bool(
        complete
        and all(
            row["worker_result"]["source_file_sha256"]
            == config["_input_sha256"]["sources"]
            and len(row["worker_result"]["output_sha256"]) == 64
            and [
                output["source_path"]
                for output in row["worker_result"]["normalized_report"]["outputs"]
            ]
            == expected_paths
            for row in runs
        )
    )
    identity_pass = bool(
        complete
        and all(
            row["worker_result"]["safety_actions"]
            == ["identity-fallback"] * 64
            for row in runs
        )
    )
    peaks = [row["monitor"]["peak_process_tree_rss_bytes"] for row in runs]
    peak_max = max(peaks) if complete else None
    peak_min = min(peaks) if complete else None
    repeat_ratio = (
        peak_max / peak_min
        if peak_max is not None and peak_min not in (None, 0)
        else None
    )
    gates = config["gates"]
    resource_pass = bool(
        peak_max is not None
        and repeat_ratio is not None
        and peak_max <= int(gates["maximum_peak_process_tree_rss_bytes"])
        and repeat_ratio <= float(gates["maximum_peak_rss_repeat_ratio"])
    )
    return {
        "complete_run_matrix": complete,
        "parity": parity,
        "ordered_binding_pass": ordered_binding_pass,
        "all_identity_fallback_pass": identity_pass,
        "peak_process_tree_rss_bytes": peaks,
        "maximum_peak_process_tree_rss_bytes": peak_max,
        "minimum_peak_process_tree_rss_bytes": peak_min,
        "peak_rss_repeat_ratio": repeat_ratio,
        "artifact_parity_pass": all(parity.values()),
        "resource_gate_pass": resource_pass,
        "automatic_pass": bool(
            complete
            and all(parity.values())
            and ordered_binding_pass
            and identity_pass
            and resource_pass
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def run_parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError("output_dir must be create-only")
    output_dir.mkdir(parents=True)
    preflight_result = preflight(config, output_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "node": "P162",
        "config_sha256": _sha256_file(config_path),
        "candidate_commit": config["candidate_commit"],
        "preflight": preflight_result,
        "input_sha256": None,
        "runs": [],
        "gate_result": None,
        "worktree_cleanup": [],
        "claim_ceiling": config["claim_ceiling"],
    }
    if not preflight_result["passed"]:
        report["gate_result"] = {
            "automatic_pass": False,
            "run_deferred_by_preflight": True,
            "claim_ceiling": config["claim_ceiling"],
        }
        _atomic_write_json(output_dir / "report.json", report)
        return report

    input_dir = output_dir / "inputs"
    report["input_sha256"] = generate_inputs(config, input_dir)
    config["_input_sha256"] = report["input_sha256"]
    worktree = output_dir / "worktree"
    added = False
    try:
        _add_worktree(worktree, config["candidate_commit"])
        added = True
        for index in range(1, int(config["repeat_count"]) + 1):
            run_dir = output_dir / "runs" / f"{index:02d}"
            result_path = output_dir / "worker-results" / f"{index:02d}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            monitor = launch_worker(
                repo_root=worktree,
                input_dir=input_dir,
                run_dir=run_dir,
                result_path=result_path,
                interval=float(config["rss_sample_interval_seconds"]),
                timeout=float(config["worker_timeout_seconds"]),
                output_bit_depth=int(config["file_match"]["output_bit_depth"]),
            )
            worker_result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if monitor["result_exists"]
                else None
            )
            staging_count = (
                len(worker_result["staging_temporaries"])
                if worker_result is not None
                else None
            )
            run_pass = bool(
                monitor["exit_code"] == 0
                and not monitor["timed_out"]
                and monitor["orphan_worker_count"]
                == int(config["gates"]["orphan_worker_count"])
                and not monitor["result_temporary_exists"]
                and worker_result is not None
                and worker_result["repo_head"] == config["candidate_commit"]
                and staging_count
                == int(config["gates"]["staging_temporary_count"])
                and worker_result["worker_wall_seconds"]
                <= float(config["gates"]["worker_wall_seconds_max"])
            )
            report["runs"].append(
                {
                    "index": index,
                    "expected_commit": config["candidate_commit"],
                    "monitor": monitor,
                    "worker_result": worker_result,
                    "run_pass": run_pass,
                }
            )
        report["gate_result"] = evaluate_runs(config, report["runs"])
    finally:
        if added:
            report["worktree_cleanup"].append(_remove_worktree(worktree))
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
        )
        cleanup_pass = bool(
            len(report["worktree_cleanup"]) == int(added)
            and all(
                row["exit_code"] == 0 and not row["path_exists_after"]
                for row in report["worktree_cleanup"]
            )
        )
        if report.get("gate_result") is not None:
            report["gate_result"]["worktree_cleanup_pass"] = cleanup_pass
            report["gate_result"]["automatic_pass"] = bool(
                report["gate_result"]["automatic_pass"] and cleanup_pass
            )
        _atomic_write_json(output_dir / "report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--output-bit-depth", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        required = (
            args.repo_root,
            args.input_dir,
            args.run_dir,
            args.worker_result,
            args.output_bit_depth,
        )
        if any(value is None for value in required):
            raise ValueError("worker arguments are incomplete")
        worker(
            repo_root=args.repo_root,
            input_dir=args.input_dir,
            run_dir=args.run_dir,
            result_path=args.worker_result,
            output_bit_depth=args.output_bit_depth,
        )
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required for the parent")
    report = run_parent(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
