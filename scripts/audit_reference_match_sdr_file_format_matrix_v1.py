"""Run the frozen P163 SDR JPEG/TIFF transaction matrix."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np
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
        preflight,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "reference_match_sdr_file_format_matrix_v1.json"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("node") != "P163":
        raise ValueError("config must be the P163 SDR format contract")
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
        or image.get("width") * image.get("height") != 3_145_728
        or image.get("pixel_count") != 3_145_728
    ):
        raise ValueError("image geometry differs from the frozen contract")
    cases = payload.get("cases")
    if not isinstance(cases, list) or [
        row.get("case_id") for row in cases
    ] != ["jpeg8", "tiff8", "tiff16"]:
        raise ValueError("case matrix differs from the frozen contract")
    if payload.get("repeat_count_per_case") != 2:
        raise ValueError("repeat count differs from the frozen contract")
    return payload


def _case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in config["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case_id must identify exactly one frozen case")
    return matches[0]


def _save_input(
    pixels: np.ndarray,
    path: Path,
    *,
    case_id: str,
) -> None:
    if case_id == "jpeg8":
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        return
    if case_id == "tiff8":
        Image.fromarray(pixels, mode="RGB").save(
            path,
            format="TIFF",
            compression="tiff_deflate",
        )
        return
    if case_id == "tiff16":
        from src.preprocess import save_srgb16_tiff

        save_srgb16_tiff(
            pixels.astype(np.float32) / np.float32(255.0),
            path,
        )
        return
    raise ValueError("unsupported frozen case")


def generate_inputs(
    config: dict[str, Any],
    input_root: Path,
) -> dict[str, dict[str, str]]:
    image = config["image"]
    input_root.mkdir(parents=True, exist_ok=False)
    identities: dict[str, dict[str, str]] = {}
    for case in config["cases"]:
        case_id = case["case_id"]
        case_dir = input_root / case_id
        case_dir.mkdir()
        extension = case["input_extension"]
        paths = {
            "reference": case_dir / f"reference{extension}",
            "source": case_dir / f"source{extension}",
        }
        for role, seed_field in (
            ("reference", "reference_seed"),
            ("source", "source_seed"),
        ):
            pixels = _generated_rgb(
                int(image["width"]),
                int(image["height"]),
                int(image[seed_field]),
            )
            _save_input(pixels, paths[role], case_id=case_id)
            del pixels
            gc.collect()
        identities[case_id] = {
            role: _sha256_file(path) for role, path in paths.items()
        }
    return identities


def _normalize_report(
    payload: dict[str, Any],
    *,
    case_id: str,
    input_extension: str,
    output_extension: str,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized["reference"]["path"] = (
        f"<INPUT>/{case_id}/reference{input_extension}"
    )
    normalized["recipe_file"]["path"] = "<RUN>/recipe.json"
    normalized["outputs"][0]["source_path"] = (
        f"<INPUT>/{case_id}/source{input_extension}"
    )
    normalized["outputs"][0]["output_path"] = (
        f"<RUN>/output{output_extension}"
    )
    return normalized


def worker(
    *,
    config_path: Path,
    case_id: str,
    repo_root: Path,
    input_root: Path,
    run_dir: Path,
    result_path: Path,
) -> None:
    config = load_config(config_path)
    case = _case(config, case_id)
    repo_root = repo_root.resolve(strict=True)
    sys.path.insert(0, str(repo_root))
    from src.color_match import match_reference_files
    from src.preprocess import inspect_input, load_working_image

    case_input = input_root / case_id
    input_extension = case["input_extension"]
    reference_path = case_input / f"reference{input_extension}"
    source_path = case_input / f"source{input_extension}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / f"output{case['output_extension']}"
    recipe_path = run_dir / "recipe.json"
    report_path = run_dir / "report.json"

    reference_inspection = inspect_input(reference_path)
    source_inspection = inspect_input(source_path)
    reference_working = load_working_image(reference_path)
    source_working = load_working_image(source_path)
    decoded = {
        "reference": {
            "format_name": reference_inspection.format_name,
            "bit_depth": reference_inspection.bit_depth,
            "working_space": reference_working.working_space,
            "transfer_state": reference_working.transfer_state,
        },
        "source": {
            "format_name": source_inspection.format_name,
            "bit_depth": source_inspection.bit_depth,
            "working_space": source_working.working_space,
            "transfer_state": source_working.transfer_state,
        },
    }
    del reference_working, source_working
    gc.collect()

    started = perf_counter()
    result = match_reference_files(
        reference_path,
        [source_path],
        [output_path],
        recipe_path=recipe_path,
        report_path=report_path,
        output_bit_depth=int(case["output_bit_depth"]),
    )
    worker_wall_seconds = perf_counter() - started
    output_inspection = inspect_input(output_path)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    normalized = _normalize_report(
        report_payload,
        case_id=case_id,
        input_extension=input_extension,
        output_extension=case["output_extension"],
    )
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
        "case_id": case_id,
        "worker_wall_seconds": worker_wall_seconds,
        "decoded": decoded,
        "output_inspection": {
            "format_name": output_inspection.format_name,
            "bit_depth": output_inspection.bit_depth,
        },
        "reference_file_sha256": result.reference_file_sha256,
        "source_file_sha256": result.outputs[0].source_file_sha256,
        "output_sha256": _sha256_file(output_path),
        "recipe_sha256": _sha256_file(recipe_path),
        "report_sha256": _sha256_file(report_path),
        "normalized_report_sha256": _sha256_bytes(_canonical_json(normalized)),
        "normalized_report": normalized,
        "output_format": result.outputs[0].output_format,
        "output_bit_depth": result.outputs[0].output_bit_depth,
        "safety_action": result.outputs[0].safety.action,
        "staging_temporaries": stage_temporaries,
    }
    _atomic_write_json(result_path, payload)


def launch_worker(
    *,
    config_path: Path,
    case_id: str,
    repo_root: Path,
    input_root: Path,
    run_dir: Path,
    result_path: Path,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--case-id",
        case_id,
        "--repo-root",
        str(repo_root),
        "--input-root",
        str(input_root),
        "--run-dir",
        str(run_dir),
        "--worker-result",
        str(result_path),
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
            observed_process_ids.update(row.pid for row in tree)
            tree_rss = 0
            for row in tree:
                try:
                    tree_rss += int(row.memory_info().rss)
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
    gates = config["gates"]
    case_results: dict[str, Any] = {}
    for case in config["cases"]:
        case_id = case["case_id"]
        rows = [row for row in runs if row["case_id"] == case_id]
        complete = bool(len(rows) == 2 and all(row["run_pass"] for row in rows))
        parity = {
            field: bool(
                complete
                and len({row["worker_result"][field] for row in rows}) == 1
            )
            for field in (
                "output_sha256",
                "recipe_sha256",
                "normalized_report_sha256",
            )
        }
        peaks = [row["monitor"]["peak_process_tree_rss_bytes"] for row in rows]
        peak_max = max(peaks) if complete else None
        peak_min = min(peaks) if complete else None
        repeat_ratio = (
            peak_max / peak_min
            if peak_max is not None and peak_min not in (None, 0)
            else None
        )
        semantic_pass = bool(
            complete
            and all(
                row["worker_result"]["decoded"]["reference"][
                    "working_space"
                ]
                == "linear_srgb"
                and row["worker_result"]["decoded"]["reference"][
                    "transfer_state"
                ]
                == "display_linear"
                and row["worker_result"]["decoded"]["source"][
                    "working_space"
                ]
                == "linear_srgb"
                and row["worker_result"]["decoded"]["source"][
                    "transfer_state"
                ]
                == "display_linear"
                and row["worker_result"]["decoded"]["reference"]["bit_depth"]
                == case["input_bit_depth"]
                and row["worker_result"]["decoded"]["source"]["bit_depth"]
                == case["input_bit_depth"]
                and row["worker_result"]["output_format"]
                == case["expected_output_format"]
                and row["worker_result"]["output_bit_depth"]
                == case["output_bit_depth"]
                and row["worker_result"]["output_inspection"]["bit_depth"]
                == case["output_bit_depth"]
                and row["worker_result"]["safety_action"]
                == "identity-fallback"
                for row in rows
            )
        )
        resource_pass = bool(
            peak_max is not None
            and repeat_ratio is not None
            and peak_max
            <= int(gates["maximum_peak_process_tree_rss_bytes_per_run"])
            and repeat_ratio
            <= float(gates["maximum_peak_rss_repeat_ratio_per_case"])
        )
        case_results[case_id] = {
            "complete": complete,
            "parity": parity,
            "semantic_pass": semantic_pass,
            "peak_process_tree_rss_bytes": peaks,
            "peak_rss_repeat_ratio": repeat_ratio,
            "resource_pass": resource_pass,
            "automatic_pass": bool(
                complete
                and all(parity.values())
                and semantic_pass
                and resource_pass
            ),
        }
    return {
        "case_results": case_results,
        "automatic_pass": all(
            row["automatic_pass"] for row in case_results.values()
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
        "node": "P163",
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

    input_root = output_dir / "inputs"
    report["input_sha256"] = generate_inputs(config, input_root)
    worktree = output_dir / "worktree"
    added = False
    try:
        _add_worktree(worktree, config["candidate_commit"])
        added = True
        run_index = 0
        for case in config["cases"]:
            for repeat in range(1, int(config["repeat_count_per_case"]) + 1):
                run_index += 1
                case_id = case["case_id"]
                run_dir = (
                    output_dir / "runs" / f"{run_index:02d}-{case_id}-{repeat}"
                )
                result_path = (
                    output_dir / "worker-results" / f"{run_index:02d}.json"
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                monitor = launch_worker(
                    config_path=config_path,
                    case_id=case_id,
                    repo_root=worktree,
                    input_root=input_root,
                    run_dir=run_dir,
                    result_path=result_path,
                    interval=float(config["rss_sample_interval_seconds"]),
                    timeout=float(config["worker_timeout_seconds"]),
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
                        "index": run_index,
                        "case_id": case_id,
                        "repeat": repeat,
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
    parser.add_argument("--case-id")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--worker-result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        required = (
            args.case_id,
            args.repo_root,
            args.input_root,
            args.run_dir,
            args.worker_result,
        )
        if any(value is None for value in required):
            raise ValueError("worker arguments are incomplete")
        worker(
            config_path=args.config.resolve(),
            case_id=args.case_id,
            repo_root=args.repo_root,
            input_root=args.input_root,
            run_dir=args.run_dir,
            result_path=args.worker_result,
        )
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required for the parent")
    report = run_parent(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
