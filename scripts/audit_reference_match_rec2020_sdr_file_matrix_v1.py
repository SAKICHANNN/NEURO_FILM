"""Run the frozen P165 BT.2020 SDR PNG/CICP file matrix."""

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

if __package__:
    from scripts.audit_reference_match_file_memory_v1 import (
        _add_worktree,
        _atomic_write_json,
        _canonical_json,
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
        _kill_process_tree,
        _remove_worktree,
        _sha256_bytes,
        _sha256_file,
        preflight,
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs" / "reference_match_rec2020_sdr_file_matrix_v1.json"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("node") != "P165":
        raise ValueError("config must be the P165 schema v1 contract")
    if [row.get("case_id") for row in config.get("cases", [])] != [
        "rec2020-single",
        "mixed-srgb-rec2020",
    ]:
        raise ValueError("config must freeze the exact two-case matrix")
    if config.get("output_bit_depth") != 16:
        raise ValueError("P165 output must remain 16-bit")
    if config.get("repeat_count_per_case") != 2:
        raise ValueError("P165 requires exactly two repeats per case")
    return config


def _case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [row for row in config["cases"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case_id must bind one frozen case")
    return matches[0]


def _pixels(seed: int, width: int, height: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(
        0.02,
        0.98,
        size=(height, width, 3),
    ).astype(np.float32)


def _save_rail(pixels: np.ndarray, rail: str, path: Path) -> None:
    from src.preprocess import (
        SourceProfile,
        WorkingImage,
        save_rec2020_16_png,
        save_srgb16_png,
    )

    if rail == "linear_srgb":
        save_srgb16_png(pixels, path)
        return
    if rail == "linear_rec2020":
        save_rec2020_16_png(
            WorkingImage(
                pixels=pixels,
                working_space="linear_rec2020",
                transfer_state="display_linear",
                source_transfer_state="display_referred",
                source_profile=SourceProfile("cicp", "P165 BT.2020 SDR"),
                hdr_metadata={},
                orientation_applied=True,
                alpha_policy="absent",
                bit_depth_in=16,
                source_path=path,
                warnings=[],
            ),
            path,
        )
        return
    raise ValueError("unsupported frozen rail")


def generate_inputs(
    config: dict[str, Any],
    input_root: Path,
) -> dict[str, Any]:
    image = config["image"]
    input_root.mkdir(parents=True, exist_ok=False)
    identities: dict[str, Any] = {}
    next_seed = int(image["source_seed_start"])
    for case in config["cases"]:
        case_id = case["case_id"]
        case_dir = input_root / case_id
        case_dir.mkdir()
        reference = case_dir / "reference.png"
        reference_pixels = _pixels(
            int(image["reference_seed"]),
            int(image["width"]),
            int(image["height"]),
        )
        _save_rail(reference_pixels, case["reference_rail"], reference)
        del reference_pixels
        source_paths: list[Path] = []
        for index, rail in enumerate(case["source_rails"]):
            source = case_dir / f"source-{index + 1}.png"
            source_pixels = _pixels(
                next_seed,
                int(image["width"]),
                int(image["height"]),
            )
            next_seed += 1
            _save_rail(source_pixels, rail, source)
            del source_pixels
            source_paths.append(source)
        gc.collect()
        identities[case_id] = {
            "reference": _sha256_file(reference),
            "sources": [_sha256_file(path) for path in source_paths],
        }
    return identities


def _normalize_report(
    payload: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    normalized["reference"]["path"] = f"<INPUT>/{case_id}/reference.png"
    normalized["recipe_file"]["path"] = "<RUN>/recipe.json"
    for index, output in enumerate(normalized["outputs"]):
        output["source_path"] = (
            f"<INPUT>/{case_id}/source-{index + 1}.png"
        )
        output["output_path"] = f"<RUN>/output-{index + 1}.png"
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
    reference = case_input / "reference.png"
    sources = tuple(
        case_input / f"source-{index + 1}.png"
        for index in range(len(case["source_rails"]))
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs = tuple(
        run_dir / f"output-{index + 1}.png"
        for index in range(len(sources))
    )
    recipe = run_dir / "recipe.json"
    report = run_dir / "report.json"

    decoded_reference = load_working_image(reference)
    decoded_sources = [load_working_image(path) for path in sources]
    decoded = {
        "reference_rail": decoded_reference.working_space,
        "source_rails": [image.working_space for image in decoded_sources],
    }
    del decoded_reference, decoded_sources
    gc.collect()

    started = perf_counter()
    result = match_reference_files(
        reference,
        sources,
        outputs,
        recipe_path=recipe,
        report_path=report,
        output_bit_depth=int(config["output_bit_depth"]),
    )
    wall = perf_counter() - started
    output_inspections = [inspect_input(path) for path in outputs]
    restored_outputs = [load_working_image(path) for path in outputs]
    normalized = _normalize_report(
        json.loads(report.read_text(encoding="utf-8")),
        case_id=case_id,
    )
    staging = sorted(
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
        "worker_wall_seconds": wall,
        "decoded": decoded,
        "output_rails": [image.working_space for image in restored_outputs],
        "output_profiles": [
            inspection.source_profile.kind
            for inspection in output_inspections
        ],
        "output_formats": [
            inspection.format_name for inspection in output_inspections
        ],
        "output_bit_depths": [
            inspection.bit_depth for inspection in output_inspections
        ],
        "output_sha256": [_sha256_file(path) for path in outputs],
        "recipe_sha256": _sha256_file(recipe),
        "normalized_report_sha256": _sha256_bytes(
            _canonical_json(normalized)
        ),
        "safety_actions": [row.safety.action for row in result.outputs],
        "staging_temporaries": staging,
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
    observed = {process.pid}
    peak_rss = 0
    timed_out = False
    while process.poll() is None:
        try:
            tree = [root_process, *root_process.children(recursive=True)]
            observed.update(row.pid for row in tree)
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
        psutil.Process(pid) for pid in observed if psutil.pid_exists(pid)
    ]
    if existing:
        psutil.wait_procs(existing, timeout=2.0)
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "parent_wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "observed_process_ids": sorted(observed),
        "orphan_worker_count": sum(
            int(psutil.pid_exists(pid)) for pid in observed
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
        rows = [row for row in runs if row["case_id"] == case["case_id"]]
        complete = bool(
            len(rows) == config["repeat_count_per_case"]
            and all(row["run_pass"] for row in rows)
        )
        parity = {
            field: bool(
                complete
                and len(
                    {
                        _sha256_bytes(_canonical_json(row["worker_result"][field]))
                        for row in rows
                    }
                )
                == 1
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
        ratio = (
            peak_max / peak_min
            if peak_max is not None and peak_min not in (None, 0)
            else None
        )
        semantic_pass = bool(
            complete
            and all(
                row["worker_result"]["decoded"]["reference_rail"]
                == case["reference_rail"]
                and row["worker_result"]["decoded"]["source_rails"]
                == case["source_rails"]
                and row["worker_result"]["output_rails"]
                == case["expected_output_rails"]
                and row["worker_result"]["output_profiles"]
                == case["expected_output_profiles"]
                and row["worker_result"]["output_formats"]
                == ["PNG"] * len(case["source_rails"])
                and row["worker_result"]["output_bit_depths"]
                == [16] * len(case["source_rails"])
                and row["worker_result"]["safety_actions"]
                == ["identity-fallback"] * len(case["source_rails"])
                for row in rows
            )
        )
        resource_pass = bool(
            peak_max is not None
            and ratio is not None
            and peak_max
            <= int(gates["maximum_peak_process_tree_rss_bytes_per_run"])
            and ratio
            <= float(gates["maximum_peak_rss_repeat_ratio_per_case"])
        )
        case_results[case["case_id"]] = {
            "complete": complete,
            "parity": parity,
            "semantic_pass": semantic_pass,
            "peak_process_tree_rss_bytes": peaks,
            "peak_rss_repeat_ratio": ratio,
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
        "node": "P165",
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
                run_dir = (
                    output_dir
                    / "runs"
                    / f"{run_index:02d}-{case['case_id']}-{repeat}"
                )
                result_path = (
                    output_dir / "worker-results" / f"{run_index:02d}.json"
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                monitor = launch_worker(
                    config_path=config_path,
                    case_id=case["case_id"],
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
                run_pass = bool(
                    monitor["exit_code"] == 0
                    and not monitor["timed_out"]
                    and monitor["orphan_worker_count"]
                    == int(config["gates"]["orphan_worker_count"])
                    and not monitor["result_temporary_exists"]
                    and worker_result is not None
                    and worker_result["repo_head"] == config["candidate_commit"]
                    and len(worker_result["staging_temporaries"])
                    == int(config["gates"]["staging_temporary_count"])
                    and worker_result["worker_wall_seconds"]
                    <= float(config["gates"]["worker_wall_seconds_max"])
                )
                report["runs"].append(
                    {
                        "index": run_index,
                        "case_id": case["case_id"],
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
        if report["gate_result"] is not None:
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
        raise ValueError("--output-dir is required")
    report = run_parent(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(report["gate_result"], indent=2))
    return 0 if report["gate_result"]["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
