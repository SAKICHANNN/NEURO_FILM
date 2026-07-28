"""Measure P151-P153 file-path lifetime changes against the frozen P150 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from time import perf_counter, sleep
from typing import Any

import numpy as np
from PIL import Image
import psutil


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "reference_match_file_memory_v1.json"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("node") != "P154":
        raise ValueError("config must be the P154 schema_version 1 contract")
    if payload.get("execution_order") != [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]:
        raise ValueError("execution_order differs from the frozen contract")
    for field in ("baseline_commit", "candidate_commit"):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{field} must be a lowercase full Git commit")
    image = payload.get("image")
    if not isinstance(image, dict):
        raise ValueError("image contract must be an object")
    if image.get("width") != 3000 or image.get("height") != 2000:
        raise ValueError("image geometry differs from the frozen 6 MP contract")
    return payload


def _generated_rgb(width: int, height: int, seed: int) -> np.ndarray:
    """Create a deterministic photographic-like RGB8 field without random state."""

    x = np.arange(width, dtype=np.uint32)[None, :]
    y = np.arange(height, dtype=np.uint32)[:, None]
    phase = np.uint32(seed)
    red = (17 + ((x * 13 + y * 7 + phase * 3) % 211)).astype(np.uint8)
    green = (19 + ((x * 5 + y * 11 + phase * 5) % 207)).astype(np.uint8)
    blue = (23 + ((x * 3 + y * 17 + phase * 7) % 199)).astype(np.uint8)
    return np.stack(
        [
            np.broadcast_to(red, (height, width)),
            np.broadcast_to(green, (height, width)),
            np.broadcast_to(blue, (height, width)),
        ],
        axis=-1,
    )


def generate_inputs(config: dict[str, Any], input_dir: Path) -> dict[str, str]:
    image = config["image"]
    input_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "reference": input_dir / "reference.png",
        "source": input_dir / "source.png",
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
        Image.fromarray(pixels, mode="RGB").save(
            paths[role],
            format="PNG",
            compress_level=6,
            optimize=False,
        )
    return {role: _sha256_file(path) for role, path in paths.items()}


def normalize_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only run-root path identity while preserving semantic fields."""

    normalized = json.loads(json.dumps(payload))
    normalized["reference"]["path"] = "<INPUT>/reference.png"
    recipe_file = normalized.get("recipe_file")
    if recipe_file is not None:
        recipe_file["path"] = "<RUN>/recipe.json"
    for index, row in enumerate(normalized["outputs"]):
        row["source_path"] = f"<INPUT>/source-{index}.png"
        row["output_path"] = f"<RUN>/output-{index}.png"
    return normalized


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
    from src.color_match import match_reference_files  # noqa: PLC0415

    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / "output.png"
    recipe_path = run_dir / "recipe.json"
    report_path = run_dir / "report.json"
    started = perf_counter()
    result = match_reference_files(
        input_dir / "reference.png",
        [input_dir / "source.png"],
        [output_path],
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
        "source_file_sha256": result.outputs[0].source_file_sha256,
        "output_sha256": _sha256_file(output_path),
        "recipe_sha256": _sha256_file(recipe_path),
        "report_sha256": _sha256_file(report_path),
        "normalized_report_sha256": _sha256_bytes(_canonical_json(normalized)),
        "normalized_report": normalized,
        "safety_action": result.outputs[0].safety.action,
        "staging_temporaries": stage_temporaries,
    }
    _atomic_write_json(result_path, payload)


def preflight(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    rules = config["preflight"]
    available = int(psutil.virtual_memory().available)
    current = psutil.Process()
    excluded = {current.pid, *(parent.pid for parent in current.parents())}
    competitors: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if process.pid in excluded:
                continue
            rss = int(process.info["memory_info"].rss)
            if rss > int(rules["competing_process_rss_bytes_max"]):
                competitors.append(
                    {
                        "pid": process.pid,
                        "name": process.info["name"],
                        "rss_bytes": rss,
                    }
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            continue
    probe = output_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_disk = int(shutil.disk_usage(probe).free)
    passed = bool(
        available >= int(rules["available_physical_memory_bytes_min"])
        and not competitors
        and free_disk >= int(rules["minimum_free_disk_bytes"])
    )
    return {
        "available_physical_memory_bytes": available,
        "free_disk_bytes": free_disk,
        "competing_processes": competitors,
        "passed": passed,
    }


def _kill_process_tree(root: psutil.Process) -> None:
    try:
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for process in reversed(children):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        root.kill()
    except psutil.NoSuchProcess:
        pass


def launch_worker(
    *,
    config_path: Path,
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
        "--config",
        str(config_path),
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
    orphan_count = sum(
        int(psutil.pid_exists(pid)) for pid in observed_process_ids
    )
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "parent_wall_seconds": perf_counter() - started,
        "peak_process_tree_rss_bytes": peak_rss,
        "observed_process_ids": sorted(observed_process_ids),
        "orphan_worker_count": orphan_count,
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
    variants = {
        name: [run for run in runs if run["variant"] == name]
        for name in ("baseline", "candidate")
    }
    complete = bool(
        len(variants["baseline"]) == 2
        and len(variants["candidate"]) == 2
        and all(run["run_pass"] for run in runs)
    )
    artifact_fields = (
        "output_sha256",
        "recipe_sha256",
        "normalized_report_sha256",
    )
    parity = {
        field: (
            len(
                {
                    run["worker_result"][field]
                    for run in runs
                    if run.get("worker_result") is not None
                }
            )
            == 1
            if complete
            else False
        )
        for field in artifact_fields
    }
    actions = {
        run["worker_result"]["safety_action"]
        for run in runs
        if run.get("worker_result") is not None
    }
    baseline_median = (
        statistics.median(
            run["monitor"]["peak_process_tree_rss_bytes"]
            for run in variants["baseline"]
        )
        if complete
        else None
    )
    candidate_median = (
        statistics.median(
            run["monitor"]["peak_process_tree_rss_bytes"]
            for run in variants["candidate"]
        )
        if complete
        else None
    )
    reduction = (
        baseline_median - candidate_median
        if baseline_median is not None and candidate_median is not None
        else None
    )
    ratio = (
        candidate_median / baseline_median
        if baseline_median not in (None, 0) and candidate_median is not None
        else None
    )
    memory_pass = bool(
        reduction is not None
        and ratio is not None
        and reduction >= int(gates["minimum_median_rss_reduction_bytes"])
        and ratio
        <= float(gates["maximum_candidate_to_baseline_median_rss_ratio"])
    )
    artifact_pass = bool(
        parity["output_sha256"]
        and parity["recipe_sha256"]
        and parity["normalized_report_sha256"]
    )
    identity_pass = actions == {"identity-fallback"}
    automatic_pass = bool(
        complete and artifact_pass and identity_pass and memory_pass
    )
    return {
        "complete_run_matrix": complete,
        "parity": parity,
        "safety_actions": sorted(actions),
        "baseline_median_peak_process_tree_rss_bytes": baseline_median,
        "candidate_median_peak_process_tree_rss_bytes": candidate_median,
        "median_rss_reduction_bytes": reduction,
        "candidate_to_baseline_median_rss_ratio": ratio,
        "artifact_parity_pass": artifact_pass,
        "identity_fallback_pass": identity_pass,
        "memory_gate_pass": memory_pass,
        "automatic_pass": automatic_pass,
        "claim_ceiling": config["claim_ceiling"],
    }


def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _add_worktree(path: Path, commit: str) -> None:
    _git("worktree", "add", "--detach", str(path), commit)
    actual = _git("rev-parse", "HEAD", cwd=path).stdout.strip()
    if actual != commit:
        raise RuntimeError(f"worktree HEAD mismatch: expected {commit}, got {actual}")


def _remove_worktree(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(path)],
        capture_output=True,
        text=True,
    )
    return {
        "path": str(path),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "path_exists_after": path.exists(),
    }


def run_parent(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if output_dir.exists():
        raise FileExistsError("output_dir must be create-only")
    output_dir.mkdir(parents=True)
    preflight_result = preflight(config, output_dir)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "node": config["node"],
        "config_sha256": _sha256_file(config_path),
        "baseline_commit": config["baseline_commit"],
        "candidate_commit": config["candidate_commit"],
        "preflight": preflight_result,
        "input_sha256": None,
        "runs": [],
        "gate_result": None,
        "worktree_cleanup": [],
        "claim_ceiling": config["claim_ceiling"],
    }
    if not preflight_result["passed"]:
        base_report["gate_result"] = {
            "automatic_pass": False,
            "run_deferred_by_preflight": True,
            "claim_ceiling": config["claim_ceiling"],
        }
        _atomic_write_json(output_dir / "report.json", base_report)
        return base_report

    input_dir = output_dir / "inputs"
    base_report["input_sha256"] = generate_inputs(config, input_dir)
    worktree_root = output_dir / "worktrees"
    worktree_root.mkdir()
    worktrees = {
        "baseline": worktree_root / "baseline",
        "candidate": worktree_root / "candidate",
    }
    added: list[Path] = []
    try:
        for variant, commit_field in (
            ("baseline", "baseline_commit"),
            ("candidate", "candidate_commit"),
        ):
            _add_worktree(worktrees[variant], config[commit_field])
            added.append(worktrees[variant])

        for index, variant in enumerate(config["execution_order"], start=1):
            run_dir = output_dir / "runs" / f"{index:02d}-{variant}"
            result_path = output_dir / "worker-results" / f"{index:02d}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            monitor = launch_worker(
                config_path=config_path,
                repo_root=worktrees[variant],
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
            expected_commit = config[
                "baseline_commit" if variant == "baseline" else "candidate_commit"
            ]
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
                and worker_result["repo_head"] == expected_commit
                and staging_count
                == int(config["gates"]["staging_temporary_count"])
                and worker_result["worker_wall_seconds"]
                <= float(config["gates"]["worker_wall_seconds_max"])
            )
            base_report["runs"].append(
                {
                    "index": index,
                    "variant": variant,
                    "expected_commit": expected_commit,
                    "monitor": monitor,
                    "worker_result": worker_result,
                    "run_pass": run_pass,
                }
            )
        base_report["gate_result"] = evaluate_runs(config, base_report["runs"])
    finally:
        for path in reversed(added):
            base_report["worktree_cleanup"].append(_remove_worktree(path))
        subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
        )
        cleanup_pass = bool(
            len(base_report["worktree_cleanup"]) == len(added)
            and all(
                row["exit_code"] == 0 and not row["path_exists_after"]
                for row in base_report["worktree_cleanup"]
            )
        )
        if base_report.get("gate_result") is not None:
            base_report["gate_result"]["worktree_cleanup_pass"] = cleanup_pass
            base_report["gate_result"]["automatic_pass"] = bool(
                base_report["gate_result"]["automatic_pass"] and cleanup_pass
            )
        _atomic_write_json(output_dir / "report.json", base_report)
    return base_report


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
