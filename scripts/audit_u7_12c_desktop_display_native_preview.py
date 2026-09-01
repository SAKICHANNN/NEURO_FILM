#!/usr/bin/env python3
"""Committed-head fidelity and latency audit for U7.12C desktop previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import (
    preview_fidelity_metrics,
    render_three_stock_previews_to_directory,
)

CONFIG_PATH = ROOT / "configs/u7_12c_desktop_display_native_preview_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
BOUND_PATHS = (
    "configs/u7_12c_desktop_display_native_preview_v1.json",
    "docs/planning/U7_12C_DESKTOP_DISPLAY_NATIVE_PREVIEW_CONTRACT.md",
    "src/inference/three_stock_preview.py",
    "src/inference/product_desktop.py",
    "src/inference/product_desktop_ui.py",
    "tests/test_u7_12c_desktop_display_native_preview.py",
    "scripts/audit_u7_12c_desktop_display_native_preview.py",
    "tests/test_u7_12c_desktop_display_native_preview_audit.py",
)
STYLE_IDS = ("ektar_100", "portra_400", "velvia_50")


class U712CError(RuntimeError):
    """Raised when the frozen display-native preview audit cannot execute."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_blob_sha256(path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _tree_rss(process: psutil.Process) -> int:
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _load_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise U712CError(f"invalid RGB8 preview: {path}")
    return np.ascontiguousarray(rgb)


def _visible_baseline_rgb(path: Path, display_box: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        visible = opened.convert("RGB")
        visible.thumbnail(display_box, Image.Resampling.LANCZOS)
        rgb = np.asarray(visible, dtype=np.uint8)
    return np.ascontiguousarray(rgb)


def _source_row(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    for row in config["sources"]:
        if row["source_id"] == source_id:
            return row
    raise U712CError(f"unknown source id: {source_id}")


def _worker(
    config_path: Path,
    source_id: str,
    variant: str,
    destination: Path,
    result_path: Path,
) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_row = _source_row(config, source_id)
    source = ROOT / source_row["path"]
    if (
        not source.is_file()
        or source.stat().st_size != int(source_row["bytes"])
        or sha256_file(source) != source_row["sha256"]
    ):
        raise U712CError(f"frozen source identity mismatch: {source_id}")
    if variant not in {"baseline", "candidate"}:
        raise U712CError("worker variant must be baseline or candidate")
    render = config["render"]
    kwargs: dict[str, Any] = {}
    if variant == "candidate":
        kwargs = {
            "max_preview_width": int(config["display_box"]["width"]),
            "max_preview_height": int(config["display_box"]["height"]),
        }
    suffix = source.suffix.casefold()
    started = time.perf_counter()
    manifest = render_three_stock_previews_to_directory(
        source,
        destination,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=int(render[f"{variant}_max_preview_pixels"]),
        look_amount=float(render["look_amount"]),
        seed=int(render["seed"]),
        tile_size=int(render["tile_size"]),
        tile_workers=int(render["tile_workers"]),
        png_compression=int(render["png_compression"]),
        jpeg_scaled_decode=suffix in {".jpg", ".jpeg"},
        raw_half_size_decode=suffix == ".dng",
        **kwargs,
    )
    renderer_wall_seconds = time.perf_counter() - started
    outputs: dict[str, dict[str, Any]] = {}
    for style_id in STYLE_IDS:
        path = destination / f"{style_id}.preview.png"
        rgb = _load_rgb8(path)
        outputs[style_id] = {
            "file_sha256": sha256_file(path),
            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        }
    normalized_manifest = {
        key: manifest[key]
        for key in (
            "schema_version",
            "input_sha256",
            "source_width",
            "source_height",
            "preview_width",
            "preview_height",
            "preview_pixels",
            "max_preview_pixels",
            "decoded_width",
            "decoded_height",
            "jpeg_scaled_decode",
            "png_compression",
            "look_amount",
            "preview_basis",
            "claim_ceiling",
        )
    }
    if "raw_half_size_decode" in manifest:
        normalized_manifest["raw_half_size_decode"] = manifest["raw_half_size_decode"]
    if variant == "candidate":
        normalized_manifest["max_preview_width"] = manifest.get("max_preview_width")
        normalized_manifest["max_preview_height"] = manifest.get("max_preview_height")
    result = {
        "variant": variant,
        "renderer_wall_seconds": renderer_wall_seconds,
        "manifest": normalized_manifest,
        "outputs": outputs,
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _run_worker(
    config_path: Path,
    source_id: str,
    variant: str,
    run_index: int,
    destination: Path,
    result_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--config",
        str(config_path),
        "--source-id",
        source_id,
        "--variant",
        variant,
        "--destination",
        str(destination),
        "--worker-result",
        str(result_path),
    ]
    started = time.perf_counter()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    monitored = psutil.Process(child.pid)
    peak_rss = 0
    while child.poll() is None:
        peak_rss = max(peak_rss, _tree_rss(monitored))
        time.sleep(0.01)
    stdout, stderr = child.communicate()
    outer_wall_seconds = time.perf_counter() - started
    if child.returncode != 0:
        raise U712CError(
            f"{source_id}/{variant}/{run_index} failed with "
            f"{child.returncode}: {stderr.strip()} {stdout.strip()}"
        )
    if not result_path.is_file():
        raise U712CError("worker did not publish its owned result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["run_index"] = run_index
    result["outer_process_wall_seconds"] = outer_wall_seconds
    result["peak_process_tree_rss_bytes"] = peak_rss
    return result


def _run_pair_fidelity(
    baseline_directory: Path,
    candidate_directory: Path,
    display_box: tuple[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for style_id in STYLE_IDS:
        baseline = _visible_baseline_rgb(
            baseline_directory / f"{style_id}.preview.png", display_box
        )
        candidate = _load_rgb8(candidate_directory / f"{style_id}.preview.png")
        if candidate.shape != baseline.shape:
            raise U712CError("candidate and baseline visible geometry differ")
        rows.append(
            {
                "style_id": style_id,
                "visible_width": int(candidate.shape[1]),
                "visible_height": int(candidate.shape[0]),
                "baseline_visible_rgb_sha256": hashlib.sha256(
                    baseline.tobytes()
                ).hexdigest(),
                "candidate_rgb_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                **preview_fidelity_metrics(
                    candidate.astype(np.float32) / 255.0,
                    baseline.astype(np.float32) / 255.0,
                ),
            }
        )
    return rows


def _source_record(
    config_path: Path,
    config: dict[str, Any],
    source_row: dict[str, Any],
    scratch: Path,
    *,
    reverse: bool,
) -> dict[str, Any]:
    source_id = str(source_row["source_id"])
    source = ROOT / source_row["path"]
    before_sha = sha256_file(source)
    variants = ["baseline", "candidate"]
    if reverse:
        variants.reverse()
    by_variant: dict[str, list[tuple[dict[str, Any], Path]]] = {
        "baseline": [],
        "candidate": [],
    }
    for variant in variants:
        for run_index in range(
            int(config["render"]["timed_fresh_processes_per_source_and_variant"])
        ):
            destination = scratch / source_id / f"{variant}_{run_index}"
            result_path = scratch / source_id / f"{variant}_{run_index}.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result = _run_worker(
                config_path,
                source_id,
                variant,
                run_index,
                destination,
                result_path,
            )
            by_variant[variant].append((result, destination))
    display_box = (
        int(config["display_box"]["width"]),
        int(config["display_box"]["height"]),
    )
    fidelity = [
        _run_pair_fidelity(
            by_variant["baseline"][index][1],
            by_variant["candidate"][index][1],
            display_box,
        )
        for index in range(len(by_variant["candidate"]))
    ]
    record = {
        "source_id": source_id,
        "source_path": source_row["path"],
        "source_bytes": source.stat().st_size,
        "source_sha256": before_sha,
        "source_unchanged": sha256_file(source) == before_sha,
        "baseline_runs": [row for row, _ in by_variant["baseline"]],
        "candidate_runs": [row for row, _ in by_variant["candidate"]],
        "fidelity": fidelity,
    }
    shutil.rmtree(scratch / source_id)
    return record


def evaluate_records(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, bool]:
    """Evaluate complete source records against the frozen U7.12C gates."""

    source_config = {row["source_id"]: row for row in config["sources"]}
    gates = config["gates"]
    identities_exact = len(records) == len(source_config) and {
        row["source_id"] for row in records
    } == set(source_config)
    geometry_exact = True
    display_contained = True
    decoder_paths_exact = True
    candidate_repeat_exact = True
    distinct_outputs = True
    fidelity_rows: list[dict[str, Any]] = []
    performance: dict[str, dict[str, float]] = {}
    rss_not_above = True
    for record in records:
        source = source_config[record["source_id"]]
        expected_baseline = tuple(source["baseline_preview_dimensions"])
        expected_candidate = tuple(source["candidate_preview_dimensions"])
        baseline_runs = record["baseline_runs"]
        candidate_runs = record["candidate_runs"]
        if len(baseline_runs) != 2 or len(candidate_runs) != 2:
            geometry_exact = False
            continue
        geometry_exact &= all(
            (
                run["manifest"]["preview_width"],
                run["manifest"]["preview_height"],
            )
            == expected_baseline
            for run in baseline_runs
        ) and all(
            (
                run["manifest"]["preview_width"],
                run["manifest"]["preview_height"],
            )
            == expected_candidate
            for run in candidate_runs
        )
        display_contained &= all(
            run["manifest"].get("max_preview_width") == config["display_box"]["width"]
            and run["manifest"].get("max_preview_height")
            == config["display_box"]["height"]
            and run["manifest"]["preview_width"] <= config["display_box"]["width"]
            and run["manifest"]["preview_height"] <= config["display_box"]["height"]
            for run in candidate_runs
        )
        is_jpeg = str(source["path"]).casefold().endswith((".jpg", ".jpeg"))
        decoder_paths_exact &= all(
            run["manifest"]["jpeg_scaled_decode"] is is_jpeg
            and (
                (is_jpeg and "raw_half_size_decode" not in run["manifest"])
                or (not is_jpeg and run["manifest"].get("raw_half_size_decode") is True)
            )
            for run in [*baseline_runs, *candidate_runs]
        )
        candidate_repeat_exact &= (
            candidate_runs[0]["outputs"] == candidate_runs[1]["outputs"]
        )
        distinct_outputs &= all(
            len({output["file_sha256"] for output in run["outputs"].values()}) == 3
            for run in [*baseline_runs, *candidate_runs]
        )
        fidelity_rows.extend(row for group in record["fidelity"] for row in group)
        baseline_wall = statistics.median(
            float(run["renderer_wall_seconds"]) for run in baseline_runs
        )
        candidate_wall = statistics.median(
            float(run["renderer_wall_seconds"]) for run in candidate_runs
        )
        performance[record["source_id"]] = {
            "candidate_median_renderer_wall_seconds": candidate_wall,
            "candidate_to_baseline_median_renderer_wall_ratio": candidate_wall
            / baseline_wall,
        }
        rss_not_above &= max(
            int(run["peak_process_tree_rss_bytes"]) for run in candidate_runs
        ) <= max(int(run["peak_process_tree_rss_bytes"]) for run in baseline_runs)
    jpeg_perf = performance.get("u4_5b_exact_jpeg_portrait", {})
    dng_perf = performance.get("u4_5f_blackmagic_dng_landscape", {})
    return {
        "required_sources_exact": identities_exact,
        "source_identities_exact": identities_exact
        and all(
            record["source_bytes"] == source_config[record["source_id"]]["bytes"]
            and record["source_sha256"] == source_config[record["source_id"]]["sha256"]
            for record in records
        ),
        "sources_immutable": identities_exact
        and all(record["source_unchanged"] for record in records),
        "exact_baseline_and_candidate_dimensions": geometry_exact,
        "display_box_containment": display_contained,
        "decoder_paths_unchanged": decoder_paths_exact,
        "candidate_repeat_pixels_and_outputs_exact": candidate_repeat_exact,
        "three_distinct_look_outputs": distinct_outputs,
        "rgb_rmse": bool(fidelity_rows)
        and max(float(row["rgb_rmse"]) for row in fidelity_rows)
        <= float(gates["maximum_rgb_rmse_vs_baseline_visible"]),
        "rgb_absolute_error_p95": bool(fidelity_rows)
        and max(float(row["rgb_absolute_error_p95"]) for row in fidelity_rows)
        <= float(gates["maximum_rgb_absolute_error_p95"]),
        "new_boundary_fraction": bool(fidelity_rows)
        and max(float(row["new_boundary_fraction"]) for row in fidelity_rows)
        <= float(gates["maximum_new_boundary_fraction"]),
        "jpeg_renderer_wall_seconds": bool(jpeg_perf)
        and jpeg_perf["candidate_median_renderer_wall_seconds"]
        <= float(gates["maximum_jpeg_candidate_median_renderer_wall_seconds"]),
        "jpeg_renderer_wall_ratio": bool(jpeg_perf)
        and jpeg_perf["candidate_to_baseline_median_renderer_wall_ratio"]
        <= float(
            gates["maximum_jpeg_candidate_to_baseline_median_renderer_wall_ratio"]
        ),
        "dng_renderer_wall_seconds": bool(dng_perf)
        and dng_perf["candidate_median_renderer_wall_seconds"]
        <= float(gates["maximum_dng_candidate_median_renderer_wall_seconds"]),
        "dng_renderer_wall_ratio": bool(dng_perf)
        and dng_perf["candidate_to_baseline_median_renderer_wall_ratio"]
        <= float(gates["maximum_dng_candidate_to_baseline_median_renderer_wall_ratio"]),
        "candidate_peak_rss_not_above_baseline": rss_not_above,
    }


def _scientific_payload(report: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for source in report["records"]:
        record = {
            key: value
            for key, value in source.items()
            if key not in {"baseline_runs", "candidate_runs"}
        }
        for variant in ("baseline_runs", "candidate_runs"):
            record[variant] = []
            for run in source[variant]:
                record[variant].append(
                    {
                        key: value
                        for key, value in run.items()
                        if key
                        not in {
                            "renderer_wall_seconds",
                            "outer_process_wall_seconds",
                            "peak_process_tree_rss_bytes",
                        }
                    }
                )
        records.append(record)
    return {
        "schema": report["schema"],
        "status": report["status"],
        "bindings": report["bindings"],
        "records": records,
        "gates": report["gates"],
        "claim_ceiling": report["claim_ceiling"],
    }


def execute(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commit = _git_output("rev-parse", "HEAD")
    tracked_clean = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        == 0
    )
    bindings = {
        "commit": commit,
        "config_sha256": sha256_file(config_path),
        "files": {path: _git_blob_sha256(path) for path in BOUND_PATHS},
    }
    parent_evidence = ROOT / config["parent_evidence"]["path"]
    parent_exact = (
        parent_evidence.is_file()
        and parent_evidence.stat().st_size == config["parent_evidence"]["bytes"]
        and sha256_file(parent_evidence) == config["parent_evidence"]["sha256"]
        and _git_output("merge-base", "--is-ancestor", config["parent_head"], "HEAD")
        == ""
    )
    source_rows = list(config["sources"])
    if reverse:
        source_rows.reverse()
    for row in source_rows:
        source = ROOT / row["path"]
        if (
            not source.is_file()
            or source.stat().st_size != int(row["bytes"])
            or sha256_file(source) != row["sha256"]
        ):
            raise U712CError(f"frozen source identity mismatch: {row['source_id']}")
    scratch = Path(tempfile.mkdtemp(prefix="u7_12c_", dir=ROOT / "tmp"))
    try:
        records = [
            _source_record(config_path, config, row, scratch, reverse=reverse)
            for row in source_rows
        ]
        records.sort(key=lambda row: row["source_id"])
        core_gates = evaluate_records(config, records)
        gates = {
            "tracked_diff_clean": tracked_clean,
            "committed_sources_bound": all(
                len(value) == 64 for value in bindings["files"].values()
            ),
            "parent_u7_12b_exact": parent_exact,
            **core_gates,
            "owned_residue_empty": not any(scratch.iterdir()),
        }
        status = (
            "PASS_PRIVATE_U7_12C_DESKTOP_DISPLAY_NATIVE_PREVIEW"
            if all(gates.values())
            else "FAIL_CLOSED_U7_12C_DESKTOP_DISPLAY_NATIVE_PREVIEW"
        )
        report = {
            "schema": "kmcfm.u7-12c-desktop-display-native-preview-result.v1",
            "status": status,
            "enumeration": "reverse" if reverse else "forward",
            "bindings": bindings,
            "runtime": {
                "platform": sys.platform,
                "python": ".".join(map(str, sys.version_info[:3])),
            },
            "records": records,
            "summary": {
                "maximum_rgb_rmse": max(
                    row["rgb_rmse"]
                    for record in records
                    for group in record["fidelity"]
                    for row in group
                ),
                "maximum_rgb_absolute_error_p95": max(
                    row["rgb_absolute_error_p95"]
                    for record in records
                    for group in record["fidelity"]
                    for row in group
                ),
                "maximum_new_boundary_fraction": max(
                    row["new_boundary_fraction"]
                    for record in records
                    for group in record["fidelity"]
                    for row in group
                ),
                "timing_and_rss_are_nondeterministic_diagnostics": True,
            },
            "gates": gates,
            "claim_ceiling": config["claim_ceiling"],
        }
        report["scientific_identity"] = "sha256:" + _canonical_sha256(
            _scientific_payload(report)
        )
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS
    )
    parser.add_argument("--destination", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not all(
            value is not None
            for value in (
                args.source_id,
                args.variant,
                args.destination,
                args.worker_result,
            )
        ):
            raise U712CError("worker arguments are incomplete")
        return _worker(
            args.config,
            args.source_id,
            args.variant,
            args.destination,
            args.worker_result,
        )
    if args.order is None or args.report is None:
        raise U712CError("controller requires --order and --report")
    report = execute(args.config, reverse=args.order == "reverse")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
