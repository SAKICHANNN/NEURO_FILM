#!/usr/bin/env python3
"""Run the frozen U4.5F RAW half-size preview audit."""

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

import cv2
import numpy as np
import psutil
import rawpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.three_stock_preview import (
    preview_dimensions,
    preview_fidelity_metrics,
)


class U45FError(RuntimeError):
    """Raised when the frozen RAW half-size audit cannot execute."""


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
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_execution_bindings(config: dict[str, Any]) -> bool:
    bindings = config.get("execution_bindings")
    if not isinstance(bindings, dict) or not bindings:
        return False
    for item in bindings.values():
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
            or _git_output("rev-parse", f"{item['commit']}^{{commit}}")
            != item["commit"]
            or _git_output("rev-parse", f"{item['commit']}:{item['path']}")
            != item["git_blob"]
        ):
            return False
    return True


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise U45FError(f"failed to decode expected RGB8 preview: {path}")
    return np.ascontiguousarray(bgr[..., ::-1].astype(np.float32) / 255.0)


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


def _run_preview(
    config: dict[str, Any], source: Path, destination: Path, *, half_size: bool
) -> dict[str, Any]:
    render = config["render"]
    command = [
        sys.executable,
        str(ROOT / "scripts/render_three_stock_preview.py"),
        str(source),
        str(destination),
        "--max-preview-pixels",
        str(render["max_preview_pixels"]),
        "--look-amount",
        str(render["look_amount"]),
        "--seed",
        str(render["seed"]),
        "--tile-size",
        str(render["tile_size"]),
        "--tile-workers",
        str(render["tile_workers"]),
        "--png-compression",
        str(render["png_compression"]),
    ]
    if half_size:
        command.append("--raw-half-size-decode")
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
    wall_seconds = time.perf_counter() - started
    if child.returncode != 0:
        raise U45FError(
            f"preview worker failed with {child.returncode}: {stderr.strip()}"
        )
    try:
        manifest = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise U45FError("preview worker emitted invalid JSON") from exc
    output_hashes = {
        str(row["style_id"]): str(row["output_sha256"])
        for row in manifest.get("rows", [])
    }
    if sorted(output_hashes) != ["ektar_100", "portra_400", "velvia_50"]:
        raise U45FError("preview worker did not emit the exact three stock rows")
    return {
        "wall_seconds": wall_seconds,
        "peak_process_tree_rss_bytes": peak_rss,
        "manifest": manifest,
        "output_hashes": output_hashes,
    }


def _source_record(
    config: dict[str, Any], row: dict[str, Any], scratch: Path, *, reverse: bool
) -> dict[str, Any]:
    source = ROOT / row["path"]
    before_sha = sha256_file(source)
    full_destination = scratch / row["source_id"] / "full"
    candidate_destination = scratch / row["source_id"] / "candidate"
    if reverse:
        full = _run_preview(config, source, full_destination, half_size=False)
        candidate = _run_preview(config, source, candidate_destination, half_size=True)
    else:
        candidate = _run_preview(config, source, candidate_destination, half_size=True)
        full = _run_preview(config, source, full_destination, half_size=False)

    fidelity: list[dict[str, Any]] = []
    for style_id in sorted(candidate["output_hashes"]):
        candidate_rgb = _load_rgb(candidate_destination / f"{style_id}.preview.png")
        full_rgb = _load_rgb(full_destination / f"{style_id}.preview.png")
        fidelity.append(
            {
                "style_id": style_id,
                **preview_fidelity_metrics(candidate_rgb, full_rgb),
            }
        )

    candidate_manifest = candidate["manifest"]
    full_manifest = full["manifest"]
    return {
        "source_id": row["source_id"],
        "source_path": row["path"],
        "source_sha256": before_sha,
        "source_unchanged": sha256_file(source) == before_sha,
        "source_width": int(candidate_manifest["source_width"]),
        "source_height": int(candidate_manifest["source_height"]),
        "preview_width": int(candidate_manifest["preview_width"]),
        "preview_height": int(candidate_manifest["preview_height"]),
        "candidate": {
            "decoded_width": int(candidate_manifest["decoded_width"]),
            "decoded_height": int(candidate_manifest["decoded_height"]),
            "raw_half_size_decode": candidate_manifest.get("raw_half_size_decode"),
            "preview_basis": candidate_manifest["preview_basis"],
            "output_hashes": candidate["output_hashes"],
            "peak_process_tree_rss_bytes": candidate["peak_process_tree_rss_bytes"],
            "wall_seconds": candidate["wall_seconds"],
        },
        "full": {
            "decoded_width": int(full_manifest["decoded_width"]),
            "decoded_height": int(full_manifest["decoded_height"]),
            "raw_half_size_decode_present": "raw_half_size_decode" in full_manifest,
            "preview_basis": full_manifest["preview_basis"],
            "output_hashes": full["output_hashes"],
            "peak_process_tree_rss_bytes": full["peak_process_tree_rss_bytes"],
            "wall_seconds": full["wall_seconds"],
        },
        "fidelity": fidelity,
        "candidate_to_full_rss_ratio": candidate["peak_process_tree_rss_bytes"]
        / full["peak_process_tree_rss_bytes"],
        "candidate_to_full_wall_ratio": candidate["wall_seconds"]
        / full["wall_seconds"],
    }


def evaluate_records(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, bool]:
    """Evaluate one complete five-source execution against frozen gates."""

    gates = config["gates"]
    rows_by_id = {row["source_id"]: row for row in config["sources"]}
    fidelity = [item for record in records for item in record["fidelity"]]
    wall_ratios = [float(record["candidate_to_full_wall_ratio"]) for record in records]
    return {
        "required_sources_exact": len(records) == len(rows_by_id) == 5
        and {record["source_id"] for record in records} == set(rows_by_id),
        "source_identities_exact": all(
            record["source_sha256"] == rows_by_id[record["source_id"]]["sha256"]
            and record["source_width"] == rows_by_id[record["source_id"]]["width"]
            and record["source_height"] == rows_by_id[record["source_id"]]["height"]
            for record in records
        ),
        "sources_immutable": all(record["source_unchanged"] for record in records),
        "candidate_decodes_before_float_expansion": all(
            record["candidate"]["raw_half_size_decode"] is True
            and record["candidate"]["decoded_width"] < record["source_width"]
            and record["candidate"]["decoded_height"] < record["source_height"]
            and record["candidate"]["decoded_width"] >= record["preview_width"]
            and record["candidate"]["decoded_height"] >= record["preview_height"]
            and record["full"]["raw_half_size_decode_present"] is False
            for record in records
        ),
        "exact_preview_dimensions": all(
            (record["preview_width"], record["preview_height"])
            == preview_dimensions(
                record["source_width"],
                record["source_height"],
                int(config["render"]["max_preview_pixels"]),
            )
            for record in records
        ),
        "three_distinct_stock_outputs": all(
            len(set(record["candidate"]["output_hashes"].values())) == 3
            and len(set(record["full"]["output_hashes"].values())) == 3
            for record in records
        ),
        "rgb_rmse": max(float(item["rgb_rmse"]) for item in fidelity)
        <= float(gates["maximum_rgb_rmse_vs_full_decode"]),
        "rgb_absolute_error_p95": max(
            float(item["rgb_absolute_error_p95"]) for item in fidelity
        )
        <= float(gates["maximum_rgb_absolute_error_p95"]),
        "new_boundary_fraction": max(
            float(item["new_boundary_fraction"]) for item in fidelity
        )
        <= float(gates["maximum_new_boundary_fraction"]),
        "candidate_process_tree_rss": max(
            int(record["candidate"]["peak_process_tree_rss_bytes"])
            for record in records
        )
        <= int(gates["maximum_candidate_process_tree_rss_bytes"]),
        "candidate_to_full_rss_ratio": all(
            float(record["candidate_to_full_rss_ratio"])
            <= float(gates["maximum_candidate_to_full_rss_ratio_per_source"])
            for record in records
        ),
        "candidate_wall_seconds": max(
            float(record["candidate"]["wall_seconds"]) for record in records
        )
        <= float(gates["maximum_candidate_wall_seconds_per_source"]),
        "candidate_to_full_wall_ratio": statistics.median(wall_ratios)
        <= float(gates["maximum_median_candidate_to_full_wall_ratio"])
        and max(wall_ratios)
        <= float(gates["maximum_worst_candidate_to_full_wall_ratio"]),
    }


def _scientific_payload(report: dict[str, Any]) -> dict[str, Any]:
    records = []
    for record in report["records"]:
        records.append(
            {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "candidate_to_full_rss_ratio",
                    "candidate_to_full_wall_ratio",
                }
            }
        )
        records[-1]["candidate"] = {
            key: value
            for key, value in records[-1]["candidate"].items()
            if key not in {"peak_process_tree_rss_bytes", "wall_seconds"}
        }
        records[-1]["full"] = {
            key: value
            for key, value in records[-1]["full"].items()
            if key not in {"peak_process_tree_rss_bytes", "wall_seconds"}
        }
    return {
        "schema": report["schema"],
        "bindings": report["bindings"],
        "runtime": report["runtime"],
        "records": records,
        "quality_and_structure_gates": {
            key: value
            for key, value in report["gates"].items()
            if key
            not in {
                "candidate_process_tree_rss",
                "candidate_to_full_rss_ratio",
                "candidate_wall_seconds",
                "candidate_to_full_wall_ratio",
                "owned_residue_empty",
            }
        },
        "claim_ceiling": report["claim_ceiling"],
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = {
        "config_sha256": sha256_file(config_path),
        "p98_config_sha256": sha256_file(ROOT / config["upstream"]["p98_config_path"]),
        "p98_evidence_sha256": sha256_file(
            ROOT / config["upstream"]["p98_evidence_path"]
        ),
        "raw_preview_decode_sha256": sha256_file(
            ROOT / "src/preprocess/raw_preview_decode.py"
        ),
        "three_stock_preview_sha256": sha256_file(
            ROOT / "src/inference/three_stock_preview.py"
        ),
        "preview_cli_sha256": sha256_file(
            ROOT / "scripts/render_three_stock_preview.py"
        ),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    upstream_exact = (
        bindings["p98_config_sha256"] == config["upstream"]["p98_config_sha256"]
        and bindings["p98_evidence_sha256"] == config["upstream"]["p98_evidence_sha256"]
    )
    execution_bindings_exact = _verify_execution_bindings(config)
    runtime = {
        "platform": sys.platform,
        "python": ".".join(map(str, sys.version_info[:3])),
        "rawpy": rawpy.__version__,
        "libraw": list(rawpy.libraw_version),
    }
    runtime_exact = (
        sys.platform == "win32"
        and config["runtime"]["platform"] == "Windows"
        and rawpy.__version__ == config["runtime"]["rawpy"]
        and list(rawpy.libraw_version) == config["runtime"]["libraw"]
    )
    rows = list(config["sources"])
    if reverse:
        rows.reverse()
    for row in rows:
        source = ROOT / row["path"]
        if (
            not source.is_file()
            or source.stat().st_size != int(row["bytes"])
            or sha256_file(source) != row["sha256"]
        ):
            raise U45FError(f"frozen source identity mismatch: {row['source_id']}")

    scratch_parent = ROOT / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="u4_5f_", dir=scratch_parent))
    try:
        records = [
            _source_record(config, row, scratch, reverse=reverse) for row in rows
        ]
        records.sort(key=lambda item: item["source_id"])
        for child in list(scratch.iterdir()):
            shutil.rmtree(child)
        gates = {
            "upstream_bindings_exact": upstream_exact,
            "execution_bindings_exact": execution_bindings_exact,
            "runtime_exact": runtime_exact,
            **evaluate_records(config, records),
            "owned_residue_empty": not any(scratch.iterdir()),
        }
        report = {
            "schema": "kmcfm.u4-5f-raw-half-size-preview-result.v1",
            "status": (
                "PASS_PRIVATE_U4_5F_RAW_HALF_SIZE_PREVIEW"
                if all(gates.values())
                else "FAIL_CLOSED_U4_5F_RAW_HALF_SIZE_PREVIEW"
            ),
            "enumeration": "reverse" if reverse else "forward",
            "bindings": bindings,
            "runtime": runtime,
            "records": records,
            "summary": {
                "maximum_rgb_rmse": max(
                    item["rgb_rmse"]
                    for record in records
                    for item in record["fidelity"]
                ),
                "maximum_rgb_absolute_error_p95": max(
                    item["rgb_absolute_error_p95"]
                    for record in records
                    for item in record["fidelity"]
                ),
                "maximum_new_boundary_fraction": max(
                    item["new_boundary_fraction"]
                    for record in records
                    for item in record["fidelity"]
                ),
                "maximum_candidate_process_tree_rss_bytes": max(
                    record["candidate"]["peak_process_tree_rss_bytes"]
                    for record in records
                ),
                "maximum_candidate_wall_seconds": max(
                    record["candidate"]["wall_seconds"] for record in records
                ),
                "maximum_candidate_to_full_rss_ratio": max(
                    record["candidate_to_full_rss_ratio"] for record in records
                ),
                "median_candidate_to_full_wall_ratio": statistics.median(
                    record["candidate_to_full_wall_ratio"] for record in records
                ),
                "maximum_candidate_to_full_wall_ratio": max(
                    record["candidate_to_full_wall_ratio"] for record in records
                ),
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
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u4_5f_raw_half_size_preview_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config, reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
