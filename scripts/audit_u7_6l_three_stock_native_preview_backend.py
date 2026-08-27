#!/usr/bin/env python3
"""Audit the opt-in exact native backend of the three-stock preview API."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.render_contract import atomic_write_json, sha256_file
from src.inference.three_stock_native_preview import (
    build_native_three_stock_preview_backend,
)
from src.inference.three_stock_preview import render_three_stock_previews_to_directory

CONTRACT_SCHEMA = "neuro-film.u7-6l-three-stock-native-preview-backend-contract.v1"
WORKER_SCHEMA = "neuro-film.u7-6l-three-stock-native-preview-backend-worker.v1"
REPORT_SCHEMA = "neuro-film.u7-6l-three-stock-native-preview-backend-result.v1"


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    contract = json.loads(raw)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("U7.6L contract schema drift")
    for parent in contract["parents"]:
        if _sha256(ROOT / parent["evidence_path"]) != parent["evidence_sha256"]:
            raise ValueError(f"U7.6L parent evidence drift: {parent['experiment_id']}")
    if _sha256(ROOT / contract["source"]["path"]) != contract["source"]["sha256"]:
        raise ValueError("U7.6L source identity drift")
    return contract, hashlib.sha256(raw).hexdigest()


def _normalized_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["input_path"] = Path(result["input_path"]).name
    rows = []
    for row in result["rows"]:
        normalized = dict(row)
        normalized["output_path"] = Path(normalized["output_path"]).name
        rows.append(normalized)
    result["rows"] = rows
    return result


def _pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def _render(
    source: Path,
    destination: Path,
    contract: dict[str, Any],
    *,
    backend=None,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    manifest = render_three_stock_previews_to_directory(
        source,
        destination,
        root=ROOT,
        profile_path=ROOT / "configs/render_profiles/safe_rich_v1.json",
        statistics_path=ROOT / "configs/film_color_stats.json",
        guardrails_path=ROOT / "configs/color_guardrails.json",
        max_preview_pixels=int(contract["execution"]["maximum_preview_pixels"]),
        look_amount=float(contract["execution"]["look_amount"]),
        seed=int(contract["execution"]["seed"]),
        tile_size=256,
        tile_workers=1,
        png_compression=int(contract["execution"]["png_compression"]),
        native_backend=backend,
    )
    return manifest, time.perf_counter() - started


def run_worker(contract_path: Path, output_path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    contract, contract_sha256 = _load_contract(contract_path)
    scratch_parent = ROOT / "tmp" / "u7_6l_three_stock_native_preview_backend"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="worker_", dir=scratch_parent))
    source = ROOT / contract["source"]["path"]
    try:
        python_manifest, python_wall = _render(source, scratch / "python", contract)
        backend = build_native_three_stock_preview_backend(
            root=ROOT,
            output_directory=scratch / "native-build",
            thread_count=int(contract["execution"]["thread_count"]),
            gamut_workers=int(contract["execution"]["gamut_workers"]),
        )
        try:
            native_manifest, native_wall = _render(
                source, scratch / "native", contract, backend=backend
            )
            build_identity = {
                "dll_sha256": backend.dll_sha256,
                "source_sha256": backend.source_sha256,
                "header_sha256": backend.header_sha256,
                "toolchain": backend.toolchain,
            }
        finally:
            backend.close()

        python_hashes = [row["output_sha256"] for row in python_manifest["rows"]]
        native_hashes = [row["output_sha256"] for row in native_manifest["rows"]]
        python_pixels = [
            _pixel_sha256(scratch / "python" / Path(row["output_path"]).name)
            for row in python_manifest["rows"]
        ]
        native_pixels = [
            _pixel_sha256(scratch / "native" / Path(row["output_path"]).name)
            for row in native_manifest["rows"]
        ]

        closed_destination = scratch / "closed"
        failure_message = ""
        try:
            _render(source, closed_destination, contract, backend=backend)
        except (RuntimeError, ValueError) as exc:
            failure_message = f"{type(exc).__name__}: {exc}"
        stage_residue = sorted(path.name for path in scratch.glob(".closed.*.stage"))
        worker = {
            "schema": WORKER_SCHEMA,
            "contract_sha256": contract_sha256,
            "source_sha256": _sha256(source),
            "build": build_identity,
            "python_manifest": _normalized_manifest(python_manifest),
            "native_manifest": _normalized_manifest(native_manifest),
            "python_output_sha256": python_hashes,
            "native_output_sha256": native_hashes,
            "python_pixel_sha256": python_pixels,
            "native_pixel_sha256": native_pixels,
            "closed_backend_failure": failure_message,
            "closed_backend_destination_exists": closed_destination.exists(),
            "closed_backend_stage_residue": stage_residue,
        }
        atomic_write_json(output_path, worker)
        return worker, {
            "python_wall_seconds": python_wall,
            "native_wall_seconds": native_wall,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _stable_id(report: dict[str, Any]) -> str:
    payload = {
        "schema": report["schema"],
        "contract_sha256": report["contract_sha256"],
        "worker_sha256": report["worker_sha256"],
        "gates": report["gates"],
        "status": report["status"],
        "claim_ceiling": report["claim_ceiling"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_controller(contract_path: Path, report_path: Path) -> dict[str, Any]:
    contract, contract_sha256 = _load_contract(contract_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    workers: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    worker_shas: list[str] = []
    temp_paths: list[Path] = []
    try:
        for index in range(int(contract["execution"]["formal_processes"])):
            with tempfile.NamedTemporaryFile(
                prefix=f"u7_6l_worker_{index}_",
                suffix=".json",
                dir=report_path.parent,
                delete=False,
            ) as handle:
                worker_path = Path(handle.name)
            worker_path.unlink()
            temp_paths.append(worker_path)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--contract",
                    str(contract_path.resolve()),
                    "--report",
                    str(worker_path.resolve()),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )
            timings.append(json.loads(completed.stdout.strip()))
            workers.append(json.loads(worker_path.read_text("utf-8")))
            worker_shas.append(_sha256(worker_path))
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)

    python_max = max(row["python_wall_seconds"] for row in timings)
    native_max = max(row["native_wall_seconds"] for row in timings)
    ratio = native_max / python_max
    first = workers[0]
    gates = {
        "cross_process_report_identity": len(set(worker_shas)) == 1,
        "reproducible_dll_identity": len({row["build"]["dll_sha256"] for row in workers}) == 1,
        "default_python_manifest_backend": all(
            "color_backend" not in row["python_manifest"] for row in workers
        ),
        "exact_native_vs_python_png_sha256": all(
            row["native_output_sha256"] == row["python_output_sha256"] for row in workers
        ),
        "exact_native_vs_python_rgb_sha256": all(
            row["native_pixel_sha256"] == row["python_pixel_sha256"] for row in workers
        ),
        "three_distinct_outputs": len(set(first["native_output_sha256"])) == 3,
        "failure_before_publication": all(
            "backend is unavailable" in row["closed_backend_failure"]
            and not row["closed_backend_destination_exists"]
            for row in workers
        ),
        "zero_stage_residue": all(not row["closed_backend_stage_residue"] for row in workers),
        "native_wall_seconds": native_max <= float(
            contract["gates"]["maximum_native_transaction_wall_seconds"]
        ),
        "native_to_python_wall_ratio": ratio
        <= float(contract["gates"]["maximum_native_to_python_transaction_wall_ratio"]),
    }
    status = (
        contract["decision_if_pass"]
        if all(gates.values())
        else contract["decision_if_fail"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": contract_sha256,
        "worker_sha256": worker_shas,
        "timings": timings,
        "maximum_python_transaction_wall_seconds": python_max,
        "maximum_native_transaction_wall_seconds": native_max,
        "native_to_python_transaction_wall_ratio": ratio,
        "build": first["build"],
        "styles": [row["style_id"] for row in first["native_manifest"]["rows"]],
        "output_sha256": first["native_output_sha256"],
        "pixel_sha256": first["native_pixel_sha256"],
        "gates": gates,
        "status": status,
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_id"] = _stable_id(report)
    atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/u7_6l_three_stock_native_preview_backend_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/eval/u7_6l_three_stock_native_preview_backend_v1/report.json",
    )
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        _, timing = run_worker(args.contract, args.report)
        print(json.dumps(timing, sort_keys=True, separators=(",", ":")))
    else:
        report = run_controller(args.contract, args.report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
