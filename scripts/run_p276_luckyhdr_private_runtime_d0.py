#!/usr/bin/env python3
"""Acquire and run the exact official LuckyHDR demo under the P276 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "neuro-film.p276-luckyhdr-private-runtime-d0.v1"


class P276Error(RuntimeError):
    """Raised when a frozen P276 acquisition or runtime gate fails."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _verify_bytes(value: bytes, binding: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "bytes": len(value),
        "git_blob": _git_blob_sha1(value),
        "sha256": _sha256(value),
    }
    if observed["bytes"] != binding["bytes"] or observed["git_blob"] != binding["git_blob"]:
        raise P276Error(f"official object differs: {binding['path']}")
    return observed


def _download_one(raw_root: str, root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    destination = root / binding["path"]
    if destination.is_file():
        observed = _verify_bytes(destination.read_bytes(), binding)
        return {**observed, "path": binding["path"], "reused": True}
    if destination.exists():
        raise P276Error(f"destination is not a regular file: {binding['path']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise P276Error(f"owned partial path already exists: {temporary}")
    try:
        subprocess.run(
            [
                "curl.exe",
                "--http1.1",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--retry-all-errors",
                "--max-time",
                "180",
                "--output",
                str(temporary),
                f"{raw_root}/{binding['path']}",
            ],
            check=True,
        )
        value = temporary.read_bytes()
        observed = _verify_bytes(value, binding)
        os.rename(temporary, destination)
        return {**observed, "path": binding["path"], "reused": False}
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire(config_path: Path, project_root: Path, workers: int) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_BINARY_ACQUISITION_OR_PIXEL_DECODE":
        raise P276Error("P276 is not frozen")
    if workers < 1 or workers > 4:
        raise P276Error("workers must be in [1,4]")
    data_root = project_root / config["data_root"]
    objects = list(config["objects"].values())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(
                lambda binding: _download_one(
                    config["repository"]["raw_root"], data_root, binding
                ),
                objects,
            )
        )
    manifest = {
        "schema": "neuro-film.p276-luckyhdr-private-runtime-source-manifest.v1",
        "repository_commit": config["repository"]["commit"],
        "repository_tree": config["repository"]["tree"],
        "config_sha256": _sha256(config_bytes),
        "objects": sorted(records, key=lambda item: item["path"]),
        "object_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "binary_pixel_decodes": 0,
        "model_loads": 0,
        "target_reads": 0,
    }
    if (
        manifest["object_count"] != config["limits"]["object_count"]
        or manifest["total_bytes"] != config["limits"]["total_bytes"]
    ):
        raise P276Error("acquired inventory differs from the frozen limits")
    manifest_path = data_root / "source_manifest_v1.json"
    manifest_bytes = _canonical(manifest)
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise P276Error("existing source manifest differs")
    else:
        manifest_path.write_bytes(manifest_bytes)
    return manifest


def _local_inventory(config: dict[str, Any], root: Path, order: str) -> dict[str, Any]:
    names = list(config["objects"])
    if order == "reverse":
        names.reverse()
    observed: dict[str, Any] = {}
    for name in names:
        binding = config["objects"][name]
        path = root / binding["path"]
        if not path.is_file():
            raise P276Error(f"acquired object is absent: {binding['path']}")
        observed[name] = {
            **_verify_bytes(path.read_bytes(), binding),
            "path": binding["path"],
        }
    return dict(sorted(observed.items()))


def execute(config_path: Path, project_root: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P276Error("order must be forward or reverse")
    import cv2
    import imageio
    import numpy as np
    import rawpy
    import torch

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    data_root = project_root / config["data_root"]
    inventory = _local_inventory(config, data_root, order)
    before = {name: record["sha256"] for name, record in inventory.items()}
    runtime = config["runtime"]
    runtime_exact = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "rawpy": rawpy.__version__,
        "opencv": cv2.__version__,
        "imageio": imageio.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    expected_runtime = {key: runtime[key] for key in runtime_exact}
    if runtime_exact != expected_runtime:
        raise P276Error("local runtime differs from the frozen runtime")

    temporary_parent = project_root / "outputs/tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="p276-luckyhdr-", dir=temporary_parent))
    output = temporary / "luckyhdr.png"
    command = [
        sys.executable,
        "-m",
        "luckyhdr.inference",
        "--frames",
        str(data_root / config["objects"]["short"]["path"]),
        str(data_root / config["objects"]["mid"]["path"]),
        str(data_root / config["objects"]["long"]["path"]),
        "--ckpt",
        str(data_root / config["objects"]["checkpoint"]["path"]),
        "--exposures",
        *(str(value) for value in runtime["exposures"]),
        "--max-size",
        str(runtime["max_size"]),
        "--out",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(data_root / "src")
    environment["PYTHONHASHSEED"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=data_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=runtime["wall_seconds_max"],
        )
        wall_seconds = time.perf_counter() - started
        if completed.returncode != 0:
            raise P276Error(
                f"official inference failed ({completed.returncode}): {completed.stderr[-1000:]}"
            )
        output_bytes = output.read_bytes()
        decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise P276Error("official output does not decode")
        after_inventory = _local_inventory(config, data_root, order)
        after = {name: record["sha256"] for name, record in after_inventory.items()}
        gates = {
            "all-source-identities-exact": len(inventory) == config["limits"]["object_count"],
            "cuda-runtime-exact": runtime_exact == expected_runtime,
            "official-inference-exit-zero": completed.returncode == 0,
            "output-rgb8": bool(decoded.ndim == 3 and decoded.shape[2] == 3 and decoded.dtype == np.uint8),
            "output-finite": bool(np.all(np.isfinite(decoded))),
            "output-dynamic-range-nonzero": int(decoded.max()) > int(decoded.min()),
            "output-long-edge-exact": max(decoded.shape[:2]) == runtime["max_size"],
            "source-immutable": before == after,
            "wall-bounded": wall_seconds <= runtime["wall_seconds_max"],
            "zero-target-metric-training-reads": all(
                config["limits"][key] == 0
                for key in ("target_reads", "metric_runs", "training_reads", "external_capture_reads")
            ),
            "product-rights-remain-false": not config["rights"]["product_dependency"],
        }
        scientific = {
            "config_sha256": _sha256(config_bytes),
            "inventory": inventory,
            "runtime": runtime_exact,
            "exposures": runtime["exposures"],
            "max_size": runtime["max_size"],
            "output_bytes": len(output_bytes),
            "output_png_sha256": _sha256(output_bytes),
            "output_rgb8_sha256": _sha256(np.ascontiguousarray(decoded).tobytes()),
            "output_shape": list(decoded.shape),
            "output_min": int(decoded.min()),
            "output_max": int(decoded.max()),
            "gates": gates,
            "target_reads": 0,
            "metric_runs": 0,
            "training_reads": 0,
            "external_capture_reads": 0,
        }
        measurement = {
            "wall_seconds": wall_seconds,
            "stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
            "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=False)

    scientific["zero_temporary_residue"] = not temporary.exists()
    scientific["gates"]["zero-temporary-residue"] = not temporary.exists()
    status = (
        "PASS_PRIVATE_LUCKYHDR_OFFICIAL_BRACKET_RUNTIME_D0"
        if all(scientific["gates"].values())
        else "FAIL_CLOSED"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "order": order,
        "scientific": scientific,
        "scientific_identity": f"sha256:{_sha256(_canonical(scientific))}",
        "measurement": measurement,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("acquire", "formal"), required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "acquire":
        result = acquire(args.config.resolve(), args.project_root.resolve(), args.workers)
    else:
        if args.output is None:
            raise P276Error("--output is required for formal mode")
        result = execute(args.config.resolve(), args.project_root.resolve(), args.order)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(_canonical(result))
    return 0 if result.get("status", "PASS_").startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
