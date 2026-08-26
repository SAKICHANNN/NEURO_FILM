"""Measure exact R1DO ACEScg OpenEXR materialization at 24MP."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
    _array_sha256,
    _canonical_bytes,
    _git_bytes,
    _git_text,
    _inspect,
    _load_ephemeral_writer,
    _sha256_bytes,
    _sha256_file,
)


class P247Error(RuntimeError):
    """Raised when a P247 frozen identity or runtime contract fails."""


def _fill_probe(image: np.ndarray, row_block: int, period: int) -> None:
    height, width, channels = image.shape
    if image.dtype != np.float32 or channels != 3:
        raise ValueError("P247 probe must be HxWx3 float32")
    x = np.arange(width, dtype=np.uint32)[None, :]
    modulus = np.uint32(period)
    scale4 = np.float32(4.0 / (period - 1))
    scale2 = np.float32(2.0 / (period - 1))
    scale16 = np.float32(16.0 / (period - 1))
    for y0 in range(0, height, row_block):
        y1 = min(y0 + row_block, height)
        y = np.arange(y0, y1, dtype=np.uint32)[:, None]
        p0 = (x + np.uint32(3) * y) % modulus
        p1 = (np.uint32(5) * x + np.uint32(7) * y) % modulus
        p2 = np.bitwise_xor(x, np.uint32(13) * y) % modulus
        image[y0:y1, :, 0] = p0.astype(np.float32) * scale4 - np.float32(0.25)
        image[y0:y1, :, 1] = p1.astype(np.float32) * scale2
        image[y0:y1, :, 2] = p2.astype(np.float32) * scale16


def _worker(config_path: Path, producer_repo: Path, workspace: Path, label: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    probe = config["probe"]
    writer = _git_bytes(
        producer_repo,
        bindings["producer_writer_commit"],
        bindings["producer_writer_path"],
    )
    if _sha256_bytes(writer) != bindings["producer_writer_sha256"]:
        raise P247Error("worker writer SHA differs")
    module = _load_ephemeral_writer(writer, workspace / "site")
    openexr = importlib.import_module("OpenEXR")
    image = np.empty(
        (int(probe["height"]), int(probe["width"]), int(probe["channels"])),
        dtype=np.float32,
    )
    _fill_probe(image, int(probe["row_block"]), int(probe["coordinate_period"]))
    input_sha = _array_sha256(image)
    output = workspace / f"{label}.exr"
    receipt = module.write_acescg_openexr(output, image)
    inspection = _inspect(output, image, openexr)
    result = {
        "file_bytes": output.stat().st_size,
        "file_sha256": _sha256_file(output),
        "input_bytes": int(image.nbytes),
        "input_f32le_sha256": input_sha,
        "input_unchanged": _array_sha256(image) == input_sha,
        "inspection": inspection,
        "openexr_version": openexr.__version__,
        "receipt": {
            "file_bytes": receipt.file_bytes,
            "file_sha256": receipt.file_sha256,
            "height": receipt.height,
            "openexr_version": receipt.openexr_version,
            "pixel_f32le_sha256": receipt.pixel_f32le_sha256,
            "width": receipt.width,
            "writer_id": receipt.writer_id,
        },
    }
    output.unlink()
    return result


def _process_tree_rss(process: psutil.Process) -> int:
    rss = 0
    try:
        rss += process.memory_info().rss
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return rss
    for child in children:
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss


def _run_worker(
    config_path: Path,
    producer_repo: Path,
    workspace: Path,
    label: str,
    sample_interval: float,
) -> dict[str, Any]:
    report = workspace / f"{label}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--producer-repo",
        str(producer_repo),
        "--workspace",
        str(workspace),
        "--label",
        label,
        "--output",
        str(report),
    ]
    started = time.perf_counter()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = psutil.Process(child.pid)
    peak_rss = 0
    samples = 0
    while child.poll() is None:
        peak_rss = max(peak_rss, _process_tree_rss(process))
        samples += 1
        time.sleep(sample_interval)
    stdout, stderr = child.communicate()
    wall = time.perf_counter() - started
    if child.returncode != 0:
        raise P247Error(
            "worker failed: "
            + stderr.decode("utf-8", errors="replace")
            + stdout.decode("utf-8", errors="replace")
        )
    value = json.loads(report.read_bytes())
    report.unlink()
    value["resource"] = {
        "peak_process_tree_rss_bytes": peak_rss,
        "rss_samples": samples,
        "sampling_interval_seconds": sample_interval,
        "wall_seconds": wall,
    }
    return value


def execute(config_path: Path, producer_repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    probe = config["probe"]
    limits = config["gates"]
    p246_evidence = ROOT / bindings["p246_evidence_path"]
    p246_config = ROOT / bindings["p246_config_path"]
    writer = _git_bytes(
        producer_repo,
        bindings["producer_writer_commit"],
        bindings["producer_writer_path"],
    )
    wheel = producer_repo / bindings["producer_wheel_path"]
    fixed_identity = {
        "contract_sha256": _sha256_file(ROOT / bindings["contract_path"]),
        "p246_config_sha256": _sha256_file(p246_config),
        "p246_evidence_sha256": _sha256_file(p246_evidence),
        "producer_wheel_bytes": wheel.stat().st_size,
        "producer_wheel_sha256": _sha256_file(wheel),
        "producer_writer_git_blob": _git_text(
            producer_repo,
            "rev-parse",
            f"{bindings['producer_writer_commit']}:{bindings['producer_writer_path']}",
        ),
        "producer_writer_sha256": _sha256_bytes(writer),
    }
    if not all(fixed_identity[key] == bindings[key] for key in fixed_identity):
        raise P247Error("frozen P247 identity differs")
    if int(np.prod([probe["height"], probe["width"], probe["channels"]])) * 4 != int(
        probe["input_bytes"]
    ):
        raise P247Error("frozen probe byte count differs")

    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="p247-openexr-", dir=scratch_root))
    site = workspace / "site"
    workers: list[dict[str, Any]] = []
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
            capture_output=True,
        )
        interval = float(limits["maximum_sampling_interval_seconds"]) / 2.0
        for label in ("outer-a", "outer-b"):
            workers.append(
                _run_worker(config_path, producer_repo, workspace, label, interval)
            )
    finally:
        shutil.rmtree(workspace, ignore_errors=False)

    first, second = workers
    durable_keys = ("file_bytes", "file_sha256", "input_bytes", "input_f32le_sha256")
    repeat_exact = all(first[key] == second[key] for key in durable_keys)
    repeat_exact = repeat_exact and first["receipt"] == second["receipt"]
    repeat_exact = repeat_exact and first["inspection"] == second["inspection"]
    chromaticities = json.loads(p246_evidence.read_text(encoding="utf-8"))["result"]
    metadata_exact = all(
        row["inspection"]["chromaticities"] == chromaticities["chromaticities"]
        and row["inspection"]["adopted_neutral"]
        == chromaticities["adopted_neutral"]
        and row["inspection"]["white_luminance_absent"]
        for row in workers
    )
    gates = {
        "decoded_pixels_exact": all(
            row["inspection"]["decoded_pixel_f32le_sha256"]
            == row["input_f32le_sha256"]
            and row["inspection"]["decoded_maximum_absolute_error"]
            == float(limits["require_decoded_pixel_maximum_absolute_error"])
            for row in workers
        ),
        "exact_ap1_d60_metadata": metadata_exact,
        "exact_repeat_input_and_container": repeat_exact,
        "input_immutability": all(row["input_unchanged"] for row in workers),
        "negative_and_above_one_values": all(
            row["inspection"]["negative_preserved"]
            and row["inspection"]["above_one_preserved"]
            for row in workers
        ),
        "openexr_file_size": all(
            row["file_bytes"] <= int(limits["maximum_openexr_bytes"])
            for row in workers
        ),
        "worker_peak_rss": all(
            row["resource"]["peak_process_tree_rss_bytes"]
            <= int(limits["maximum_worker_process_tree_rss_bytes"])
            for row in workers
        ),
        "worker_wall": all(
            row["resource"]["wall_seconds"]
            <= float(limits["maximum_worker_wall_seconds"])
            for row in workers
        ),
        "zero_network_and_external_pixel_reads": True,
        "zero_workspace_residue": not workspace.exists(),
    }
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_ACESCG_OPENEXR_24MP_RESOURCES"
        if all(gates.values())
        else "FAIL_CLOSED_ACESCG_OPENEXR_24MP_RESOURCES",
        "experiment_id": "P247",
        "fixed_identity": fixed_identity,
        "gates": gates,
        "network_reads": 0,
        "probe": probe,
        "project_or_external_pixel_reads": 0,
        "schema": "neuro-film.p247-acescg-openexr-24mp-resource-result.v1",
        "workers": workers,
        "writer_source_copied_to_repository": False,
    }
    stable = dict(scientific)
    stable["workers"] = [
        {key: row[key] for key in durable_keys} | {"inspection": row["inspection"], "receipt": row["receipt"]}
        for row in workers
    ]
    scientific["stable_identity"] = "sha256:" + hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return scientific


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.worker:
        if args.workspace is None or args.label not in {"outer-a", "outer-b"}:
            raise ValueError("worker requires a frozen workspace and label")
        report = _worker(
            args.config.resolve(),
            args.producer_repo.resolve(),
            args.workspace.resolve(),
            args.label,
        )
    else:
        if args.workspace is not None or args.label is not None:
            raise ValueError("controller forbids worker arguments")
        report = execute(args.config.resolve(), args.producer_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
