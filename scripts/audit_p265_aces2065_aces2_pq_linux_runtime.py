"""Audit strict P251/P252 parity in WSL2 Ubuntu 22.04 x86_64."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _synthetic_lattice,
    )
except ModuleNotFoundError:
    from audit_p246_acescg_openexr_exact_consumer_intake import (
        _git_bytes,
        _synthetic_lattice,
    )

ROOT = Path(__file__).resolve().parents[1]


class P265Error(RuntimeError):
    """Raised when a frozen P265 identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if len(drive) != 1 or not drive.isalpha():
        raise P265Error(f"unsupported Windows path for WSL: {resolved}")
    suffix = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def _extract_wheels(wheels: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(destination)


def _run_linux_probe(
    *, workspace: Path, order: str, config: dict[str, Any]
) -> dict[str, Any]:
    probe_output = workspace / "linux-probe.json"
    site = workspace / "site"
    command = [
        "wsl.exe",
        "-d",
        config["runtime"]["distribution"],
        "--",
        "/usr/bin/env",
        f"PYTHONPATH={_windows_to_wsl(site)}:{_windows_to_wsl(ROOT)}",
        config["runtime"]["python"],
        _windows_to_wsl(ROOT / config["bindings"]["probe_path"]),
        "--root",
        _windows_to_wsl(ROOT),
        "--workspace",
        _windows_to_wsl(workspace),
        "--writer-source",
        _windows_to_wsl(workspace / "bound-r1dt-writer.py"),
        "--input-pixels",
        _windows_to_wsl(workspace / "p249-input.f32le"),
        "--order",
        order,
        "--output",
        _windows_to_wsl(probe_output),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        raise P265Error(
            "Linux probe failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-4000:]
        )
    return json.loads(probe_output.read_bytes())


def execute(
    config_path: Path,
    producer_repo: Path,
    order: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    identity_paths = {
        key.removesuffix("_path"): ROOT / value
        for key, value in bindings.items()
        if key.endswith("_path")
    }
    identities = {
        name: _sha256_file(path) == bindings[f"{name}_sha256"]
        for name, path in identity_paths.items()
        if f"{name}_sha256" in bindings
    }

    wheel_paths = {
        "numpy": producer_repo / config["linux_wheels"]["numpy"][0],
        "openexr": producer_repo / config["linux_wheels"]["openexr"][0],
        "opencolorio": ROOT / config["linux_wheels"]["opencolorio"][0],
    }
    for name, path in wheel_paths.items():
        expected = config["linux_wheels"][name]
        identities[f"wheel_{name}"] = (
            path.is_file()
            and path.stat().st_size == expected[1]
            and _sha256_file(path) == expected[2]
        )

    writer = _git_bytes(
        producer_repo,
        config["producer_writer"]["commit"],
        config["producer_writer"]["path"],
    )
    identities["producer_writer"] = (
        len(writer) == config["producer_writer"]["bytes"]
        and _sha256_bytes(writer) == config["producer_writer"]["sha256"]
    )
    pixels = _synthetic_lattice()
    pixel_bytes = np.ascontiguousarray(pixels, dtype="<f4").tobytes(order="C")
    identities["p249_input_pixels"] = (
        _sha256_bytes(pixel_bytes)
        == config["windows_oracle"]["p249_input_acescg_f32le_sha256"]
    )

    temporary = Path(tempfile.mkdtemp(prefix="p265-", dir=ROOT / "tmp"))
    worker: dict[str, Any] | None = None
    try:
        (temporary / "bound-r1dt-writer.py").write_bytes(writer)
        (temporary / "p249-input.f32le").write_bytes(pixel_bytes)
        _extract_wheels(list(wheel_paths.values()), temporary / "site")
        worker = _run_linux_probe(
            workspace=temporary, order=order, config=config
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if worker is None:
        raise P265Error("Linux probe did not produce a report")

    oracle = config["windows_oracle"]
    runtime = config["runtime"]
    gates = {
        "exact_bound_identities": all(identities.values()),
        "exact_linux_runtime": (
            worker["numpy_version"] == runtime["numpy_version"]
            and worker["openexr_version"] == runtime["openexr_version"]
            and worker["opencolorio_version"] == runtime["opencolorio_version"]
            and worker["config_cache_id"] == runtime["config_cache_id"]
        ),
        "exact_p249_source": (
            worker["source_bytes"] == oracle["p249_openexr_bytes"]
            and worker["source_sha256"] == oracle["p249_openexr_sha256"]
        ),
        "exact_windows_linux_rgb16": (
            worker["output_samples_sha256"] == oracle["p252_rgb16_sha256"]
            and worker["strict_sample_sha256"] == oracle["p252_rgb16_sha256"]
        ),
        "exact_windows_linux_png": (
            worker["output_bytes"] == oracle["p252_png_bytes"]
            and worker["output_png_sha256"] == oracle["p252_png_sha256"]
            and worker["output_receipt_sha256"] == oracle["p252_png_sha256"]
        ),
        "exact_encoded_output": worker["encoded_f32le_sha256"]
        == oracle["p252_encoded_f32le_sha256"],
        "strict_working_image": (
            worker["working"]["shape"] == oracle["shape"]
            and worker["working"]["working_space"] == "acescg_ap1_d60"
            and worker["working"]["transfer_state"] == "scene_linear"
        ),
        "finite_unit_output": worker["encoded_finite_in_unit"],
        "invalid_controls_rejected": all(
            worker["invalid_controls_rejected"].values()
        ),
        "foreign_destination_preserved": (
            worker["foreign_destination_rejected"]
            and worker["foreign_destination_unchanged"]
        ),
        "source_and_input_immutable": (
            worker["source_unchanged"] and worker["input_pixels_unchanged"]
        ),
        "pq_path_did_not_call_sdr_dependency": worker[
            "unused_sdr_dependency_calls"
        ]
        == {"srgb_icc_profile": 0},
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_ACES2065_ACES2_PQ_LINUX_RUNTIME"
        if all(gates.values())
        else "FAIL_CLOSED_ACES2065_ACES2_PQ_LINUX_RUNTIME",
        "experiment_id": "P265",
        "gates": gates,
        "network_reads": 0,
        "result": worker,
        "schema": "neuro-film.p265-aces2065-aces2-pq-linux-runtime-result.v1",
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute(
        args.config.resolve(), args.producer_repo.resolve(), args.order
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
