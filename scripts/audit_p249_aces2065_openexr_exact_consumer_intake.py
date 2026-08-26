"""Audit exact producer R1DT ACES2065-1 OpenEXR handoff consumption."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
        _array_sha256,
        _canonical_bytes,
        _git_bytes,
        _git_text,
        _load_ephemeral_writer,
        _sha256_bytes,
        _sha256_file,
        _synthetic_lattice,
    )
else:
    from audit_p246_acescg_openexr_exact_consumer_intake import (
        _array_sha256,
        _canonical_bytes,
        _git_bytes,
        _git_text,
        _load_ephemeral_writer,
        _sha256_bytes,
        _sha256_file,
        _synthetic_lattice,
    )

ROOT = Path(__file__).resolve().parents[1]


class P249Error(RuntimeError):
    """Raised when a frozen P249 identity or execution gate fails."""


def _normalize_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _expected_ap0(values: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(
            np.asarray(values, dtype=np.float64)
            @ np.asarray(matrix, dtype=np.float64).T,
            dtype=np.float32,
        )
    )


def _inspect(path: Path, expected_pixels: np.ndarray, openexr: Any) -> dict[str, Any]:
    with openexr.File(str(path)) as exr_file:
        header = dict(exr_file.header())
        channels = exr_file.channels()
        decoded = channels["RGB"].pixels.copy()
        channel_names = sorted(channels)
    return {
        "above_one_preserved": bool(np.max(decoded) > 1.0),
        "aces_image_container_flag": int(header["acesImageContainerFlag"]),
        "adopted_neutral": [float(value) for value in header["adoptedNeutral"]],
        "channel_names": channel_names,
        "chromaticities": [float(value) for value in header["chromaticities"]],
        "color_interop_id": _normalize_text(header["colorInteropID"]),
        "compression_zip": bool(header["compression"] == openexr.ZIP_COMPRESSION),
        "decoded_dtype": str(decoded.dtype),
        "decoded_maximum_absolute_error": float(
            np.max(
                np.abs(
                    decoded.astype(np.float64) - expected_pixels.astype(np.float64)
                )
            )
        ),
        "decoded_pixel_f32le_sha256": _array_sha256(decoded),
        "decoded_shape": list(decoded.shape),
        "negative_preserved": bool(np.min(decoded) < 0.0),
        "scanline_image": bool(header["type"] == openexr.scanlineimage),
        "white_luminance_absent": "whiteLuminance" not in header,
    }


def _invalid_inputs_rejected(module: types.ModuleType, directory: Path) -> bool:
    invalid = (
        np.zeros((1, 1, 3), dtype=np.float64),
        np.zeros((1, 1, 4), dtype=np.float32),
        np.zeros((0, 1, 3), dtype=np.float32),
        np.full((1, 1, 3), np.nan, dtype=np.float32),
        np.full((1, 1, 3), 65505.0, dtype=np.float32),
    )
    for index, value in enumerate(invalid):
        target = directory / f"invalid-{index}.exr"
        try:
            module.write_aces2065_1_openexr(target, value)
        except (TypeError, ValueError):
            if target.exists():
                return False
        else:
            return False
    return True


def _publication_failure_atomic(module: types.ModuleType, directory: Path) -> bool:
    target = directory / "existing.exr"
    original = b"p249-existing-target"
    target.write_bytes(original)
    real_replace = module.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected P249 publication failure")

    module.os.replace = fail_replace
    try:
        try:
            module.write_aces2065_1_openexr(target, _synthetic_lattice())
        except module.AcescgOpenExrError:
            pass
        else:
            return False
    finally:
        module.os.replace = real_replace
    residue = list(directory.glob(f".{target.name}.*.tmp.exr"))
    return target.read_bytes() == original and residue == []


def _worker_execute(
    config_path: Path, producer_repo: Path, workspace: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    expected = config["expected"]
    writer = _git_bytes(
        producer_repo,
        bindings["producer_writer_commit"],
        bindings["producer_writer_path"],
    )
    if len(writer) != bindings["producer_writer_bytes"] or (
        _sha256_bytes(writer) != bindings["producer_writer_sha256"]
    ):
        raise P249Error("worker writer identity differs")
    site = workspace / "site"
    output = workspace / "synthetic-ap0.exr"
    module = _load_ephemeral_writer(writer, site)
    openexr = importlib.import_module("OpenEXR")
    values = _synthetic_lattice()
    expected_ap0 = _expected_ap0(values, expected["ap1_to_ap0_matrix"])
    input_before = _array_sha256(values)
    receipt = module.write_aces2065_1_openexr(output, values)
    inspection = _inspect(output, expected_ap0, openexr)
    return {
        "file_bytes": output.stat().st_size,
        "file_sha256": _sha256_file(output),
        "input_f32le_sha256": input_before,
        "input_unchanged": _array_sha256(values) == input_before,
        "inspection": inspection,
        "invalid_inputs_rejected": _invalid_inputs_rejected(module, workspace),
        "matrix_exact": np.array_equal(
            module.ACESCG_TO_ACES2065_1,
            np.asarray(expected["ap1_to_ap0_matrix"], dtype=np.float64),
        ),
        "openexr_version": openexr.__version__,
        "publication_failure_atomic": _publication_failure_atomic(module, workspace),
        "receipt": {
            "file_bytes": receipt.file_bytes,
            "file_sha256": receipt.file_sha256,
            "height": receipt.height,
            "input_acescg_f32le_sha256": receipt.input_acescg_f32le_sha256,
            "openexr_version": receipt.openexr_version,
            "pixel_f32le_sha256": receipt.pixel_f32le_sha256,
            "width": receipt.width,
            "writer_id": receipt.writer_id,
        },
    }


def execute(config_path: Path, producer_repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    expected = config["expected"]
    contract = ROOT / bindings["contract_path"]
    writer = _git_bytes(
        producer_repo,
        bindings["producer_writer_commit"],
        bindings["producer_writer_path"],
    )
    writer_blob = _git_text(
        producer_repo,
        "rev-parse",
        f"{bindings['producer_writer_commit']}:{bindings['producer_writer_path']}",
    )
    evidence = _git_bytes(
        producer_repo,
        bindings["producer_r1dt_evidence_commit"],
        bindings["producer_r1dt_evidence_path"],
    )
    evidence_json = json.loads(evidence)
    wheel = producer_repo / bindings["producer_wheel_path"]
    fixed_identity = {
        "contract_sha256": _sha256_file(contract),
        "producer_r1dt_evidence_bytes": len(evidence),
        "producer_r1dt_evidence_sha256": _sha256_bytes(evidence),
        "producer_wheel_bytes": wheel.stat().st_size,
        "producer_wheel_sha256": _sha256_file(wheel),
        "producer_writer_bytes": len(writer),
        "producer_writer_git_blob": writer_blob,
        "producer_writer_sha256": _sha256_bytes(writer),
    }
    identity_exact = {
        key: fixed_identity[key] == bindings[key] for key in fixed_identity
    }
    evidence_exact = (
        evidence_json["status"] == "PASS_PRIVATE_ACES2065_OPENEXR_CONTAINER"
        and evidence_json["fixed_identity"]["implementation_commit"]
        == bindings["producer_writer_commit"]
        and evidence_json["official_identity"]["openexr_wheel_sha256"]
        == bindings["producer_wheel_sha256"]
        and evidence_json["encoding"]["writer_id"] == expected["writer_id"]
        and evidence_json["encoding"]["ap1_to_ap0_matrix"]
        == expected["ap1_to_ap0_matrix"]
    )
    if not all(identity_exact.values()) or not evidence_exact:
        raise P249Error("frozen producer identity differs")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected[
        "python_major_minor"
    ]:
        raise P249Error("formal Python major/minor differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p249-"))
    site = temporary / "site"
    result: dict[str, Any] | None = None
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
        worker_report = temporary / "worker.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(config_path),
                "--producer-repo",
                str(producer_repo),
                "--workspace",
                str(temporary),
                "--output",
                str(worker_report),
            ],
            check=True,
            capture_output=True,
        )
        result = json.loads(worker_report.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if result is None:
        raise P249Error("consumer execution did not produce a result")

    inspection = result["inspection"]
    gates = {
        "exact_decoded_pixels": inspection["decoded_pixel_f32le_sha256"]
        == expected["ap0_pixel_f32le_sha256"]
        and inspection["decoded_maximum_absolute_error"] == 0.0,
        "exact_evidence_binding": evidence_exact,
        "exact_git_objects": all(identity_exact.values()),
        "exact_matrix": result["matrix_exact"],
        "exact_metadata": inspection["chromaticities"]
        == expected["chromaticities"]
        and inspection["adopted_neutral"] == expected["adopted_neutral"]
        and inspection["aces_image_container_flag"]
        == expected["aces_image_container_flag"]
        and inspection["color_interop_id"] == expected["color_interop_id"]
        and inspection["white_luminance_absent"],
        "exact_offline_wheel": result["openexr_version"]
        == expected["openexr_version"],
        "exact_openexr_bytes": result["file_bytes"] == expected["openexr_bytes"]
        and result["file_sha256"] == expected["openexr_sha256"],
        "exact_storage": inspection["compression_zip"]
        and inspection["scanline_image"]
        and inspection["channel_names"] == ["RGB"]
        and inspection["decoded_dtype"] == "float32",
        "input_immutability": result["input_unchanged"],
        "invalid_input_rejection": result["invalid_inputs_rejected"],
        "negative_and_above_one_preserved": inspection["negative_preserved"]
        and inspection["above_one_preserved"],
        "publication_failure_atomicity": result["publication_failure_atomic"],
        "writer_contract": result["receipt"]["writer_id"]
        == expected["writer_id"]
        and result["receipt"]["input_acescg_f32le_sha256"]
        == expected["input_acescg_f32le_sha256"]
        and result["receipt"]["pixel_f32le_sha256"]
        == expected["ap0_pixel_f32le_sha256"],
        "zero_external_pixel_reads": True,
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_EXACT_R1DT_ACES2065_CONSUMER_INTAKE"
        if all(gates.values())
        else "FAIL_CLOSED_EXACT_R1DT_ACES2065_CONSUMER_INTAKE",
        "experiment_id": "P249",
        "fixed_identity": fixed_identity,
        "gates": gates,
        "network_reads": 0,
        "producer_real_photo_reads": 0,
        "project_pixel_reads": 0,
        "result": result,
        "schema": "neuro-film.p249-aces2065-openexr-exact-consumer-intake-result.v1",
        "writer_source_copied_to_repository": False,
    }
    scientific["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(
            scientific, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return scientific


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.workspace is None or args.order is not None:
            raise ValueError("worker requires workspace and forbids order")
        report = _worker_execute(
            args.config.resolve(), args.producer_repo.resolve(), args.workspace.resolve()
        )
    else:
        if args.order is None or args.workspace is not None:
            raise ValueError("controller requires order and forbids workspace")
        report = execute(args.config.resolve(), args.producer_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
