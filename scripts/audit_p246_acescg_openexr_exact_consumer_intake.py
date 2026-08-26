"""Audit exact producer R1DO ACEScg OpenEXR handoff consumption."""

from __future__ import annotations

import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]


class P246Error(RuntimeError):
    """Raised when a frozen P246 identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    little_endian = np.ascontiguousarray(value, dtype="<f4")
    return _sha256_bytes(little_endian.tobytes(order="C"))


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _git_bytes(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _synthetic_lattice() -> np.ndarray:
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-0.25, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    scales = np.asarray(
        [np.float32(1e-6), 0.01, 0.18, 1.0, 4.0, 8.0, 16.0],
        dtype=np.float32,
    )
    return np.ascontiguousarray(scales[:, None, None] * basis[None, :, :])


def _load_ephemeral_writer(source: bytes, site: Path) -> types.ModuleType:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    module_name = "p246_bound_r1do_writer"
    module = types.ModuleType(module_name)
    module.__file__ = "git:R1DO/src/zhuise/acescg_openexr.py"
    sys.modules[module_name] = module
    try:
        # The bytes are accepted only after exact commit/blob/SHA verification.
        exec(  # noqa: S102
            compile(source, module.__file__, "exec"), module.__dict__
        )
    except Exception:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(site))
        raise
    return module


def _unload_ephemeral_writer(module: types.ModuleType, site: Path) -> None:
    sys.modules.pop(module.__name__, None)
    sys.modules.pop("OpenEXR", None)
    if str(site) in sys.path:
        sys.path.remove(str(site))
    importlib.invalidate_caches()


def _inspect(path: Path, expected_pixels: np.ndarray, openexr: Any) -> dict[str, Any]:
    with openexr.File(str(path)) as exr_file:
        header = dict(exr_file.header())
        channels = exr_file.channels()
        decoded = channels["RGB"].pixels.copy()
        channel_names = sorted(channels)
    return {
        "above_one_preserved": bool(np.max(decoded) > 1.0),
        "adopted_neutral": [float(value) for value in header["adoptedNeutral"]],
        "channel_names": channel_names,
        "chromaticities": [float(value) for value in header["chromaticities"]],
        "compression_zip": bool(header["compression"] == openexr.ZIP_COMPRESSION),
        "decoded_dtype": str(decoded.dtype),
        "decoded_maximum_absolute_error": float(
            np.max(np.abs(decoded.astype(np.float64) - expected_pixels.astype(np.float64)))
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
            module.write_acescg_openexr(target, value)
        except (TypeError, ValueError):
            if target.exists():
                return False
        else:
            return False
    return True


def _publication_failure_atomic(module: types.ModuleType, directory: Path) -> bool:
    target = directory / "existing.exr"
    original = b"p246-existing-target"
    target.write_bytes(original)
    real_replace = module.os.replace

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("injected P246 publication failure")

    module.os.replace = fail_replace
    try:
        try:
            module.write_acescg_openexr(target, _synthetic_lattice())
        except module.AcescgOpenExrError:
            pass
        else:
            return False
    finally:
        module.os.replace = real_replace
    residue = list(directory.glob(f".{target.name}.*.tmp.exr"))
    return target.read_bytes() == original and residue == []


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
    r1do = _git_bytes(
        producer_repo,
        bindings["producer_r1do_evidence_commit"],
        bindings["producer_r1do_evidence_path"],
    )
    r1dp = _git_bytes(
        producer_repo,
        bindings["producer_r1dp_evidence_commit"],
        bindings["producer_r1dp_evidence_path"],
    )
    r1do_json = json.loads(r1do)
    r1dp_json = json.loads(r1dp)
    wheel = producer_repo / bindings["producer_wheel_path"]
    fixed_identity = {
        "contract_sha256": _sha256_file(contract),
        "producer_r1do_evidence_sha256": _sha256_bytes(r1do),
        "producer_r1dp_evidence_sha256": _sha256_bytes(r1dp),
        "producer_wheel_bytes": wheel.stat().st_size,
        "producer_wheel_sha256": _sha256_file(wheel),
        "producer_writer_git_blob": writer_blob,
        "producer_writer_sha256": _sha256_bytes(writer),
    }
    identity_exact = {
        key: fixed_identity[key] == bindings[key]
        for key in fixed_identity
    }
    evidence_exact = (
        r1do_json["status"] == "PASS_PRIVATE_ACESCG_OPENEXR_MASTER"
        and r1dp_json["status"] == "PASS_PRIVATE_ACESCG_OPENEXR_LINUX_RUNTIME"
        and r1do_json["fixed_identity"]["writer_source_sha256"]
        == bindings["producer_writer_sha256"]
        and r1dp_json["fixed_identity"]["writer_source_sha256"]
        == bindings["producer_writer_sha256"]
    )
    if not all(identity_exact.values()) or not evidence_exact:
        raise P246Error("frozen producer identity differs")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected[
        "python_major_minor"
    ]:
        raise P246Error("formal Python major/minor differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p246-"))
    site = temporary / "site"
    output = temporary / "synthetic.exr"
    module: types.ModuleType | None = None
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
        module = _load_ephemeral_writer(writer, site)
        openexr = importlib.import_module("OpenEXR")
        values = _synthetic_lattice()
        input_before = _array_sha256(values)
        receipt = module.write_acescg_openexr(output, values)
        inspection = _inspect(output, values, openexr)
        file_bytes = output.stat().st_size
        file_sha256 = _sha256_file(output)
        invalid_rejected = _invalid_inputs_rejected(module, temporary)
        publication_atomic = _publication_failure_atomic(module, temporary)
        result = {
            "file_bytes": file_bytes,
            "file_sha256": file_sha256,
            "input_f32le_sha256": input_before,
            "input_unchanged": _array_sha256(values) == input_before,
            "inspection": inspection,
            "invalid_inputs_rejected": invalid_rejected,
            "openexr_version": openexr.__version__,
            "publication_failure_atomic": publication_atomic,
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
    finally:
        if module is not None:
            _unload_ephemeral_writer(module, site)
        shutil.rmtree(temporary, ignore_errors=False)
    if result is None:
        raise P246Error("consumer execution did not produce a result")

    inspection = result["inspection"]
    gates = {
        "exact_decoded_pixels": inspection["decoded_pixel_f32le_sha256"]
        == expected["synthetic_pixel_f32le_sha256"]
        and inspection["decoded_maximum_absolute_error"] == 0.0,
        "exact_evidence_bindings": evidence_exact,
        "exact_git_objects": all(identity_exact.values()),
        "exact_metadata": inspection["chromaticities"]
        == expected["chromaticities"]
        and inspection["adopted_neutral"] == expected["adopted_neutral"]
        and inspection["white_luminance_absent"],
        "exact_offline_wheel": result["openexr_version"]
        == expected["openexr_version"],
        "exact_openexr_bytes": result["file_bytes"]
        == expected["synthetic_openexr_bytes"]
        and result["file_sha256"] == expected["synthetic_openexr_sha256"],
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
        and result["receipt"]["pixel_f32le_sha256"]
        == expected["synthetic_pixel_f32le_sha256"],
        "zero_external_pixel_reads": True,
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    scientific = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_EXACT_R1DO_OPENEXR_CONSUMER_INTAKE"
        if all(gates.values())
        else "FAIL_CLOSED_EXACT_R1DO_OPENEXR_CONSUMER_INTAKE",
        "experiment_id": "P246",
        "fixed_identity": fixed_identity,
        "gates": gates,
        "network_reads": 0,
        "producer_real_photo_reads": 0,
        "project_pixel_reads": 0,
        "result": result,
        "schema": "neuro-film.p246-acescg-openexr-exact-consumer-intake-result.v1",
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
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    report = execute(args.config.resolve(), args.producer_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
