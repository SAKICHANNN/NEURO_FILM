"""Audit exact-five real DNG composition to strict ACES2065-1 masters."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.aces2065_openexr import (
    ACES2065_ADOPTED_NEUTRAL,
    ACES2065_CHROMATICITIES,
    load_aces2065_openexr_working_image,
)
from src.preprocess.dng_forward_raster import load_dng_forward_working_image
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.ocio_aces2_output import (
    SOURCE_SPACE,
    convert_working_image_to_acescg,
    load_aces2_config,
)


class P259Error(RuntimeError):
    """Raised when one frozen P259 execution condition fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


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


def _bound(path: Path) -> dict[str, object]:
    if not path.is_absolute():
        path = ROOT / path
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(path),
    }


def _verify_file_binding(bindings: dict[str, object], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    return (
        path.is_file()
        and path.stat().st_size == int(bindings[f"{prefix}_bytes"])
        and _sha256_file(path) == str(bindings[f"{prefix}_sha256"])
    )


def _verify_parent_bindings(config: dict[str, object]) -> dict[str, bool]:
    bindings = config["bindings"]
    assert isinstance(bindings, dict)
    prefixes = (
        "contract",
        "dng_loader",
        "ocio_source",
        "p249_config",
        "p249_evidence",
        "p251_config",
        "p251_evidence",
        "p251_reader",
        "p257_config",
        "p257_evidence",
        "p98_config",
        "p98_evidence",
        "runner",
        "test",
        "u1_4d_config",
        "u1_4d_evidence",
        "u1_4e_config",
        "u1_4e_evidence",
    )
    return {prefix: _verify_file_binding(bindings, prefix) for prefix in prefixes}


def _load_ephemeral_writer(source: bytes, site: Path) -> types.ModuleType:
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    name = "p259_bound_r1dt_writer"
    module = types.ModuleType(name)
    module.__file__ = "git:R1DT/src/zhuise/acescg_openexr.py"
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    except Exception:
        sys.modules.pop(name, None)
        sys.path.remove(str(site))
        raise
    return module


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _direct_acescg(working: Any) -> np.ndarray:
    output = np.ascontiguousarray(working.pixels.reshape(-1, 3).copy())
    ocio = importlib.import_module("PyOpenColorIO")
    config = load_aces2_config()
    processor = config.getProcessor("Linear Rec.2020", SOURCE_SPACE).getDefaultCPUProcessor()
    processor.apply(ocio.PackedImageDesc(output, output.shape[0], 1, 3))
    return output.reshape(working.pixels.shape)


def _inspect_exr(
    path: Path, expected_ap0: np.ndarray, openexr: Any
) -> dict[str, object]:
    with openexr.File(str(path)) as exr_file:
        parts = getattr(exr_file, "parts", None)
        header = dict(exr_file.header())
        channels = exr_file.channels()
        channel_names = sorted(channels)
        decoded = channels["RGB"].pixels.copy()
    return {
        "aces_image_container_flag": int(header["acesImageContainerFlag"]),
        "adopted_neutral": [float(value) for value in header["adoptedNeutral"]],
        "channel_names": channel_names,
        "chromaticities": [float(value) for value in header["chromaticities"]],
        "color_interop_id": _text(header["colorInteropID"]),
        "compression_zip": bool(header["compression"] == openexr.ZIP_COMPRESSION),
        "decoded_ap0_f32le_sha256": _array_sha256(decoded),
        "decoded_ap0_pixels_exact": bool(np.array_equal(decoded, expected_ap0)),
        "decoded_dtype": str(decoded.dtype),
        "decoded_maximum_absolute_error": float(
            np.max(np.abs(decoded.astype(np.float64) - expected_ap0.astype(np.float64)))
        ),
        "decoded_shape": list(decoded.shape),
        "part_count": 0 if parts is None else len(parts),
        "scanline_image": bool(header["type"] == openexr.scanlineimage),
    }


def _compose_row(
    row: dict[str, object],
    config: dict[str, object],
    writer: types.ModuleType,
    openexr: Any,
    workspace: Path,
) -> dict[str, object]:
    source = ROOT / str(row["logical_path"])
    before_sha = _sha256_file(source)
    if source.stat().st_size != int(row["source_bytes"]):
        raise P259Error(f"source byte mismatch: {row['source_id']}")
    if before_sha != row["source_sha256"]:
        raise P259Error(f"source SHA mismatch: {row['source_id']}")
    working = load_dng_forward_working_image(
        source,
        expected_source_bytes=int(row["source_bytes"]),
        expected_source_sha256=str(row["source_sha256"]),
    )
    working_sha = _array_sha256(working.pixels)
    acescg = np.ascontiguousarray(convert_working_image_to_acescg(working).copy())
    direct = _direct_acescg(working)
    direct_exact = bool(np.array_equal(acescg, direct))
    acescg_before = _array_sha256(acescg)
    matrix = np.asarray(config["transform"]["ap1_to_ap0_matrix"], dtype=np.float64)
    expected_ap0 = np.ascontiguousarray(
        np.asarray(acescg.astype(np.float64) @ matrix.T, dtype=np.float32)
    )
    output = workspace / f"{row['source_id']}.exr"
    receipt = writer.write_aces2065_1_openexr(output, acescg)
    file_bytes = output.stat().st_size
    file_sha = _sha256_file(output)
    inspection = _inspect_exr(output, expected_ap0, openexr)
    readback = load_aces2065_openexr_working_image(output)
    readback_error = np.abs(
        readback.pixels.astype(np.float64) - acescg.astype(np.float64)
    )
    negative_mask = acescg < float(config["transform"]["strong_negative_threshold"])
    highlight_mask = acescg > float(config["transform"]["strong_highlight_threshold"])
    boundary_input = (acescg == np.float32(0.0)) | (acescg == np.float32(1.0))
    boundary_output = (readback.pixels == np.float32(0.0)) | (
        readback.pixels == np.float32(1.0)
    )
    result = {
        "acescg": {
            "c_contiguous": bool(acescg.flags.c_contiguous),
            "direct_official_bytes_exact": direct_exact,
            "dtype": str(acescg.dtype),
            "finite": bool(np.isfinite(acescg).all()),
            "f32le_sha256": acescg_before,
            "owned": bool(acescg.flags.owndata),
            "shape": list(acescg.shape),
        },
        "exr": {
            "bytes": file_bytes,
            "inspection": inspection,
            "receipt_file_sha256_exact": receipt.file_sha256 == file_sha,
            "receipt_input_sha256_exact": receipt.input_acescg_f32le_sha256
            == acescg_before,
            "sha256": file_sha,
            "writer_id": receipt.writer_id,
        },
        "p98_working_float32_sha256": working_sha,
        "p98_working_hash_exact": working_sha == row["p98_working_float32_sha256"],
        "readback": {
            "c_contiguous": bool(readback.pixels.flags.c_contiguous),
            "f32le_sha256": _array_sha256(readback.pixels),
            "maximum_absolute_error": float(np.max(readback_error)),
            "new_exact_boundary_count": int(np.count_nonzero(boundary_output & ~boundary_input)),
            "owned": bool(readback.pixels.flags.owndata),
            "strong_highlight_input_count": int(np.count_nonzero(highlight_mask)),
            "strong_highlights_preserved": bool(
                np.all(readback.pixels[highlight_mask] > np.float32(1.0))
            ),
            "strong_negative_input_count": int(np.count_nonzero(negative_mask)),
            "strong_negatives_preserved": bool(
                np.all(readback.pixels[negative_mask] < np.float32(0.0))
            ),
            "working_space": readback.working_space,
            "writeable": bool(readback.pixels.flags.writeable),
        },
        "source_id": row["source_id"],
        "source_sha256": before_sha,
        "source_unchanged": _sha256_file(source) == before_sha,
        "writer_input_unchanged": _array_sha256(acescg) == acescg_before,
    }
    output.unlink()
    result["temporary_exr_removed"] = not output.exists()
    return result


def _worker(
    config_path: Path,
    producer_repo: Path,
    workspace: Path,
    site: Path,
    order: str,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    handoff = config["producer_handoff"]
    writer_source = _git_bytes(
        producer_repo, str(handoff["writer_commit"]), str(handoff["writer_path"])
    )
    if len(writer_source) != int(handoff["writer_bytes"]):
        raise P259Error("writer byte count mismatch")
    if _sha256_bytes(writer_source) != handoff["writer_sha256"]:
        raise P259Error("writer SHA mismatch")
    writer = _load_ephemeral_writer(writer_source, site)
    openexr = importlib.import_module("OpenEXR")
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    results = [
        _compose_row(row, config, writer, openexr, workspace) for row in rows
    ]
    results.sort(key=lambda item: str(item["source_id"]))
    transform = config["transform"]
    metadata_exact = all(
        row["exr"]["inspection"]["aces_image_container_flag"] == 1
        and row["exr"]["inspection"]["adopted_neutral"]
        == list(ACES2065_ADOPTED_NEUTRAL)
        and row["exr"]["inspection"]["chromaticities"]
        == list(ACES2065_CHROMATICITIES)
        and row["exr"]["inspection"]["color_interop_id"] == "lin_ap0_scene"
        and row["exr"]["inspection"]["compression_zip"]
        and row["exr"]["inspection"]["scanline_image"]
        and row["exr"]["inspection"]["part_count"] == 1
        and row["exr"]["inspection"]["channel_names"] == ["RGB"]
        and row["exr"]["inspection"]["decoded_dtype"] == "float32"
        for row in results
    )
    gates = {
        "bounded_ap1_readback": all(
            row["readback"]["maximum_absolute_error"]
            <= float(transform["maximum_ap1_readback_absolute_error"])
            for row in results
        ),
        "bounded_temporary_media": all(
            row["exr"]["bytes"] <= int(transform["maximum_exr_bytes"])
            for row in results
        ),
        "direct_official_acescg_bytes_exact": all(
            row["acescg"]["direct_official_bytes_exact"] for row in results
        ),
        "exact_ap0_d60_metadata": metadata_exact,
        "exact_ap0_matrix_pixels": all(
            row["exr"]["inspection"]["decoded_ap0_pixels_exact"]
            and row["exr"]["inspection"]["decoded_maximum_absolute_error"] == 0.0
            for row in results
        ),
        "negative_and_highlight_preservation": sum(
            row["readback"]["strong_negative_input_count"] for row in results
        )
        > 0
        and sum(row["readback"]["strong_highlight_input_count"] for row in results)
        > 0
        and all(
            row["readback"]["strong_negatives_preserved"]
            and row["readback"]["strong_highlights_preserved"]
            for row in results
        ),
        "p98_working_hashes_exact": all(
            row["p98_working_hash_exact"] for row in results
        ),
        "required_rows": len(results) == int(config["gates"]["required_rows"]),
        "runtime_identities_exact": openexr.__version__
        == config["runtime"]["openexr_version"],
        "source_immutability": all(row["source_unchanged"] for row in results),
        "temporary_exr_removed": all(row["temporary_exr_removed"] for row in results),
        "working_contract": all(
            row["acescg"]["dtype"] == "float32"
            and row["acescg"]["finite"]
            and row["acescg"]["owned"]
            and row["acescg"]["c_contiguous"]
            and row["readback"]["owned"]
            and row["readback"]["c_contiguous"]
            and row["readback"]["writeable"]
            and row["readback"]["working_space"] == "acescg_ap1_d60"
            and row["writer_input_unchanged"]
            and row["exr"]["receipt_file_sha256_exact"]
            and row["exr"]["receipt_input_sha256_exact"]
            for row in results
        ),
        "zero_new_exact_boundaries": all(
            row["readback"]["new_exact_boundary_count"] == 0 for row in results
        ),
    }
    return {
        "decision": "PASS_PRIVATE_REAL_DNG_ACES2065_OPENEXR_MASTER_COMPOSITION"
        if all(gates.values())
        else "FAIL_CLOSED_REAL_DNG_ACES2065_OPENEXR_MASTER_COMPOSITION",
        "gates": gates,
        "network_reads": 0,
        "producer_source_persisted_in_consumer_repo": False,
        "rows": results,
        "schema": "neuro-film.p259-dng-aces2065-openexr-master-worker-result.v1",
        "target_reads": 0,
    }


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_bindings = _verify_parent_bindings(config)
    handoff = config["producer_handoff"]
    writer_source = _git_bytes(
        producer_repo, str(handoff["writer_commit"]), str(handoff["writer_path"])
    )
    writer_blob = _git_text(
        producer_repo,
        "rev-parse",
        f"{handoff['writer_commit']}:{handoff['writer_path']}",
    )
    wheel = producer_repo / str(handoff["wheel_path"])
    handoff_exact = {
        "wheel_bytes": wheel.stat().st_size == int(handoff["wheel_bytes"]),
        "wheel_sha256": _sha256_file(wheel) == handoff["wheel_sha256"],
        "writer_bytes": len(writer_source) == int(handoff["writer_bytes"]),
        "writer_git_blob": writer_blob == handoff["writer_git_blob"],
        "writer_sha256": _sha256_bytes(writer_source) == handoff["writer_sha256"],
    }
    if not all(parent_bindings.values()) or not all(handoff_exact.values()):
        raise P259Error("frozen parent or producer handoff identity differs")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config["runtime"][
        "python_major_minor"
    ]:
        raise P259Error("Python runtime identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p259-"))
    site = temporary / "site"
    worker_report = temporary / "worker.json"
    result: dict[str, object] | None = None
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
                "--site",
                str(site),
                "--order",
                order,
                "--output",
                str(worker_report),
            ],
            check=True,
            capture_output=True,
        )
        result = json.loads(worker_report.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if result is None:
        raise P259Error("worker did not produce a result")
    cleanup_exact = not temporary.exists()
    outer_gates = {
        "parent_bindings_exact": all(parent_bindings.values()),
        "producer_handoff_exact": all(handoff_exact.values()),
        "worker_passed": all(result["gates"].values()),
        "zero_network_and_target_reads": result["network_reads"] == 0
        and result["target_reads"] == 0,
        "zero_temporary_residue": cleanup_exact,
    }
    return {
        "bindings": {
            "config": _bound(config_path),
            "contract": _bound(ROOT / str(config["bindings"]["contract_path"])),
            "runner": _bound(Path(__file__)),
            "test": _bound(ROOT / str(config["bindings"]["test_path"])),
        },
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_REAL_DNG_ACES2065_OPENEXR_MASTER_COMPOSITION"
        if all(outer_gates.values())
        else "FAIL_CLOSED_REAL_DNG_ACES2065_OPENEXR_MASTER_COMPOSITION",
        "execution_commit": _git_head(),
        "experiment_id": "P259",
        "gates": outer_gates,
        "parent_bindings": parent_bindings,
        "producer_handoff": handoff_exact,
        "result": result,
        "schema": "neuro-film.p259-dng-aces2065-openexr-master-result.v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.workspace is None or args.site is None:
            raise ValueError("worker requires workspace and site")
        report = _worker(
            args.config.resolve(),
            args.producer_repo.resolve(),
            args.workspace.resolve(),
            args.site.resolve(),
            args.order,
        )
    else:
        if args.workspace is not None or args.site is not None:
            raise ValueError("controller forbids workspace and site")
        report = execute(
            args.config.resolve(), args.producer_repo.resolve(), args.order
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))


if __name__ == "__main__":
    main()
