"""Audit one real DNG through direct and AP0-interchange ACES 2 PQ routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p259_dng_aces2065_openexr_master import (
    _git_bytes,
    _load_ephemeral_writer,
)
from src.preprocess.aces2_canonical_pq_png import (
    publish_working_image_aces2_canonical_hdr_pq_png_v1,
)
from src.preprocess.aces2065_aces2_pq import (
    publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
)
from src.preprocess.dng_forward_raster import (
    load_dng_forward_working_image,
)
from src.preprocess.ocio_aces2_output import (
    convert_working_image_to_acescg,
)
from src.preprocess.png_stream import (
    sha256_rec2100_pq_rgb16_png_samples,
)


class P264Error(RuntimeError):
    """Raised when a frozen P264 execution condition is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _bound(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256_file(path),
    }


def _verify_tuple(binding: list[object]) -> bool:
    path = ROOT / str(binding[0])
    return (
        path.is_file()
        and path.stat().st_size == int(binding[1])
        and _sha256_file(path) == str(binding[2])
    )


def _compare_samples(direct: np.ndarray, interchange: np.ndarray) -> dict[str, object]:
    delta = np.abs(direct.astype(np.int32) - interchange.astype(np.int32))
    return {
        "different_component_count": int(np.count_nonzero(delta)),
        "maximum_code_difference": int(np.max(delta)),
        "rgb16_bytes_exact": bool(np.array_equal(direct, interchange)),
    }


def _worker(config_path: Path, workspace: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_info = config["input"]
    source = ROOT / source_info["logical_path"]
    if source.stat().st_size != source_info["source_bytes"] or (
        _sha256_file(source) != source_info["source_sha256"]
    ):
        raise P264Error("frozen DNG identity differs")
    source_before = _sha256_file(source)

    writer_info = config["producer_writer"]
    writer_bytes = _git_bytes(
        Path(writer_info["repository"]),
        writer_info["commit"],
        writer_info["path"],
    )
    if len(writer_bytes) != writer_info["bytes"] or (
        _sha256_bytes(writer_bytes) != writer_info["sha256"]
    ):
        raise P264Error("producer writer identity differs")
    writer = _load_ephemeral_writer(writer_bytes, workspace / "writer-site")

    working = load_dng_forward_working_image(source)
    working_before = _array_sha256(working.pixels)
    if list(working.pixels.shape) != source_info["shape"] or (
        working_before != source_info["p98_working_float32_sha256"]
    ):
        raise P264Error("P98 WorkingImage identity differs")
    acescg = convert_working_image_to_acescg(working)
    acescg_before = _array_sha256(acescg)
    if acescg_before != source_info["p259_acescg_float32_sha256"]:
        raise P264Error("P259 ACEScg identity differs")

    master = workspace / "master.exr"
    direct_png = workspace / "direct.png"
    interchange_png = workspace / "interchange.png"
    writer.write_aces2065_1_openexr(master, acescg)
    master_sha = _sha256_file(master)

    direct_partition_reverse = order == "reverse"
    interchange_partition_reverse = not direct_partition_reverse
    if order == "forward":
        direct_receipt, direct_samples, direct_encoded = (
            publish_working_image_aces2_canonical_hdr_pq_png_v1(
                working,
                direct_png,
                row_count=64,
                reverse_partition=direct_partition_reverse,
            )
        )
        interchange_receipt, interchange_samples, interchange_encoded = (
            publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
                master,
                interchange_png,
                row_count=37,
                reverse_partition=interchange_partition_reverse,
            )
        )
    else:
        interchange_receipt, interchange_samples, interchange_encoded = (
            publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
                master,
                interchange_png,
                row_count=37,
                reverse_partition=interchange_partition_reverse,
            )
        )
        direct_receipt, direct_samples, direct_encoded = (
            publish_working_image_aces2_canonical_hdr_pq_png_v1(
                working,
                direct_png,
                row_count=64,
                reverse_partition=direct_partition_reverse,
            )
        )

    comparison = _compare_samples(direct_samples, interchange_samples)
    encoded_delta = np.abs(
        direct_encoded.astype(np.float64) - interchange_encoded.astype(np.float64)
    )
    direct_png_sha = _sha256_file(direct_png)
    interchange_png_sha = _sha256_file(interchange_png)
    direct_sample_sha = _array_sha256(direct_samples)
    interchange_sample_sha = _array_sha256(interchange_samples)

    foreign = workspace / "foreign.png"
    foreign.write_bytes(b"foreign-destination")
    try:
        publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(master, foreign)
    except ValueError:
        foreign_rejected = True
    else:
        foreign_rejected = False

    return {
        "acescg": {
            "f32le_sha256": acescg_before,
            "input_unchanged": _array_sha256(acescg) == acescg_before,
            "maximum": float(np.max(acescg)),
            "minimum": float(np.min(acescg)),
            "strong_highlight_count": int(np.count_nonzero(acescg > 1.0)),
            "strong_negative_count": int(np.count_nonzero(acescg < 0.0)),
        },
        "comparison": {
            **comparison,
            "encoded_maximum_absolute_difference": float(np.max(encoded_delta)),
            "encoded_rmse": float(np.sqrt(np.mean(encoded_delta * encoded_delta))),
            "png_bytes_exact": direct_png_sha == interchange_png_sha,
        },
        "direct": {
            "encoded_finite_unit": bool(
                np.isfinite(direct_encoded).all()
                and np.all((direct_encoded >= 0.0) & (direct_encoded <= 1.0))
            ),
            "output_bytes": direct_png.stat().st_size,
            "png_sha256": direct_png_sha,
            "receipt_exact": direct_receipt == direct_png_sha,
            "rgb16_sha256": direct_sample_sha,
            "strict_readback_sha256": sha256_rec2100_pq_rgb16_png_samples(
                direct_png,
                width=working.pixels.shape[1],
                height=working.pixels.shape[0],
            ),
        },
        "foreign_destination_preserved": foreign.read_bytes() == b"foreign-destination",
        "foreign_destination_rejected": foreign_rejected,
        "interchange": {
            "encoded_finite_unit": bool(
                np.isfinite(interchange_encoded).all()
                and np.all((interchange_encoded >= 0.0) & (interchange_encoded <= 1.0))
            ),
            "output_bytes": interchange_png.stat().st_size,
            "png_sha256": interchange_png_sha,
            "receipt_exact": interchange_receipt == interchange_png_sha,
            "rgb16_sha256": interchange_sample_sha,
            "strict_readback_sha256": sha256_rec2100_pq_rgb16_png_samples(
                interchange_png,
                width=working.pixels.shape[1],
                height=working.pixels.shape[0],
            ),
        },
        "master": {
            "bytes": master.stat().st_size,
            "sha256": master_sha,
        },
        "source_unchanged": _sha256_file(source) == source_before,
        "working_input_unchanged": _array_sha256(working.pixels) == working_before,
    }


def _gates(config: dict[str, Any], result: dict[str, Any]) -> dict[str, bool]:
    source_info = config["input"]
    return {
        "acescg_identity": result["acescg"]["f32le_sha256"]
        == source_info["p259_acescg_float32_sha256"],
        "create_only_atomicity": bool(
            result["foreign_destination_rejected"]
            and result["foreign_destination_preserved"]
        ),
        "direct_interchange_png_exact": bool(result["comparison"]["png_bytes_exact"]),
        "direct_interchange_rgb16_exact": bool(
            result["comparison"]["rgb16_bytes_exact"]
            and result["comparison"]["maximum_code_difference"] == 0
        ),
        "finite_unit_outputs": bool(
            result["direct"]["encoded_finite_unit"]
            and result["interchange"]["encoded_finite_unit"]
        ),
        "negative_highlight_support": bool(
            result["acescg"]["strong_negative_count"]
            == source_info["p259_strong_negative_count"]
            and result["acescg"]["strong_highlight_count"]
            == source_info["p259_strong_highlight_count"]
        ),
        "receipts_and_readback": bool(
            result["direct"]["receipt_exact"]
            and result["interchange"]["receipt_exact"]
            and result["direct"]["strict_readback_sha256"]
            == result["direct"]["rgb16_sha256"]
            and result["interchange"]["strict_readback_sha256"]
            == result["interchange"]["rgb16_sha256"]
        ),
        "source_and_inputs_immutable": bool(
            result["source_unchanged"]
            and result["working_input_unchanged"]
            and result["acescg"]["input_unchanged"]
        ),
    }


def execute(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_bindings = {
        name: _verify_tuple(binding) for name, binding in config["parents"].items()
    }
    if not all(parent_bindings.values()):
        raise P264Error("one or more parent bindings differ")
    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="p264-", dir=scratch_root))
    try:
        result = _worker(config_path, workspace, order)
    finally:
        shutil.rmtree(workspace, ignore_errors=False)
    gates = _gates(config, result)
    gates["parents_exact"] = all(parent_bindings.values())
    gates["temporary_residue_zero"] = not workspace.exists()
    decision = (
        "PASS_PRIVATE_REAL_DNG_AP0_PQ_EXACT_INTERCHANGE"
        if all(gates.values())
        else "FAIL_CLOSED_REAL_DNG_AP0_PQ_EXACT_INTERCHANGE"
    )
    return {
        "bindings": {
            "config": _bound(config_path),
            "contract": _bound(
                ROOT / "docs/planning/P264_REAL_DNG_AP0_PQ_INTERCHANGE_CONTRACT.md"
            ),
            "runner": _bound(Path(__file__).resolve()),
            "test": _bound(ROOT / "tests/test_p264_real_dng_ap0_pq_interchange.py"),
        },
        "claim_ceiling": config["claim_ceiling"],
        "decision": decision,
        "execution_commit": _git_head(),
        "experiment_id": "P264",
        "gates": gates,
        "network_reads": 0,
        "parent_bindings": parent_bindings,
        "result": result,
        "schema": "neuro-film.p264-real-dng-ap0-pq-interchange-result.v1",
    }


def _fresh(config: Path, order: str) -> bytes:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config),
        "--order",
        order,
        "--worker",
    ]
    return subprocess.run(command, check=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = args.config.resolve()
    if args.worker:
        sys.stdout.buffer.write(_canonical_bytes(execute(config, args.order)))
        return
    if args.output is None:
        raise SystemExit("--output is required outside worker mode")
    payload = _fresh(config, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    sys.stdout.write(_sha256_bytes(payload) + "\n")


if __name__ == "__main__":
    main()
