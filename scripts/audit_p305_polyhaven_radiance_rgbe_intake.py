"""Audit one exact CC0 Poly Haven Radiance RGBE source and decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.radiance_rgbe import (  # noqa: I001
    RadianceRgbeError,
    _decode_radiance_rgbe_codes_bytes,
    decode_radiance_rgbe_bytes,
)

class P305Error(RuntimeError):
    """Raised when a frozen P305 identity or execution gate differs."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(value).tobytes(order="C"))


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _verify_file(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and _file_hash(path) == str(expected["sha256"])
    )


def _verify_bindings(config: dict[str, Any]) -> dict[str, bool]:
    return {
        name: _verify_file(ROOT / item["path"], item)
        for name, item in sorted(config["bindings"].items())
    }


def _source_facts(config: dict[str, Any]) -> dict[str, object]:
    snapshots = config["source_snapshots"]
    if not all(
        _verify_file(ROOT / item["path"], item) for item in snapshots.values()
    ):
        raise P305Error("frozen official-source snapshot identity differs")
    info = json.loads((ROOT / snapshots["asset_info"]["path"]).read_bytes())
    files = json.loads((ROOT / snapshots["asset_files"]["path"]).read_bytes())
    readme = (ROOT / snapshots["public_api_readme"]["path"]).read_text(
        encoding="utf-8"
    )
    selected = files["hdri"]["1k"]["hdr"]
    source = config["source"]
    return {
        "asset_files_hash": info["files_hash"],
        "asset_name": info["name"],
        "cc0_and_commercial_use_declared": all(
            phrase in readme
            for phrase in (
                "Free to use for any purpose, personal or commercial, forever.",
                "The assets themselves are CC0",
            )
        ),
        "selected_md5": selected["md5"],
        "selected_size": selected["size"],
        "selected_url": selected["url"],
        "source_lock_exact": (
            info["files_hash"] == source["files_hash"]
            and selected["url"] == source["url"]
            and selected["size"] == source["bytes"]
            and selected["md5"] == source["md5"]
        ),
    }


def _invalid_controls(payload: bytes, *, reverse: bool) -> dict[str, bool]:
    mutations = {
        "malformed_magic": b"X" + payload[1:],
        "orientation_variant": payload.replace(
            b"-Y 512 +X 1024", b"+Y 512 +X 1024", 1
        ),
        "scanline_width_mismatch": payload.replace(
            b"-Y 512 +X 1024", b"-Y 512 +X 1025", 1
        ),
        "trailing_data": payload + b"x",
        "truncated_payload": payload[:-1],
    }
    order = tuple(mutations)
    if reverse:
        order = tuple(reversed(order))
    results: dict[str, bool] = {}
    for name in order:
        try:
            decode_radiance_rgbe_bytes(mutations[name])
        except RadianceRgbeError:
            results[name] = True
        else:
            results[name] = False
    return dict(sorted(results.items()))


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = _verify_bindings(config)
    if not all(bindings.values()):
        raise P305Error("frozen local binding differs")
    source_facts = _source_facts(config)
    source_config = config["source"]
    source_path = ROOT / source_config["path"]
    source_before = _file_hash(source_path)
    if (
        source_path.stat().st_size != int(source_config["bytes"])
        or source_before != source_config["sha256"]
        or _file_hash(source_path, "md5") != source_config["md5"]
    ):
        raise P305Error("frozen Radiance source identity differs")

    payload = source_path.read_bytes()
    candidate = decode_radiance_rgbe_bytes(payload)
    codes = _decode_radiance_rgbe_codes_bytes(payload)
    decoded_bgr = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if decoded_bgr is None:
        raise P305Error("OpenCV did not decode the frozen Radiance source")
    opencv_rgb = np.ascontiguousarray(decoded_bgr[..., ::-1], dtype=np.float32)
    if opencv_rgb.shape != candidate.shape or opencv_rgb.dtype != np.float32:
        raise P305Error("OpenCV Radiance decode shape or dtype differs")

    exponent = codes[..., 3]
    correction = np.zeros(exponent.shape, dtype=np.float32)
    nonzero = exponent != 0
    correction[nonzero] = np.ldexp(
        np.full(np.count_nonzero(nonzero), 0.5, dtype=np.float32),
        exponent[nonzero].astype(np.int32) - 136,
    )
    official_from_opencv = np.ascontiguousarray(
        opencv_rgb + correction[..., None], dtype=np.float32
    )
    official_delta = candidate - official_from_opencv
    raw_delta = candidate - opencv_rgb
    official_abs = np.abs(official_delta)
    raw_abs = np.abs(raw_delta)
    official_max = float(np.max(official_abs))
    official_rmse = float(
        np.sqrt(np.mean(official_delta.astype(np.float64) ** 2))
    )
    raw_max = float(np.max(raw_abs))
    raw_rmse = float(np.sqrt(np.mean(raw_delta.astype(np.float64) ** 2)))
    invalid_controls = _invalid_controls(payload, reverse=reverse)

    expected = config["expected"]
    gates = {
        "cc0_and_commercial_use_declared": bool(
            source_facts["cc0_and_commercial_use_declared"]
        ),
        "decoded_shape_dtype": list(candidate.shape) == expected["shape"]
        and str(candidate.dtype) == expected["dtype"],
        "finite_nonnegative": bool(
            np.all(np.isfinite(candidate)) and np.all(candidate >= 0)
        ),
        "invalid_controls_reject": all(invalid_controls.values()),
        "official_semantics_residual": official_max
        <= float(config["gates"]["maximum_absolute_official_semantics_residual"]),
        "opencv_code_oracle_difference_detected": int(np.count_nonzero(raw_delta))
        == int(expected["raw_opencv_difference_components"]),
        "owned_contiguous_writable": bool(
            candidate.flags.owndata
            and candidate.flags.c_contiguous
            and candidate.flags.writeable
        ),
        "source_immutable": _file_hash(source_path) == source_before,
        "source_lock_exact": bool(source_facts["source_lock_exact"]),
    }
    decision = (
        "PASS_PRIVATE_POLYHAVEN_RADIANCE_RGBE_INTAKE"
        if all(gates.values())
        else "FAIL_CLOSED_POLYHAVEN_RADIANCE_RGBE_INTAKE"
    )
    report: dict[str, object] = {
        "bindings": bindings,
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "decode": {
            "candidate_f32le_sha256": _array_hash(candidate.astype("<f4", copy=False)),
            "codes_rgbe_sha256": _array_hash(codes),
            "dtype": str(candidate.dtype),
            "maximum": float(np.max(candidate)),
            "minimum": float(np.min(candidate)),
            "nonzero_exponent_pixels": int(np.count_nonzero(nonzero)),
            "shape": list(candidate.shape),
        },
        "decision": decision,
        "experiment_id": "P305",
        "gates": gates,
        "invalid_controls": invalid_controls,
        "opencv_oracle": {
            "corrected_f32le_sha256": _array_hash(
                official_from_opencv.astype("<f4", copy=False)
            ),
            "official_semantics_difference_components": int(
                np.count_nonzero(official_delta)
            ),
            "official_semantics_max_abs": official_max,
            "official_semantics_rmse": official_rmse,
            "raw_difference_components": int(np.count_nonzero(raw_delta)),
            "raw_max_abs": raw_max,
            "raw_rmse": raw_rmse,
            "version": cv2.__version__,
        },
        "schema": "neuro-film.p305-polyhaven-radiance-rgbe-intake-result.v1",
        "source": {
            "bytes": source_path.stat().st_size,
            "md5": _file_hash(source_path, "md5"),
            "path": source_config["path"],
            "sha256": source_before,
        },
        "source_facts": source_facts,
        "network_requests": 0,
        "source_file_reads": 1,
    }
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
