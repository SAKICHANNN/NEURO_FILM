"""Audit the executable blue-noise term in the AbsoluteDegradation release."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

SCHEMA = "neuro_film.u6_p4ck_absolute_degradation_blue_noise_release_audit_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ck_absolute_degradation_blue_noise_release_audit_report.v1"


class AbsoluteDegradationBlueNoiseReleaseError(RuntimeError):
    """Raised when the frozen release or audit contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(contract: Mapping[str, Any]) -> None:
    checks = contract.get("frozen_checks", {})
    files = contract.get("released_files", {})
    if (
        contract.get("schema") != SCHEMA
        or files.get("blue_noise_file_bytes") != 41943168
        or checks.get("expected_shape") != [2048, 2048, 5]
        or checks.get("expected_dtype") != "float16"
        or checks.get("fixed_channels") != [0, 1, 2, 3, 4]
        or checks.get("fixed_grain_sizes") != [1.0, 1.15, 1.3]
        or checks.get("scanner_noise_sigma") != 0.0
    ):
        raise AbsoluteDegradationBlueNoiseReleaseError("P4CK frozen contract drift")


def _download_exact(url: str, destination: Path, expected_bytes: int | None) -> None:
    temporary = destination.with_suffix(destination.suffix + ".download")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    total = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open("xb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if expected_bytes is not None and total > expected_bytes:
                        raise AbsoluteDegradationBlueNoiseReleaseError(
                            "released file exceeded frozen byte count"
                        )
                    handle.write(chunk)
        if expected_bytes is not None and total != expected_bytes:
            raise AbsoluteDegradationBlueNoiseReleaseError(
                f"released file byte count drift: {total}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_release(contract: Mapping[str, Any], scratch: Path) -> tuple[Path, Path]:
    files = contract["released_files"]
    code_path = scratch / "film_gain_step.py"
    asset_path = scratch / "blue_noise_2048_5ch.npy"
    _download_exact(str(files["film_grain_code_url"]), code_path, None)
    _download_exact(
        str(files["blue_noise_url"]),
        asset_path,
        int(files["blue_noise_file_bytes"]),
    )
    for path, key in (
        (code_path, "film_grain_code_sha256"),
        (asset_path, "blue_noise_sha256"),
    ):
        actual = sha256_file(path)
        if actual != files[key]:
            raise AbsoluteDegradationBlueNoiseReleaseError(
                f"released file hash drift for {path.name}: {actual}"
            )
    return code_path, asset_path


def _summarize_array(values: np.ndarray) -> dict[str, Any]:
    raw = np.ascontiguousarray(values).view(np.uint16)
    unique, counts = np.unique(raw, return_counts=True)
    return {
        "finite": bool(np.isfinite(values).all()),
        "numeric_nonzero_count": int(np.count_nonzero(values)),
        "standard_deviation": float(np.std(values.astype(np.float64))),
        "raw_float16_bit_counts": {
            f"0x{int(key):04x}": int(count)
            for key, count in zip(unique, counts, strict=True)
        },
    }


def _official_grain_layer(
    basis: np.ndarray,
    *,
    channel: int,
    offset_yx: Sequence[int],
    grain_size: float,
    height: int,
    width: int,
) -> np.ndarray:
    noise = basis[:, :, channel].astype(np.float32, copy=False)
    rolled = np.roll(noise, shift=(int(offset_yx[0]), int(offset_yx[1])), axis=(0, 1))
    crop_h = int(min(height / grain_size, noise.shape[0]))
    crop_w = int(min(width / grain_size, noise.shape[1]))
    crop = rolled[:crop_h, :crop_w]
    if grain_size != 1.0:
        layer = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        layer = crop
    if layer.shape[0] < height or layer.shape[1] < width:
        layer = np.pad(
            layer,
            ((0, max(0, height - layer.shape[0])), (0, max(0, width - layer.shape[1]))),
            mode="wrap",
        )
    return np.ascontiguousarray(layer[:height, :width], dtype=np.float32)


def audit_release_files(
    contract: Mapping[str, Any], code_path: Path, asset_path: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    files = contract["released_files"]
    checks = contract["frozen_checks"]
    if sha256_file(code_path) != files["film_grain_code_sha256"]:
        raise AbsoluteDegradationBlueNoiseReleaseError("film-grain code hash drift")
    if sha256_file(asset_path) != files["blue_noise_sha256"]:
        raise AbsoluteDegradationBlueNoiseReleaseError("blue-noise asset hash drift")

    source = code_path.read_text(encoding="utf-8")
    required_source_fragments = [
        "np.load(abs_path).astype(",
        "grain_layer = cv2.resize(",
        "noisy_img = image + (grain_layer * variance_map) + scanner_floor",
    ]
    source_path_exact = all(fragment in source for fragment in required_source_fragments)

    basis = np.load(asset_path, allow_pickle=False)
    expected_shape = tuple(int(value) for value in checks["expected_shape"])
    shape_dtype_exact = (
        basis.shape == expected_shape
        and str(basis.dtype) == checks["expected_dtype"]
        and basis.nbytes == int(checks["expected_array_payload_bytes"])
    )
    if not shape_dtype_exact:
        raise AbsoluteDegradationBlueNoiseReleaseError(
            f"blue-noise array contract drift: {basis.shape} {basis.dtype} {basis.nbytes}"
        )

    array_summary = _summarize_array(basis)
    branch_rows: list[dict[str, Any]] = []
    signal_nonzero_total = 0
    for channel in checks["fixed_channels"]:
        for offset in checks["fixed_offsets_yx"]:
            for grain_size in checks["fixed_grain_sizes"]:
                layer = _official_grain_layer(
                    basis,
                    channel=int(channel),
                    offset_yx=offset,
                    grain_size=float(grain_size),
                    height=int(checks["probe_height"]),
                    width=int(checks["probe_width"]),
                )
                signal = np.asarray(checks["signal_lattice_values"], dtype=np.float32)
                variance = float(checks["iso"]) * np.power(signal, float(checks["gamma"]))
                term = layer[..., None] * variance[None, None, :]
                nonzero = int(np.count_nonzero(term))
                signal_nonzero_total += nonzero
                branch_rows.append(
                    {
                        "channel": int(channel),
                        "grain_size": float(grain_size),
                        "offset_yx": [int(offset[0]), int(offset[1])],
                        "layer_nonzero_count": int(np.count_nonzero(layer)),
                        "layer_standard_deviation": float(np.std(layer.astype(np.float64))),
                        "signal_dependent_nonzero_count": nonzero,
                    }
                )

    gates = {
        "source_hashes_exact": True,
        "source_path_exact": source_path_exact,
        "array_shape_dtype_and_payload_exact": shape_dtype_exact,
        "array_has_finite_nonzero_variance": bool(
            array_summary["finite"] and array_summary["standard_deviation"] > 0.0
        ),
        "all_channels_have_nonzero_values": all(
            np.count_nonzero(basis[:, :, channel]) > 0
            for channel in checks["fixed_channels"]
        ),
        "all_roll_resize_branches_have_nonzero_variance": all(
            row["layer_standard_deviation"] > 0.0 for row in branch_rows
        ),
        "signal_dependent_term_has_nonzero_values": signal_nonzero_total > 0,
    }
    automatic_pass = all(gates.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_bytes(canonical_json(contract)),
        "released_files": {
            "film_grain_code_sha256": files["film_grain_code_sha256"],
            "blue_noise_sha256": files["blue_noise_sha256"],
            "blue_noise_file_bytes": int(files["blue_noise_file_bytes"]),
        },
        "source_path_exact": source_path_exact,
        "array": {
            "shape": list(basis.shape),
            "dtype": str(basis.dtype),
            "payload_bytes": int(basis.nbytes),
            **array_summary,
        },
        "branch_count": len(branch_rows),
        "branches": branch_rows,
        "signal_dependent_nonzero_count": signal_nonzero_total,
        "gates": gates,
        "automatic_pass": automatic_pass,
        "decision": (
            "PASS_RELEASED_SIGNAL_DEPENDENT_BLUE_NOISE_EXECUTABLE"
            if automatic_pass
            else "FAIL_CLOSED_RELEASED_BLUE_NOISE_TERM_IDENTICALLY_ZERO"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    report["stable_evidence_id"] = sha256_bytes(canonical_json(report))
    return report


def evaluate_release(contract_path: Path, scratch: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    code_path, asset_path = acquire_release(contract, scratch)
    return audit_release_files(contract, code_path, asset_path)
