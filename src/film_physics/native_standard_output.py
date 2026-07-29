"""Final sRGB16 PNG transaction for verified native Standard staging."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid
from typing import Any

import numpy as np

from src.preprocess.output_encode import (
    save_srgb16_png,
    srgb_icc_profile_sha256,
)

from .native_standard_staging import (
    _absolute_unresolved,
    _bounded_read,
    _canonical_bytes,
    _is_sha256,
    _sha256_file,
    _strict_json_loads,
    verify_native_standard_staging,
)


OUTPUT_SCHEMA = "neuro_film.native_standard_png16_report.v1"
OUTPUT_VERIFICATION_SCHEMA = (
    "neuro_film.native_standard_png16_verification.v1"
)
_MAX_REPORT_BYTES = 2 * 1024 * 1024


def commit_verified_native_standard_png16(
    *,
    staging_report_path: Path,
    expected_staging_report_sha256: str,
    expected_staging_run_id: str,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Quantize one verified raw stage exactly once and commit PNG/report."""

    verification = verify_native_standard_staging(
        report_path=staging_report_path,
        expected_report_sha256=expected_staging_report_sha256,
        expected_run_id=expected_staging_run_id,
    )
    staging_raw = _bounded_read(
        _absolute_unresolved(staging_report_path),
        _MAX_REPORT_BYTES,
    )
    if hashlib.sha256(staging_raw).hexdigest() != (
        expected_staging_report_sha256
    ):
        raise ValueError("native Standard staging changed after verification")
    staging = _strict_json_loads(staging_raw)
    raw_output = Path(staging["output_path"])
    shape = tuple(
        staging["working_image_receipt"]["output"]["shape"]
    )
    if (
        len(shape) != 3
        or shape[2] != 3
        or raw_output.stat().st_size != int(np.prod(shape)) * 4
    ):
        raise ValueError("native Standard raw staging shape drift")

    output = _absolute_unresolved(output_path)
    report = _absolute_unresolved(report_path)
    if (
        output == report
        or output.parent != report.parent
        or output.suffix.lower() != ".png"
        or report.suffix.lower() != ".json"
        or not output.parent.is_dir()
        or output.parent.resolve(strict=True) != output.parent
        or output.exists()
        or report.exists()
    ):
        raise ValueError(
            "native Standard PNG/report must be absent canonical siblings"
        )

    token = uuid.uuid4().hex
    output_temp = output.parent / f".{output.stem}.{token}.png"
    report_temp = report.parent / f".{report.name}.{token}.stage"
    output_committed = False
    try:
        pixels = np.memmap(
            raw_output,
            dtype="<f4",
            mode="r",
            shape=shape,
            order="C",
        )
        if (
            not np.all(np.isfinite(pixels))
            or np.any(pixels < 0.0)
            or np.any(pixels > 1.0)
        ):
            raise ValueError(
                "native Standard final values require clipping"
            )
        save_srgb16_png(pixels, output_temp)
        del pixels
        if _sha256_file(raw_output) != staging["output_sha256"]:
            raise ValueError(
                "native Standard raw staging changed during encoding"
            )
        png_sha = _sha256_file(output_temp)
        png_bytes = output_temp.stat().st_size
        core = {
            "schema": OUTPUT_SCHEMA,
            "state": "committed-native-standard-png16",
            "claim_ceiling": (
                "native-standard-srgb16-png-committed-not-delivered"
            ),
            "staging_verification_id": verification["verification_id"],
            "staging_run_id": expected_staging_run_id,
            "staging_report_sha256": expected_staging_report_sha256,
            "staging_output_sha256": staging["output_sha256"],
            "output_path": str(output),
            "output_sha256": png_sha,
            "output_bytes": png_bytes,
            "format": "PNG",
            "bit_depth": 16,
            "colour_space": "sRGB",
            "icc_profile_sha256": srgb_icc_profile_sha256(),
            "single_final_quantization": True,
            "production_default_changed": False,
        }
        delivery_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        payload = {**core, "delivery_id": delivery_id}
        report_bytes = _canonical_bytes(payload)
        with report_temp.open("xb") as handle:
            handle.write(report_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(output_temp, output)
        output_committed = True
        try:
            os.link(report_temp, report)
        except BaseException:
            if (
                output.exists()
                and os.path.samefile(output, output_temp)
                and _sha256_file(output) == png_sha
            ):
                output.unlink()
                output_committed = False
            raise
        return {
            "schema": OUTPUT_SCHEMA,
            "state": payload["state"],
            "delivery_id": delivery_id,
            "report_path": str(report),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "output_path": str(output),
            "output_sha256": png_sha,
            "claim_ceiling": payload["claim_ceiling"],
        }
    finally:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        if output_committed and not report.exists() and output.exists():
            output.unlink()


def verify_native_standard_png16(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_delivery_id: str,
) -> dict[str, Any]:
    """Restart-verify the final encoded staging file."""

    if not _is_sha256(expected_report_sha256) or not _is_sha256(
        expected_delivery_id
    ):
        raise ValueError("expected PNG report identities must be SHA-256")
    raw = _bounded_read(
        _absolute_unresolved(report_path),
        _MAX_REPORT_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != expected_report_sha256:
        raise ValueError("native Standard PNG report hash mismatch")
    payload = _strict_json_loads(raw)
    if (
        _canonical_bytes(payload) != raw
        or payload["schema"] != OUTPUT_SCHEMA
        or payload["state"] != "committed-native-standard-png16"
        or payload["delivery_id"] != expected_delivery_id
        or payload["single_final_quantization"] is not True
        or payload["production_default_changed"] is not False
    ):
        raise ValueError("native Standard PNG report contract drift")
    core = {
        key: value
        for key, value in payload.items()
        if key != "delivery_id"
    }
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != (
        expected_delivery_id
    ):
        raise ValueError("native Standard PNG delivery identity drift")
    output = Path(payload["output_path"])
    if (
        not output.is_absolute()
        or not output.is_file()
        or output.stat().st_size != payload["output_bytes"]
        or _sha256_file(output) != payload["output_sha256"]
    ):
        raise ValueError("native Standard PNG output drift")
    core_receipt = {
        "schema": OUTPUT_VERIFICATION_SCHEMA,
        "state": "verified-native-standard-png16",
        "delivery_id": expected_delivery_id,
        "report_sha256": expected_report_sha256,
        "output_sha256": payload["output_sha256"],
        "claim_ceiling": (
            "verified-native-standard-srgb16-png-not-delivered"
        ),
    }
    return {
        **core_receipt,
        "verification_id": hashlib.sha256(
            _canonical_bytes(core_receipt)
        ).hexdigest(),
    }


__all__ = [
    "commit_verified_native_standard_png16",
    "verify_native_standard_png16",
]
