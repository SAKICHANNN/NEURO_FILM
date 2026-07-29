"""Create-only local staging and restart verification for native Standard."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

from src.preprocess.types import WorkingImage

from .native_standard_consumer import (
    render_native_standard_working_image_to_sink,
)
from .native_standard_runtime import NativeStandardRuntime


STAGING_SCHEMA = "neuro_film.native_standard_staging_report.v1"
VERIFICATION_SCHEMA = (
    "neuro_film.native_standard_staging_verification.v1"
)
_MAX_REPORT_BYTES = 2 * 1024 * 1024


def stage_native_standard_working_image(
    runtime: NativeStandardRuntime,
    working: WorkingImage,
    *,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Stage one raw float32 output and commit its report last."""

    if sys.byteorder != "little":
        raise RuntimeError("native Standard staging requires little endian")
    output = _absolute_unresolved(output_path)
    report = _absolute_unresolved(report_path)
    if (
        output == report
        or output.parent != report.parent
        or output.suffix.lower() != ".f32"
        or report.suffix.lower() != ".json"
    ):
        raise ValueError(
            "native Standard output/report must be distinct .f32/.json "
            "siblings"
        )
    parent = output.parent
    if (
        not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise ValueError(
            "native Standard staging parent must be an existing "
            "canonical directory"
        )
    if output.exists() or report.exists():
        raise FileExistsError("native Standard staging is create-only")

    token = uuid.uuid4().hex
    output_temp = parent / f".{output.name}.{token}.stage"
    report_temp = parent / f".{report.name}.{token}.stage"
    output_committed = False
    try:
        with output_temp.open("xb") as handle:
            staged_bytes = 0

            def sink(y0: int, y1: int, rows: Any) -> None:
                nonlocal staged_bytes
                raw = memoryview(rows).cast("B")
                written = handle.write(raw)
                if written != len(raw):
                    raise OSError("short native Standard staging write")
                staged_bytes += written

            working_receipt = (
                render_native_standard_working_image_to_sink(
                    runtime,
                    working,
                    output_sink=sink,
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        output_sha = _sha256_file(output_temp)
        if (
            output_sha
            != working_receipt["output"]["array_sha256"]
            or staged_bytes
            != working_receipt["output"]["shape"][0]
            * working_receipt["output"]["shape"][1]
            * 3
            * 4
        ):
            raise RuntimeError("native Standard staged output drift")
        core = {
            "schema": STAGING_SCHEMA,
            "state": "committed-to-native-standard-staging",
            "claim_ceiling": "native-standard-staging-not-quantized-not-delivered",
            "output_path": str(output),
            "output_sha256": output_sha,
            "output_bytes": staged_bytes,
            "output_encoding": "float32-little-endian-c-order-rgb",
            "working_image_receipt": working_receipt,
            "production_default_changed": False,
        }
        run_id = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        payload = {**core, "run_id": run_id}
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
                and _sha256_file(output) == output_sha
            ):
                output.unlink()
                output_committed = False
            raise
        report_sha = hashlib.sha256(report_bytes).hexdigest()
        return {
            "schema": STAGING_SCHEMA,
            "state": payload["state"],
            "run_id": run_id,
            "report_path": str(report),
            "report_sha256": report_sha,
            "output_path": str(output),
            "output_sha256": output_sha,
            "claim_ceiling": payload["claim_ceiling"],
        }
    finally:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        if output_committed and not report.exists():
            if output.exists():
                output.unlink()


def verify_native_standard_staging(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    """Restart-verify a committed staging report and its raw output."""

    report = _absolute_unresolved(report_path)
    if not _is_sha256(expected_report_sha256) or not _is_sha256(
        expected_run_id
    ):
        raise ValueError("expected report and run identities must be SHA-256")
    raw = _bounded_read(report, _MAX_REPORT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_report_sha256:
        raise ValueError("native Standard staging report hash mismatch")
    payload = _strict_json_loads(raw)
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "state",
            "claim_ceiling",
            "output_path",
            "output_sha256",
            "output_bytes",
            "output_encoding",
            "working_image_receipt",
            "production_default_changed",
            "run_id",
        }
        or _canonical_bytes(payload) != raw
        or payload["schema"] != STAGING_SCHEMA
        or payload["state"] != "committed-to-native-standard-staging"
        or payload["claim_ceiling"]
        != "native-standard-staging-not-quantized-not-delivered"
        or payload["production_default_changed"] is not False
        or payload["output_encoding"]
        != "float32-little-endian-c-order-rgb"
        or payload["run_id"] != expected_run_id
    ):
        raise ValueError("native Standard staging report contract drift")
    core = {key: value for key, value in payload.items() if key != "run_id"}
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != expected_run_id:
        raise ValueError("native Standard staging run identity drift")
    output = Path(payload["output_path"])
    if not output.is_absolute() or not output.is_file():
        raise ValueError("native Standard staged output missing")
    if (
        output.stat().st_size != payload["output_bytes"]
        or _sha256_file(output) != payload["output_sha256"]
        or payload["working_image_receipt"]["output"]["array_sha256"]
        != payload["output_sha256"]
    ):
        raise ValueError("native Standard staged output drift")
    core_receipt = {
        "schema": VERIFICATION_SCHEMA,
        "state": "verified-native-standard-staging",
        "run_id": expected_run_id,
        "report_sha256": expected_report_sha256,
        "output_sha256": payload["output_sha256"],
        "claim_ceiling": (
            "verified-native-standard-staging-not-quantized-not-delivered"
        ),
    }
    return {
        **core_receipt,
        "verification_id": hashlib.sha256(
            _canonical_bytes(core_receipt)
        ).hexdigest(),
    }


def _absolute_unresolved(path: Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = Path.cwd() / value
    return Path(os.path.abspath(value))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_read(path: Path, maximum: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) <= 0 or len(raw) > maximum:
        raise ValueError("native Standard staging report size rejected")
    return raw


def _strict_json_loads(raw: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw.decode("ascii"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {value}")
        ),
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


__all__ = [
    "stage_native_standard_working_image",
    "verify_native_standard_staging",
]
