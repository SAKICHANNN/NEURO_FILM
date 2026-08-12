"""Create-only staging for correct-domain cloud plus Standard display rows."""

from __future__ import annotations

import hashlib
import os
import sys
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any

from src.eval.native_cloud_standard_display import (
    render_cloud_scan_with_standard_display,
)

from .native_cloud_scan_runtime_v2 import WindowedNativeCloudScanRuntime
from .native_standard_runtime import NativeStandardRuntime
from .native_standard_staging import (
    STAGING_SCHEMA,
    _absolute_unresolved,
    _canonical_bytes,
    _sha256_file,
)


def _stage_cloud_display_rows(
    standard: NativeStandardRuntime,
    cloud: WindowedNativeCloudScanRuntime,
    *,
    renderer: Callable[..., dict[str, Any]],
    receipt_schema: str,
    height: int,
    width: int,
    source_rows: Any,
    expected_input_sha256: str,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Atomically publish raw display rows and the established report last."""

    if sys.byteorder != "little":
        raise RuntimeError("cloud Standard staging requires little endian")
    output = _absolute_unresolved(output_path)
    report = _absolute_unresolved(report_path)
    if (
        output == report
        or output.parent != report.parent
        or output.suffix.lower() != ".f32"
        or report.suffix.lower() != ".json"
        or not output.parent.is_dir()
        or output.parent.resolve(strict=True) != output.parent
    ):
        raise ValueError("cloud Standard output/report staging paths rejected")
    if output.exists() or report.exists():
        raise FileExistsError("cloud Standard staging is create-only")
    token = uuid.uuid4().hex
    output_temp = output.parent / f".{output.name}.{token}.stage"
    report_temp = output.parent / f".{report.name}.{token}.stage"
    output_committed = False
    try:
        with output_temp.open("xb") as handle:
            staged_bytes = 0

            def sink(_y0: int, _y1: int, rows: Any) -> None:
                nonlocal staged_bytes
                raw = memoryview(rows).cast("B")
                written = handle.write(raw)
                if written != len(raw):
                    raise OSError("short cloud Standard staging write")
                staged_bytes += written

            receipt = renderer(
                standard,
                cloud,
                height=height,
                width=width,
                source_rows=source_rows,
                expected_input_sha256=expected_input_sha256,
                output_sink=sink,
            )
            handle.flush()
            os.fsync(handle.fileno())
        output_sha = _sha256_file(output_temp)
        if output_sha != receipt["output_sha256"] or staged_bytes != height * width * 12:
            raise RuntimeError("cloud Standard staged output drift")
        working_receipt = {
            "schema": receipt_schema,
            "package_sha256": receipt["standard_package_sha256"],
            "artifact_sha256": receipt["standard_artifact_sha256"],
            "physical_component_sha256": receipt["cloud_receipt"][
                "physical_component_sha256"
            ],
            "physical_order": receipt["physical_order"],
            "input": {
                "array_sha256": receipt["input_sha256"],
                "dtype": "float32",
                "shape": receipt["shape"],
                "domain": "scene-linear-relative-exposure",
                "working_space": "linear-srgb",
            },
            "output": {
                "array_sha256": output_sha,
                "dtype": "float32",
                "shape": receipt["shape"],
                "domain": "display-encoded-rgb",
                "quantized": False,
            },
            "source_passes": receipt["source_passes"],
            "production_default_changed": False,
        }
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
            if output.exists() and os.path.samefile(output, output_temp):
                output.unlink()
                output_committed = False
            raise
        return {
            "schema": STAGING_SCHEMA,
            "state": payload["state"],
            "run_id": run_id,
            "report_path": str(report),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "output_path": str(output),
            "output_sha256": output_sha,
            "claim_ceiling": payload["claim_ceiling"],
        }
    finally:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        if output_committed and not report.exists() and output.exists():
            output.unlink()


def stage_cloud_scan_with_standard_display(
    standard: NativeStandardRuntime,
    cloud: WindowedNativeCloudScanRuntime,
    *,
    height: int,
    width: int,
    source_rows: Any,
    expected_input_sha256: str,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Stage the historical full-AO6 composition for exact replay only."""

    return _stage_cloud_display_rows(
        standard,
        cloud,
        renderer=render_cloud_scan_with_standard_display,
        receipt_schema="neuro_film.cloud_standard_display_receipt.v1",
        height=height,
        width=width,
        source_rows=source_rows,
        expected_input_sha256=expected_input_sha256,
        output_path=output_path,
        report_path=report_path,
    )


__all__ = ["stage_cloud_scan_with_standard_display"]
