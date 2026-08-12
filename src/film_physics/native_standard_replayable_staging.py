"""Create-only staging companion for replayable Native Standard rows."""

from __future__ import annotations

import hashlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from src.eval.native_standard_replayable_rows import (
    render_native_standard_replayable_rows,
)

from .native_standard_runtime import NativeStandardRuntime
from .native_standard_staging import (
    STAGING_SCHEMA,
    _absolute_unresolved,
    _canonical_bytes,
    _sha256_file,
)


def stage_native_standard_replayable_rows(
    runtime: NativeStandardRuntime,
    *,
    height: int,
    width: int,
    source_rows: Any,
    expected_input_sha256: str,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Stage replayable rows with the frozen report-last staging schema."""

    if sys.byteorder != "little":
        raise RuntimeError("native Standard staging requires little endian")
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
        raise ValueError("native Standard output/report staging paths rejected")
    if output.exists() or report.exists():
        raise FileExistsError("native Standard staging is create-only")
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
                    raise OSError("short native Standard staging write")
                staged_bytes += written

            receipt = render_native_standard_replayable_rows(
                runtime,
                height=height,
                width=width,
                source_rows=source_rows,
                expected_input_sha256=expected_input_sha256,
                output_sink=sink,
            )
            handle.flush()
            os.fsync(handle.fileno())
        output_sha = _sha256_file(output_temp)
        if (
            output_sha != receipt["output_sha256"]
            or staged_bytes != height * width * 12
        ):
            raise RuntimeError("native Standard staged output drift")
        working_receipt = {
            "schema": "neuro_film.native_standard_replayable_receipt.v1",
            "package_sha256": receipt["package_sha256"],
            "artifact_sha256": receipt["artifact_sha256"],
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


__all__ = ["stage_native_standard_replayable_rows"]
