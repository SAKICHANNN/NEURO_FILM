#!/usr/bin/env python3
"""Audit the frozen P229 exact-five-DNG opt-in P3-PQ callable."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_forward_aces2_pq import (
    DngForwardAces2PqError,
    render_dng_forward_to_aces2_p3_pq,
)
from src.preprocess.dng_forward_raster import load_dng_forward_working_image
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.ocio_aces2_output import apply_working_image_aces2_output

SCHEMA = "neuro-film.p229-dng-forward-aces2-p3-pq-callable-result.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _bindings(config: dict[str, Any]) -> dict[str, dict[str, object]]:
    checked: dict[str, dict[str, object]] = {}
    for name in (
        "p98_config",
        "p98_evidence",
        "p98_source",
        "p226_source",
        "p226_evidence",
        "default_pipeline",
        "generic_raw",
    ):
        path = ROOT / config["bindings"][f"{name}_path"]
        actual = _sha256_file(path)
        expected = config["bindings"][f"{name}_sha256"]
        if actual != expected:
            raise RuntimeError(f"P229 binding mismatch: {name}")
        checked[name] = {
            "path": config["bindings"][f"{name}_path"],
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return checked


def _record(row: dict[str, Any], target: str) -> dict[str, Any]:
    source = ROOT / row["logical_path"]
    source_before = _sha256_file(source)
    base = {
        "source_id": row["source_id"],
        "camera_make": row["camera_make"],
        "source_bytes": source.stat().st_size,
        "source_sha256": source_before,
    }
    try:
        callable_output = render_dng_forward_to_aces2_p3_pq(
            source,
            expected_source_bytes=row["source_bytes"],
            expected_source_sha256=row["source_sha256"],
        )
    except DngForwardAces2PqError as exc:
        return {
            **base,
            "callable_returned": False,
            "callable_error": str(exc),
            "decode_reads": 1,
            "source_unchanged": _sha256_file(source) == source_before,
        }
    working = load_dng_forward_working_image(
        source,
        expected_source_bytes=row["source_bytes"],
        expected_source_sha256=row["source_sha256"],
    )
    working_before = _array_sha256(working.pixels)
    direct = apply_working_image_aces2_output(working, target)
    difference = callable_output.astype(np.float64) - direct.astype(np.float64)
    return {
        **base,
        "callable_returned": True,
        "decode_reads": 2,
        "source_unchanged": _sha256_file(source) == source_before,
        "working_pixels_unchanged": _array_sha256(working.pixels) == working_before,
        "shape": list(callable_output.shape),
        "dtype": str(callable_output.dtype),
        "c_contiguous": bool(callable_output.flags.c_contiguous),
        "owns_data": bool(callable_output.flags.owndata),
        "finite": bool(np.isfinite(callable_output).all()),
        "in_unit": bool(np.all((callable_output >= 0.0) & (callable_output <= 1.0))),
        "minimum": float(np.min(callable_output)),
        "maximum": float(np.max(callable_output)),
        "output_sha256": _array_sha256(callable_output),
        "direct_sha256": _array_sha256(direct),
        "direct_bytes_exact": bool(np.array_equal(callable_output, direct)),
        "direct_max_abs_error": float(np.max(np.abs(difference))),
    }


def _negative_controls(row: dict[str, Any]) -> dict[str, bool]:
    source = ROOT / row["logical_path"]
    results: dict[str, bool] = {}
    for name, source_bytes, source_sha in (
        ("wrong_bytes", row["source_bytes"] + 1, row["source_sha256"]),
        ("wrong_sha", row["source_bytes"], "0" * 64),
    ):
        try:
            render_dng_forward_to_aces2_p3_pq(
                source,
                expected_source_bytes=source_bytes,
                expected_source_sha256=source_sha,
            )
        except DngForwardAces2PqError:
            results[name] = True
        else:
            results[name] = False
    invalid_bytes = b"P229 non-DNG predecode control\n"
    with tempfile.TemporaryDirectory(prefix="p229-invalid-") as temporary:
        invalid_path = Path(temporary) / "not-a-dng.bin"
        invalid_path.write_bytes(invalid_bytes)
        try:
            render_dng_forward_to_aces2_p3_pq(
                invalid_path,
                expected_source_bytes=len(invalid_bytes),
                expected_source_sha256=hashlib.sha256(invalid_bytes).hexdigest(),
            )
        except DngForwardAces2PqError:
            results["non_dng"] = True
        else:
            results["non_dng"] = False
    results["temporary_residue_zero"] = not Path(temporary).exists()
    return results


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _json(config_path)
    bindings = _bindings(config)
    p98 = _json(ROOT / config["bindings"]["p98_config_path"])
    rows = list(p98["rows"])
    if reverse:
        rows.reverse()
    records = sorted(
        (_record(row, config["target"]) for row in rows),
        key=lambda value: value["source_id"],
    )
    negative_controls = _negative_controls(p98["rows"][0])
    gates = {
        "bindings_exact": True,
        "required_rows_exact": len(records) == config["required_rows"],
        "source_identities_and_bytes_unchanged": all(
            row["source_unchanged"]
            and row["source_sha256"]
            == next(
                item["source_sha256"]
                for item in p98["rows"]
                if item["source_id"] == row["source_id"]
            )
            and row["source_bytes"]
            == next(
                item["source_bytes"]
                for item in p98["rows"]
                if item["source_id"] == row["source_id"]
            )
            for row in records
        ),
        "callable_equals_direct_retained_composition_byte_exact": all(
            row.get("direct_bytes_exact", False)
            and row.get("direct_max_abs_error") == 0.0
            for row in records
        ),
        "outputs_float32_contiguous_finite_in_unit": all(
            row.get("dtype") == "float32"
            and row.get("c_contiguous", False)
            and row.get("finite", False)
            and row.get("in_unit", False)
            for row in records
        ),
        "caller_owns_output": all(row.get("owns_data", False) for row in records),
        "wrong_identity_and_invalid_input_fail_closed": all(negative_controls.values()),
        "default_pipeline_and_generic_raw_sources_unchanged": True,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "protocol": config["schema"],
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "bindings": bindings,
        "target": config["target"],
        "records": records,
        "negative_controls": negative_controls,
        "gates": gates,
        "status": (
            "PASS_PRIVATE_DNG_FORWARD_ACES2_P3_PQ_CALLABLE"
            if all(gates.values())
            else "FAIL_CLOSED_DNG_FORWARD_ACES2_P3_PQ_CALLABLE"
        ),
        "pixel_reads": sum(int(row["decode_reads"]) for row in records),
        "network_reads": 0,
        "repository_image_writes": 0,
        "claim_ceiling": config["claim_ceiling"],
    }
    scientific = dict(report)
    scientific.pop("implementation_commit")
    report["stable_identity"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(scientific)
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p229_dng_forward_aces2_p3_pq_callable_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
    payload = canonical_json_bytes(report) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "stable_identity": report["stable_identity"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
