#!/usr/bin/env python3
"""Run the frozen P95 Adobe dng_validate bounded runtime audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "neuro_film.p95_adobe_dng_validate_runtime_result.v1"


class P95Error(RuntimeError):
    pass


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_bytes: int | None, expected_sha256: str) -> None:
    if not path.is_file():
        raise P95Error(f"bound file is absent: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise P95Error(f"bound file size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise P95Error(f"bound file hash mismatch: {path}")


def _output_path(base: Path) -> Path:
    appended = Path(f"{base}.tif")
    if appended.is_file():
        return appended
    if base.is_file():
        return base
    raise P95Error(f"dng_validate output is absent: {base}")


def _inspect_tiff(path: Path, *, expected_components: int) -> dict[str, Any]:
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise P95Error(f"output must contain exactly one TIFF page: {path}")
        page = document.pages[0]
        image = page.asarray()
        samples = int(page.samplesperpixel)
        bits = int(page.bitspersample)
        photometric = int(page.photometric)
    if image.size == 0 or not np.all(np.isfinite(image)):
        raise P95Error(f"output is empty or non-finite: {path}")
    if samples != expected_components:
        raise P95Error(f"unexpected component count {samples}: {path}")
    return {
        "bits_per_sample": bits,
        "bytes": path.stat().st_size,
        "dtype": str(image.dtype),
        "height": int(image.shape[0]),
        "maximum": int(np.max(image)),
        "minimum": int(np.min(image)),
        "photometric": photometric,
        "samples_per_pixel": samples,
        "sha256": _sha256(path),
        "width": int(image.shape[1]),
    }


def _run_row(binary: Path, row: dict[str, Any], output_root: Path) -> dict[str, Any]:
    source = Path(row["logical_path"])
    _verify_file(source, row["source_bytes"], row["source_sha256"])
    row_root = output_root / row["source_id"]
    row_root.mkdir(parents=True, exist_ok=True)
    bases = {name: row_root / name for name in ("stage2", "stage3", "final")}
    command = [
        str(binary),
        "-size",
        "1024",
        "-16",
        "-cs1",
        "-2",
        str(bases["stage2"]),
        "-3",
        str(bases["stage3"]),
        "-tif",
        str(bases["final"]),
        str(source),
    ]
    process = subprocess.run(command, check=False, capture_output=True)
    (row_root / "stdout.bin").write_bytes(process.stdout)
    (row_root / "stderr.bin").write_bytes(process.stderr)
    stdout_text = process.stdout.decode("utf-8", errors="replace")
    stderr_text = process.stderr.decode("utf-8", errors="replace")
    error_lines = [
        line
        for line in (stdout_text + "\n" + stderr_text).splitlines()
        if "error" in line.casefold() or "exception" in line.casefold()
    ]
    if process.returncode != 0 or error_lines:
        raise P95Error(
            f"dng_validate rejected {row['source_id']}: "
            f"returncode={process.returncode}, error_lines={len(error_lines)}"
        )
    output_paths = {name: _output_path(base) for name, base in bases.items()}
    outputs = {
        "final": _inspect_tiff(output_paths["final"], expected_components=3),
        "stage2": _inspect_tiff(output_paths["stage2"], expected_components=1),
        "stage3": _inspect_tiff(output_paths["stage3"], expected_components=3),
    }
    if (
        outputs["final"]["dtype"] != "uint16"
        or max(outputs["final"]["height"], outputs["final"]["width"]) > 1024
    ):
        raise P95Error(f"final output contract mismatch: {row['source_id']}")
    _verify_file(source, row["source_bytes"], row["source_sha256"])
    return {
        "camera_make": row["camera_make"],
        "outputs": outputs,
        "outcome": "success",
        "process_returncode": process.returncode,
        "source_bytes": row["source_bytes"],
        "source_id": row["source_id"],
        "source_sha256": row["source_sha256"],
        "stdout_nonempty_lines": sum(
            bool(line.strip()) for line in stdout_text.splitlines()
        ),
        "stderr_nonempty_lines": sum(
            bool(line.strip()) for line in stderr_text.splitlines()
        ),
        "validation_error_lines": len(error_lines),
    }


def run(config_path: Path, output_root: Path, order: str) -> dict[str, Any]:
    if output_root.exists():
        raise P95Error(f"formal output root must not already exist: {output_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    for name in ("contract", "runner", "binary", "sdk_archive"):
        binding = bindings[name]
        _verify_file(
            Path(binding["path"]),
            binding.get("bytes"),
            binding["sha256"],
        )
    rows = list(config["rows"])
    if order == "reverse":
        rows.reverse()
    scientific_rows = []
    for row in rows:
        try:
            result = _run_row(Path(bindings["binary"]["path"]), row, output_root)
        except P95Error as error:
            _verify_file(
                Path(row["logical_path"]), row["source_bytes"], row["source_sha256"]
            )
            result = {
                "camera_make": row["camera_make"],
                "failure": str(error),
                "outcome": "rejected",
                "outputs": {},
                "source_bytes": row["source_bytes"],
                "source_id": row["source_id"],
                "source_sha256": row["source_sha256"],
            }
        scientific_rows.append(result)
    scientific_rows.sort(key=lambda item: item["source_id"])
    successful_rows = [row for row in scientific_rows if row["outcome"] == "success"]
    metrics = {
        "distinct_camera_makes": len({row["camera_make"] for row in scientific_rows}),
        "maximum_final_long_side": max(
            (
                max(
                    row["outputs"]["final"]["height"],
                    row["outputs"]["final"]["width"],
                )
                for row in successful_rows
            ),
            default=None,
        ),
        "output_file_count": sum(len(row["outputs"]) for row in scientific_rows),
        "rejected_row_count": len(scientific_rows) - len(successful_rows),
        "row_count": len(scientific_rows),
        "successful_row_count": len(successful_rows),
        "validation_error_lines": sum(
            row["validation_error_lines"] for row in successful_rows
        ),
    }
    gates = config["gates"]
    gate_results = {
        "camera_make_count": metrics["distinct_camera_makes"]
        == gates["required_camera_makes"],
        "final_long_side": metrics["maximum_final_long_side"] is not None
        and metrics["maximum_final_long_side"] <= gates["maximum_final_long_side"],
        "output_file_count": metrics["output_file_count"]
        == gates["required_output_files"],
        "row_count": metrics["row_count"] == gates["required_rows"],
        "successful_rows": metrics["successful_row_count"] == gates["required_rows"],
        "validation_errors": metrics["validation_error_lines"] == 0,
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_ADOBE_DNG_VALIDATE_RUNTIME"
        if all(gate_results.values())
        else "FAIL_CLOSED_ADOBE_DNG_VALIDATE_RUNTIME",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": metrics,
        "rows": scientific_rows,
        "schema": SCHEMA,
    }
    return {
        **scientific,
        "stable_identity_sha256": hashlib.sha256(
            _canonical_bytes(scientific)
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    report = run(args.config, args.output_root, args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(_canonical_bytes(report))
    print(report["stable_identity_sha256"])
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
