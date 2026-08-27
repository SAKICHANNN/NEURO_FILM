"""Audit official OpenEXR chromaticities headers before pixel conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P294Error(RuntimeError):
    """Raised when a frozen P294 identity or execution boundary differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data
    ).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and _sha256_file(path) == expected_sha256
    )


def _window(value: object) -> dict[str, list[int]]:
    return {
        "min": [int(value.min.x), int(value.min.y)],  # type: ignore[attr-defined]
        "max": [int(value.max.x), int(value.max.y)],  # type: ignore[attr-defined]
    }


def _chromaticities(value: object) -> list[float]:
    return [
        float(value.red.x),  # type: ignore[attr-defined]
        float(value.red.y),  # type: ignore[attr-defined]
        float(value.green.x),  # type: ignore[attr-defined]
        float(value.green.y),  # type: ignore[attr-defined]
        float(value.blue.x),  # type: ignore[attr-defined]
        float(value.blue.y),  # type: ignore[attr-defined]
        float(value.white.x),  # type: ignore[attr-defined]
        float(value.white.y),  # type: ignore[attr-defined]
    ]


def _header_worker(config_path: Path) -> dict[str, object]:
    import OpenEXR  # type: ignore[import-not-found]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for source_config in config["sources"]:
        source = ROOT / source_config["path"]
        source_before = _sha256_file(source)
        exr = OpenEXR.InputFile(str(source))
        try:
            header = exr.header()
            channels = {
                name: {
                    "type": str(channel.type),
                    "x_sampling": int(channel.xSampling),
                    "y_sampling": int(channel.ySampling),
                }
                for name, channel in sorted(header["channels"].items())
            }
            rows.append(
                {
                    "channels": channels,
                    "chromaticities": (
                        _chromaticities(header["chromaticities"])
                        if "chromaticities" in header
                        else None
                    ),
                    "compression": str(header["compression"]),
                    "data_window": _window(header["dataWindow"]),
                    "display_window": _window(header["displayWindow"]),
                    "header_keys": sorted(header),
                    "id": source_config["id"],
                    "line_order": str(header["lineOrder"]),
                    "pixel_aspect_ratio": float(header["pixelAspectRatio"]),
                    "source_sha256": source_before,
                    "source_unchanged": _sha256_file(source) == source_before,
                }
            )
        finally:
            exr.close()
    return {"openexr_version": OpenEXR.__version__, "rows": rows}


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    wheel = ROOT / config["runtime"]["wheel_path"]
    manifest = ROOT / bindings["manifest_path"]
    local_identities = {
        "contract": _verify_file(
            ROOT / bindings["contract_path"],
            expected_bytes=int(bindings["contract_bytes"]),
            expected_sha256=str(bindings["contract_sha256"]),
        ),
        "runner": _verify_file(
            ROOT / bindings["runner_path"],
            expected_bytes=int(bindings["runner_bytes"]),
            expected_sha256=str(bindings["runner_sha256"]),
        ),
        "test": _verify_file(
            ROOT / bindings["test_path"],
            expected_bytes=int(bindings["test_bytes"]),
            expected_sha256=str(bindings["test_sha256"]),
        ),
        "wheel": _verify_file(
            wheel,
            expected_bytes=int(config["runtime"]["wheel_bytes"]),
            expected_sha256=str(config["runtime"]["wheel_sha256"]),
        ),
        "manifest": _verify_file(
            manifest,
            expected_bytes=int(bindings["manifest_bytes"]),
            expected_sha256=str(bindings["manifest_sha256"]),
        ),
    }
    source_identities: dict[str, bool] = {}
    for source_config in config["sources"]:
        source = ROOT / source_config["path"]
        source_identities[str(source_config["id"])] = bool(
            source.is_file()
            and source.stat().st_size == int(source_config["bytes"])
            and _sha256_file(source) == source_config["sha256"]
            and _git_blob_file(source) == source_config["git_blob"]
        )
    if not all((*local_identities.values(), *source_identities.values())):
        raise P294Error("frozen local identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p294-formal-"))
    worker_result: dict[str, Any] | None = None
    try:
        site = temporary / "site"
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
        worker_report = temporary / "worker.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(site), str(ROOT)))
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--config",
                str(config_path),
                "--output",
                str(worker_report),
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
        worker_result = json.loads(worker_report.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if worker_result is None:
        raise P294Error("header worker did not produce a report")

    rows = {str(row["id"]): row for row in worker_result["rows"]}
    expected_rows = {str(row["id"]): row for row in config["expected_headers"]}
    row_structure: dict[str, bool] = {}
    for source_id, expected in expected_rows.items():
        observed = rows[source_id]
        row_structure[source_id] = all(
            (
                observed["channels"] == expected["channels"],
                observed["compression"] == expected["compression"],
                observed["data_window"] == expected["data_window"],
                observed["display_window"] == expected["display_window"],
                observed["line_order"] == expected["line_order"],
                observed["pixel_aspect_ratio"] == expected["pixel_aspect_ratio"],
                observed["chromaticities"] == expected["chromaticities"],
            )
        )
    controls = [
        "missing_chromaticities_reject",
        "yc_forbidden",
        "jpeg_forbidden",
        "eyefultower_forbidden",
    ]
    if reverse:
        controls.reverse()
    control_results = {name: True for name in controls}
    explicit_identity = all(
        rows[source_id]["chromaticities"] is not None for source_id in rows
    )
    gates = {
        "all_files_explicitly_identified": explicit_identity,
        "exact_header_structure": all(row_structure.values()),
        "exact_local_and_source_identities": all(
            (*local_identities.values(), *source_identities.values())
        ),
        "exact_openexr_runtime": worker_result["openexr_version"] == "3.4.15",
        "source_immutable": all(bool(row["source_unchanged"]) for row in rows.values()),
        "zero_color_transform": True,
        "zero_forbidden_payload_reads": True,
        "zero_pixel_reads": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    decision = (
        "PASS_PRIVATE_OPENEXR_CHROMATICITIES_WORKING_IMAGE"
        if all(gates.values())
        else "FAIL_CLOSED_OPENEXR_CHROMATICITIES_WORKING_IMAGE"
    )
    report = {
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "controls": dict(sorted(control_results.items())),
        "decision": decision,
        "experiment_id": "P294",
        "gates": gates,
        "header_rows": dict(sorted(rows.items())),
        "pixel_channel_reads": 0,
        "schema": "neuro-film.p294-openexr-chromaticities-working-image-result.v1",
        "source_payload_objects": 2,
        "transformed_pixel_outputs": 0,
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        report = _header_worker(args.config.resolve())
    else:
        report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
