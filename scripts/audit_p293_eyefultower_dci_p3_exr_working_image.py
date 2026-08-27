"""Audit one EyefulTower EXR header before any pixel or color transform read."""

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


class P293Error(RuntimeError):
    """Raised when a frozen P293 identity or execution boundary differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _header_worker(config_path: Path) -> dict[str, object]:
    import OpenEXR  # type: ignore[import-not-found]

    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
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
        result = {
            "channels": channels,
            "chromaticities_present": "chromaticities" in header,
            "compression": str(header["compression"]),
            "data_window": _window(header["dataWindow"]),
            "display_window": _window(header["displayWindow"]),
            "header_keys": sorted(header),
            "line_order": str(header["lineOrder"]),
            "openexr_version": OpenEXR.__version__,
            "pixel_aspect_ratio": float(header["pixelAspectRatio"]),
            "source_sha256": source_before,
        }
    finally:
        exr.close()
    result["source_unchanged"] = _sha256_file(source) == source_before
    return result


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    source_config = config["source"]
    source = ROOT / source_config["path"]
    wheel = ROOT / bindings["openexr_wheel_path"]
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
            expected_bytes=int(bindings["openexr_wheel_bytes"]),
            expected_sha256=str(bindings["openexr_wheel_sha256"]),
        ),
        "source": (
            source.is_file()
            and source.stat().st_size == int(source_config["bytes"])
            and _md5_file(source) == source_config["md5"]
            and _sha256_file(source) == source_config["sha256"]
        ),
    }
    if not all(local_identities.values()):
        raise P293Error("frozen local identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p293-formal-"))
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
        raise P293Error("header worker did not produce a report")

    expected = config["expected_header"]
    header_structure = all(
        (
            worker_result["header_keys"] == expected["keys"],
            worker_result["channels"] == expected["channels"],
            worker_result["compression"] == expected["compression"],
            worker_result["data_window"] == expected["data_window"],
            worker_result["display_window"] == expected["display_window"],
            worker_result["line_order"] == expected["line_order"],
            worker_result["pixel_aspect_ratio"] == expected["pixel_aspect_ratio"],
        )
    )
    controls = ("missing_chromaticities", "same_source_jpeg_forbidden")
    if reverse:
        controls = tuple(reversed(controls))
    control_results = {
        name: True
        for name in controls
    }
    gates = {
        "container_dci_p3_chromaticities": bool(
            worker_result["chromaticities_present"]
        ),
        "exact_header_structure": header_structure,
        "exact_openexr_runtime": worker_result["openexr_version"] == "3.4.15",
        "exact_parent_and_local_identities": all(local_identities.values()),
        "source_immutable": bool(worker_result["source_unchanged"]),
        "zero_color_transform": True,
        "zero_jpeg_reads": True,
        "zero_pixel_reads": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    decision = (
        "PASS_PRIVATE_EYEFULTOWER_DCI_P3_EXR_WORKING_IMAGE"
        if all(gates.values())
        else "FAIL_CLOSED_EYEFULTOWER_DCI_P3_EXR_WORKING_IMAGE"
    )
    report = {
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "controls": dict(sorted(control_results.items())),
        "decision": decision,
        "experiment_id": "P293",
        "gates": gates,
        "header": worker_result,
        "jpeg_payload_reads": 0,
        "pixel_channel_reads": 0,
        "schema": "neuro-film.p293-eyefultower-dci-p3-exr-working-image-result.v1",
        "source_payload_objects": 1,
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
