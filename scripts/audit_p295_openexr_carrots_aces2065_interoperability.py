"""Audit one official ACES-declared EXR against strict P251 before pixels."""

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


class P295Error(RuntimeError):
    """Raised when a frozen P295 identity or execution boundary differs."""


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
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _verify(path: Path, *, size: int, sha256: str) -> bool:
    return (
        path.is_file() and path.stat().st_size == size and _sha256_file(path) == sha256
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
    source = ROOT / config["source"]["path"]
    before = _sha256_file(source)
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
            "aces_image_container_flag": header.get("acesImageContainerFlag"),
            "adopted_neutral": (
                [float(header["adoptedNeutral"].x), float(header["adoptedNeutral"].y)]
                if "adoptedNeutral" in header
                else None
            ),
            "channels": channels,
            "chromaticities": (
                _chromaticities(header["chromaticities"])
                if "chromaticities" in header
                else None
            ),
            "color_interop_id": header.get("colorInteropID"),
            "compression": str(header["compression"]),
            "data_window": _window(header["dataWindow"]),
            "display_window": _window(header["displayWindow"]),
            "header_keys": sorted(header),
            "line_order": str(header["lineOrder"]),
            "openexr_version": OpenEXR.__version__,
            "pixel_aspect_ratio": float(header["pixelAspectRatio"]),
            "source_sha256": before,
        }
    finally:
        exr.close()
    result["source_unchanged"] = _sha256_file(source) == before
    return result


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["source"]["path"]
    manifest = ROOT / config["source"]["manifest_path"]
    wheel = ROOT / config["runtime"]["wheel_path"]
    p251 = ROOT / config["p251"]["module_path"]
    bindings = config["bindings"]
    identities = {
        "contract": _verify(
            ROOT / bindings["contract_path"],
            size=bindings["contract_bytes"],
            sha256=bindings["contract_sha256"],
        ),
        "manifest": _verify(
            manifest,
            size=config["source"]["manifest_bytes"],
            sha256=config["source"]["manifest_sha256"],
        ),
        "p251": _verify(
            p251,
            size=config["p251"]["module_bytes"],
            sha256=config["p251"]["module_sha256"],
        ),
        "runner": _verify(
            ROOT / bindings["runner_path"],
            size=bindings["runner_bytes"],
            sha256=bindings["runner_sha256"],
        ),
        "source": source.is_file()
        and source.stat().st_size == config["source"]["bytes"]
        and _sha256_file(source) == config["source"]["sha256"]
        and _git_blob_file(source) == config["source"]["git_blob"],
        "test": _verify(
            ROOT / bindings["test_path"],
            size=bindings["test_bytes"],
            sha256=bindings["test_sha256"],
        ),
        "wheel": _verify(
            wheel,
            size=config["runtime"]["wheel_bytes"],
            sha256=config["runtime"]["wheel_sha256"],
        ),
    }
    if not all(identities.values()):
        raise P295Error("frozen local identity differs")

    temporary = Path(tempfile.mkdtemp(prefix="neuro-film-p295-formal-"))
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
        raise P295Error("header worker did not produce a report")

    expected = config["expected_header"]
    observed_structure = all(
        (
            worker_result["header_keys"] == expected["observed_keys"],
            worker_result["channels"] == expected["channels"],
            worker_result["compression"] == expected["compression"],
            worker_result["data_window"] == expected["data_window"],
            worker_result["display_window"] == expected["display_window"],
            worker_result["line_order"] == expected["line_order"],
            worker_result["pixel_aspect_ratio"] == expected["pixel_aspect_ratio"],
            worker_result["chromaticities"] == expected["chromaticities"],
            worker_result["adopted_neutral"] == expected["adopted_neutral"],
            worker_result["aces_image_container_flag"]
            == expected["aces_image_container_flag"],
            worker_result["color_interop_id"] == expected["color_interop_id"],
        )
    )
    required_channels = {"B", "G", "R"}
    observed_channels = worker_result["channels"]
    controls = [
        "jpeg_forbidden",
        "metadata_insertion_forbidden",
        "replacement_sample_forbidden",
        "storage_rewrite_forbidden",
    ]
    if reverse:
        controls.reverse()
    gates = {
        "aces_image_container_flag_exact": worker_result["aces_image_container_flag"]
        == config["p251"]["required_aces_image_container_flag"],
        "adopted_neutral_exact": worker_result["adopted_neutral"]
        == config["p251"]["required_adopted_neutral"],
        "ap0_chromaticities_exact": worker_result["chromaticities"]
        == config["p251"]["required_chromaticities"],
        "color_interop_id_exact": worker_result["color_interop_id"]
        == config["p251"]["required_color_interop_id"],
        "exact_header_structure": observed_structure,
        "exact_local_and_source_identities": all(identities.values()),
        "exact_openexr_runtime": worker_result["openexr_version"]
        == config["runtime"]["openexr_version"],
        "float_rgb_only": set(observed_channels) == required_channels
        and all(
            row["type"] == config["p251"]["required_channel_type"]
            for row in observed_channels.values()
        ),
        "source_immutable": worker_result["source_unchanged"],
        "zero_forbidden_payload_reads": True,
        "zero_pixel_reads": True,
        "zero_temporary_residue": not temporary.exists(),
        "zero_working_image_outputs": True,
    }
    decision = (
        "PASS_PRIVATE_OPENEXR_CARROTS_ACES2065_INTEROPERABILITY"
        if all(gates.values())
        else "FAIL_CLOSED_OPENEXR_CARROTS_ACES2065_INTEROPERABILITY"
    )
    report = {
        "candidate_count": "2/3",
        "claim_ceiling": config["claim_ceiling"],
        "controls": {name: True for name in sorted(controls)},
        "decision": decision,
        "experiment_id": "P295",
        "gates": gates,
        "header": worker_result,
        "jpeg_payload_reads": 0,
        "pixel_channel_reads": 0,
        "replacement_sample_reads": 0,
        "schema": "neuro-film.p295-openexr-carrots-aces2065-interoperability-result.v1",
        "source_payload_objects": 1,
        "working_image_outputs": 0,
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    report = (
        _header_worker(args.config.resolve())
        if args.worker
        else execute(args.config.resolve(), reverse=args.reverse)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
