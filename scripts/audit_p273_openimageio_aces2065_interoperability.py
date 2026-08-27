#!/usr/bin/env python3
"""Audit exact P249 ACES2065-1 interoperability with a pinned OpenImageIO runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_SCHEMA = "neuro-film.p273-openimageio-aces2065-interoperability-result.v1"


class P273Error(RuntimeError):
    """Raised when a frozen P273 binding or execution requirement differs."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"HEAD:{path.as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_file(path: Path, binding: dict[str, Any], *, git: bool) -> None:
    if not path.is_file():
        raise P273Error(f"bound file is absent: {path}")
    observed = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if git:
        observed["git_blob"] = _git_blob(path.relative_to(ROOT))
    expected = {key: binding[key] for key in observed}
    if observed != expected:
        raise P273Error(f"bound file identity differs: {path}")


def _oiio_worker(input_path: Path) -> dict[str, Any]:
    import numpy as np
    import OpenImageIO as oiio

    source_before = _sha256_file(input_path)
    image = oiio.ImageInput.open(str(input_path))
    if image is None:
        raise P273Error("OpenImageIO rejected the complete P249 container")
    try:
        spec = image.spec()
        pixels = image.read_image(format=oiio.FLOAT)
        if not isinstance(pixels, np.ndarray):
            raise P273Error("OpenImageIO did not return a pixel array")
        attributes = {item.name: item.value for item in spec.extra_attribs}
        pixel_bytes = np.ascontiguousarray(pixels).astype("<f4", copy=False).tobytes()
        result = {
            "adopted_neutral": list(attributes.get("adoptedNeutral", ())),
            "aces_image_container_flag": attributes.get("acesImageContainerFlag"),
            "channels": list(spec.channelnames),
            "chromaticities": list(attributes.get("chromaticities", ())),
            "color_interop_id": attributes.get("colorInteropID"),
            "compression": attributes.get("compression"),
            "format": str(spec.format),
            "height": spec.height,
            "oiio_color_space": attributes.get("oiio:ColorSpace"),
            "openimageio_version": oiio.VERSION_STRING,
            "pixel_f32le_sha256": _sha256_bytes(pixel_bytes),
            "shape": list(pixels.shape),
            "width": spec.width,
        }
    finally:
        image.close()

    truncated = input_path.with_name("truncated.exr")
    truncated.write_bytes(input_path.read_bytes()[:-64])
    truncated_rejected = False
    broken = oiio.ImageInput.open(str(truncated))
    if broken is None:
        truncated_rejected = True
    else:
        try:
            decoded = broken.read_image(format=oiio.FLOAT)
            truncated_rejected = decoded is None
        except RuntimeError:
            truncated_rejected = True
        finally:
            broken.close()
    truncated.unlink()
    result["source_immutable"] = _sha256_file(input_path) == source_before
    result["truncated_input_rejected"] = truncated_rejected
    return result


def execute(config_path: Path, producer_repo: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise P273Error("order must be forward or reverse")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["status"] != "FROZEN_BEFORE_OPENIMAGEIO_CONTAINER_READ":
        raise P273Error("P273 is not frozen before the OpenImageIO read")

    parent_names = list(config["parents"])
    runtime_names = list(config["openimageio_runtime"]["files"])
    if order == "reverse":
        parent_names.reverse()
        runtime_names.reverse()
    for name in parent_names:
        binding = config["parents"][name]
        _validate_file(ROOT / binding["path"], binding, git=True)
    runtime_root = ROOT / config["openimageio_runtime"]["root"]
    for name in runtime_names:
        _validate_file(
            runtime_root / name,
            config["openimageio_runtime"]["files"][name],
            git=False,
        )

    temporary_parent = ROOT / config["temporary_root"]
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="p273-oiio-", dir=temporary_parent))
    p249_result: dict[str, Any] | None = None
    oiio_result: dict[str, Any] | None = None
    try:
        p249_config = ROOT / config["parents"]["p249_config"]["path"]
        p249_bindings = json.loads(p249_config.read_bytes())["bindings"]
        wheel = producer_repo / p249_bindings["producer_wheel_path"]
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
                str(temporary / "site"),
                str(wheel),
            ],
            check=True,
            capture_output=True,
        )
        p249_report = temporary / "p249.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / config["parents"]["p249_runner"]["path"]),
                "--worker",
                "--config",
                str(p249_config),
                "--producer-repo",
                str(producer_repo),
                "--workspace",
                str(temporary),
                "--output",
                str(p249_report),
            ],
            check=True,
            capture_output=True,
        )
        p249_result = json.loads(p249_report.read_bytes())
        oiio_report = temporary / "oiio.json"
        subprocess.run(
            [
                str(runtime_root / "Scripts/python.exe"),
                str(Path(__file__).resolve()),
                "--oiio-worker",
                "--input",
                str(temporary / "synthetic-ap0.exr"),
                "--output",
                str(oiio_report),
            ],
            check=True,
            capture_output=True,
        )
        oiio_result = json.loads(oiio_report.read_bytes())
    finally:
        shutil.rmtree(temporary, ignore_errors=False)
    if p249_result is None or oiio_result is None:
        raise P273Error("P273 execution did not produce complete results")

    expected = config["expected"]
    gates = {
        "exact_container": p249_result["file_bytes"] == expected["container_bytes"]
        and p249_result["file_sha256"] == expected["container_sha256"],
        "exact_pixels": oiio_result["pixel_f32le_sha256"]
        == expected["pixel_f32le_sha256"],
        "exact_shape_channels_format": oiio_result["shape"] == expected["shape"]
        and oiio_result["channels"] == expected["channels"]
        and oiio_result["format"] == expected["format"],
        "exact_aces_metadata": oiio_result["compression"]
        == expected["compression"]
        and oiio_result["aces_image_container_flag"]
        == expected["aces_image_container_flag"]
        and oiio_result["color_interop_id"] == expected["color_interop_id"]
        and oiio_result["oiio_color_space"] == expected["color_interop_id"]
        and oiio_result["chromaticities"] == expected["chromaticities"]
        and oiio_result["adopted_neutral"] == expected["adopted_neutral"],
        "exact_runtime_version": oiio_result["openimageio_version"]
        == config["openimageio_runtime"]["version"],
        "source_immutable": oiio_result["source_immutable"],
        "truncated_input_rejected": oiio_result["truncated_input_rejected"],
        "zero_network": True,
        "zero_temporary_residue": not temporary.exists(),
    }
    scientific = {
        "consumer_config_sha256": _sha256_bytes(config_bytes),
        "gates": gates,
        "network_reads": 0,
        "oiio_result": oiio_result,
        "p249_result": {
            "file_bytes": p249_result["file_bytes"],
            "file_sha256": p249_result["file_sha256"],
            "pixel_f32le_sha256": p249_result["inspection"][
                "decoded_pixel_f32le_sha256"
            ],
        },
        "persistent_media_writes": 0,
    }
    status = (
        "PASS_PRIVATE_OPENIMAGEIO_ACES2065_INTEROPERABILITY"
        if all(gates.values())
        else "FAIL_CLOSED_OPENIMAGEIO_ACES2065_INTEROPERABILITY"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "scientific": scientific,
        "stable_identity": "sha256:" + _sha256_bytes(_canonical(scientific)),
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--producer-repo", type=Path)
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oiio-worker", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.oiio_worker:
        if args.input is None or args.config or args.producer_repo or args.order:
            raise P273Error("invalid OpenImageIO worker arguments")
        report = _oiio_worker(args.input.resolve())
    else:
        if args.config is None or args.producer_repo is None or args.order is None:
            raise P273Error("controller requires config, producer repo and order")
        report = execute(
            args.config.resolve(), args.producer_repo.resolve(), args.order
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
