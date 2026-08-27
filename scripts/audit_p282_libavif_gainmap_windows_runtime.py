#!/usr/bin/env python3
"""Audit the official libavif Windows gain-map runtime on P278 fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p282-libavif-gainmap-windows-runtime-result.v1"
_ALTERNATE_HEADROOM = re.compile(
    r"Alternate headroom:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def parse_alternate_headroom(output: str) -> float:
    match = _ALTERNATE_HEADROOM.search(output)
    if match is None:
        raise ValueError("alternate headroom absent")
    value = float(match.group(1))
    if not np.isfinite(value):
        raise ValueError("alternate headroom nonfinite")
    return value


def build_info_command(executable: Path, input_path: Path) -> list[str]:
    return [str(executable), "--info", str(input_path)]


def build_metadata_command(executable: Path, input_path: Path) -> list[str]:
    return [str(executable), "printmetadata", str(input_path), "--jobs", "1"]


def build_base_command(
    executable: Path, input_path: Path, output_path: Path
) -> list[str]:
    return [
        str(executable),
        "--jobs",
        "1",
        "--depth",
        "16",
        "--png-compress",
        "0",
        str(input_path),
        str(output_path),
    ]


def build_tonemap_command(
    executable: Path,
    input_path: Path,
    output_path: Path,
    alternate_headroom: float,
) -> list[str]:
    return [
        str(executable),
        "tonemap",
        str(input_path),
        str(output_path),
        "--headroom",
        format(alternate_headroom, ".17g"),
        "--jobs",
        "1",
        "--depth",
        "12",
        "--speed",
        "10",
    ]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def _decode_png(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"PNG decode failed: {path.name}")
    if value.ndim != 3 or value.shape[2] not in {3, 4}:
        raise ValueError(f"unexpected PNG layout: {value.shape}")
    if value.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise ValueError(f"unexpected PNG dtype: {value.dtype}")
    return np.ascontiguousarray(value)


def _validate_runtime(config: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    runtime = config["runtime"]
    root = ROOT / runtime["root"]
    manifest_path = ROOT / runtime["manifest"]
    if _sha256_file(manifest_path) != runtime["manifest_sha256"]:
        raise ValueError("runtime manifest SHA-256 mismatch")
    manifest = _load_object(manifest_path)
    expected = {row["name"]: row for row in runtime["files"]}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != set(expected) | {"SOURCE.json"}:
        raise ValueError("runtime retained-member set mismatch")
    for name, row in expected.items():
        path = root / name
        if path.stat().st_size != row["size"] or _sha256_file(path) != row["sha256"]:
            raise ValueError(f"runtime member identity mismatch: {name}")
    return root / "avifdec.exe", root / "avifgainmaputil.exe", manifest


def _validate_fixture_manifest(config: dict[str, Any]) -> dict[str, Any]:
    fixtures = config["fixtures"]
    manifest_path = ROOT / fixtures["manifest"]
    if _sha256_file(manifest_path) != fixtures["manifest_sha256"]:
        raise ValueError("fixture manifest SHA-256 mismatch")
    manifest = _load_object(manifest_path)
    rows = {row["name"]: row for row in manifest["files"]}
    expected = {row["name"]: row for row in fixtures["valid"] + fixtures["invalid"]}
    for name, row in expected.items():
        source = ROOT / fixtures["root"] / name
        if _sha256_file(source) != row["sha256"]:
            raise ValueError(f"fixture SHA-256 mismatch: {name}")
        if rows[name]["sha256"] != row["sha256"]:
            raise ValueError(f"fixture manifest/config mismatch: {name}")
    return manifest


def _valid_record(
    row: dict[str, Any],
    source: Path,
    scratch: Path,
    avifdec: Path,
    gainmaputil: Path,
) -> dict[str, Any]:
    info = _run(build_info_command(avifdec, source))
    metadata = _run(build_metadata_command(gainmaputil, source))
    metadata_text = metadata.stdout + "\n" + metadata.stderr
    alternate_headroom: float | None = None
    if metadata.returncode == 0:
        alternate_headroom = parse_alternate_headroom(metadata_text)
    base_path = scratch / f"{row['name']}.base.png"
    tone_path = scratch / f"{row['name']}.tone.png"
    base = _run(build_base_command(avifdec, source, base_path))
    tone = (
        _run(build_tonemap_command(gainmaputil, source, tone_path, alternate_headroom))
        if alternate_headroom is not None
        else None
    )
    base_exists = base_path.is_file()
    tone_exists = tone_path.is_file()
    base_pixels = _decode_png(base_path) if base_exists else None
    tone_pixels = _decode_png(tone_path) if tone_exists else None
    comparable = (
        base_pixels is not None
        and tone_pixels is not None
        and base_pixels.shape == tone_pixels.shape
    )
    differing = int(np.count_nonzero(base_pixels != tone_pixels)) if comparable else 0
    maximum_difference = (
        int(np.max(np.abs(base_pixels.astype(np.int32) - tone_pixels.astype(np.int32))))
        if comparable
        else 0
    )
    expected_shape = (int(row["height"]), int(row["width"]))
    dimensions_exact = bool(
        comparable
        and base_pixels is not None
        and base_pixels.shape[:2] == expected_shape
    )
    return {
        "alternate_headroom": alternate_headroom,
        "base_decode_returncode": base.returncode,
        "base_png_sha256": _sha256_file(base_path) if base_exists else "",
        "base_pixel_sha256": _sha256_bytes(base_pixels.tobytes())
        if base_pixels is not None
        else "",
        "dimensions_exact": dimensions_exact,
        "gain_map_metadata_exposed": metadata.returncode == 0
        and "Gain Map" in metadata_text,
        "info_returncode": info.returncode,
        "info_gain_map_exposed": info.returncode == 0
        and "Gain map" in (info.stdout + "\n" + info.stderr),
        "maximum_rgb_code_difference": maximum_difference,
        "metadata_returncode": metadata.returncode,
        "name": row["name"],
        "nonzero_rgb_difference_components": differing,
        "role": "valid",
        "source_sha256": row["sha256"],
        "tone_map_returncode": tone.returncode if tone is not None else -1,
        "tone_png_sha256": _sha256_file(tone_path) if tone_exists else "",
        "tone_pixel_sha256": _sha256_bytes(tone_pixels.tobytes())
        if tone_pixels is not None
        else "",
    }


def _invalid_record(
    row: dict[str, Any],
    source: Path,
    scratch: Path,
    gainmaputil: Path,
) -> dict[str, Any]:
    metadata = _run(build_metadata_command(gainmaputil, source))
    tone_path = scratch / f"{row['name']}.invalid-tone.png"
    tone = _run(build_tonemap_command(gainmaputil, source, tone_path, 1.0))
    return {
        "atomic_gain_map_rejection": metadata.returncode != 0
        and tone.returncode != 0
        and not tone_path.exists(),
        "metadata_returncode": metadata.returncode,
        "name": row["name"],
        "reason": row["reason"],
        "role": "invalid",
        "source_sha256": row["sha256"],
        "tone_map_returncode": tone.returncode,
        "tone_output_exists": tone_path.exists(),
    }


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    avifdec, gainmaputil, runtime_manifest = _validate_runtime(config)
    fixture_manifest = _validate_fixture_manifest(config)
    fixture_root = ROOT / config["fixtures"]["root"]
    roles = [("valid", row) for row in config["fixtures"]["valid"]] + [
        ("invalid", row) for row in config["fixtures"]["invalid"]
    ]
    if reverse:
        roles.reverse()
    source_before = {
        row["name"]: _sha256_file(fixture_root / row["name"]) for _, row in roles
    }
    runtime_before = {path.name: _sha256_file(path) for path in (avifdec, gainmaputil)}
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p282_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for role, row in roles:
            source = fixture_root / row["name"]
            if role == "valid":
                records.append(
                    _valid_record(row, source, scratch, avifdec, gainmaputil)
                )
            else:
                records.append(_invalid_record(row, source, scratch, gainmaputil))

        truncated = scratch / "truncated.avif"
        truncated.write_bytes(
            (fixture_root / config["fixtures"]["valid"][0]["name"]).read_bytes()[:64]
        )
        truncated_output = scratch / "truncated.png"
        truncated_result = _run(
            build_tonemap_command(gainmaputil, truncated, truncated_output, 1.0)
        )
        truncation_atomic = (
            truncated_result.returncode != 0 and not truncated_output.exists()
        )
        missing_output = scratch / "missing.png"
        try:
            _run(
                build_tonemap_command(
                    scratch / "missing-avifgainmaputil.exe",
                    fixture_root / config["fixtures"]["valid"][0]["name"],
                    missing_output,
                    1.0,
                )
            )
            missing_runtime_atomic = False
        except FileNotFoundError:
            missing_runtime_atomic = not missing_output.exists()

    records.sort(key=lambda value: value["name"])
    source_after = {name: _sha256_file(fixture_root / name) for name in source_before}
    runtime_after = {path.name: _sha256_file(path) for path in (avifdec, gainmaputil)}
    valid = [row for row in records if row["role"] == "valid"]
    invalid = [row for row in records if row["role"] == "invalid"]
    gates = {
        "all_invalid_gain_maps_reject_atomically": len(invalid)
        == int(config["gates"]["required_invalid_count"])
        and all(row["atomic_gain_map_rejection"] for row in invalid),
        "all_valid_base_decode": len(valid)
        == int(config["gates"]["required_valid_count"])
        and all(row["base_decode_returncode"] == 0 for row in valid),
        "all_valid_dimensions_exact": all(row["dimensions_exact"] for row in valid),
        "all_valid_gain_map_metadata": all(
            row["gain_map_metadata_exposed"] and row["info_gain_map_exposed"]
            for row in valid
        ),
        "all_valid_tone_map": all(row["tone_map_returncode"] == 0 for row in valid),
        "all_valid_tone_map_material": all(
            row["nonzero_rgb_difference_components"] > 0 for row in valid
        ),
        "missing_runtime_atomic": missing_runtime_atomic,
        "runtime_immutable": runtime_before == runtime_after,
        "source_immutable": source_before == source_after,
        "truncation_atomic": truncation_atomic,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha256_file(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "records": records,
        "runtime": {
            "asset_sha256": config["runtime"]["asset_sha256"],
            "manifest_sha256": config["runtime"]["manifest_sha256"],
            "retained_file_count": len(runtime_manifest["files"]),
            "runtime_file_sha256": runtime_before,
            "tag": config["runtime"]["tag"],
        },
        "schema": REPORT_SCHEMA,
        "source": {
            "fixture_manifest_sha256": config["fixtures"]["manifest_sha256"],
            "retained_file_count": len(fixture_manifest["files"]),
            "runtime_network_requests": 0,
        },
        "status": "PASS_PRIVATE_LIBAVIF_GAINMAP_WINDOWS_RUNTIME"
        if passed
        else "FAIL_CLOSED_LIBAVIF_GAINMAP_WINDOWS_RUNTIME",
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p282_libavif_gainmap_windows_runtime_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(config, reverse=args.reverse)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": _sha256_bytes(payload),
                "stable_evidence_id": report["stable_evidence_id"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
