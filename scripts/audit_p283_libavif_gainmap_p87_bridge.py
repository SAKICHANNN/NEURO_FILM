#!/usr/bin/env python3
"""Audit the private libavif gain-map higher-rendition to P87 bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.audit_p282_libavif_gainmap_windows_runtime import (
    build_base_command,
    build_info_command,
    build_tonemap_command,
)
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from src.color_match.libavif_gainmap_ingress import (
    LIBAVIF_GAINMAP_RENDER_BRIDGE_ID,
    prepare_libavif_gainmap_match_view_v1,
)
from src.preprocess.color_management import linear_rgb_matrix
from src.preprocess.dng_metadata import canonical_json_bytes
from src.preprocess.rec2100_pq_transfer import absolute_rec2020_cdm2_to_pq

REPORT_SCHEMA = "neuro-film.p283-libavif-gainmap-p87-bridge-result.v1"


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


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def parse_chosen_cicp(output: str, role: str) -> tuple[int, int, int]:
    if role not in {"base", "alternate"}:
        raise ValueError("role must be base or alternate")
    if " * Alternate image:" not in output:
        raise ValueError("alternate image section absent")
    base, alternate = output.split(" * Alternate image:", 1)
    section = base if role == "base" else alternate

    def value(label: str) -> int:
        prefix = f"* {label}:"
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return int(stripped.split(":", 1)[1].strip())
        raise ValueError(f"{label} absent")

    return (
        value("Color Primaries"),
        value("Transfer Char. "),
        value("Matrix Coeffs. "),
    )


def _decode_rgb16(path: Path) -> tuple[np.ndarray, str]:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.shape != (300, 400, 3):
        raise ValueError("official libavif PNG must decode as 400x300 uint16 BGR")
    bgr = np.ascontiguousarray(decoded)
    bgr_sha = _sha256_bytes(bgr.tobytes())
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return rgb, bgr_sha


def roundtrip_rgb16(absolute_rec2020: np.ndarray) -> np.ndarray:
    if absolute_rec2020.dtype != np.float32:
        raise TypeError("roundtrip input must be float32")
    matrix = linear_rgb_matrix("linear_rec2020", "linear_srgb")
    absolute_srgb = np.matmul(absolute_rec2020.astype(np.float64), matrix.T)
    encoded = absolute_rec2020_cdm2_to_pq(absolute_srgb)
    return np.ascontiguousarray(np.floor(encoded * 65535.0 + 0.5).astype(np.uint16))


def _invalid_controls(source_bytes: bytes, source_sha: str) -> dict[str, bool]:
    samples = np.zeros((1, 1, 3), dtype=np.uint16)
    base: dict[str, Any] = {
        "source_asset": source_bytes,
        "expected_source_sha256": source_sha,
        "encoded_rgb16": samples,
        "decoder_version": "1.4.2",
        "source_color_primaries": 1,
        "source_transfer_characteristics": 16,
        "source_full_range": True,
        "higher_rendition_role": "base",
    }
    controls: dict[str, dict[str, Any]] = {
        "wrong_source_hash": {"expected_source_sha256": "0" * 64},
        "wrong_decoder": {"decoder_version": "1.4.1"},
        "wrong_primaries": {"source_color_primaries": 9},
        "wrong_transfer": {"source_transfer_characteristics": 13},
        "wrong_range": {"source_full_range": False},
        "wrong_role": {"higher_rendition_role": "lower"},
        "wrong_dtype": {"encoded_rgb16": samples.astype(np.float32)},
        "wrong_shape": {"encoded_rgb16": samples[:, :, :2]},
    }
    results: dict[str, bool] = {}
    for name, override in controls.items():
        arguments = dict(base)
        arguments.update(override)
        try:
            prepare_libavif_gainmap_match_view_v1(**arguments)
        except ReferenceMatchContractError:
            results[name] = True
        else:
            results[name] = False
    return results


def _validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for key in ("p282", "p87"):
        path = ROOT / config["bindings"][f"{key}_evidence"]
        expected = config["bindings"][f"{key}_evidence_sha256"]
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(f"{key.upper()} evidence SHA-256 mismatch")
        bindings[f"{key}_evidence_sha256"] = actual
    p282 = _load_object(ROOT / config["bindings"]["p282_evidence"])
    p87 = _load_object(ROOT / config["bindings"]["p87_evidence"])
    if p282["status"] != "PASS_PRIVATE_LIBAVIF_GAINMAP_WINDOWS_RUNTIME":
        raise ValueError("P282 parent status mismatch")
    if p87["status"] != "PASS_PRIVATE_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_INGRESS":
        raise ValueError("P87 parent status mismatch")
    return bindings


def run(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = _load_object(config_path)
    bindings = _validate_bindings(config)
    runtime_root = ROOT / config["runtime"]["root"]
    manifest = ROOT / config["runtime"]["manifest"]
    if _sha256_file(manifest) != config["runtime"]["manifest_sha256"]:
        raise ValueError("runtime manifest SHA-256 mismatch")
    avifdec = runtime_root / "avifdec.exe"
    gainmaputil = runtime_root / "avifgainmaputil.exe"
    if _sha256_file(avifdec) != config["runtime"]["avifdec_sha256"]:
        raise ValueError("avifdec identity mismatch")
    if _sha256_file(gainmaputil) != config["runtime"]["avifgainmaputil_sha256"]:
        raise ValueError("avifgainmaputil identity mismatch")
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    source_root = ROOT / config["source_root"]
    source_before = {
        row["name"]: _sha256_file(source_root / row["name"]) for row in rows
    }
    runtime_before = {
        "avifdec.exe": _sha256_file(avifdec),
        "avifgainmaputil.exe": _sha256_file(gainmaputil),
    }
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p283_", dir=ROOT / "tmp") as directory:
        scratch = Path(directory)
        for row in rows:
            source = source_root / row["name"]
            source_bytes = source.read_bytes()
            if _sha256_bytes(source_bytes) != row["source_sha256"]:
                raise ValueError(f"source identity mismatch: {row['name']}")
            info = _run(build_info_command(avifdec, source))
            if info.returncode != 0:
                raise RuntimeError(f"avifdec info failed: {row['name']}")
            cicp = parse_chosen_cicp(info.stdout, row["higher_role"])
            output = scratch / f"{row['name']}.higher.png"
            command = (
                build_base_command(avifdec, source, output)
                if row["higher_role"] == "base"
                else build_tonemap_command(
                    gainmaputil, source, output, float(row["alternate_headroom"])
                )
            )
            completed = _run(command)
            if completed.returncode != 0 or not output.is_file():
                raise RuntimeError(f"higher rendition failed: {row['name']}")
            rgb16, bgr_sha = _decode_rgb16(output)
            if bgr_sha != row["p282_bgr16_sample_sha256"]:
                raise ValueError(f"P282 sample identity mismatch: {row['name']}")
            input_before = rgb16.copy()
            prepared = prepare_libavif_gainmap_match_view_v1(
                source_asset=source_bytes,
                expected_source_sha256=row["source_sha256"],
                encoded_rgb16=rgb16,
                decoder_version=config["runtime"]["decoder_version"],
                source_color_primaries=cicp[0],
                source_transfer_characteristics=cicp[1],
                source_full_range=True,
                higher_rendition_role=row["higher_role"],
            )
            replay = roundtrip_rgb16(prepared.pixels)
            error = np.abs(replay.astype(np.int32) - rgb16.astype(np.int32))
            records.append(
                {
                    "bgr16_parent_sha256": bgr_sha,
                    "chosen_cicp": list(cicp),
                    "descriptor_profile_id": prepared.descriptor.profile_id,
                    "encoded_rgb16_sha256": prepared.encoded_sample_sha256,
                    "higher_role": row["higher_role"],
                    "ingress_id": prepared.ingress_id,
                    "input_unchanged": bool(np.array_equal(rgb16, input_before)),
                    "maximum_nits": float(np.max(prepared.pixels)),
                    "minimum_nits": float(np.min(prepared.pixels)),
                    "name": row["name"],
                    "output_contiguous": bool(prepared.pixels.flags.c_contiguous),
                    "output_owned": bool(prepared.pixels.flags.owndata),
                    "output_read_only": not prepared.pixels.flags.writeable,
                    "pixel_sha256": prepared.descriptor.pixel_sha256,
                    "reference_white_nits": prepared.descriptor.reference_white_nits,
                    "render_bridge_id": prepared.descriptor.render_bridge_id,
                    "roundtrip_maximum_rgb16_error": int(np.max(error)),
                    "roundtrip_median_rgb16_error": float(np.median(error)),
                    "shape": list(prepared.pixels.shape),
                    "source_sha256": row["source_sha256"],
                }
            )
        invalid_controls = _invalid_controls(
            (source_root / config["rows"][0]["name"]).read_bytes(),
            config["rows"][0]["source_sha256"],
        )

    records.sort(key=lambda value: value["name"])
    source_after = {name: _sha256_file(source_root / name) for name in source_before}
    runtime_after = {
        "avifdec.exe": _sha256_file(avifdec),
        "avifgainmaputil.exe": _sha256_file(gainmaputil),
    }
    gates = {
        "all_cicp_exact": all(row["chosen_cicp"] == [1, 16, 6] for row in records),
        "all_descriptors_exact": all(
            row["descriptor_profile_id"] == MATCH_PROFILE_ABSOLUTE_REC2020
            and row["reference_white_nits"] == 203.0
            and row["render_bridge_id"] == LIBAVIF_GAINMAP_RENDER_BRIDGE_ID
            for row in records
        ),
        "all_inputs_unchanged": all(row["input_unchanged"] for row in records),
        "all_outputs_owned_read_only": all(
            row["output_owned"] and row["output_contiguous"] and row["output_read_only"]
            for row in records
        ),
        "all_ranges_valid": all(
            row["minimum_nits"] >= 0.0
            and row["maximum_nits"] <= float(config["gates"]["maximum_maximum_nits"])
            and row["maximum_nits"] > float(config["gates"]["minimum_maximum_nits"])
            for row in records
        ),
        "all_roundtrips_within_bound": all(
            row["roundtrip_maximum_rgb16_error"]
            <= int(config["gates"]["maximum_rgb16_roundtrip_error"])
            and row["roundtrip_median_rgb16_error"]
            <= float(config["gates"]["maximum_median_rgb16_roundtrip_error"])
            for row in records
        ),
        "all_shapes_exact": len(records) == int(config["gates"]["required_row_count"])
        and all(row["shape"] == [300, 400, 3] for row in records),
        "invalid_controls_atomic": all(invalid_controls.values()),
        "runtime_immutable": runtime_before == runtime_after,
        "source_immutable": source_before == source_after,
    }
    passed = all(gates.values())
    report: dict[str, Any] = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha256_file(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "invalid_controls": invalid_controls,
        "records": records,
        "runtime": {
            "decoder_version": config["runtime"]["decoder_version"],
            "runtime_file_sha256": runtime_before,
            "runtime_network_requests": 0,
        },
        "schema": REPORT_SCHEMA,
        "status": "PASS_PRIVATE_LIBAVIF_GAINMAP_P87_BRIDGE"
        if passed
        else "FAIL_CLOSED_LIBAVIF_GAINMAP_P87_BRIDGE",
    }
    report["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p283_libavif_gainmap_p87_bridge_v1.json"),
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
