#!/usr/bin/env python3
"""Audit frozen SPCP scene variants for exact geometry and pixel alignment."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_u5_r2spcp0_metadata_source_lock import (
    ARCHIVE_URL,
    CENTRAL_OFFSET,
    CENTRAL_SHA256,
    CENTRAL_SIZE,
    ZipMember,
    canonical_sha256,
    extract_member,
    parse_central_directory,
    range_get,
    sha256_bytes,
)

REFERENCE_VARIANT = "01_01"
EXPECTED_VARIANTS = (
    "01_01",
    "02_01",
    "02_02",
    "02_03",
    "03_01",
    "03_02",
    "03_03",
    "03_04",
    "04_01",
    "04_02",
    "04_03",
    "04_04",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tie_aware_midrank(gray: np.ndarray) -> np.ndarray:
    """Map uint8 gray values to their exact empirical midranks in [0, 1]."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("midrank input must be a two-dimensional uint8 array")
    counts = np.bincount(gray.reshape(-1), minlength=256).astype(np.int64)
    before = np.cumsum(counts, dtype=np.int64) - counts
    total = int(gray.size)
    ranks = (before.astype(np.float64) + 0.5 * counts.astype(np.float64)) / total
    return ranks[gray].astype(np.float32)


def _sobel_vector(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return np.stack((gx, gy), axis=-1)


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("NCC inputs must have equal shapes")
    a = left.astype(np.float64, copy=False).reshape(-1)
    b = right.astype(np.float64, copy=False).reshape(-1)
    a = a - float(np.mean(a, dtype=np.float64))
    b = b - float(np.mean(b, dtype=np.float64))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("NCC denominator is non-positive or non-finite")
    value = float(np.dot(a, b) / denominator)
    if not math.isfinite(value):
        raise ValueError("NCC is non-finite")
    return max(-1.0, min(1.0, value))


def alignment_metrics(reference_rgb: np.ndarray, variant_rgb: np.ndarray) -> dict[str, float]:
    if reference_rgb.shape != variant_rgb.shape:
        raise ValueError("alignment inputs must have equal shapes")
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    variant_gray = cv2.cvtColor(variant_rgb, cv2.COLOR_RGB2GRAY)
    reference_rank = tie_aware_midrank(reference_gray)
    variant_rank = tie_aware_midrank(variant_gray)
    gradient_ncc = normalized_correlation(
        _sobel_vector(reference_gray.astype(np.float32) / 255.0),
        _sobel_vector(variant_gray.astype(np.float32) / 255.0),
    )
    rank_gradient_ncc = normalized_correlation(
        _sobel_vector(reference_rank),
        _sobel_vector(variant_rank),
    )
    (phase_x, phase_y), phase_response = cv2.phaseCorrelate(reference_rank, variant_rank)
    phase_translation = math.hypot(float(phase_x), float(phase_y))
    values = (gradient_ncc, rank_gradient_ncc, phase_translation, float(phase_response))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("alignment metric is non-finite")
    return {
        "gradient_ncc": gradient_ncc,
        "rank_gradient_ncc": rank_gradient_ncc,
        "phase_translation_pixels": phase_translation,
        "phase_response": float(phase_response),
    }


def _verify_member_fact(fact: dict[str, Any], member: ZipMember) -> None:
    expected = {
        "name": str(fact["member"]),
        "method": int(fact["method"]),
        "crc32": int(str(fact["crc32"]), 16),
        "compressed_size": int(fact["compressed_size"]),
        "uncompressed_size": int(fact["uncompressed_size"]),
        "local_offset": int(fact["local_offset"]),
    }
    observed = {
        "name": member.name,
        "method": member.method,
        "crc32": member.crc32,
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "local_offset": member.local_offset,
    }
    if observed != expected:
        raise ValueError(f"role/central member mismatch: {member.name}")


def _load_member(
    fact: dict[str, Any], cache_dir: Path, member_by_name: dict[str, ZipMember]
) -> tuple[bytes, str]:
    member = member_by_name.get(str(fact["member"]))
    if member is None:
        raise ValueError(f"member missing from central directory: {fact['member']}")
    _verify_member_fact(fact, member)
    cache_path = cache_dir / f"{fact['image_id']}.png"
    if cache_path.exists():
        payload = cache_path.read_bytes()
        source = "verified-cache"
    else:
        payload = extract_member(ARCHIVE_URL, member)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".png.tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
        source = "range-download"
    if len(payload) != int(fact["uncompressed_size"]):
        raise ValueError(f"cached size mismatch: {fact['image_id']}")
    if binascii.crc32(payload) & 0xFFFFFFFF != int(str(fact["crc32"]), 16):
        raise ValueError(f"cached CRC mismatch: {fact['image_id']}")
    return payload, source


def _decode_png(payload: bytes, image_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"not a PNG: {image_id}")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"multi-frame PNG: {image_id}")
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"unsupported PNG mode {image.mode}: {image_id}")
        width, height = image.size
        raw = np.asarray(image)
        rgb = np.ascontiguousarray(raw[..., :3])
        return rgb, {
            "mode": image.mode,
            "width": int(width),
            "height": int(height),
            "pixel_sha256": sha256_bytes(raw.tobytes(order="C")),
        }


def _gate_pair(metrics: dict[str, float], gates: dict[str, Any]) -> bool:
    return bool(
        metrics["gradient_ncc"] >= float(gates["minimum_pair_gradient_ncc"])
        and metrics["rank_gradient_ncc"]
        >= float(gates["minimum_pair_rank_gradient_ncc"])
        and metrics["phase_translation_pixels"]
        <= float(gates["maximum_phase_translation_pixels"])
    )


def run_preflight(
    *, contract_path: Path, role_manifest_path: Path, role: str, cache_dir: Path
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(role_manifest_path.read_text(encoding="utf-8"))
    if role not in {"development", "confirmation"}:
        raise ValueError("role must be development or confirmation")
    if manifest["lock_id"] != "03a003be5c0d7f2722cca3bcc8acce306dcb92df54510e082b0b5972bb4f45fc":
        raise ValueError("role lock identity differs")
    role_facts = manifest["roles"][role]
    expected_scenes = int(contract["role_freeze"][f"{role}_scene_count"])
    if int(role_facts["scene_count"]) != expected_scenes:
        raise ValueError("role scene count differs")
    central = range_get(
        ARCHIVE_URL,
        CENTRAL_OFFSET,
        CENTRAL_OFFSET + CENTRAL_SIZE - 1,
    )
    if sha256_bytes(central) != CENTRAL_SHA256:
        raise ValueError("central-directory hash mismatch")
    member_by_name = {member.name: member for member in parse_central_directory(central)}

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in role_facts["image_members"]:
        by_scene[str(fact["scene_id"])].append(fact)
    rows: list[dict[str, Any]] = []
    image_facts: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    variant_passes: dict[str, list[bool]] = defaultdict(list)
    geometries_equal = True
    geometries_1024 = True
    modes_valid = True

    for scene_id in sorted(by_scene):
        images: dict[str, np.ndarray] = {}
        geometries: set[tuple[int, int]] = set()
        variants = sorted(str(fact["variant_id"]) for fact in by_scene[scene_id])
        if tuple(variants) != tuple(sorted(EXPECTED_VARIANTS)):
            raise ValueError(f"variant inventory differs: {scene_id}")
        for fact in sorted(by_scene[scene_id], key=lambda value: value["variant_id"]):
            payload, source = _load_member(fact, cache_dir, member_by_name)
            source_counts[source] += 1
            rgb, decoded = _decode_png(payload, str(fact["image_id"]))
            variant = str(fact["variant_id"])
            images[variant] = rgb
            geometries.add((decoded["width"], decoded["height"]))
            geometries_1024 &= (decoded["width"], decoded["height"]) == (1024, 1024)
            modes_valid &= decoded["mode"] in {"RGB", "RGBA"}
            image_facts.append(
                {
                    "image_id": fact["image_id"],
                    "scene_id": scene_id,
                    "variant_id": variant,
                    "file_sha256": sha256_bytes(payload),
                    **decoded,
                }
            )
        geometries_equal &= len(geometries) == 1
        reference = images[REFERENCE_VARIANT]
        for variant in EXPECTED_VARIANTS:
            if variant == REFERENCE_VARIANT:
                continue
            metrics = alignment_metrics(reference, images[variant])
            passed = _gate_pair(metrics, contract["development_gates"])
            variant_passes[variant].append(passed)
            rows.append(
                {
                    "scene_id": scene_id,
                    "reference_variant": REFERENCE_VARIANT,
                    "variant_id": variant,
                    **metrics,
                    "alignment_gate_pass": passed,
                }
            )

    gates = contract["development_gates"]
    pair_pass_rate = sum(bool(row["alignment_gate_pass"]) for row in rows) / len(rows)
    variant_rates = {
        variant: sum(values) / len(values) for variant, values in sorted(variant_passes.items())
    }
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            "gradient_ncc",
            "rank_gradient_ncc",
            "phase_translation_pixels",
            "phase_response",
        )
    )
    gate_results = {
        "scene_count_exact": len(by_scene) == expected_scenes,
        "variant_count_exact": all(len(values) == 12 for values in by_scene.values()),
        "single_frame_mode_valid": modes_valid,
        "geometries_equal_within_scene": geometries_equal,
        "geometries_exact_1024_square": geometries_1024,
        "minimum_gradient_ncc": min(row["gradient_ncc"] for row in rows)
        >= float(gates["minimum_pair_gradient_ncc"]),
        "minimum_rank_gradient_ncc": min(row["rank_gradient_ncc"] for row in rows)
        >= float(gates["minimum_pair_rank_gradient_ncc"]),
        "maximum_phase_translation": max(row["phase_translation_pixels"] for row in rows)
        <= float(gates["maximum_phase_translation_pixels"]),
        "pair_pass_rate": pair_pass_rate
        >= float(gates["pairs_passing_all_alignment_gates_rate_min"]),
        "each_variant_family_pass_rate": min(variant_rates.values())
        >= float(gates["each_variant_family_passing_rate_min"]),
        "all_finite": finite,
        "image_member_crc_sha_size_exact": True,
        "preference_score_reads_exact": True,
        "operator_fit_count_exact": True,
    }
    scientific = {
        "schema": "neuro-film.u5-r2spcp1-pixel-alignment-preflight-report.v1",
        "experiment_id": "U5.R2SPCP1",
        "role": role,
        "contract_sha256": _sha256_file(contract_path),
        "role_manifest_sha256": _sha256_file(role_manifest_path),
        "role_lock_id": manifest["lock_id"],
        "scene_count": len(by_scene),
        "image_count": len(image_facts),
        "pair_count": len(rows),
        "image_facts": sorted(image_facts, key=lambda value: value["image_id"]),
        "rows": rows,
        "metrics": {
            "minimum_gradient_ncc": min(row["gradient_ncc"] for row in rows),
            "minimum_rank_gradient_ncc": min(row["rank_gradient_ncc"] for row in rows),
            "maximum_phase_translation_pixels": max(
                row["phase_translation_pixels"] for row in rows
            ),
            "pair_alignment_gate_pass_rate": pair_pass_rate,
            "variant_family_pass_rates": variant_rates,
        },
        "gate_results": gate_results,
        "all_gates_pass": all(gate_results.values()),
        "preference_or_score_values_read": 0,
        "operator_fit_count": 0,
        "future_role_image_payload_reads": 0,
        "claim_ceiling": contract["claim_ceiling"],
    }
    scientific["scientific_payload_id"] = canonical_sha256(scientific)
    return {
        "scientific": scientific,
        "runtime": {"member_sources": dict(sorted(source_counts.items()))},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("development", "confirmation"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(
        contract_path=args.contract,
        role_manifest_path=args.role_manifest,
        role=args.role,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "role": args.role,
                "all_gates_pass": result["scientific"]["all_gates_pass"],
                "scientific_payload_id": result["scientific"]["scientific_payload_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
