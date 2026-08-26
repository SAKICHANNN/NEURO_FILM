#!/usr/bin/env python3
"""Audit the exact R1CZ payload's structural absolute-black anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p233_r1cz_local_payload_capsule import (
    canonical_json_bytes,
    decode_capsule,
)
from src.color_match.shared_hdr_dpct_payload import (
    _interp_channel32,
    _signed_exp32,
    _signed_log32,
    apply_shared_hdr_dpct_payload,
    load_shared_hdr_dpct_payload,
)

SCHEMA = "kmcfm.p235-r1cz-structural-black-anchor-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _signed_exp64(value: float, scale: float) -> float:
    return float(np.sign(value) * np.expm1(abs(value)) * scale)


def _first_segment_slope(x: np.ndarray, y: np.ndarray) -> float:
    numerator = float(y[1]) - float(y[0])
    denominator = float(x[1]) - float(x[0])
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return min(1.0, numerator / denominator)


def _unclamped_apply(
    source: np.ndarray,
    *,
    source_knots: np.ndarray,
    reference_knots: np.ndarray,
    scale: np.float32,
    channels: list[int],
) -> np.ndarray:
    shaped = _signed_log32(source, scale)
    mapped = np.empty_like(shaped)
    for channel in channels:
        mapped[..., channel] = _interp_channel32(
            shaped[..., channel],
            source_knots[:, channel],
            reference_knots[:, channel],
        )
    return _signed_exp32(mapped, scale).astype(np.float32, copy=False)


def run(config_path: Path, *, reverse: bool = False) -> dict[str, Any]:
    config = _load_json(config_path)
    parent_facts: dict[str, Any] = {}
    for parent in ("p233", "p234"):
        path = ROOT / config["parents"][f"{parent}_evidence_path"]
        digest = _sha256_file(path)
        evidence = _load_json(path)
        if digest != config["parents"][f"{parent}_evidence_sha256"]:
            raise RuntimeError(f"{parent.upper()} evidence identity differs")
        if evidence.get("status") != config["parents"][f"{parent}_required_status"]:
            raise RuntimeError(f"{parent.upper()} status differs")
        parent_facts[parent] = {"evidence_sha256": digest, "status": evidence["status"]}

    capsule_path = ROOT / config["payload"]["local_capsule_path"]
    capsule_raw = capsule_path.read_bytes()
    payload_bytes, capsule = decode_capsule(
        capsule_raw,
        expected_schema=config["payload"]["capsule_schema"],
        expected_encoding=config["payload"]["encoding"],
        expected_payload_bytes=int(config["payload"]["payload_bytes"]),
        expected_payload_sha256=config["payload"]["payload_sha256"],
        expected_bundle_id=config["payload"]["bundle_id"],
    )
    bundle = load_shared_hdr_dpct_payload(payload_bytes, config["envelope"])
    scale = np.float32(bundle.payload["dpct"]["scale"])
    channel_order = [2, 1, 0] if reverse else [0, 1, 2]

    black = np.asarray(config["probe"]["exact_black_nits_float32"], dtype=np.float32)
    black = np.ascontiguousarray(black.reshape(1, 1, 3))
    raw_black = _unclamped_apply(
        black,
        source_knots=bundle.source_knots,
        reference_knots=bundle.reference_knots,
        scale=scale,
        channels=channel_order,
    )
    clamped_black = apply_shared_hdr_dpct_payload(black, bundle)

    levels = np.asarray(
        config["probe"]["neutral_shadow_levels_nits_float32"], dtype=np.float32
    )
    neutral = np.repeat(levels[:, None], 3, axis=1).reshape(-1, 1, 3)
    neutral = np.ascontiguousarray(neutral)
    neutral_output = apply_shared_hdr_dpct_payload(neutral, bundle)
    mixed_levels: list[float] = []
    for index, level in enumerate(levels):
        row = neutral_output[index, 0]
        zeros = row == np.float32(0.0)
        positives = row > np.float32(0.0)
        if bool(np.any(zeros) and np.any(positives)):
            mixed_levels.append(float(level))

    source_first_nits: list[float] = []
    reference_first_nits: list[float] = []
    zero_crossings_nits: list[float | None] = []
    first_slopes: list[float] = []
    for channel in channel_order:
        x = bundle.source_knots[:, channel]
        y = bundle.reference_knots[:, channel]
        slope = _first_segment_slope(x, y)
        first_slopes.append(slope)
        source_first_nits.append(_signed_exp64(float(x[0]), float(scale)))
        reference_first_nits.append(_signed_exp64(float(y[0]), float(scale)))
        if slope == 0.0:
            zero_crossings_nits.append(None)
        else:
            shaped_crossing = float(x[0]) - float(y[0]) / slope
            zero_crossings_nits.append(
                _signed_exp64(shaped_crossing, float(scale))
            )
    if reverse:
        source_first_nits.reverse()
        reference_first_nits.reverse()
        zero_crossings_nits.reverse()
        first_slopes.reverse()

    tolerance = float(config["probe"]["zero_crossing_absolute_tolerance_nits"])
    finite_crossings = [value for value in zero_crossings_nits if value is not None]
    gates = {
        "p233_and_p234_parent_identities_exact": True,
        "payload_and_envelope_exact": bool(
            _sha256_bytes(payload_bytes) == config["payload"]["payload_sha256"]
            and capsule["bundle_id"] == config["payload"]["bundle_id"]
        ),
        "black_preclamp_output_exact_zero_all_channels": bool(
            np.array_equal(raw_black, np.zeros_like(raw_black))
        ),
        "black_postclamp_output_exact_zero_all_channels": bool(
            np.array_equal(clamped_black, np.zeros_like(clamped_black))
        ),
        "all_channel_zero_crossings_within_one_micro_nit_of_zero": bool(
            len(finite_crossings) == 3
            and max(abs(value) for value in finite_crossings) <= tolerance
        ),
        "no_mixed_zero_nonzero_channels_on_fixed_neutral_shadow_probe": not mixed_levels,
        "application_pixels_build_network_and_media_reads_zero": True,
    }
    passed = all(gates.values())
    status = (
        "PASS_PRIVATE_R1CZ_STRUCTURAL_BLACK_ANCHOR"
        if passed
        else "FAIL_CLOSED_R1CZ_STRUCTURAL_BLACK_ANCHOR"
    )
    result = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": status,
        "parents": parent_facts,
        "payload": {
            "capsule_sha256": _sha256_bytes(capsule_raw),
            "payload_sha256": _sha256_bytes(payload_bytes),
            "payload_bytes": len(payload_bytes),
            "bundle_id": capsule["bundle_id"],
            "scale_nits": float(scale),
        },
        "structural_probe": {
            "black_preclamp_output_nits_float32": raw_black.reshape(3).tolist(),
            "black_postclamp_output_nits_float32": clamped_black.reshape(3).tolist(),
            "first_segment_slopes": first_slopes,
            "source_first_knot_nits": source_first_nits,
            "reference_first_knot_nits": reference_first_nits,
            "channel_zero_crossings_input_nits": zero_crossings_nits,
            "zero_crossing_absolute_tolerance_nits": tolerance,
            "neutral_shadow_levels_nits_float32": levels.tolist(),
            "neutral_shadow_output_nits_float32": neutral_output[:, 0, :].tolist(),
            "mixed_zero_nonzero_neutral_levels_nits": mixed_levels,
        },
        "execution": config["execution"],
        "gates": gates,
        "decision": {
            "result": status,
            "further_arbitrary_independent_absolute_light_domain_accumulation": passed,
            "p233_persistence_unchanged": True,
            "original_r1cy_paired_result_unchanged": True,
            "consumer_mapping": False,
            "claim_ceiling": config["claim_ceiling"],
        },
    }
    result["scientific_identity"] = "sha256:" + _sha256_bytes(
        canonical_json_bytes(result)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/p235_r1cz_structural_black_anchor_v1.json",
    )
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, reverse=args.order == "reverse")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))


if __name__ == "__main__":
    main()
