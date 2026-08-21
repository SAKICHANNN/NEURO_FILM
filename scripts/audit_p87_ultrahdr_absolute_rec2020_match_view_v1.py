#!/usr/bin/env python3
"""Formal P87 Ultra HDR decoded-payload MatchView ingress audit."""

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

from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.ultra_hdr_ingress import (
    ULTRAHDR_DECODER_VERSION,
    ULTRAHDR_EXTERNAL_PROFILE_ID,
    ULTRAHDR_MAXIMUM_RELATIVE_LINEAR,
    ULTRAHDR_REFERENCE_WHITE_NITS,
    prepare_ultrahdr_match_view_v1,
)
from src.preprocess.dng_metadata import canonical_json_bytes

REPORT_SCHEMA = "neuro-film.p87-ultrahdr-absolute-rec2020-result.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _validate_source_facts(
    config: dict[str, Any], producer_evidence_path: Path
) -> dict[str, Any]:
    producer = config["producer"]
    official = config["official_decoder"]
    consumer = config["consumer"]
    actual_evidence_sha256 = _sha256_file(producer_evidence_path)
    if actual_evidence_sha256 != producer["evidence_sha256"]:
        raise ValueError("producer evidence SHA-256 mismatch")
    evidence = _load_object(producer_evidence_path)
    expected = {
        "experiment_id": producer["experiment_id"],
        "producer_commit": producer["commit"],
        "status": "PASS_PRIVATE_ULTRAHDR_MATCH_VIEW_CANONICALIZATION",
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"producer evidence {key} mismatch")
    if evidence["official_source"]["commit"] != official["commit"]:
        raise ValueError("official decoder commit mismatch")
    if evidence["official_source"]["decoder_version"] != official["version"]:
        raise ValueError("official decoder version mismatch")
    if evidence["canonical_view"]["profile_id"] != producer["external_profile_id"]:
        raise ValueError("producer profile mismatch")
    if (
        evidence["canonical_view"]["reference_white_nits"]
        != consumer["reference_white_nits"]
    ):
        raise ValueError("producer reference white mismatch")
    return {
        "evidence_sha256": actual_evidence_sha256,
        "experiment_id": evidence["experiment_id"],
        "official_commit": evidence["official_source"]["commit"],
        "producer_commit": evidence["producer_commit"],
        "producer_stable_identity": evidence["formal_execution"]["stable_identity"],
    }


def _fixtures() -> list[tuple[str, np.ndarray]]:
    return [
        (
            "extrema",
            np.array(
                [[[0.0, 0.5, 1.0, 1.0], [49.25, 2.0, 8.0, 1.0]]],
                dtype="<f2",
            ),
        ),
        (
            "fractional",
            np.array(
                [[[0.125, 1.5, 3.25, 1.0], [16.0, 24.0, 32.0, 1.0]]],
                dtype="<f2",
            ),
        ),
        (
            "subnormal-and-black",
            np.array(
                [[[2**-24, 0.0, 2**-14, 1.0], [4.0, 1.0, 0.25, 1.0]]],
                dtype="<f2",
            ),
        ),
    ]


def _expected_pixel_sha256(rgba: np.ndarray) -> str:
    expected = rgba[..., :3].astype(np.float32)
    expected *= np.float32(ULTRAHDR_REFERENCE_WHITE_NITS)
    return _sha256_bytes(np.asarray(expected, dtype=">f4").tobytes(order="C"))


def _expect_rejection(**overrides: Any) -> str:
    rgba = np.ones((1, 1, 4), dtype="<f2")
    source = b"p87-negative-control"
    arguments: dict[str, Any] = {
        "source_asset": source,
        "expected_source_sha256": _sha256_bytes(source),
        "decoded_rgba16f": rgba.tobytes(order="C"),
        "width": 1,
        "height": 1,
        "decoder_version": ULTRAHDR_DECODER_VERSION,
        "producer_profile_id": ULTRAHDR_EXTERNAL_PROFILE_ID,
    }
    arguments.update(overrides)
    try:
        prepare_ultrahdr_match_view_v1(**arguments)
    except ReferenceMatchContractError as error:
        return str(error)
    raise AssertionError("negative control was accepted")


def _negative_controls() -> dict[str, str]:
    nonfinite = np.ones((1, 1, 4), dtype="<f2")
    nonfinite[0, 0, 0] = np.inf
    out_of_range = np.ones((1, 1, 4), dtype="<f2")
    out_of_range[0, 0, 0] = np.float16(ULTRAHDR_MAXIMUM_RELATIVE_LINEAR + 1.0)
    alpha = np.ones((1, 1, 4), dtype="<f2")
    alpha[0, 0, 3] = np.float16(0.5)
    return {
        "alpha": _expect_rejection(decoded_rgba16f=alpha.tobytes()),
        "length": _expect_rejection(decoded_rgba16f=b"\x00"),
        "nonfinite": _expect_rejection(decoded_rgba16f=nonfinite.tobytes()),
        "profile": _expect_rejection(producer_profile_id="unsupported"),
        "range": _expect_rejection(decoded_rgba16f=out_of_range.tobytes()),
        "source": _expect_rejection(expected_source_sha256="0" * 64),
        "version": _expect_rejection(decoder_version="2.0.1"),
    }


def run(
    config_path: Path,
    producer_evidence_path: Path,
    *,
    reverse: bool,
) -> dict[str, Any]:
    config = _load_object(config_path)
    source_facts = _validate_source_facts(config, producer_evidence_path)
    fixtures = _fixtures()
    if reverse:
        fixtures.reverse()
    records: list[dict[str, Any]] = []
    input_unchanged = True
    for label, rgba in fixtures:
        payload = rgba.tobytes(order="C")
        source = f"self-authored-p87-{label}".encode("ascii")
        source_before = bytes(source)
        payload_before = bytes(payload)
        prepared = prepare_ultrahdr_match_view_v1(
            source_asset=source,
            expected_source_sha256=_sha256_bytes(source),
            decoded_rgba16f=payload,
            width=rgba.shape[1],
            height=rgba.shape[0],
            decoder_version=config["official_decoder"]["version"],
            producer_profile_id=config["producer"]["external_profile_id"],
        )
        input_unchanged &= source == source_before and payload == payload_before
        records.append(
            {
                "decoded_payload_sha256": prepared.decoded_payload_sha256,
                "expected_pixel_sha256": _expected_pixel_sha256(rgba),
                "ingress_id": prepared.ingress_id,
                "label": label,
                "maximum_nits": float(np.max(prepared.pixels)),
                "minimum_nits": float(np.min(prepared.pixels)),
                "pixel_sha256": prepared.descriptor.pixel_sha256,
                "profile_id": prepared.descriptor.profile_id,
                "reference_white_nits": prepared.descriptor.reference_white_nits,
                "view_id": prepared.descriptor.view_id,
            }
        )
    records.sort(key=lambda value: value["label"])
    controls = _negative_controls()
    policy = _load_object(ROOT / "configs/reference_match_successor_admission_v1.json")
    gates = {
        "all_pixel_hashes_exact": all(
            row["pixel_sha256"] == row["expected_pixel_sha256"] for row in records
        ),
        "consumer_profile_exact": all(
            row["profile_id"] == config["consumer"]["profile_id"]
            and row["reference_white_nits"]
            == config["consumer"]["reference_white_nits"]
            for row in records
        ),
        "input_bytes_unchanged": input_unchanged,
        "negative_controls_rejected": len(controls) == 7,
        "producer_evidence_bound": source_facts["evidence_sha256"]
        == config["producer"]["evidence_sha256"],
        "successor_admission_remains_sdr": policy["evaluation_readiness"]["profile_id"]
        == "zhuise.display-linear-srgb-d65-relative-f32.v1",
    }
    scientific: dict[str, Any] = {
        "claim_ceiling": config["claim_ceiling"],
        "config_sha256": _sha256_file(config_path),
        "contract_id": config["contract_id"],
        "gates": gates,
        "negative_controls": controls,
        "records": records,
        "schema": REPORT_SCHEMA,
        "source_facts": source_facts,
        "status": (
            "PASS_PRIVATE_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_INGRESS"
            if all(gates.values())
            else "FAIL_CLOSED_ULTRAHDR_ABSOLUTE_REC2020_MATCH_VIEW_INGRESS"
        ),
    }
    scientific["stable_evidence_id"] = _sha256_bytes(canonical_json_bytes(scientific))
    return scientific


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p87_ultrahdr_absolute_rec2020_match_view_v1.json"),
    )
    parser.add_argument("--producer-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    evidence_path = (
        args.producer_evidence
        if args.producer_evidence.is_absolute()
        else ROOT / args.producer_evidence
    )
    report = run(config_path, evidence_path, reverse=args.reverse)
    output = args.output if args.output.is_absolute() else ROOT / args.output
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
