#!/usr/bin/env python3
"""Audit the frozen P99 DNG baseline-exposure stage."""

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

from src.preprocess.dng_baseline_exposure import (
    DngBaselineExposureError,
    _apply_baseline_exposure,
    _read_baseline_exposure,
    _single_rational,
    load_dng_baseline_exposed_working_image,
)
from src.preprocess.dng_forward_raster import load_dng_forward_working_image

SCHEMA = "neuro_film.p99_dng_baseline_exposure_result.v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise DngBaselineExposureError(f"bound file mismatch: {path}")


def _expected_value(value: dict[str, Any]) -> tuple[float, str]:
    provenance = str(value["provenance"])
    if provenance == "explicit_ifd_tag":
        return float(value["numerator"]) / float(value["denominator"]), provenance
    if provenance == "dng_standard_default":
        return float(value["value"]), provenance
    raise DngBaselineExposureError("unknown frozen metadata provenance")


def _row(
    frozen: dict[str, Any],
    p98: dict[str, Any],
) -> dict[str, Any]:
    source = ROOT / p98["logical_path"]
    before_sha = _sha256(source)
    if (
        source.stat().st_size != p98["source_bytes"]
        or before_sha != p98["source_sha256"]
    ):
        raise DngBaselineExposureError("P98 source binding mismatch")
    expected_ev, expected_ev_provenance = _expected_value(frozen["baseline_exposure"])
    expected_offset, expected_offset_provenance = _expected_value(
        frozen["baseline_exposure_offset"]
    )
    facts = _read_baseline_exposure(source)
    parent = load_dng_forward_working_image(
        source,
        expected_source_bytes=p98["source_bytes"],
        expected_source_sha256=p98["source_sha256"],
    )
    parent_frozen = parent.pixels.copy()
    candidate = load_dng_baseline_exposed_working_image(
        source,
        expected_source_bytes=p98["source_bytes"],
        expected_source_sha256=p98["source_sha256"],
    )
    oracle = (parent.pixels.astype(np.float64) * np.exp2(facts.total_ev)).astype(
        np.float32
    )
    parent_boundary = (parent.pixels == 0.0) | (parent.pixels == 1.0)
    candidate_boundary = (candidate.pixels == 0.0) | (candidate.pixels == 1.0)
    new_boundary = int(np.count_nonzero(candidate_boundary & ~parent_boundary))
    if not np.array_equal(parent.pixels, parent_frozen):
        raise DngBaselineExposureError("P98 parent changed during P99 audit")
    if _sha256(source) != before_sha:
        raise DngBaselineExposureError("source changed during P99 audit")
    receipt = candidate.hdr_metadata.get("dng_baseline_exposure")
    if not isinstance(receipt, dict):
        raise DngBaselineExposureError("candidate omitted exposure receipt")
    return {
        "camera_make": p98["camera_make"],
        "candidate": {
            "finite": bool(np.all(np.isfinite(candidate.pixels))),
            "maximum": float(np.max(candidate.pixels)),
            "minimum": float(np.min(candidate.pixels)),
            "new_exact_boundary_count": new_boundary,
            "sha256": _array_sha256(candidate.pixels),
            "warning_codes": [warning.code for warning in candidate.warnings],
        },
        "expected": {
            "baseline_exposure_ev": expected_ev,
            "baseline_exposure_provenance": expected_ev_provenance,
            "baseline_exposure_offset_ev": expected_offset,
            "baseline_exposure_offset_provenance": expected_offset_provenance,
        },
        "facts": facts.to_dict(),
        "metadata_exact": (
            facts.baseline_exposure_ev == expected_ev
            and facts.baseline_exposure_provenance == expected_ev_provenance
            and facts.baseline_exposure_offset_ev == expected_offset
            and facts.baseline_exposure_offset_provenance == expected_offset_provenance
        ),
        "oracle_float32_exact": np.array_equal(candidate.pixels, oracle),
        "parent": {
            "sha256": _array_sha256(parent.pixels),
            "unchanged": np.array_equal(parent.pixels, parent_frozen),
        },
        "parent_equal": np.array_equal(candidate.pixels, parent.pixels),
        "receipt": receipt,
        "source_id": p98["source_id"],
        "source_sha256": before_sha,
        "source_unchanged": True,
    }


def _invalid_rejection_count() -> int:
    rejected = 0
    for action in (
        lambda: _single_rational((1, 0), name="invalid"),
        lambda: _single_rational((1, 2, 3, 4), name="invalid"),
        lambda: _apply_baseline_exposure(np.zeros((1, 1, 3), dtype=np.float64), 1.0),
        lambda: _apply_baseline_exposure(np.zeros((1, 1, 3), dtype=np.float32), 257.0),
    ):
        try:
            action()
        except DngBaselineExposureError:
            rejected += 1
    return rejected


def run(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    for path_key, sha_key in (
        ("contract_path", "contract_sha256"),
        ("dng_spec_path", "dng_spec_sha256"),
        ("p98_evidence_path", "p98_evidence_sha256"),
        ("p98_implementation_path", "p98_implementation_sha256"),
        ("p98_config_path", "p98_config_sha256"),
        ("implementation_path", "implementation_sha256"),
        ("runner_path", "runner_sha256"),
        ("test_path", "test_sha256"),
    ):
        _verify_file(ROOT / bindings[path_key], bindings[sha_key])
    p98_config = json.loads((ROOT / bindings["p98_config_path"]).read_text())
    p98_by_id = {row["source_id"]: row for row in p98_config["rows"]}
    frozen_rows = list(config["rows"])
    if order == "reverse":
        frozen_rows.reverse()
    rows = sorted(
        (_row(row, p98_by_id[row["source_id"]]) for row in frozen_rows),
        key=lambda value: value["source_id"],
    )
    invalid_rejections = _invalid_rejection_count()
    maximum_absolute_output = max(
        max(abs(row["candidate"]["minimum"]), abs(row["candidate"]["maximum"]))
        for row in rows
    )
    gates = config["gates"]
    gate_results = {
        "camera_make_count": len({row["camera_make"] for row in rows})
        == gates["required_camera_makes"],
        "finite": all(row["candidate"]["finite"] for row in rows),
        "invalid_inputs": invalid_rejections == 4,
        "metadata_exact": all(row["metadata_exact"] for row in rows),
        "nonzero_row_changed": all(
            row["parent_equal"] == (row["facts"]["total_ev"] == 0.0) for row in rows
        ),
        "no_new_exact_boundary": all(
            row["candidate"]["new_exact_boundary_count"] == 0 for row in rows
        ),
        "oracle_float32_exact": all(row["oracle_float32_exact"] for row in rows),
        "output_absolute_range": maximum_absolute_output
        <= gates["maximum_absolute_output"],
        "parent_unchanged": all(row["parent"]["unchanged"] for row in rows),
        "row_count": len(rows) == gates["required_rows"],
        "source_unchanged": all(row["source_unchanged"] for row in rows),
        "total_ev": all(
            gates["minimum_total_ev"]
            <= row["facts"]["total_ev"]
            <= gates["maximum_total_ev"]
            for row in rows
        ),
        "working_contract": all(
            "private_dng_baseline_exposure" in row["candidate"]["warning_codes"]
            for row in rows
        ),
    }
    scientific = {
        "bindings": bindings,
        "claim_ceiling": config["claim_ceiling"],
        "decision": "PASS_PRIVATE_DNG_BASELINE_EXPOSURE"
        if all(gate_results.values())
        else "FAIL_CLOSED_DNG_BASELINE_EXPOSURE",
        "experiment_id": config["experiment_id"],
        "gate_results": gate_results,
        "metrics": {
            "distinct_camera_makes": len({row["camera_make"] for row in rows}),
            "invalid_rejections": invalid_rejections,
            "maximum_absolute_output": maximum_absolute_output,
            "maximum_scale": max(row["facts"]["scale"] for row in rows),
            "row_count": len(rows),
        },
        "rows": rows,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    report = run(config, args.order)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(report)
    output.write_bytes(payload)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "stable_identity_sha256": report["stable_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
