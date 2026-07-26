"""Fail-closed evaluator for U5.R2L1 full-frame strength preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps

from src.eval.density_strength_oracle import (
    DensityStrengthOracleError,
    sha256_file,
)
from src.roll2film.adaptive_density_strength import (
    apply_strength_preflight,
    render_strength_rgb8,
)
from src.roll2film.density_domain import operator_from_config


class FullFrameStrengthPreflightError(DensityStrengthOracleError):
    """Raised when the frozen L1 replay contract or evidence drifts."""


def _load(root: Path, relative: str, expected_hash: str) -> tuple[Path, dict[str, Any]]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise FullFrameStrengthPreflightError("input path escapes root")
    if sha256_file(path) != expected_hash:
        raise FullFrameStrengthPreflightError(f"hash mismatch: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FullFrameStrengthPreflightError("JSON input must be an object")
    return path, value


def _decode_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            ImageOps.exif_transpose(image).convert("RGB"),
            dtype=np.uint8,
        )


def evaluate_full_frame_preflight(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    software_commit: str,
) -> dict[str, Any]:
    parent = config["parent"]
    _, decision = _load(
        root,
        str(parent["decision"]),
        str(parent["decision_sha256"]),
    )
    _, selected_manifest = _load(
        root,
        str(parent["selected_manifest"]),
        str(parent["selected_manifest_sha256"]),
    )
    frozen_path, frozen = _load(
        root,
        str(config["frozen_set"]),
        str(config["frozen_set_sha256"]),
    )
    _, density = _load(
        root,
        str(config["density_operator_config"]),
        str(config["density_operator_config_sha256"]),
    )
    archive_path, archive = _load(
        root,
        str(config["archive_render_manifest"]),
        str(config["archive_render_manifest_sha256"]),
    )
    if decision.get("decision") != (
        "retain_oracle_feasibility_and_open_simplest_inference_time_policy_contract"
    ):
        raise FullFrameStrengthPreflightError("parent decision does not open L1")

    samples_raw = frozen.get("frozen_set", frozen).get("samples", [])
    samples = {
        str(row["id"]): dict(row)
        for row in samples_raw
        if row.get("availability") == "available"
        and row.get("split") in {"gold", "stress"}
    }
    if len(samples) != int(config["expected_samples"]):
        raise FullFrameStrengthPreflightError("frozen sample count drift")
    selected = {
        str(row["sample_id"]): dict(row)
        for row in selected_manifest.get("records", [])
    }
    if selected.keys() != samples.keys():
        raise FullFrameStrengthPreflightError("selected membership drift")

    witness = str(config["witness_id"])
    baseline_strength = float(config["baseline_strength"])
    challenger_strength = float(config["challenger_strength"])
    baseline_id = f"{witness}__s{int(round(baseline_strength * 100)):02d}"
    challenger_id = f"{witness}__s{int(round(challenger_strength * 100)):02d}"
    archive_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in archive.get("records", []):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id not in {baseline_id, challenger_id}:
            continue
        key = (candidate_id, str(row.get("sample_id", "")))
        if key in archive_rows:
            raise FullFrameStrengthPreflightError(f"duplicate archive row: {key}")
        archive_rows[key] = dict(row)
    if len(archive_rows) != 2 * len(samples):
        raise FullFrameStrengthPreflightError("archive candidate rows incomplete")

    bounds = density["parameter_bounds"]
    operator = operator_from_config(
        density["witnesses"][witness],
        exposure_floor=float(density["exposure_floor"]),
        matrix_minimum_determinant=float(bounds["matrix_minimum_determinant"]),
        minimum_endpoint_span=float(bounds["minimum_endpoint_span"]),
    )
    epsilon = float(config["new_hard_clipping_epsilon"])
    ceiling = float(config["maximum_full_frame_new_hard_clipping_fraction"])
    records = []
    for sample_id in sorted(samples):
        sample = samples[sample_id]
        source_path = (root / str(sample["source_path"])).resolve()
        if sha256_file(source_path) != str(sample["source_sha256"]):
            raise FullFrameStrengthPreflightError(
                f"source hash mismatch: {sample_id}"
            )
        source = _decode_rgb8(source_path)
        baseline_pixels = render_strength_rgb8(
            source,
            operator,
            strength=baseline_strength,
        )
        policy = apply_strength_preflight(
            source,
            operator,
            challenger_strength=challenger_strength,
            baseline_strength=baseline_strength,
            epsilon=epsilon,
            maximum_new_hard_clipping_fraction=ceiling,
        )
        archived_arrays = {}
        for candidate_id in (baseline_id, challenger_id):
            row = archive_rows[(candidate_id, sample_id)]
            if str(row.get("source_sha256", "")) != str(
                sample["source_sha256"]
            ):
                raise FullFrameStrengthPreflightError(
                    f"archive source lineage mismatch: {candidate_id}/{sample_id}"
                )
            path = (archive_path.parent / str(row["output"])).resolve()
            if not path.is_relative_to(archive_path.parent.resolve()):
                raise FullFrameStrengthPreflightError("archive output escapes root")
            if sha256_file(path) != str(row["output_sha256"]):
                raise FullFrameStrengthPreflightError(
                    f"archive output hash mismatch: {candidate_id}/{sample_id}"
                )
            archived_arrays[candidate_id] = _decode_rgb8(path)
        selected_row = selected[sample_id]
        selected_path = (root / str(selected_row["selected_output"])).resolve()
        if sha256_file(selected_path) != str(
            selected_row["selected_output_sha256"]
        ):
            raise FullFrameStrengthPreflightError(
                f"selected output hash mismatch: {sample_id}"
            )
        selected_pixels = _decode_rgb8(selected_path)
        baseline_match = np.array_equal(
            baseline_pixels,
            archived_arrays[baseline_id],
        )
        challenger_match = np.array_equal(
            policy.challenger_rgb8,
            archived_arrays[challenger_id],
        )
        assignment_match = policy.selected_strength == float(
            selected_row["selected_strength"]
        )
        selected_match = np.array_equal(policy.selected_rgb8, selected_pixels)
        records.append(
            {
                "sample_id": sample_id,
                "split": str(sample["split"]),
                "full_frame_challenger_new_hard_clipping_fraction": (
                    policy.challenger_new_hard_clipping_fraction
                ),
                "selected_strength": policy.selected_strength,
                "oracle_strength": float(selected_row["selected_strength"]),
                "fallback_applied": policy.fallback_applied,
                "baseline_archive_rgb8_match": baseline_match,
                "challenger_archive_rgb8_match": challenger_match,
                "oracle_assignment_match": assignment_match,
                "selected_output_rgb8_match": selected_match,
            }
        )

    counts = {
        "baseline_archive_rgb8_matches": sum(
            row["baseline_archive_rgb8_match"] for row in records
        ),
        "challenger_archive_rgb8_matches": sum(
            row["challenger_archive_rgb8_match"] for row in records
        ),
        "oracle_assignment_matches": sum(
            row["oracle_assignment_match"] for row in records
        ),
        "selected_output_rgb8_matches": sum(
            row["selected_output_rgb8_match"] for row in records
        ),
        "challenger_selected_count": sum(
            row["selected_strength"] == challenger_strength for row in records
        ),
        "baseline_fallback_count": sum(row["fallback_applied"] for row in records),
    }
    gates = config["gates"]
    checks = {
        "baseline_archive_rgb8_replay": (
            counts["baseline_archive_rgb8_matches"]
            == int(gates["required_baseline_archive_byte_matches"])
        ),
        "challenger_archive_rgb8_replay": (
            counts["challenger_archive_rgb8_matches"]
            == int(gates["required_challenger_archive_byte_matches"])
        ),
        "oracle_assignment_replay": (
            counts["oracle_assignment_matches"]
            == int(gates["required_oracle_assignment_matches"])
        ),
        "selected_output_rgb8_replay": (
            counts["selected_output_rgb8_matches"]
            == int(gates["required_selected_output_byte_matches"])
        ),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "frozen_set": frozen_path.relative_to(root.resolve()).as_posix(),
        "sample_count": len(records),
        "counts": counts,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "per_image": records,
        "claim_ceiling": config["claim_ceiling"],
    }
