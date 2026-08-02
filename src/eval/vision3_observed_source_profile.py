"""U5.R2BU3 compiler for non-renderable VISION3 source observations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.observed_source_profile import (
    BUNDLE_SCHEMA,
    CHANNELS,
    EXECUTION_AUTHORITY,
    GRANULARITY_DOMAIN,
    MTF_DOMAIN,
    STOCKS,
    ObservedCurve,
    Vision3ObservedSourceProfileBundle,
    Vision3ObservedStockSourceProfile,
)

SCHEMA = "neuro_film.u5_r2bu3_vision3_observed_source_profile_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2bu3_vision3_observed_source_profile_report.v1"


class ObservedSourceCompilerError(RuntimeError):
    """Raised when frozen evidence or the non-renderable profile contract drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ObservedSourceCompilerError("BU3 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bundle = payload.get("bundle", {})
    gates = payload.get("gates", {})
    parents = payload.get("parents", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != "U5.R2BU3"
        or parents.get("mtf_decision", {}).get("sha256")
        != "eae84870c4c3706228fecbe5696825c7d500020558dab94d8b6177baabe1848e"
        or parents.get("granularity_decision", {}).get("sha256")
        != "d9d55f2d04107fa52c82c5edec219fe9e04cb5634c1ccf6f5ab27e5cc4ebace1"
        or parents.get("mtf_trace", {}).get("sha256")
        != "d88c0e698c0e0969696224df50f1573ed78a6de0a894075f6502d9b3f658a794"
        or parents.get("granularity_trace", {}).get("sha256")
        != "00b3436c9619a477bf13dcae0ab90df6fd9470bc1976e77484983a4a0029f145"
        or bundle
        != {
            "schema": BUNDLE_SCHEMA,
            "stocks": list(STOCKS),
            "channels": list(CHANNELS),
            "mtf_domain": MTF_DOMAIN,
            "granularity_domain": GRANULARITY_DOMAIN,
            "curve_interpolation": "piecewise_linear_in_stored_coordinate_domain",
            "execution_authority": EXECUTION_AUTHORITY,
        }
        or gates
        != {
            "required_stock_count": 3,
            "required_channel_count": 3,
            "minimum_mtf_samples_per_channel": 8,
            "minimum_granularity_samples_per_channel": 16,
            "strictly_increasing_coordinates": True,
            "finite_values": True,
            "mtf_response_fraction_bounds": [0.0, 1.1],
            "granularity_sigma_d_bounds": [0.001, 0.05],
            "exact_roundtrip": True,
            "exact_recompile_identity": True,
            "reject_parent_or_trace_drift": True,
            "forbidden_serialized_fields": [
                "psf_sigma",
                "nps",
                "grain_radius",
                "particle_count",
                "dye_cloud_radius",
                "physical_placement",
                "render_operator",
                "rgb_transform",
            ],
        }
    ):
        raise ObservedSourceCompilerError("BU3 frozen contract drift")
    for lock in parents.values():
        _relative_path(str(lock.get("path", "")))
    return payload


def _linear_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if pixel0 == pixel1:
        raise ObservedSourceCompilerError("BU3 invalid linear axis")
    return float(value0 + (pixel - pixel0) / (pixel1 - pixel0) * (value1 - value0))


def _log_value_from_pixel(pixel: float, anchors: Sequence[Sequence[float]]) -> float:
    (value0, pixel0), (value1, pixel1) = anchors
    if value0 <= 0.0 or value1 <= 0.0 or pixel0 == pixel1:
        raise ObservedSourceCompilerError("BU3 invalid logarithmic axis")
    fraction = (pixel - pixel0) / (pixel1 - pixel0)
    return float(
        10.0
        ** (
            math.log10(float(value0))
            + fraction * (math.log10(float(value1)) - math.log10(float(value0)))
        )
    )


def _load_locked_json(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    path = root / _relative_path(str(lock["path"]))
    if not path.is_file() or hash_file(path) != lock["sha256"]:
        raise ObservedSourceCompilerError(f"BU3 locked input mismatch: {lock['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _forbidden_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                found.add(str(key).lower())
            found.update(_forbidden_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child, forbidden))
    return found


def compile_observed_bundle(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = config["parents"]
    mtf_decision = _load_locked_json(root, parents["mtf_decision"])
    granularity_decision = _load_locked_json(root, parents["granularity_decision"])
    mtf_trace = _load_locked_json(root, parents["mtf_trace"])
    granularity_trace = _load_locked_json(root, parents["granularity_trace"])
    mtf_parent_gate = (
        mtf_decision.get("decision") == parents["mtf_decision"]["required_decision"]
    )
    granularity_parent_gate = (
        granularity_decision.get("decision")
        == parents["granularity_decision"]["required_decision"]
    )
    mtf_evidence_id = str(mtf_decision["formal_evidence"]["stable_evidence_id"])
    granularity_evidence_id = str(
        granularity_decision["formal_evidence"]["stable_evidence_id"]
    )

    profiles: list[Vision3ObservedStockSourceProfile] = []
    sample_counts: dict[str, Any] = {}
    for stock in STOCKS:
        mtf_row = mtf_trace["stocks"][stock]
        granularity_row = granularity_trace["stocks"][stock]
        mtf_curves: dict[str, ObservedCurve] = {}
        granularity_curves: dict[str, ObservedCurve] = {}
        for channel in CHANNELS:
            mtf_coordinates = np.asarray(mtf_row["curves"][channel], dtype=np.float64)
            mtf_curves[channel] = ObservedCurve(
                tuple(
                    _log_value_from_pixel(x, mtf_row["graph_axes"]["x_value_pixels"])
                    for x in mtf_coordinates[:, 0]
                ),
                tuple(
                    _log_value_from_pixel(y, mtf_row["graph_axes"]["y_value_pixels"])
                    / 100.0
                    for y in mtf_coordinates[:, 1]
                ),
            )
            granularity_coordinates = np.asarray(
                granularity_row["curves"][channel], dtype=np.float64
            )
            granularity_curves[channel] = ObservedCurve(
                tuple(
                    _linear_value_from_pixel(
                        x, granularity_row["graph_axes"]["x_value_pixels"]
                    )
                    for x in granularity_coordinates[:, 0]
                ),
                tuple(
                    _log_value_from_pixel(
                        y, granularity_row["graph_axes"]["y_value_pixels"]
                    )
                    for y in granularity_coordinates[:, 1]
                ),
            )
        profile = Vision3ObservedStockSourceProfile(
            stock_id=stock,
            mtf_source_pdf_sha256=str(mtf_row["source_pdf"]["sha256"]),
            mtf_source_graph_sha256=str(mtf_row["source_graph"]["sha256"]),
            granularity_source_pdf_sha256=str(
                granularity_row["source_pdf"]["sha256"]
            ),
            granularity_source_graph_sha256=str(
                granularity_row["source_graph"]["sha256"]
            ),
            mtf_measurement_context=dict(mtf_row["measurement_context"]),
            granularity_measurement_context=dict(
                granularity_row["measurement_context"]
            ),
            mtf_curves=mtf_curves,
            granularity_curves=granularity_curves,
            mtf_evidence_id=mtf_evidence_id,
            granularity_evidence_id=granularity_evidence_id,
        )
        profiles.append(profile)
        sample_counts[stock] = {
            "mtf": {channel: len(mtf_curves[channel].coordinates) for channel in CHANNELS},
            "granularity": {
                channel: len(granularity_curves[channel].coordinates)
                for channel in CHANNELS
            },
        }

    bundle = Vision3ObservedSourceProfileBundle(
        profiles=tuple(profiles),
        mtf_trace_sha256=hash_file(root / _relative_path(parents["mtf_trace"]["path"])),
        granularity_trace_sha256=hash_file(
            root / _relative_path(parents["granularity_trace"]["path"])
        ),
        mtf_evidence_id=mtf_evidence_id,
        granularity_evidence_id=granularity_evidence_id,
    )
    payload = bundle.to_dict()
    rebuilt = Vision3ObservedSourceProfileBundle.from_dict(payload)
    roundtrip_gate = rebuilt.to_dict() == payload and rebuilt.identity() == bundle.identity()
    forbidden = _forbidden_keys(
        payload, {str(value).lower() for value in config["gates"]["forbidden_serialized_fields"]}
    )
    sample_gate = all(
        all(
            sample_counts[stock]["mtf"][channel]
            >= int(config["gates"]["minimum_mtf_samples_per_channel"])
            and sample_counts[stock]["granularity"][channel]
            >= int(config["gates"]["minimum_granularity_samples_per_channel"])
            for channel in CHANNELS
        )
        for stock in STOCKS
    )
    gate_results = {
        "mtf_parent": mtf_parent_gate,
        "granularity_parent": granularity_parent_gate,
        "stock_and_channel_counts": len(profiles) == 3 and sample_gate,
        "exact_roundtrip": roundtrip_gate,
        "forbidden_serialized_fields_absent": not forbidden,
        "execution_authority_is_non_renderable": payload["execution_authority"]
        == EXECUTION_AUTHORITY,
    }
    compiler_pass = all(gate_results.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "bundle_id": bundle.identity(),
        "bundle_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "sample_counts": sample_counts,
        "profile_ids": {profile.stock_id: profile.identity() for profile in profiles},
        "forbidden_serialized_fields_found": sorted(forbidden),
        "gate_results": gate_results,
        "compiler_pass": compiler_pass,
        "decision": (
            "retain_non_renderable_observed_source_profile"
            if compiler_pass
            else "close_observed_source_profile_compiler_without_rescue"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    stable = dict(report)
    stable["stable_evidence_id"] = hashlib.sha256(canonical_json(stable)).hexdigest()
    return stable, payload


__all__ = [
    "ObservedSourceCompilerError",
    "canonical_json",
    "compile_observed_bundle",
    "hash_file",
    "load_contract",
]
