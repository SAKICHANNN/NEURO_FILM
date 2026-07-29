"""U6.P3M spatial attribution for the closed P3L backing-return candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

from src.eval.fresh_native_standard_confirmation import (
    load_confirmation_working_image,
    sha256_file,
)
from src.eval.physical_backing_return_combined_ablation import (
    _compile_profiles,
)
from src.eval.scene_linear_backing_return import (
    _canonical_bytes,
    _resize_scene_linear,
    _variants_fft,
    validate_contract as validate_p3l_contract,
)
from src.eval.sensitometry_primitive import build_operator


SCHEMA = (
    "neuro_film.u6_p3m_scene_linear_backing_return_attribution_contract.v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3M contract")
    return payload


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, contract: dict[str, Any]
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if contract.get("schema") != SCHEMA:
        raise ValueError("invalid U6.P3M contract")
    parent = contract["parent"]
    p3l = _load_exact_json(
        root, parent["contract_path"], parent["contract_sha256"]
    )
    decision = _load_exact_json(
        root, parent["decision_path"], parent["decision_sha256"]
    )
    report = _load_exact_json(
        root, parent["report_path"], parent["report_sha256"]
    )
    if (
        decision["decision"]
        != "close_fixed_scene_linear_backing_return_topology"
        or decision["automatic_gate"]["passed"]
        or report["automatic_pass"]
        or tuple(decision["automatic_gate"]["failed"])
        != ("candidate_bound", "isolated_excursions")
    ):
        raise ValueError("P3L closure drift")
    manifest, p8bp, p1, p3d, sensitometry = validate_p3l_contract(
        root, p3l
    )
    return p3l, manifest, p8bp, p1, p3d, sensitometry


def _edge_distance(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.arange(height, dtype=np.int32)[:, None]
    x = np.arange(width, dtype=np.int32)[None, :]
    return np.minimum.reduce(
        (
            np.broadcast_to(y, (height, width)),
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(height - 1 - y, (height, width)),
            np.broadcast_to(width - 1 - x, (height, width)),
        )
    )


def _isolated_mask(
    delta: np.ndarray, *, threshold: float, radius: int, minimum_support: int
) -> np.ndarray:
    excursion = np.max(delta, axis=-1) > threshold
    width = 2 * radius + 1
    support = (
        uniform_filter(
            excursion.astype(np.float32),
            size=width,
            mode="constant",
            cval=0.0,
        )
        * float(width * width)
    )
    return excursion & (support < minimum_support)


def _fraction(mask: np.ndarray, selection: np.ndarray) -> float:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(mask & selection) / count)


def _classify(
    aggregate: dict[str, float], *, dominance_fraction: float
) -> str:
    for label, key in (
        ("frame-boundary-associated", "bound_fail_within_halo_fraction"),
        ("exact-zero-associated", "bound_fail_exact_zero_fraction"),
        ("highlight-associated", "bound_fail_highlight_fraction"),
        ("shadow-associated", "bound_fail_shadow_fraction"),
    ):
        if aggregate[key] >= dominance_fraction:
            return label
    return "distributed-or-content-associated"


def evaluate_attribution(
    *, root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    p3l, manifest, p8bp, p1, p3d, sensitometry = validate_contract(
        root, contract
    )
    legacy, forward, backing = _compile_profiles(p1, p3d)
    operator = build_operator(sensitometry)
    candidates = {row["id"]: row for row in p8bp["candidates"]}
    analysis = contract["analysis"]
    halo = int(backing.required_halo)
    if halo != int(analysis["edge_distance_bands_pixels"][-1]):
        raise ValueError("attribution halo drift")
    rows: list[dict[str, Any]] = []
    total_bound = 0
    total_isolated = 0
    aggregate_counts = {
        "bound_within_halo": 0,
        "bound_exact_zero": 0,
        "bound_highlight": 0,
        "bound_shadow": 0,
        "isolated_within_halo": 0,
        "isolated_exact_zero": 0,
        "isolated_highlight": 0,
        "isolated_shadow": 0,
    }
    for source_row in manifest:
        candidate = candidates[source_row["id"]]
        raw_path = root / candidate["path"]
        if sha256_file(raw_path) != candidate["sha256"]:
            raise ValueError("RAW source identity drift")
        working = load_confirmation_working_image(raw_path)
        source = _resize_scene_linear(
            working.pixels, int(p3l["input"]["maximum_long_edge"])
        )
        variants = _variants_fft(
            source,
            legacy=legacy,
            forward=forward,
            backing=backing,
            operator=operator,
        )
        delta = np.abs(
            variants["split_candidate"] - variants["no_spatial"]
        )
        pixel_delta = np.max(delta, axis=-1)
        bound = pixel_delta > float(analysis["candidate_bound_threshold"])
        isolated = _isolated_mask(
            delta,
            threshold=float(analysis["isolated_excursion_threshold"]),
            radius=int(analysis["isolated_support_radius_pixels"]),
            minimum_support=int(analysis["minimum_isolated_support_count"]),
        )
        distance = _edge_distance(source.shape[:2])
        exact_zero = np.any(source == 0.0, axis=-1)
        highlight = np.max(source, axis=-1) >= float(
            analysis["highlight_minimum"]
        )
        shadow = np.max(source, axis=-1) <= float(
            analysis["shadow_maximum"]
        )
        within_halo = distance <= halo
        bound_count = int(np.count_nonzero(bound))
        isolated_count = int(np.count_nonzero(isolated))
        total_bound += bound_count
        total_isolated += isolated_count
        selections = {
            "within_halo": within_halo,
            "exact_zero": exact_zero,
            "highlight": highlight,
            "shadow": shadow,
        }
        for name, selection in selections.items():
            aggregate_counts[f"bound_{name}"] += int(
                np.count_nonzero(bound & selection)
            )
            aggregate_counts[f"isolated_{name}"] += int(
                np.count_nonzero(isolated & selection)
            )
        flat_order = np.argsort(pixel_delta, axis=None)[
            -int(analysis["top_locations_per_source"]) :
        ][::-1]
        top = []
        for flat in flat_order:
            y, x = np.unravel_index(int(flat), pixel_delta.shape)
            top.append(
                {
                    "y": int(y),
                    "x": int(x),
                    "edge_distance": int(distance[y, x]),
                    "source_rgb": [float(value) for value in source[y, x]],
                    "delta_rgb": [float(value) for value in delta[y, x]],
                    "maximum_delta": float(pixel_delta[y, x]),
                }
            )
        rows.append(
            {
                "id": source_row["id"],
                "scene_linear_sha256": hashlib.sha256(
                    source.tobytes()
                ).hexdigest(),
                "bound_failure_pixels": bound_count,
                "isolated_excursion_pixels": isolated_count,
                "bound_failure_fractions": {
                    f"within_{band}px": _fraction(
                        bound, distance <= int(band)
                    )
                    for band in analysis["edge_distance_bands_pixels"]
                },
                "bound_failure_exact_zero_fraction": _fraction(
                    bound, exact_zero
                ),
                "bound_failure_highlight_fraction": _fraction(
                    bound, highlight
                ),
                "bound_failure_shadow_fraction": _fraction(bound, shadow),
                "isolated_within_halo_fraction": _fraction(
                    isolated, within_halo
                ),
                "isolated_exact_zero_fraction": _fraction(
                    isolated, exact_zero
                ),
                "isolated_highlight_fraction": _fraction(
                    isolated, highlight
                ),
                "isolated_shadow_fraction": _fraction(isolated, shadow),
                "top_locations": top,
            }
        )
    aggregate = {
        "bound_failure_pixels": total_bound,
        "isolated_excursion_pixels": total_isolated,
        "bound_fail_within_halo_fraction": (
            aggregate_counts["bound_within_halo"] / total_bound
            if total_bound
            else 0.0
        ),
        "bound_fail_exact_zero_fraction": (
            aggregate_counts["bound_exact_zero"] / total_bound
            if total_bound
            else 0.0
        ),
        "bound_fail_highlight_fraction": (
            aggregate_counts["bound_highlight"] / total_bound
            if total_bound
            else 0.0
        ),
        "bound_fail_shadow_fraction": (
            aggregate_counts["bound_shadow"] / total_bound
            if total_bound
            else 0.0
        ),
        "isolated_within_halo_fraction": (
            aggregate_counts["isolated_within_halo"] / total_isolated
            if total_isolated
            else 0.0
        ),
        "isolated_exact_zero_fraction": (
            aggregate_counts["isolated_exact_zero"] / total_isolated
            if total_isolated
            else 0.0
        ),
        "isolated_highlight_fraction": (
            aggregate_counts["isolated_highlight"] / total_isolated
            if total_isolated
            else 0.0
        ),
        "isolated_shadow_fraction": (
            aggregate_counts["isolated_shadow"] / total_isolated
            if total_isolated
            else 0.0
        ),
    }
    classification = _classify(
        aggregate,
        dominance_fraction=float(analysis["dominance_fraction"]),
    )
    core = {
        "schema": "neuro_film.u6_p3m_scene_linear_backing_return_attribution_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "backing_required_halo_pixels": halo,
        "rows": rows,
        "aggregate": aggregate,
        "classification": classification,
        "branch": contract["branch_rule"][
            "frame-boundary-associated"
            if classification == "frame-boundary-associated"
            else "other"
        ],
    }
    return {
        **core,
        "stable_evidence_id": hashlib.sha256(
            _canonical_bytes(core)
        ).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "_classify",
    "_edge_distance",
    "_isolated_mask",
    "evaluate_attribution",
    "load_contract",
    "validate_contract",
    "write_report",
]
