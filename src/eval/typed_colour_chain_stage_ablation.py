"""Frozen U6.P4DN typed colour/cloud/scanner stage ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_cloud_capacity_v2 import evaluate as evaluate_capacity
from src.eval.sensitometry_primitive import build_operator
from src.eval.typed_sensitometry_cloud_chain import _exposure
from src.film_physics.cross_layer_cloud_profile import CrossLayerCloudReferenceProfile
from src.film_physics.exposure_development import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
)
from src.film_physics.scanner import apply_scanner_profile
from src.film_physics.sensitometry_cloud_chain import (
    render_typed_sensitometry_cloud_chain,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    parent = root / contract["parent"]["path"]
    evidence = json.loads(parent.read_text(encoding="utf-8"))
    if (
        _sha(parent) != contract["parent"]["sha256"]
        or evidence["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4DN parent drift")
    return contract


def _render_route(
    exposure: Any,
    operator: Any,
    profile: CrossLayerCloudReferenceProfile,
    route: InterpretationRoute,
    *,
    seed: int,
    rows: int,
) -> Any:
    interpretation = DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=(
            EmulsionFamily.COLOR_NEGATIVE
            if route is InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN
            else EmulsionFamily.SLIDE
        ),
        interpretation_route=route,
    )
    return render_typed_sensitometry_cloud_chain(
        exposure,
        operator,
        interpretation,
        profile,
        seed=seed,
        row_tile_height=rows,
    )


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    fixture = contract["fixture"]
    shape = (fixture["height"], fixture["width"])
    operator = build_operator(
        json.loads((root / "configs/u2_2a_sensitometry_primitive_v1.json").read_text())
    )
    capacity = evaluate_capacity(
        root, root / "configs/u6_p4di_sensitometry_cloud_capacity_v2.json"
    )
    profile = CrossLayerCloudReferenceProfile.from_payload(capacity["compiled_profile"])
    exposure = _exposure(shape, fixture["pixel_pitch_um"])
    negative = _render_route(
        exposure,
        operator,
        profile,
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        seed=fixture["seed"],
        rows=fixture["height"],
    )
    negative_tiled = _render_route(
        exposure,
        operator,
        profile,
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    negative_repeat = _render_route(
        exposure,
        operator,
        profile,
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    slide = _render_route(
        exposure,
        operator,
        profile,
        InterpretationRoute.SLIDE_DIRECT_SCAN,
        seed=fixture["seed"],
        rows=fixture["row_partition_height"],
    )
    mean_transmittance = np.exp(-negative.mean_developed_density.values)
    cloud_transmittance = negative.transmittance.values.astype(np.float64)
    scanner_contract = json.loads(
        (root / "configs/u6_p6a_scanner_profile_boundary_v1.json").read_text()
    )
    scanner_a = _profile(scanner_contract["profiles"]["scanner_a"])
    scanner_b = _profile(scanner_contract["profiles"]["scanner_b"])
    scanner_only = apply_scanner_profile(
        mean_transmittance, scanner_a, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    combined = apply_scanner_profile(
        cloud_transmittance, scanner_a, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    combined_repeat = apply_scanner_profile(
        cloud_transmittance, scanner_a, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    wrong = apply_scanner_profile(
        cloud_transmittance, scanner_b, pixel_pitch_um=fixture["pixel_pitch_um"]
    )
    margin = fixture["measurement_margin"]
    region = np.s_[margin:-margin, margin:-margin, :]
    cloud_residual = cloud_transmittance - mean_transmittance
    scanner_residual = scanner_only - mean_transmittance
    combined_vs_cloud = combined - cloud_transmittance
    combined_vs_scanner = combined - scanner_only
    wrong_vs_scanner = wrong - scanner_only
    residual_chroma = combined_vs_scanner - np.mean(
        combined_vs_scanner, axis=-1, keepdims=True
    )
    neutral_chroma_p99 = float(
        np.quantile(np.linalg.norm(residual_chroma[region], axis=-1), 0.99)
    )
    correct_rms = _rms(combined_vs_scanner[region])
    wrong_ratio = _rms(wrong_vs_scanner[region]) / correct_rms
    new_boundary = float(
        np.mean(
            ((combined <= 0.0) | (combined >= 1.0))
            & ~((scanner_only <= 0.0) | (scanner_only >= 1.0))
        )
    )
    metrics = {
        "cloud_stage_rms": _rms(cloud_residual[region]),
        "scanner_stage_rms": _rms(scanner_residual[region]),
        "combined_vs_cloud_only_rms": _rms(combined_vs_cloud[region]),
        "combined_vs_scanner_only_rms": correct_rms,
        "wrong_scanner_residual_ratio": wrong_ratio,
        "combined_new_boundary_fraction": new_boundary,
        "neutral_cloud_residual_chroma_p99": neutral_chroma_p99,
    }
    gates = contract["gates"]
    decisions = {
        "cloud_stage": metrics["cloud_stage_rms"] >= gates["minimum_cloud_stage_rms"],
        "scanner_stage": metrics["scanner_stage_rms"]
        >= gates["minimum_scanner_stage_rms"],
        "combined_vs_cloud": metrics["combined_vs_cloud_only_rms"]
        >= gates["minimum_combined_vs_cloud_only_rms"],
        "combined_vs_scanner": metrics["combined_vs_scanner_only_rms"]
        >= gates["minimum_combined_vs_scanner_only_rms"],
        "wrong_scanner": wrong_ratio >= gates["minimum_wrong_scanner_residual_ratio"],
        "boundary": new_boundary <= gates["maximum_combined_new_boundary_fraction"],
        "neutral_chroma": neutral_chroma_p99 <= gates["maximum_neutral_chroma_p99"],
        "route_identity": bool(
            np.array_equal(negative.cloud_density.values, slide.cloud_density.values)
            and np.array_equal(
                negative.transmittance.values, slide.transmittance.values
            )
        ),
        "partition_identity": bool(
            np.array_equal(
                negative.cloud_density.values, negative_tiled.cloud_density.values
            )
            and np.array_equal(
                negative.transmittance.values, negative_tiled.transmittance.values
            )
        ),
        "repeat_identity": bool(
            np.array_equal(
                negative_tiled.cloud_density.values,
                negative_repeat.cloud_density.values,
            )
            and np.array_equal(combined, combined_repeat)
        ),
    }
    stable = {
        "contract_sha256": _sha(contract_path),
        "cloud_profile_identity": profile.identity(),
        "scanner_profile_id": scanner_a.profile_id,
        "metrics": metrics,
        "combined_sha256": hashlib.sha256(combined.astype("<f8").tobytes()).hexdigest(),
        "gates": decisions,
        "decision": contract["decision_if_pass"]
        if all(decisions.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "neuro_film.u6_p4dn_typed_colour_chain_stage_ablation_report.v1",
        "automatic_pass": all(decisions.values()),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(_canonical(stable)).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
