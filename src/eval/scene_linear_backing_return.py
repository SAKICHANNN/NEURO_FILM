"""U6.P3L fixed backing-return audit on RAW-derived scene-linear inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.eval.fresh_native_standard_confirmation import (
    load_confirmation_working_image,
    sha256_file,
    validate_preflight,
)
from src.eval.physical_backing_return_combined_ablation import (
    _array,
    _compile_profiles,
    _interpret,
)
from src.eval.physical_backing_return_photographic_stress import (
    _isolated_excursions,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    apply_compiled_scatter,
    apply_fft_backing_return,
)


SCHEMA = "neuro_film.u6_p3l_scene_linear_backing_return_contract.v1"


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P3L contract")
    return payload


def _load_exact_json(root: Path, path: str, expected: str) -> Any:
    resolved = root / path
    if sha256_file(resolved) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, contract: dict[str, Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        contract.get("schema") != SCHEMA
        or contract.get("production_integration_allowed")
        or contract["fixed_chain"]["profile_fitting"]
        or contract["fixed_chain"]["parameter_selection"]
        or contract["input"]["physical_scale_claimed"]
    ):
        raise ValueError("invalid U6.P3L contract")
    parents = contract["parents"]
    p1 = _load_exact_json(
        root,
        parents["p1_contract_path"],
        parents["p1_contract_sha256"],
    )
    p3d = _load_exact_json(
        root,
        parents["p3d_contract_path"],
        parents["p3d_contract_sha256"],
    )
    sensitometry = _load_exact_json(
        root,
        parents["sensitometry_path"],
        parents["sensitometry_sha256"],
    )
    p8bp = _load_exact_json(
        root,
        parents["p8bp_contract_path"],
        parents["p8bp_contract_sha256"],
    )
    _load_exact_json(
        root,
        parents["p8bp_preflight_decision_path"],
        parents["p8bp_preflight_decision_sha256"],
    )
    manifest = validate_preflight(root, p8bp)
    if sha256_file(root / parents["p8bp_manifest_path"]) != parents[
        "p8bp_manifest_sha256"
    ]:
        raise ValueError("P8BP manifest drift")
    if (
        len(manifest) != int(contract["input"]["source_count"])
        or len({row["make"] for row in manifest})
        != int(contract["input"]["camera_make_count"])
    ):
        raise ValueError("P8BP support drift")
    return manifest, p8bp, p1, p3d, sensitometry


def _resize_scene_linear(values: np.ndarray, maximum_long_edge: int) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or not np.all(np.isfinite(source))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
    ):
        raise ValueError("scene-linear source must be finite bounded HxWx3")
    height, width = source.shape[:2]
    scale = min(1.0, maximum_long_edge / max(height, width))
    if scale == 1.0:
        return source.copy()
    resized = cv2.resize(
        source,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized, dtype=np.float32)


def _variants_fft(
    source: np.ndarray,
    *,
    legacy: Any,
    forward: Any,
    backing: Any,
    operator: Any,
) -> dict[str, np.ndarray]:
    source_array = _array(source, legacy.pixel_pitch_um)
    legacy_exposure = apply_compiled_scatter(source_array, legacy)
    forward_exposure = apply_compiled_scatter(source_array, forward)
    split_exposure = apply_fft_backing_return(
        forward_exposure, backing
    ).values
    double_exposure = apply_fft_backing_return(
        legacy_exposure, backing
    ).values
    return {
        "no_spatial": _interpret(source, operator),
        "legacy_full": _interpret(legacy_exposure.values, operator),
        "split_candidate": _interpret(split_exposure, operator),
        "forbidden_double": _interpret(double_exposure, operator),
    }


def evaluate_scene_linear_backing_return(
    *, root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    manifest, p8bp, p1, p3d, sensitometry = validate_contract(root, contract)
    legacy, forward, backing = _compile_profiles(p1, p3d)
    operator = build_operator(sensitometry)
    candidates = {row["id"]: row for row in p8bp["candidates"]}
    gates = contract["automatic_gates"]
    rows: list[dict[str, Any]] = []
    for source_row in manifest:
        candidate = candidates[source_row["id"]]
        raw_path = root / candidate["path"]
        if sha256_file(raw_path) != candidate["sha256"]:
            raise ValueError("RAW source identity drift")
        working = load_confirmation_working_image(raw_path)
        if (
            working.transfer_state != "scene_linear"
            or working.working_space not in {"linear_srgb", "linear_srgb_d65"}
            or not working.orientation_applied
        ):
            raise ValueError("scene-linear WorkingImage contract drift")
        source = _resize_scene_linear(
            working.pixels,
            int(contract["input"]["maximum_long_edge"]),
        )
        variants = _variants_fft(
            source,
            legacy=legacy,
            forward=forward,
            backing=backing,
            operator=operator,
        )
        no_spatial = variants["no_spatial"]
        split = variants["split_candidate"]
        delta = np.abs(split - no_spatial)
        new_boundary = (
            ((split <= 0.0) | (split >= 1.0))
            & ~((no_spatial <= 0.0) | (no_spatial >= 1.0))
        )
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "raw_sha256": candidate["sha256"],
                "scene_linear_sha256": hashlib.sha256(
                    source.tobytes()
                ).hexdigest(),
                "shape": list(source.shape),
                "input_exact_zero_channel_fraction": float(
                    np.mean(source == 0.0)
                ),
                "input_exact_one_channel_fraction": float(
                    np.mean(source == 1.0)
                ),
                "finite_domain": all(
                    np.all(np.isfinite(value))
                    and np.all(value > 0.0)
                    and np.all(value <= 1.0)
                    for value in variants.values()
                ),
                "candidate_vs_no_max_abs": float(np.max(delta)),
                "candidate_vs_no_p95_abs": float(
                    np.quantile(delta, 0.95)
                ),
                "candidate_vs_legacy_p95_abs": float(
                    np.quantile(
                        np.abs(split - variants["legacy_full"]), 0.95
                    )
                ),
                "candidate_vs_forbidden_double_p95_abs": float(
                    np.quantile(
                        np.abs(split - variants["forbidden_double"]), 0.95
                    )
                ),
                "new_hard_boundary_fraction": float(np.mean(new_boundary)),
                "isolated_candidate_excursion_count": _isolated_excursions(
                    delta,
                    threshold=float(gates["isolated_excursion_threshold"]),
                    radius=int(gates["isolated_support_radius_pixels"]),
                    minimum_support=int(
                        gates["minimum_isolated_support_count"]
                    ),
                ),
            }
        )
    p95_no = float(
        np.quantile(
            [row["candidate_vs_no_p95_abs"] for row in rows], 0.95
        )
    )
    p95_legacy = float(
        np.quantile(
            [row["candidate_vs_legacy_p95_abs"] for row in rows], 0.95
        )
    )
    p95_double = float(
        np.quantile(
            [
                row["candidate_vs_forbidden_double_p95_abs"]
                for row in rows
            ],
            0.95,
        )
    )
    decisions = {
        "source_support": len(rows) == int(contract["input"]["source_count"]),
        "source_hashes": all(
            row["raw_sha256"] == candidates[row["id"]]["sha256"]
            for row in rows
        ),
        "finite_domain": all(row["finite_domain"] for row in rows),
        "candidate_bound": max(
            row["candidate_vs_no_max_abs"] for row in rows
        )
        <= float(gates["maximum_candidate_vs_no_spatial_abs"]),
        "candidate_nontrivial": p95_no
        >= float(
            gates["minimum_population_p95_candidate_vs_no_spatial_abs"]
        ),
        "legacy_distinct": p95_legacy
        >= float(
            gates["minimum_population_p95_candidate_vs_legacy_abs"]
        ),
        "double_counting_distinct": p95_double
        >= float(
            gates[
                "minimum_population_p95_candidate_vs_forbidden_double_abs"
            ]
        ),
        "new_boundaries": max(
            row["new_hard_boundary_fraction"] for row in rows
        )
        <= float(gates["maximum_new_hard_boundary_fraction"]),
        "isolated_excursions": sum(
            row["isolated_candidate_excursion_count"] for row in rows
        )
        <= int(gates["maximum_isolated_candidate_excursion_count"]),
    }
    core = {
        "schema": "neuro_film.u6_p3l_scene_linear_backing_return_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "rows": rows,
        "metrics": {
            "maximum_input_exact_zero_channel_fraction": max(
                row["input_exact_zero_channel_fraction"] for row in rows
            ),
            "maximum_candidate_vs_no_spatial_abs": max(
                row["candidate_vs_no_max_abs"] for row in rows
            ),
            "population_p95_candidate_vs_no_spatial_abs": p95_no,
            "population_p95_candidate_vs_legacy_abs": p95_legacy,
            "population_p95_candidate_vs_forbidden_double_abs": p95_double,
            "maximum_new_hard_boundary_fraction": max(
                row["new_hard_boundary_fraction"] for row in rows
            ),
            "total_isolated_candidate_excursion_count": sum(
                row["isolated_candidate_excursion_count"] for row in rows
            ),
        },
        "decisions": decisions,
        "automatic_pass": all(decisions.values()),
        "branch": contract["branch_rule"][
            "pass" if all(decisions.values()) else "fail"
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
    "_resize_scene_linear",
    "_variants_fft",
    "evaluate_scene_linear_backing_return",
    "load_contract",
    "validate_contract",
    "write_report",
]
