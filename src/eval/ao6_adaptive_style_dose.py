"""Development-only selection along the frozen AO6 residual-strength path."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "neuro-film.u5-r2at0-ao6-adaptive-style-dose-contract.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _median_absolute_deviation(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array))))


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported adaptive-style-dose contract")
    return contract


def evaluate_adaptive_style_dose(
    *,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Select one already-rendered AO6 strength per image and audit the policy."""

    policy = contract["policy"]
    if (
        policy.get("content_features_allowed")
        or policy.get("scene_or_camera_labels_allowed")
        or policy.get("operator_refit_allowed")
        or policy.get("inter_candidate_blending_allowed")
        or policy.get("post_selection_pixel_change_allowed")
        or contract.get("training_allowed")
        or contract.get("production_integration_allowed")
    ):
        raise ValueError("adaptive-style-dose boundary drift")

    report_path = root / contract["parent_automatic_report"]
    manifest_path = root / contract["parent_render_manifest"]
    parent = _load_hashed_json(
        report_path, contract["parent_automatic_report_sha256"]
    )
    manifest = _load_hashed_json(
        manifest_path, contract["parent_render_manifest_sha256"]
    )

    eligible_ids = [str(value) for value in policy["eligible_candidate_ids"]]
    if (
        len(eligible_ids) != len(set(eligible_ids))
        or set(eligible_ids) != set(parent["candidates"])
    ):
        raise ValueError("eligible AO6 candidate set drift")
    comparator_id = str(policy["fixed_global_comparator_id"])
    if comparator_id not in eligible_ids:
        raise ValueError("fixed comparator is not eligible")

    candidate_rows: dict[str, dict[str, dict[str, Any]]] = {}
    strengths: dict[str, tuple[float, float]] = {}
    sample_ids: list[str] | None = None
    for candidate_id in eligible_ids:
        candidate = parent["candidates"][candidate_id]
        rows = {
            str(row["sample_id"]): dict(row)
            for row in candidate["per_image"]
        }
        if len(rows) != len(candidate["per_image"]):
            raise ValueError(f"duplicate sample in {candidate_id}")
        if sample_ids is None:
            sample_ids = list(rows)
        elif list(rows) != sample_ids:
            raise ValueError("candidate sample order drift")
        candidate_rows[candidate_id] = rows
        strengths[candidate_id] = (
            float(candidate["tone_strength"]),
            float(candidate["chroma_strength"]),
        )
    assert sample_ids is not None

    manifest_hashes: dict[tuple[str, str], str] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key in manifest_hashes:
            raise ValueError(f"duplicate parent manifest record: {key}")
        manifest_hashes[key] = str(row.get("output_sha256"))
    expected = {
        (candidate_id, sample_id)
        for candidate_id in eligible_ids
        for sample_id in sample_ids
    }
    if set(manifest_hashes) != expected:
        raise ValueError("parent manifest AO6 record set drift")

    target = float(policy["target_median_delta_e76_from_b0"])
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("invalid residual-dose target")
    selections: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        ranked = sorted(
            eligible_ids,
            key=lambda candidate_id: (
                abs(
                    float(
                        candidate_rows[candidate_id][sample_id][
                            "median_real_film_delta_e76_from_base"
                        ]
                    )
                    - target
                ),
                strengths[candidate_id][0],
                strengths[candidate_id][1],
                candidate_id,
            ),
        )
        selected_id = ranked[0]
        selected = candidate_rows[selected_id][sample_id]
        if (
            selected["output_sha256"]
            != manifest_hashes[(selected_id, sample_id)]
        ):
            raise ValueError(f"selected output hash drift: {sample_id}")
        residual = float(selected["median_real_film_delta_e76_from_base"])
        selections.append(
            {
                "sample_id": sample_id,
                "split": selected["split"],
                "candidate_id": selected_id,
                "tone_strength": strengths[selected_id][0],
                "chroma_strength": strengths[selected_id][1],
                "median_real_film_delta_e76_from_base": residual,
                "absolute_target_error": abs(residual - target),
                "output_sha256": selected["output_sha256"],
            }
        )
        selected_rows.append(selected)
        fixed_rows.append(candidate_rows[comparator_id][sample_id])

    def subset(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["split"] == split]

    residual_values = [
        float(row["median_real_film_delta_e76_from_base"])
        for row in selected_rows
    ]
    fixed_residual_values = [
        float(row["median_real_film_delta_e76_from_base"])
        for row in fixed_rows
    ]
    errors = [abs(value - target) for value in residual_values]
    selected_counts = Counter(
        str(row["candidate_id"]) for row in selections
    )
    gold = subset(selected_rows, "gold")
    stress = subset(selected_rows, "stress")
    metrics = {
        "sample_count": len(selected_rows),
        "gold_sample_count": len(gold),
        "stress_sample_count": len(stress),
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "selected_strength_level_count": len(selected_counts),
        "largest_strength_share": max(selected_counts.values())
        / len(selected_rows),
        "median_absolute_target_error": float(np.median(errors)),
        "p90_absolute_target_error": _percentile(errors, 0.9),
        "selected_residual_median": float(np.median(residual_values)),
        "selected_residual_mad": _median_absolute_deviation(residual_values),
        "fixed_global_residual_mad": _median_absolute_deviation(
            fixed_residual_values
        ),
        "dose_mad_ratio_vs_fixed_global": (
            _median_absolute_deviation(residual_values)
            / _median_absolute_deviation(fixed_residual_values)
        ),
        "gold_median_style_delta_e76": float(
            np.median([row["median_style_delta_e76"] for row in gold])
        ),
        "gold_median_non_basic_residual_delta_e76": float(
            np.median(
                [row["median_non_basic_residual_delta_e76"] for row in gold]
            )
        ),
        "gold_median_real_film_delta_e76_from_base": float(
            np.median(
                [
                    row["median_real_film_delta_e76_from_base"]
                    for row in gold
                ]
            )
        ),
        "worst_gold_new_hard_clipping_fraction": float(
            max(row["new_hard_clipping_fraction"] for row in gold)
        ),
        "worst_stress_new_hard_clipping_fraction": float(
            max(row["new_hard_clipping_fraction"] for row in stress)
        ),
    }
    gates = contract["automatic_gates"]
    checks = {
        "sample_count": metrics["sample_count"] == int(gates["sample_count"]),
        "gold_sample_count": metrics["gold_sample_count"]
        == int(gates["gold_sample_count"]),
        "stress_sample_count": metrics["stress_sample_count"]
        == int(gates["stress_sample_count"]),
        "selected_strength_levels": metrics["selected_strength_level_count"]
        >= int(gates["minimum_selected_strength_levels"]),
        "largest_strength_share": metrics["largest_strength_share"]
        <= float(gates["maximum_largest_strength_share"]),
        "median_absolute_target_error": metrics[
            "median_absolute_target_error"
        ]
        <= float(gates["median_absolute_target_error_maximum"]),
        "p90_absolute_target_error": metrics["p90_absolute_target_error"]
        <= float(gates["p90_absolute_target_error_maximum"]),
        "dose_mad_ratio": metrics["dose_mad_ratio_vs_fixed_global"]
        <= float(gates["dose_mad_ratio_vs_fixed_global_maximum"]),
        "gold_style": metrics["gold_median_style_delta_e76"]
        >= float(gates["gold_median_style_delta_e76_minimum"]),
        "gold_non_basic": metrics[
            "gold_median_non_basic_residual_delta_e76"
        ]
        >= float(
            gates["gold_median_non_basic_residual_delta_e76_minimum"]
        ),
        "gold_real_film_delta": metrics[
            "gold_median_real_film_delta_e76_from_base"
        ]
        >= float(
            gates["gold_median_real_film_delta_e76_from_base_minimum"]
        ),
        "gold_clipping": metrics["worst_gold_new_hard_clipping_fraction"]
        <= float(gates["worst_gold_new_hard_clipping_fraction_maximum"]),
        "stress_clipping": metrics["worst_stress_new_hard_clipping_fraction"]
        <= float(gates["worst_stress_new_hard_clipping_fraction_maximum"]),
        "selected_hashes": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    stable_evidence = {
        "contract_sha256": contract_sha256,
        "parent_automatic_report_sha256": contract[
            "parent_automatic_report_sha256"
        ],
        "parent_render_manifest_sha256": contract[
            "parent_render_manifest_sha256"
        ],
        "policy_id": policy["policy_id"],
        "metrics": metrics,
        "checks": checks,
        "selections": selections,
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable_evidence, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "schema": "neuro-film.u5-r2at0-ao6-adaptive-style-dose-report.v1",
        "experiment_id": contract["experiment_id"],
        "stable_evidence_id": stable_id,
        "automatic_passed": passed,
        "automatic_decision": (
            "open_development_visual_gate"
            if passed
            else "close_adaptive_dose"
        ),
        **stable_evidence,
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "SCHEMA",
    "evaluate_adaptive_style_dose",
    "load_contract",
]
