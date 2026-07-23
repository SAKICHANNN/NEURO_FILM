"""Data-independent projected-LUT frontier for U5.R2F2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.canoncgt_reference_condition import out_of_range_fraction
from src.eval.global_frontier import new_hard_clipping_fraction, sha256_file
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.lut import DenseLUT3D


class ProjectedLUTError(ValueError):
    """Raised when frozen projection evidence is invalid."""


def identity_lut(size: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def convert_public_lut(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 4 or array.shape[0] != 3 or len(set(array.shape[1:])) != 1:
        raise ProjectedLUTError("public LUT must have shape (3,N,N,N)")
    return np.transpose(array, (3, 2, 1, 0))


def lut_diagnostics(values: np.ndarray) -> dict[str, float | bool]:
    array = np.asarray(values, dtype=np.float64)
    lut = DenseLUT3D(
        array,
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        "trilinear",
    )
    steps = (
        np.diff(array[..., 0], axis=0),
        np.diff(array[..., 1], axis=1),
        np.diff(array[..., 2], axis=2),
    )
    determinants = lut.tetrahedron_jacobian_determinants()
    return {
        "finite": bool(np.all(np.isfinite(array))),
        "minimum_node": float(np.min(array)),
        "maximum_node": float(np.max(array)),
        "minimum_corresponding_channel_grid_step": float(
            min(np.min(step) for step in steps)
        ),
        "minimum_tetrahedron_jacobian_determinant": float(
            np.min(determinants)
        ),
        "negative_tetrahedron_fraction": float(np.mean(determinants < 0.0)),
    }


def structure_safe(
    diagnostics: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> bool:
    return bool(
        diagnostics["finite"]
        and float(diagnostics["minimum_node"])
        >= float(projection["node_minimum"])
        and float(diagnostics["maximum_node"])
        <= float(projection["node_maximum"])
        and float(diagnostics["minimum_corresponding_channel_grid_step"])
        >= float(projection["minimum_corresponding_channel_grid_step"])
        and float(diagnostics["minimum_tetrahedron_jacobian_determinant"])
        >= float(projection["minimum_tetrahedron_jacobian_determinant"])
    )


def project_lut(
    public_values: np.ndarray,
    policy: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    converted = convert_public_lut(public_values)
    clipped = np.clip(
        converted,
        float(projection["node_minimum"]),
        float(projection["node_maximum"]),
    )
    identity = identity_lut(converted.shape[0])
    cap = float(policy["identity_contraction_cap"])

    def at(alpha: float) -> np.ndarray:
        return identity + alpha * (clipped - identity)

    if not bool(policy["enforce_structure_prefix"]):
        alpha = cap
    else:
        samples = int(projection["alpha_prefix_grid_samples"])
        grid = np.linspace(0.0, cap, samples)
        low, high = 0.0, cap
        found_unsafe = False
        for index, candidate in enumerate(grid[1:], start=1):
            if not structure_safe(lut_diagnostics(at(float(candidate))), projection):
                low = float(grid[index - 1])
                high = float(candidate)
                found_unsafe = True
                break
            low = float(candidate)
        if found_unsafe:
            for _ in range(int(projection["alpha_bisection_iterations"])):
                middle = 0.5 * (low + high)
                if structure_safe(lut_diagnostics(at(middle)), projection):
                    low = middle
                else:
                    high = middle
            alpha = low
        else:
            alpha = cap
    result = at(alpha)
    diagnostics = lut_diagnostics(result)
    diagnostics.update(
        {
            "applied_alpha": float(alpha),
            "cap": cap,
            "structure_safe": structure_safe(diagnostics, projection),
        }
    )
    return result, diagnostics


def _load_inputs(root: Path, f1: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    spec = f1["input_set"]
    path = root / str(spec["manifest"])
    if sha256_file(path) != str(spec["manifest_sha256"]):
        raise ProjectedLUTError("input manifest hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frozen_set", payload)["samples"]
    by_id = {str(row["id"]): dict(row) for row in rows}
    result = {str(value): by_id[str(value)] for value in spec["sample_ids"]}
    for sample_id, row in result.items():
        if sha256_file(root / str(row["source_path"])) != row["source_sha256"]:
            raise ProjectedLUTError(f"source hash mismatch: {sample_id}")
    return result


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = (
        ("source_f1_config", "source_f1_config_sha256"),
        ("source_f1_manifest", "source_f1_manifest_sha256"),
        ("source_f1_report", "source_f1_report_sha256"),
    )
    for path_key, hash_key in paths:
        if sha256_file(root / str(config[path_key])) != str(config[hash_key]):
            raise ProjectedLUTError(f"{path_key} hash mismatch")
    f1 = json.loads(
        (root / str(config["source_f1_config"])).read_text(encoding="utf-8")
    )
    manifest_path = root / str(config["source_f1_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        (str(row["reference_id"]), str(row["sample_id"])): dict(row)
        for row in manifest["records"]
    }
    references = {str(row["reference_id"]): dict(row) for row in f1["references"]}
    if set(references) != set(str(v) for v in config["reference_ids"]):
        raise ProjectedLUTError("reference bank drift")
    samples = _load_inputs(root, f1)
    if len(samples) != int(config["sample_count"]):
        raise ProjectedLUTError("sample count drift")
    expected = {(reference, sample) for reference in references for sample in samples}
    if records.keys() != expected:
        raise ProjectedLUTError("F1 manifest is incomplete")
    for row in records.values():
        for key in ("canonical_lut", "restyle_lut"):
            path = manifest_path.parent / str(row[key])
            if sha256_file(path) != str(row[f"{key}_file_sha256"]):
                raise ProjectedLUTError("source LUT hash mismatch")
    return {
        "f1": f1,
        "manifest_path": manifest_path,
        "records": records,
        "references": references,
        "samples": samples,
    }


def candidate_bank(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for reference in config["reference_ids"]:
        for policy in config["projection_policies"]:
            rows.append(
                {
                    "candidate_id": str(config["candidate_id_format"]).format(
                        reference_id=reference,
                        policy_id=policy["policy_id"],
                    ),
                    "reference_id": str(reference),
                    "policy": dict(policy),
                }
            )
    if len(rows) != int(config["candidate_count"]):
        raise ProjectedLUTError("candidate count mismatch")
    return rows


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for candidate in candidate_bank(config):
        candidate_dir = output_dir / candidate["candidate_id"]
        lut_dir = candidate_dir / "luts"
        lut_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, sample in validated["samples"].items():
            source_row = validated["records"][
                (candidate["reference_id"], sample_id)
            ]
            source_luts = []
            projected = []
            diagnostics = []
            for key in ("canonical_lut", "restyle_lut"):
                source_path = validated["manifest_path"].parent / str(source_row[key])
                raw = np.load(source_path, allow_pickle=False)
                values, audit = project_lut(
                    raw,
                    candidate["policy"],
                    config["projection"],
                )
                source_luts.append(source_path)
                projected.append(values)
                diagnostics.append(audit)
            with Image.open(root / str(sample["source_path"])) as image:
                source = (
                    np.asarray(
                        ImageOps.exif_transpose(image).convert("RGB"),
                        dtype=np.float64,
                    )
                    / 255.0
                )
            current = source
            for values in projected:
                current = DenseLUT3D(
                    values,
                    np.zeros(3),
                    np.ones(3),
                    "trilinear",
                ).apply(current)
            if not np.all(np.isfinite(current)):
                raise ProjectedLUTError("non-finite projected output")
            output_path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(
                np.rint(np.clip(current, 0.0, 1.0) * 255.0).astype(np.uint8),
                mode="RGB",
            ).save(output_path, format="PNG", compress_level=6)
            lut_paths = []
            for stage, values in zip(("canonical", "restyle"), projected):
                path = lut_dir / f"{sample_id}_{stage}.npy"
                np.save(path, np.asarray(values, dtype="<f8"), allow_pickle=False)
                lut_paths.append(path)
            records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reference_id": candidate["reference_id"],
                    "provenance_bucket": validated["references"][
                        candidate["reference_id"]
                    ]["provenance_bucket"],
                    "policy_id": candidate["policy"]["policy_id"],
                    "sample_id": sample_id,
                    "source_sha256": sample["source_sha256"],
                    "source_canonical_lut_sha256": sha256_file(source_luts[0]),
                    "source_restyle_lut_sha256": sha256_file(source_luts[1]),
                    "projected_canonical_lut": (
                        f"{candidate['candidate_id']}/luts/{sample_id}_canonical.npy"
                    ),
                    "projected_canonical_lut_sha256": sha256_file(lut_paths[0]),
                    "projected_restyle_lut": (
                        f"{candidate['candidate_id']}/luts/{sample_id}_restyle.npy"
                    ),
                    "projected_restyle_lut_sha256": sha256_file(lut_paths[1]),
                    "canonical_projection": diagnostics[0],
                    "restyle_projection": diagnostics[1],
                    "raw_final_minimum": float(np.min(current)),
                    "raw_final_maximum": float(np.max(current)),
                    "raw_final_out_of_range_fraction": out_of_range_fraction(current),
                    "output": f"{candidate['candidate_id']}/{sample_id}.png",
                    "output_sha256": sha256_file(output_path),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "source_f1_manifest_sha256": config["source_f1_manifest_sha256"],
        "candidate_count": int(config["candidate_count"]),
        "sample_count": len(validated["samples"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "manifest": manifest,
    }


def _representatives(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    by_reference: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for candidate_id, row in summaries.items():
        if row["automatic_survivor"]:
            by_reference.setdefault(str(row["reference_id"]), []).append(
                (candidate_id, row)
            )
    result = {}
    for reference, rows in by_reference.items():
        rows.sort(
            key=lambda item: (
                -float(item[1]["gold_median_non_basic_residual_delta_e76"]),
                -float(item[1]["gold_median_style_delta_e76"]),
                -float(item[1]["applied_alpha_minimum"]),
                item[0],
            )
        )
        result[reference] = rows[0][0]
    return result


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = candidate_bank(config)
    expected = {
        (row["candidate_id"], sample_id)
        for row in candidates
        for sample_id in validated["samples"]
    }
    records = {
        (str(row["candidate_id"]), str(row["sample_id"])): dict(row)
        for row in manifest["records"]
    }
    if records.keys() != expected:
        raise ProjectedLUTError("projected manifest is incomplete")
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries = {}
    for candidate in candidates:
        per_image = []
        structure = []
        alphas = []
        for sample_id, sample in validated["samples"].items():
            row = records[(candidate["candidate_id"], sample_id)]
            output_path = manifest_path.parent / str(row["output"])
            if sha256_file(output_path) != row["output_sha256"]:
                raise ProjectedLUTError("output hash mismatch")
            source_path = root / str(sample["source_path"])
            source = sample_rgb_image(source_path, budget)
            output = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(source, output)
            per_image.append(
                {
                    "sample_id": sample_id,
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "new_hard_clipping_fraction": new_hard_clipping_fraction(
                        source, output, epsilon
                    ),
                    "raw_final_out_of_range_fraction": row[
                        "raw_final_out_of_range_fraction"
                    ],
                }
            )
            for key in ("canonical_projection", "restyle_projection"):
                structure.append(bool(row[key]["structure_safe"]))
                alphas.append(float(row[key]["applied_alpha"]))
        summary = {
            "reference_id": candidate["reference_id"],
            "provenance_bucket": records[
                (candidate["candidate_id"], next(iter(validated["samples"])))
            ]["provenance_bucket"],
            "policy_id": candidate["policy"]["policy_id"],
            "all_projected_luts_structure_safe": all(structure),
            "applied_alpha_minimum": min(alphas),
            "applied_alpha_median": float(np.median(alphas)),
            "gold_median_style_delta_e76": float(
                np.median([row["median_style_delta_e76"] for row in per_image])
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [row["median_non_basic_residual_delta_e76"] for row in per_image]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": max(
                row["new_hard_clipping_fraction"] for row in per_image
            ),
            "worst_gold_raw_final_out_of_range_fraction": max(
                row["raw_final_out_of_range_fraction"] for row in per_image
            ),
            "per_image": per_image,
        }
        gates = {
            "structure": summary["all_projected_luts_structure_safe"],
            "style": summary["gold_median_style_delta_e76"]
            >= float(config["metrics"]["minimum_gold_median_style_delta_e76"]),
            "non_basic": summary["gold_median_non_basic_residual_delta_e76"]
            >= float(
                config["metrics"][
                    "minimum_gold_median_non_basic_residual_delta_e76"
                ]
            ),
            "clipping": summary["worst_gold_new_hard_clipping_fraction"]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_new_hard_clipping_fraction"
                ]
            ),
            "raw_range": summary["worst_gold_raw_final_out_of_range_fraction"]
            <= float(
                config["metrics"][
                    "maximum_worst_gold_raw_final_out_of_range_fraction"
                ]
            ),
        }
        summary["automatic_gates"] = gates
        summary["automatic_survivor"] = all(gates.values())
        summaries[candidate["candidate_id"]] = summary
    representatives = _representatives(summaries)
    sensitivity_values = []
    if len(representatives) >= 2:
        reference_ids = sorted(representatives)
        for sample_id in validated["samples"]:
            arrays = {
                ref: sample_rgb_image(
                    manifest_path.parent
                    / records[(representatives[ref], sample_id)]["output"],
                    budget,
                )
                for ref in reference_ids
            }
            labs = {
                ref: rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
                for ref, arr in arrays.items()
            }
            for index, first in enumerate(reference_ids):
                for second in reference_ids[index + 1 :]:
                    sensitivity_values.append(
                        float(
                            np.median(
                                np.linalg.norm(
                                    labs[first] - labs[second], axis=1
                                )
                            )
                        )
                    )
    sensitivity = (
        float(np.median(sensitivity_values)) if sensitivity_values else 0.0
    )
    sensitive = sensitivity >= float(
        config["metrics"][
            "minimum_reference_bank_median_pairwise_output_delta_e76"
        ]
    )
    reps = [
        (candidate_id, summaries[candidate_id])
        for candidate_id in representatives.values()
    ]
    reps.sort(
        key=lambda item: (
            -float(item[1]["gold_median_non_basic_residual_delta_e76"]),
            -float(item[1]["gold_median_style_delta_e76"]),
            item[0],
        )
    )
    seen = set()
    shortlist = []
    if sensitive:
        for candidate_id, row in reps:
            bucket = row["provenance_bucket"]
            if bucket in seen:
                continue
            seen.add(bucket)
            shortlist.append(candidate_id)
            if len(shortlist) >= int(config["shortlist"]["maximum_candidates"]):
                break
    return {
        "candidate_count": len(candidates),
        "sample_count": len(validated["samples"]),
        "candidates": summaries,
        "automatic_survivors": sorted(
            key for key, row in summaries.items() if row["automatic_survivor"]
        ),
        "reference_representatives": representatives,
        "reference_sensitivity_median_pairwise_output_delta_e76": sensitivity,
        "reference_sensitivity_gate_passed": sensitive,
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required"
            if shortlist
            else (
                "reference_conditioning_too_weak"
                if representatives and not sensitive
                else "no_automatic_survivor"
            )
        ),
    }


__all__ = [
    "ProjectedLUTError",
    "candidate_bank",
    "convert_public_lut",
    "evaluate_bank",
    "identity_lut",
    "lut_diagnostics",
    "project_lut",
    "render_bank",
    "structure_safe",
    "validate_contract",
]
