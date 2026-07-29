"""Isolated photo frontier for the recorder-proxy code-reuse hypothesis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.eval.density_witness_frontier import (
    encoded_srgb_to_linear,
    linear_srgb_to_encoded,
)
from src.eval.global_frontier import (
    load_frozen_samples,
    new_hard_clipping_fraction,
    sha256_file,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.colorreference_bounded_bernstein import (
    BoundedBernsteinModel,
    apply_bounded_bernstein,
)
from src.roll2film.factorized_boundary_guard import (
    apply_target_residual_boundary_guard,
)


_D50_TO_D65 = np.asarray(
    [
        [0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098],
    ],
    dtype=np.float64,
)
_XYZ_D65_TO_LINEAR_SRGB = np.asarray(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


class RecorderProxyReuseError(ValueError):
    """Raised when the isolated code-reuse experiment drifts."""


def _load_bundle(
    root: Path, config: Mapping[str, Any]
) -> BoundedBernsteinModel:
    path = root / str(config["parent"]["bundle"])
    if sha256_file(path) != config["parent"]["bundle_sha256"]:
        raise RecorderProxyReuseError("recorder proxy bundle drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload["model_id"] != "bernstein_d3"
        or "recorder-device" not in payload["input_semantics"]
        or "not applicable to ordinary digital photographs"
        not in payload["claim_ceiling"]
    ):
        raise RecorderProxyReuseError("recorder proxy semantics drift")
    return BoundedBernsteinModel(
        degree=int(payload["degree"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        lower_bound=float(payload["lower_bound"]),
        upper_bound=float(payload["upper_bound"]),
    )


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    operator = config["operator_contract"]
    candidates = [dict(row) for row in config["strength_bank"]]
    if (
        config.get("experiment_id")
        != "u5.r2aq4c-recorder-proxy-code-reuse-frontier-v1"
        or config.get("production_integration_allowed")
        or config.get("operator_refit_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or operator.get("input_semantics_identified")
        or operator.get("post_operator_clipping_allowed")
        or operator.get("per_image_fit_or_adjustment_allowed")
        or len(candidates) != config["candidate_count"]
        or [row["strength"] for row in candidates] != [0.5, 0.75, 1.0]
    ):
        raise RecorderProxyReuseError("AQ4C frozen contract drift")
    return {
        "model": _load_bundle(root, config),
        "samples": load_frozen_samples(root, config),
        "candidates": candidates,
    }


def recorder_target_linear_srgb(
    model: BoundedBernsteinModel,
    encoded_srgb: np.ndarray,
) -> np.ndarray:
    values = np.asarray(encoded_srgb, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise RecorderProxyReuseError(
            "recorder proxy input must be finite encoded RGB"
        )
    xyz_d50 = apply_bounded_bernstein(model, values)
    xyz_d65 = xyz_d50 @ _D50_TO_D65.T
    return xyz_d65 @ _XYZ_D65_TO_LINEAR_SRGB.T


def _read_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        if transposed.mode != "RGB":
            raise RecorderProxyReuseError(
                f"expected exact RGB source: {path}"
            )
        return np.asarray(transposed, dtype=np.uint8)


def render_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    model = validated["model"]
    controls = config["operator_contract"]
    records = []
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in sorted(validated["samples"]):
            sample = validated["samples"][sample_id]
            source_path = root / str(sample["source_path"])
            if sha256_file(source_path) != sample["source_sha256"]:
                raise RecorderProxyReuseError(
                    f"source hash mismatch: {sample_id}"
                )
            source_uint8 = _read_rgb8(source_path)
            output_uint8 = np.empty_like(source_uint8)
            limited_count = 0
            total_pixels = 0
            minimum_scale = 1.0
            raw_oog_count = 0
            for start in range(0, source_uint8.shape[0], 128):
                stop = min(start + 128, source_uint8.shape[0])
                encoded_image = (
                    source_uint8[start:stop].astype(np.float64) / 255.0
                )
                encoded = encoded_image.reshape(-1, 3)
                source_linear = encoded_srgb_to_linear(
                    encoded_image
                ).reshape(-1, 3)
                target_linear = recorder_target_linear_srgb(model, encoded)
                raw_oog_count += int(
                    np.sum(
                        (target_linear < 0.0) | (target_linear > 1.0)
                    )
                )
                guarded = apply_target_residual_boundary_guard(
                    source_linear,
                    target_linear,
                    strength=float(candidate["strength"]),
                    hard_boundary_epsilon_encoded_srgb=float(
                        controls[
                            "hard_boundary_epsilon_encoded_srgb"
                        ]
                    ),
                    guard_boundary_epsilon_encoded_srgb=float(
                        controls[
                            "guard_boundary_epsilon_encoded_srgb"
                        ]
                    ),
                )
                output_uint8[start:stop] = np.rint(
                    linear_srgb_to_encoded(guarded.output).reshape(
                        stop - start, source_uint8.shape[1], 3
                    )
                    * 255.0
                ).astype(np.uint8)
                limited_count += int(
                    np.sum(guarded.residual_scale < 1.0 - 1e-12)
                )
                total_pixels += int(guarded.residual_scale.size)
                minimum_scale = min(
                    minimum_scale, float(np.min(guarded.residual_scale))
                )
            path = candidate_dir / f"{sample_id}.png"
            Image.fromarray(output_uint8, mode="RGB").save(
                path, format="PNG", compress_level=6
            )
            records.append(
                {
                    **candidate,
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "source_sha256": sample["source_sha256"],
                    "output": f"{candidate_id}/{sample_id}.png",
                    "output_sha256": sha256_file(path),
                    "residual_limited_fraction": limited_count
                    / total_pixels,
                    "minimum_residual_scale": minimum_scale,
                    "raw_target_oog_component_fraction": raw_oog_count
                    / (3 * total_pixels),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "bundle_sha256": config["parent"]["bundle_sha256"],
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(validated["samples"]),
        "records": records,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = output_dir / "manifest.json"
    path.write_bytes(encoded)
    return {
        "manifest": manifest,
        "manifest_path": path,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _median_delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    first_lab = rgb2lab(
        np.clip(first, 0.0, 1.0).reshape(-1, 1, 3)
    ).reshape(-1, 3)
    second_lab = rgb2lab(
        np.clip(second, 0.0, 1.0).reshape(-1, 1, 3)
    ).reshape(-1, 3)
    return float(np.median(np.linalg.norm(second_lab - first_lab, axis=1)))


def evaluate_bank(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("frozen_set_sha256")
        != config["frozen_set_sha256"]
        or manifest.get("bundle_sha256")
        != config["parent"]["bundle_sha256"]
    ):
        raise RecorderProxyReuseError("AQ4C manifest lineage drift")
    expected = {
        (str(candidate["candidate_id"]), sample_id)
        for candidate in validated["candidates"]
        for sample_id in validated["samples"]
    }
    records = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise RecorderProxyReuseError(
                f"unexpected or duplicate AQ4C record: {key}"
            )
        records[key] = dict(row)
    if records.keys() != expected:
        raise RecorderProxyReuseError("AQ4C manifest is incomplete")
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries = {}
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        per_image = []
        for sample_id, sample in validated["samples"].items():
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            if (
                record["source_sha256"] != sample["source_sha256"]
                or sha256_file(output_path) != record["output_sha256"]
            ):
                raise RecorderProxyReuseError(
                    f"AQ4C output lineage drift: {candidate_id}/{sample_id}"
                )
            source_path = root / str(sample["source_path"])
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(source_path) as source_image:
                    source_size = ImageOps.exif_transpose(
                        source_image
                    ).size
                if output.mode != "RGB" or output.size != source_size:
                    raise RecorderProxyReuseError(
                        f"AQ4C output shape drift: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels, output_pixels
            )
            raw_target = recorder_target_linear_srgb(
                validated["model"], source_pixels
            )
            raw_preview = linear_srgb_to_encoded(
                np.clip(raw_target, 0.0, 1.0)
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "median_unconstrained_operator_delta_e76": (
                        _median_delta_e76(source_pixels, raw_preview)
                    ),
                    "new_hard_clipping_fraction": (
                        new_hard_clipping_fraction(
                            source_pixels, output_pixels, epsilon
                        )
                    ),
                    "residual_limited_fraction": record[
                        "residual_limited_fraction"
                    ],
                    "minimum_residual_scale": record[
                        "minimum_residual_scale"
                    ],
                    "raw_target_oog_component_fraction": record[
                        "raw_target_oog_component_fraction"
                    ],
                    "output_sha256": record["output_sha256"],
                }
            )
        gold = [row for row in per_image if row["split"] == "gold"]
        stress = [row for row in per_image if row["split"] == "stress"]
        summary = {
            **candidate,
            "gold_median_style_delta_e76": float(
                np.median(
                    [row["median_style_delta_e76"] for row in gold]
                )
            ),
            "gold_median_non_basic_residual_delta_e76": float(
                np.median(
                    [
                        row["median_non_basic_residual_delta_e76"]
                        for row in gold
                    ]
                )
            ),
            "gold_median_unconstrained_operator_delta_e76": float(
                np.median(
                    [
                        row["median_unconstrained_operator_delta_e76"]
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
            "maximum_gold_residual_limited_fraction": float(
                max(row["residual_limited_fraction"] for row in gold)
            ),
            "per_image": per_image,
        }
        metrics = config["metrics"]
        summary["automatic_gates"] = {
            "style_floor": summary["gold_median_style_delta_e76"]
            >= metrics["minimum_gold_median_style_delta_e76"],
            "non_basic_floor": summary[
                "gold_median_non_basic_residual_delta_e76"
            ]
            >= metrics[
                "minimum_gold_median_non_basic_residual_delta_e76"
            ],
            "operator_materiality": summary[
                "gold_median_unconstrained_operator_delta_e76"
            ]
            >= metrics[
                "minimum_gold_median_unconstrained_operator_delta_e76"
            ],
            "gold_clipping": summary[
                "worst_gold_new_hard_clipping_fraction"
            ]
            <= metrics[
                "maximum_worst_gold_new_hard_clipping_fraction"
            ],
            "stress_clipping": summary[
                "worst_stress_new_hard_clipping_fraction"
            ]
            <= metrics[
                "maximum_worst_stress_new_hard_clipping_fraction"
            ],
        }
        summary["automatic_survivor"] = all(
            summary["automatic_gates"].values()
        )
        summaries[candidate_id] = summary
    survivors = [
        row for row in summaries.values() if row["automatic_survivor"]
    ]
    survivors.sort(
        key=lambda row: (
            -row["gold_median_non_basic_residual_delta_e76"],
            -row["gold_median_style_delta_e76"],
            row["strength"],
            row["candidate_id"],
        )
    )
    shortlist = [
        row["candidate_id"]
        for row in survivors[
            : int(config["shortlist"]["maximum_candidates"])
        ]
    ]
    return {
        "candidate_count": len(validated["candidates"]),
        "sample_count": len(validated["samples"]),
        "gold_sample_count": sum(
            row["split"] == "gold"
            for row in validated["samples"].values()
        ),
        "stress_sample_count": sum(
            row["split"] == "stress"
            for row in validated["samples"].values()
        ),
        "all_source_and_output_hashes_verified": True,
        "all_outputs_rgb_and_dimension_matched": True,
        "candidates": summaries,
        "automatic_survivors": sorted(
            row["candidate_id"] for row in survivors
        ),
        "shortlist": shortlist,
        "automatic_decision": (
            "visual_gate_required" if shortlist else "no_automatic_survivor"
        ),
    }


__all__ = [
    "RecorderProxyReuseError",
    "evaluate_bank",
    "recorder_target_linear_srgb",
    "render_bank",
    "validate_contract",
]
