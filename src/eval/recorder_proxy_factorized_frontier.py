"""Factorized photo frontier for the bounded recorder-proxy hypothesis."""

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
from src.eval.recorder_proxy_code_reuse_frontier import (
    _load_bundle,
    recorder_target_linear_srgb,
)
from src.real_film.gold_matrix_transplant import (
    sample_rgb_image,
    style_and_basic_residual,
)
from src.roll2film.factorized_boundary_guard import (
    apply_target_factorized_boundary_guard,
)


class RecorderProxyFactorizedError(ValueError):
    """Raised when the frozen factorized recorder-proxy contract drifts."""


def _load_direct_records(
    root: Path, config: Mapping[str, Any]
) -> tuple[Path, dict[str, dict[str, Any]]]:
    path = root / str(config["parent"]["direct_manifest"])
    if sha256_file(path) != config["parent"]["direct_manifest_sha256"]:
        raise RecorderProxyFactorizedError("AQ4C direct manifest drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_id = str(config["parent"]["direct_candidate_id"])
    records: dict[str, dict[str, Any]] = {}
    for row in payload.get("records", []):
        if str(row.get("candidate_id")) != candidate_id:
            continue
        sample_id = str(row.get("sample_id"))
        if sample_id in records:
            raise RecorderProxyFactorizedError(
                f"duplicate AQ4C direct record: {sample_id}"
            )
        records[sample_id] = dict(row)
    return path, records


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    factor = config["factorization"]
    candidates = [dict(row) for row in config["candidate_bank"]]
    expected = [
        ("recorder_proxy_factor_t50_c125", 0.5, 1.25),
        ("recorder_proxy_factor_t50_c175", 0.5, 1.75),
        ("recorder_proxy_factor_t75_c200", 0.75, 2.0),
    ]
    actual = [
        (
            str(row["candidate_id"]),
            float(row["tone_strength"]),
            float(row["chroma_strength"]),
        )
        for row in candidates
    ]
    if (
        config.get("experiment_id")
        != "u5.r2aq4f-recorder-proxy-factorized-frontier-v1"
        or config.get("production_integration_allowed")
        or config.get("operator_refit_allowed")
        or config.get("stock_response_claim_allowed")
        or config.get("calibrated_reference_claim_allowed")
        or factor.get("post_operator_clipping_allowed")
        or factor.get("per_image_fit_or_adjustment_allowed")
        or factor.get("execution")
        != "sequential source-inclusive analytical maximum safe scale for tone then chroma"
        or len(candidates) != config["candidate_count"]
        or actual != expected
    ):
        raise RecorderProxyFactorizedError("AQ4F frozen contract drift")
    samples = load_frozen_samples(root, config)
    direct_path, direct_records = _load_direct_records(root, config)
    if set(direct_records) != set(samples):
        raise RecorderProxyFactorizedError(
            "AQ4C direct control does not cover the frozen set"
        )
    for sample_id, row in direct_records.items():
        output = direct_path.parent / str(row["output"])
        if (
            row.get("source_sha256") != samples[sample_id]["source_sha256"]
            or sha256_file(output) != row.get("output_sha256")
        ):
            raise RecorderProxyFactorizedError(
                f"AQ4C direct output drift: {sample_id}"
            )
    return {
        "model": _load_bundle(root, config),
        "samples": samples,
        "candidates": candidates,
        "direct_manifest_path": direct_path,
        "direct_records": direct_records,
    }


def _read_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image)
        if transposed.mode != "RGB":
            raise RecorderProxyFactorizedError(
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
    controls = config["factorization"]
    weights = np.asarray(controls["luma_weights"], dtype=np.float64)
    records = []
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in sorted(validated["samples"]):
            sample = validated["samples"][sample_id]
            source_path = root / str(sample["source_path"])
            if sha256_file(source_path) != sample["source_sha256"]:
                raise RecorderProxyFactorizedError(
                    f"source hash mismatch: {sample_id}"
                )
            source_uint8 = _read_rgb8(source_path)
            output_uint8 = np.empty_like(source_uint8)
            tone_limited = 0
            chroma_limited = 0
            total_pixels = 0
            minimum_tone_scale = 1.0
            minimum_chroma_scale = 1.0
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
                target_linear = recorder_target_linear_srgb(
                    validated["model"], encoded
                )
                raw_oog_count += int(
                    np.sum(
                        (target_linear < 0.0) | (target_linear > 1.0)
                    )
                )
                guarded = apply_target_factorized_boundary_guard(
                    source_linear,
                    target_linear,
                    tone_strength=float(candidate["tone_strength"]),
                    chroma_strength=float(candidate["chroma_strength"]),
                    luma_weights=weights,
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
                    linear_srgb_to_encoded(
                        guarded.output.reshape(
                            stop - start, source_uint8.shape[1], 3
                        )
                    )
                    * 255.0
                ).astype(np.uint8)
                tone_limited += int(
                    np.sum(guarded.tone_scale < 1.0 - 1e-12)
                )
                chroma_limited += int(
                    np.sum(guarded.chroma_scale < 1.0 - 1e-12)
                )
                total_pixels += int(guarded.tone_scale.size)
                minimum_tone_scale = min(
                    minimum_tone_scale, float(np.min(guarded.tone_scale))
                )
                minimum_chroma_scale = min(
                    minimum_chroma_scale,
                    float(np.min(guarded.chroma_scale)),
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
                    "tone_limited_fraction": tone_limited / total_pixels,
                    "chroma_limited_fraction": chroma_limited / total_pixels,
                    "minimum_tone_scale": minimum_tone_scale,
                    "minimum_chroma_scale": minimum_chroma_scale,
                    "raw_target_oog_component_fraction": (
                        raw_oog_count / (3 * total_pixels)
                    ),
                }
            )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_set_sha256": config["frozen_set_sha256"],
        "bundle_sha256": config["parent"]["bundle_sha256"],
        "direct_manifest_sha256": config["parent"][
            "direct_manifest_sha256"
        ],
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
        or manifest.get("direct_manifest_sha256")
        != config["parent"]["direct_manifest_sha256"]
    ):
        raise RecorderProxyFactorizedError("AQ4F manifest lineage drift")
    expected = {
        (str(candidate["candidate_id"]), sample_id)
        for candidate in validated["candidates"]
        for sample_id in validated["samples"]
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest.get("records", []):
        key = (str(row.get("candidate_id")), str(row.get("sample_id")))
        if key not in expected or key in records:
            raise RecorderProxyFactorizedError(
                f"unexpected or duplicate AQ4F record: {key}"
            )
        records[key] = dict(row)
    if set(records) != expected:
        raise RecorderProxyFactorizedError("AQ4F manifest is incomplete")
    budget = int(config["metrics"]["maximum_pixels_per_image"])
    epsilon = float(config["metrics"]["new_hard_clipping_epsilon"])
    summaries = {}
    for candidate in validated["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        per_image = []
        for sample_id, sample in validated["samples"].items():
            record = records[(candidate_id, sample_id)]
            output_path = manifest_path.parent / str(record["output"])
            direct_record = validated["direct_records"][sample_id]
            direct_path = (
                validated["direct_manifest_path"].parent
                / str(direct_record["output"])
            )
            if (
                record["source_sha256"] != sample["source_sha256"]
                or sha256_file(output_path) != record["output_sha256"]
            ):
                raise RecorderProxyFactorizedError(
                    f"AQ4F output lineage drift: {candidate_id}/{sample_id}"
                )
            source_path = root / str(sample["source_path"])
            with Image.open(output_path) as image:
                output = ImageOps.exif_transpose(image)
                with Image.open(source_path) as source_image:
                    source_size = ImageOps.exif_transpose(
                        source_image
                    ).size
                if output.mode != "RGB" or output.size != source_size:
                    raise RecorderProxyFactorizedError(
                        f"AQ4F output shape drift: {candidate_id}/{sample_id}"
                    )
            source_pixels = sample_rgb_image(source_path, budget)
            output_pixels = sample_rgb_image(output_path, budget)
            direct_pixels = sample_rgb_image(direct_path, budget)
            style, residual = style_and_basic_residual(
                source_pixels, output_pixels
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "split": sample["split"],
                    "median_style_delta_e76": style,
                    "median_non_basic_residual_delta_e76": residual,
                    "median_factorized_delta_e76": _median_delta_e76(
                        direct_pixels, output_pixels
                    ),
                    "new_hard_clipping_fraction": (
                        new_hard_clipping_fraction(
                            source_pixels, output_pixels, epsilon
                        )
                    ),
                    "tone_limited_fraction": record[
                        "tone_limited_fraction"
                    ],
                    "chroma_limited_fraction": record[
                        "chroma_limited_fraction"
                    ],
                    "minimum_tone_scale": record["minimum_tone_scale"],
                    "minimum_chroma_scale": record[
                        "minimum_chroma_scale"
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
            "gold_median_factorized_delta_e76": float(
                np.median(
                    [row["median_factorized_delta_e76"] for row in gold]
                )
            ),
            "worst_gold_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in gold)
            ),
            "worst_stress_new_hard_clipping_fraction": float(
                max(row["new_hard_clipping_fraction"] for row in stress)
            ),
            "maximum_gold_tone_limited_fraction": float(
                max(row["tone_limited_fraction"] for row in gold)
            ),
            "maximum_gold_chroma_limited_fraction": float(
                max(row["chroma_limited_fraction"] for row in gold)
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
            "factorized_delta": summary[
                "gold_median_factorized_delta_e76"
            ]
            >= metrics["minimum_gold_median_factorized_delta_e76"],
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
            -row["gold_median_factorized_delta_e76"],
            -row["gold_median_style_delta_e76"],
            row["tone_strength"],
            row["chroma_strength"],
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
        "all_source_direct_and_output_hashes_verified": True,
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
    "RecorderProxyFactorizedError",
    "evaluate_bank",
    "render_bank",
    "validate_contract",
]
