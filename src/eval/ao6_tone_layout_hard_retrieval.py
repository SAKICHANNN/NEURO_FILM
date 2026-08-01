"""Classical grayscale tone/layout hard retrieval for AO6 cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.ao6_content_hard_retrieval import (
    _run_audit,
    validate_contract as validate_bm2_contract,
)
from src.eval.ao6_global_lut_distillation import _load_exact_json


SCHEMA = "neuro_film.u5_r2bm3_ao6_tone_layout_hard_retrieval_report.v1"


class AO6ToneLayoutHardRetrievalError(RuntimeError):
    """Raised when BM3 contract or fixed descriptor semantics drift."""


def _extract_tone_layout_features(
    *, root: Path, views: list[np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    del root
    spec = config["feature_model"]
    quantile_count = int(spec["luma_quantiles"])
    grid_size = int(spec["spatial_grid"])
    gradient_count = int(spec["gradient_magnitude_quantiles"])
    features = []
    for view in views:
        luma = np.asarray(view[..., 0], dtype=np.float64)
        if (
            view.ndim != 3
            or view.shape[-1] != 3
            or not np.array_equal(view[..., 0], view[..., 1])
            or not np.array_equal(view[..., 0], view[..., 2])
            or not np.all(np.isfinite(luma))
            or np.any(luma < 0.0)
            or np.any(luma > 1.0)
        ):
            raise AO6ToneLayoutHardRetrievalError(
                "descriptor input must be replicated finite luma"
            )
        tone = np.quantile(
            luma,
            np.linspace(0.0, 1.0, quantile_count),
            method="linear",
        )
        spatial = []
        y_edges = np.linspace(0, luma.shape[0], grid_size + 1, dtype=int)
        x_edges = np.linspace(0, luma.shape[1], grid_size + 1, dtype=int)
        for y in range(grid_size):
            for x in range(grid_size):
                block = luma[
                    y_edges[y] : y_edges[y + 1],
                    x_edges[x] : x_edges[x + 1],
                ]
                spatial.extend((float(np.mean(block)), float(np.std(block))))
        delta_x = np.diff(luma, axis=1)[:-1]
        delta_y = np.diff(luma, axis=0)[:, :-1]
        magnitude = np.sqrt(delta_x * delta_x + delta_y * delta_y) / np.sqrt(2.0)
        gradients = np.quantile(
            magnitude,
            np.linspace(0.0, 1.0, gradient_count),
            method="linear",
        )
        feature = np.concatenate(
            (tone, np.asarray(spatial, dtype=np.float64), gradients)
        )
        if not np.all(np.isfinite(feature)) or np.any(feature < 0.0) or np.any(feature > 1.0):
            raise AO6ToneLayoutHardRetrievalError("descriptor left fixed bounds")
        features.append(feature.astype(np.float32))
    output = np.stack(features)
    expected = quantile_count + 2 * grid_size * grid_size + gradient_count
    if output.shape != (len(views), expected):
        raise AO6ToneLayoutHardRetrievalError("descriptor shape drift")
    return output


def validate_contract(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    feature = config["feature_model"]
    selector = config["selector"]
    if (
        config.get("schema") != "neuro_film.u5_r2bm3_ao6_tone_layout_hard_retrieval.v1"
        or config.get("status") != "contract_frozen_implementation_ready"
        or config.get("final_rgb_learning_allowed")
        or config.get("real_film_operator_fitting_allowed")
        or config.get("latent_mode_claim_allowed")
        or config.get("product_integration_allowed")
        or feature.get("training_allowed")
        or feature.get("fitting_allowed")
        or feature.get("colour_channels_used")
        or selector.get("target_pixels_used")
        or selector.get("source_chroma_used")
        or selector.get("operator_signatures_used_at_inference")
        or selector.get("dense_blending_allowed")
    ):
        raise AO6ToneLayoutHardRetrievalError("BM3 frozen contract drift")
    parent = config["parent"]
    decision = _load_exact_json(
        root, parent["decision"], parent["decision_sha256"]
    )
    if decision.get("decision") != parent["required_decision"]:
        raise AO6ToneLayoutHardRetrievalError("BM2 decision does not open BM3")
    bm2_config = _load_exact_json(
        root,
        decision["contract"]["path"],
        decision["contract"]["sha256"],
    )
    validated = validate_bm2_contract(root, bm2_config)
    bm1_decision = _load_exact_json(
        root, parent["bm1_decision"], parent["bm1_decision_sha256"]
    )
    bm1_report = _load_exact_json(
        root, parent["bm1_report"], parent["bm1_report_sha256"]
    )
    if (
        bm1_decision.get("decision")
        != "retain_case_oracle_open_content_space_hard_retrieval"
        or not bm1_report.get("automatic_pass")
        or bm1_report["aggregate"]["source_count"] != parent["expected_sources"]
    ):
        raise AO6ToneLayoutHardRetrievalError("BM1 Oracle drift")
    return {
        **validated,
        "bm1_report": bm1_report,
    }


def run_audit(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    return _run_audit(
        root=root,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        software_commit=software_commit,
        validated=validate_contract(root, config),
        feature_extractor=_extract_tone_layout_features,
        report_schema=SCHEMA,
        experiment_label="BM3",
    )


__all__ = [
    "AO6ToneLayoutHardRetrievalError",
    "_extract_tone_layout_features",
    "run_audit",
    "validate_contract",
]
