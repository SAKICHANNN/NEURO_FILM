"""U1.4C10 rerun of the frozen C4 evaluation through product ICC semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from src.eval import rec2020_native_prophoto_confirmation as c4
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json, hash_file
from src.preprocess import load_working_image
from src.preprocess.color_management import convert_linear_rgb

SCHEMA = "neuro-film.u1-4c10-icc-semantic-c4-recheck-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c10-icc-semantic-c4-recheck-report.v1"
EXPERIMENT_ID = "U1.4C10"
CONTRACT_SHA256 = "e2ef4852a59944ab9733f54604ae06544b741ac93be86b4715bbe8bcd4a7a6a5"


class ICCSemanticC4RecheckError(RuntimeError):
    """Raised when the frozen C10 contract or adapter drifts."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ICCSemanticC4RecheckError("C10 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise ICCSemanticC4RecheckError("C10 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ICCSemanticC4RecheckError("C10 contract structure drift")
    if (
        payload["execution"]["parameter_change_allowed"] is not False
        or payload["execution"]["gate_change_allowed"] is not False
        or payload["execution"]["require_two_process_report_and_png_identity"]
        is not True
    ):
        raise ICCSemanticC4RecheckError("C10 execution policy drift")
    for parent in payload["parents"].values():
        _relative(parent["path"])
    return payload


def _validate_parents(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for parent in contract["parents"].values():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise ICCSemanticC4RecheckError("C10 parent identity drift")
    c9 = json.loads(
        (root / _relative(contract["parents"]["c9_evidence"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if c9.get("status") != contract["parents"]["c9_evidence"]["required_status"]:
        raise ICCSemanticC4RecheckError("C10 requires the passing C9 evidence")
    return c4.load_contract(root / _relative(contract["parents"]["c4_contract"]["path"]))


def _load_semantic_source(
    row: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = root / c4._relative_path(row["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(row["bytes"])
        or hash_file(path) != row["sha256"]
    ):
        raise ICCSemanticC4RecheckError("C10 source identity drift")
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise ICCSemanticC4RecheckError("C10 source TIFF must contain one page")
        page = document.pages[0]
        tags = {tag.name: tag.value for tag in page.tags.values()}
        profile = tags.get("InterColorProfile")
        if (
            page.dtype != np.dtype(np.uint16)
            or tuple(page.shape) != (int(row["height"]), int(row["width"]), 3)
            or not isinstance(profile, bytes)
            or hashlib.sha256(profile).hexdigest()
            != config["source"]["expected_embedded_icc_sha256"]
            or str(tags.get("Make")) != row["make"]
            or str(tags.get("Model")) != row["model"]
        ):
            raise ICCSemanticC4RecheckError("C10 source TIFF structure drift")
    working = load_working_image(path)
    if (
        working.working_space != "linear_rec2020"
        or working.transfer_state != "display_linear"
        or working.pixels.dtype != np.float32
    ):
        raise ICCSemanticC4RecheckError("C10 product semantic ingress drift")
    extended = c4.resize_float(
        working.pixels, int(config["source"]["maximum_evaluation_side"])
    )
    linear_srgb = convert_linear_rgb(
        extended,
        source_space="linear_rec2020",
        destination_space="linear_srgb",
    )
    weights = np.asarray(
        config["ingress"]["rec2020_luminance_weights"], dtype=np.float64
    )
    mapped, ingress_scale, out_of_gamut = c4.luminance_axis_interior_compress(
        extended,
        luminance_weights=weights,
        margin=2.0 / 65535.0,
    )
    source_luminance = np.matmul(extended.astype(np.float64), weights)
    mapped_luminance = np.matmul(mapped.astype(np.float64), weights)
    facts = {
        "native_srgb_out_of_gamut_fraction": float(
            np.mean(np.any((linear_srgb < 0.0) | (linear_srgb > 1.0), axis=-1))
        ),
        "precompression_rec2020_out_of_gamut_fraction": float(np.mean(out_of_gamut)),
        "ingress_scale_median_on_out_of_gamut": float(
            np.median(ingress_scale[out_of_gamut])
        ),
        "ingress_scale_minimum": float(np.min(ingress_scale)),
        "maximum_luminance_preservation_absolute_error": float(
            np.max(np.abs(mapped_luminance - source_luminance))
        ),
        "mapped_source_array_sha256": c4._array_sha256(mapped),
        "mapped_source_boundary_fraction": float(
            np.mean(
                np.any(
                    (mapped <= 1.0 / 65536.0) | (mapped >= 1.0 - 1.0 / 65536.0),
                    axis=-1,
                )
            )
        ),
        "shape": list(mapped.shape),
    }
    return mapped, extended, facts


def _semantic_luminance_facts(
    row: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    path = root / c4._relative_path(row["path"])
    working = load_working_image(path)
    if (
        working.working_space != "linear_rec2020"
        or working.transfer_state != "display_linear"
        or working.pixels.dtype != np.float32
    ):
        raise ICCSemanticC4RecheckError("C10 product semantic ingress drift")
    extended = c4.resize_float(
        working.pixels, int(config["source"]["maximum_evaluation_side"])
    )
    weights = np.asarray(
        config["ingress"]["rec2020_luminance_weights"], dtype=np.float64
    )
    luminance = np.matmul(extended.astype(np.float64), weights)
    below = luminance < -2e-6
    above = luminance > 1.0 + 2e-6
    return {
        "id": row["id"],
        "source_sha256": row["sha256"],
        "shape": list(extended.shape),
        "minimum_luminance": float(np.min(luminance)),
        "maximum_luminance": float(np.max(luminance)),
        "fraction_luminance_below_c4_domain": float(np.mean(below)),
        "fraction_luminance_above_c4_domain": float(np.mean(above)),
        "c4_luminance_domain_pass": not bool(np.any(below) or np.any(above)),
        "extended_rec2020_array_sha256": c4._array_sha256(extended),
    }


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    c4_config = _validate_parents(contract, root)
    source_rows = c4._validate_inputs(c4_config, root)
    luminance_facts = [
        _semantic_luminance_facts(row, c4_config, root) for row in source_rows
    ]
    if not all(row["c4_luminance_domain_pass"] for row in luminance_facts):
        if output_dir.exists():
            raise ICCSemanticC4RecheckError("C10 output directory must be create-only")
        output_dir.mkdir(parents=True)
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": CONTRACT_SHA256,
            "parent_c4_contract_sha256": c4.CONTRACT_SHA256,
            "source_decode": contract["execution"]["source_decode"],
            "evaluation_resize": contract["execution"]["evaluation_resize"],
            "sources": luminance_facts,
            "rows": [],
            "metrics": {
                "source_count": len(luminance_facts),
                "luminance_domain_failure_count": sum(
                    not row["c4_luminance_domain_pass"] for row in luminance_facts
                ),
                "minimum_luminance": min(
                    row["minimum_luminance"] for row in luminance_facts
                ),
                "maximum_luminance": max(
                    row["maximum_luminance"] for row in luminance_facts
                ),
                "maximum_fraction_luminance_below_c4_domain": max(
                    row["fraction_luminance_below_c4_domain"]
                    for row in luminance_facts
                ),
                "maximum_fraction_luminance_above_c4_domain": max(
                    row["fraction_luminance_above_c4_domain"]
                    for row in luminance_facts
                ),
                "render_count": 0,
            },
            "checks": {"c4_luminance_domain": False},
            "failed_checks": ["c4_luminance_domain"],
            "automatic_pass": False,
            "decision": contract["decisions"]["fail"],
            "claim_ceiling": contract["claim_ceiling"],
            "product_ingress_opened": False,
            "production_default_changed": False,
        }
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
        return report
    original_loader = c4._load_native_source
    c4._load_native_source = _load_semantic_source
    try:
        report = c4.evaluate(c4_config, root, output_dir)
    finally:
        c4._load_native_source = original_loader
    automatic_pass = report["automatic_pass"] is True
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": CONTRACT_SHA256,
            "parent_c4_contract_sha256": c4.CONTRACT_SHA256,
            "source_decode": contract["execution"]["source_decode"],
            "evaluation_resize": contract["execution"]["evaluation_resize"],
            "decision": contract["decisions"]["pass" if automatic_pass else "fail"],
            "claim_ceiling": contract["claim_ceiling"],
            "product_ingress_opened": False,
            "production_default_changed": False,
        }
    )
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "CONTRACT_SHA256",
    "ICCSemanticC4RecheckError",
    "evaluate",
    "load_contract",
]
