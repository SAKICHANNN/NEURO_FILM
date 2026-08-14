"""U1.4C13 C11 stress confirmation on CC0 scene structure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tifffile

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.oklab_analytical_interior_confirmation import _evaluate_prevalidated

SCHEMA = "neuro-film.u1-4c13-cc0-synthetic-romm-stress-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c13-cc0-synthetic-romm-stress-report.v1"
EXPERIMENT_ID = "U1.4C13"
CONTRACT_SHA256 = "7eb63fd2edc482e6557d35c0e77a293f323951634b559e2050e74c5f09a9840f"


class CC0SyntheticROMMConfirmationError(RuntimeError):
    """Raised when a frozen C13 identity or source invariant fails."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise CC0SyntheticROMMConfirmationError(
            "C13 paths must be repository-relative"
        )
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise CC0SyntheticROMMConfirmationError("C13 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CC0SyntheticROMMConfirmationError("C13 contract structure drift")
    mapper = payload["mapper"]
    if (
        mapper.get("softness") != 1.0 / 64.0
        or mapper.get("rgb16_margin") != 2.0 / 65535.0
        or mapper.get("bisection_iterations") != 24
        or mapper.get("map_only_out_of_gamut_pixels") is not True
        or mapper.get("preserve_oklab_ab_direction") is not True
        or mapper.get("cohort_fitting_allowed") is not False
        or mapper.get("hard_component_clipping_allowed") is not False
        or mapper.get("posthoc_limiting_allowed") is not False
        or payload["source"].get("require_oog_per_source") is not True
    ):
        raise CC0SyntheticROMMConfirmationError("C13 frozen policy drift")
    for parent in payload["parents"].values():
        _relative(parent["path"])
    return payload


def _validate_sources(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name, parent in contract["parents"].items():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise CC0SyntheticROMMConfirmationError(f"C13 parent drift: {name}")
        if path.suffix == ".json":
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    if payloads["c12_evidence"].get("status") != contract["parents"][
        "c12_evidence"
    ]["required_status"]:
        raise CC0SyntheticROMMConfirmationError("C13 requires the C12 pass branch")
    if payloads["c12_report"].get("automatic_pass") is not contract["parents"][
        "c12_report"
    ]["required_automatic_pass"]:
        raise CC0SyntheticROMMConfirmationError("C13 C12 report decision drift")
    manifest = payloads["source_manifest"]
    source = contract["source"]
    rows = manifest.get("rows", [])
    if (
        manifest.get("stable_source_id")
        != contract["parents"]["source_manifest"]["required_stable_source_id"]
        or manifest.get("official_icc_sha256")
        != source["expected_embedded_icc_sha256"]
        or manifest.get("synthetic_transform")
        != source["required_synthetic_transform"]
        or manifest.get("minimum_rec2020_out_of_gamut_fraction", 0.0)
        < float(source["minimum_rec2020_out_of_gamut_fraction"])
        or len(rows) != int(source["expected_rows"])
        or len({row.get("id") for row in rows}) != len(rows)
        or len({row.get("path") for row in rows}) != len(rows)
        or len({row.get("sha256") for row in rows}) != len(rows)
        or len({row.get("make") for row in rows}) != int(source["expected_makes"])
    ):
        raise CC0SyntheticROMMConfirmationError("C13 source manifest drift")
    expected_profile = contract["parents"]["official_icc"]["sha256"]
    for row in rows:
        path = root / _relative(row["path"])
        if (
            not path.is_file()
            or hash_file(path) != row["sha256"]
            or row.get("embedded_icc_sha256") != expected_profile
            or row.get("allowed_use") != source["required_allowed_use"]
            or row.get("rights_scope") != source["required_rights_scope"]
            or row.get("synthetic_transform")
            != source["required_synthetic_transform"]
            or float(row.get("rec2020_out_of_gamut_fraction", 0.0))
            < float(source["minimum_rec2020_out_of_gamut_fraction"])
            or row.get("dtype") != "uint16"
        ):
            raise CC0SyntheticROMMConfirmationError("C13 source row drift")
        with tifffile.TiffFile(path) as document:
            profile_tag = document.pages[0].tags.get(34675)
            profile = bytes(profile_tag.value) if profile_tag is not None else b""
        if hashlib.sha256(profile).hexdigest() != expected_profile:
            raise CC0SyntheticROMMConfirmationError("C13 embedded profile drift")
    return {
        "source": {
            "maximum_evaluation_side": int(source["maximum_evaluation_side"])
        }
    }, rows


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise CC0SyntheticROMMConfirmationError(
            "C13 output directory must be create-only"
        )
    source_config, rows = _validate_sources(contract, root)
    return _evaluate_prevalidated(
        contract,
        root,
        output_dir,
        source_config,
        rows,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_sha256=CONTRACT_SHA256,
    )


__all__ = [
    "CONTRACT_SHA256",
    "CC0SyntheticROMMConfirmationError",
    "evaluate",
    "load_contract",
]
