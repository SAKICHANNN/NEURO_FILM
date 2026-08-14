"""U1.4C12 confirmation of C11 under the official ROMM RGB ICC profile."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tifffile

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.oklab_analytical_interior_confirmation import _evaluate_prevalidated

SCHEMA = "neuro-film.u1-4c12-official-romm-cross-dataset-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c12-official-romm-cross-dataset-report.v1"
EXPERIMENT_ID = "U1.4C12"
CONTRACT_SHA256 = "dd434aba470f9cc7ba55ef51efd13fdbc8848d5c897b7a723f2101a9c1b86f3f"


class OfficialROMMConfirmationError(RuntimeError):
    """Raised when a frozen C12 identity or source invariant fails."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise OfficialROMMConfirmationError("C12 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise OfficialROMMConfirmationError("C12 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise OfficialROMMConfirmationError("C12 contract structure drift")
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
    ):
        raise OfficialROMMConfirmationError("C12 mapper policy drift")
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
            raise OfficialROMMConfirmationError(f"C12 parent drift: {name}")
        if path.suffix == ".json":
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
    if payloads["c11_evidence"].get("status") != contract["parents"][
        "c11_evidence"
    ]["required_status"]:
        raise OfficialROMMConfirmationError("C12 requires the C11 pass branch")
    if payloads["c11_report"].get("automatic_pass") is not contract["parents"][
        "c11_report"
    ]["required_automatic_pass"]:
        raise OfficialROMMConfirmationError("C12 C11 report decision drift")
    manifest = payloads["source_manifest"]
    source = contract["source"]
    if (
        manifest.get("stable_source_id")
        != contract["parents"]["source_manifest"]["required_stable_source_id"]
        or manifest.get("official_icc_sha256")
        != source["expected_embedded_icc_sha256"]
        or manifest.get("pixel_roundtrip_exact") is not True
        or manifest.get("embedded_profile_exact") is not True
        or manifest.get("product_decode_pass") is not True
    ):
        raise OfficialROMMConfirmationError("C12 source manifest identity drift")
    rows = manifest.get("rows", [])
    if (
        len(rows) != int(source["expected_rows"])
        or len({row.get("id") for row in rows}) != len(rows)
        or len({row.get("path") for row in rows}) != len(rows)
        or len({row.get("sha256") for row in rows}) != len(rows)
        or len({row.get("camera") for row in rows})
        != int(source["expected_cameras"])
    ):
        raise OfficialROMMConfirmationError("C12 source inventory drift")
    expected_profile = contract["parents"]["official_icc"]["sha256"]
    for row in rows:
        path = root / _relative(row["path"])
        if (
            not path.is_file()
            or hash_file(path) != row["sha256"]
            or row.get("embedded_icc_sha256") != expected_profile
            or row.get("allowed_use") != source["required_allowed_use"]
            or row.get("rights_scope") != source["required_rights_scope"]
            or row.get("dtype") != "uint16"
            or [row.get("height"), row.get("width")] != [512, 512]
        ):
            raise OfficialROMMConfirmationError("C12 source row drift")
        with tifffile.TiffFile(path) as document:
            profile_tag = document.pages[0].tags.get(34675)
            profile = bytes(profile_tag.value) if profile_tag is not None else b""
        if hash_file(root / _relative(contract["parents"]["official_icc"]["path"])) != expected_profile or hash_file(path) != row["sha256"] or not profile:
            raise OfficialROMMConfirmationError("C12 source profile drift")
        import hashlib

        if hashlib.sha256(profile).hexdigest() != expected_profile:
            raise OfficialROMMConfirmationError("C12 embedded profile drift")
    return {"source": {"maximum_evaluation_side": int(source["maximum_evaluation_side"])}}, rows


def evaluate(
    contract: Mapping[str, Any], root: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise OfficialROMMConfirmationError("C12 output directory must be create-only")
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
    "OfficialROMMConfirmationError",
    "evaluate",
    "load_contract",
]
