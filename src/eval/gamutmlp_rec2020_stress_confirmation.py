"""U1.4C7 unchanged C4 execution on reviewed Rec.2020-stress sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.eval.fujifilm_dye_basis_measured_conformance import hash_file
from src.eval.gamutmlp_prophoto_rec2020_confirmation import (
    evaluate_rows,
    write_report,
)

SCHEMA = "neuro-film.u1-4c7-gamutmlp-rec2020-stress-confirmation-contract.v1"
REPORT_SCHEMA = "neuro-film.u1-4c7-gamutmlp-rec2020-stress-confirmation-report.v1"
EXPERIMENT_ID = "U1.4C7"
CONTRACT_SHA256 = "ecf68184e637525dcc0ff08626153e66fa227fc532e65048476356e929a4ab84"


class GamutMLPStressConfirmationError(RuntimeError):
    """Raised when a frozen C7 identity or source inventory drifts."""


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise GamutMLPStressConfirmationError("C7 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise GamutMLPStressConfirmationError("C7 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GamutMLPStressConfirmationError("C7 contract structure drift")
    _relative(payload["source"]["manifest"])
    for parent in payload["parents"].values():
        _relative(parent["path"])
    return payload


def _validate_inputs(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    for parent in config["parents"].values():
        path = root / _relative(parent["path"])
        if not path.is_file() or hash_file(path) != parent["sha256"]:
            raise GamutMLPStressConfirmationError("C7 parent identity drift")
        if parent.get("required_automatic_pass") is True:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("automatic_pass") is not True:
                raise GamutMLPStressConfirmationError("C7 requires a passing parent")
    source = config["source"]
    reviewed_path = root / _relative(source["manifest"])
    if (
        not reviewed_path.is_file()
        or hash_file(reviewed_path) != source["manifest_sha256"]
    ):
        raise GamutMLPStressConfirmationError("C7 reviewed manifest identity drift")
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    audit_parent = config["parents"]["source_audit_report"]
    report = json.loads(
        (root / _relative(audit_parent["path"])).read_text(encoding="utf-8")
    )
    selected = report.get("selected_manifest")
    rows = selected.get("rows") if isinstance(selected, dict) else None
    expected_order = (
        [
            [f"{row['camera']}-{row['source_id']}-{row['style']}", row["member_sha256"]]
            for row in rows
        ]
        if isinstance(rows, list)
        else None
    )
    if (
        reviewed.get("schema")
        != "neuro-film.u1-4c7-gamutmlp-rec2020-stress-reviewed-source-manifest.v1"
        or reviewed.get("automatic_pass") is not True
        or reviewed.get("visual_pass") is not source["required_visual_pass"]
        or reviewed.get("allowed_use") != source["required_allowed_use"]
        or reviewed.get("rights_scope") != source["required_rights_scope"]
        or reviewed.get("selected_manifest_canonical_sha256")
        != report.get("selected_manifest_canonical_sha256")
        or reviewed.get("ordered_rows") != expected_order
        or not isinstance(rows, list)
        or len(rows) != int(source["expected_rows"])
        or len({row["camera"] for row in rows}) != int(source["expected_camera_models"])
        or len({row["source_id"] for row in rows}) != len(rows)
        or any(row.get("dtype") != source["expected_dtype"] for row in rows)
        or any(
            float(row.get("precompression_rec2020_out_of_gamut_fraction", -1.0))
            < float(
                config["automatic_gates"][
                    "minimum_precompression_rec2020_out_of_gamut_fraction_per_source"
                ]
            )
            for row in rows
        )
    ):
        raise GamutMLPStressConfirmationError("C7 source inventory drift")
    return [dict(row) for row in rows]


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    return evaluate_rows(
        config,
        root,
        output_dir,
        _validate_inputs(config, root),
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_sha256=CONTRACT_SHA256,
    )


__all__ = [
    "CONTRACT_SHA256",
    "GamutMLPStressConfirmationError",
    "evaluate",
    "load_contract",
    "write_report",
]
