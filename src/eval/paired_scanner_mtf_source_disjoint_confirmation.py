"""P4HF source-disjoint confirmation of the exact P4HE bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.layer_gamma_photographic_development import sha256_file
from src.eval.paired_scanner_mtf_photographic_development import _evaluate_paired

SCHEMA = (
    "neuro-film.u6-p4hf-paired-scanner-mtf-source-disjoint-confirmation-contract.v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HF contract")
    return payload


def evaluate(
    contract: dict[str, Any], *, root: Path, contact_sheet_path: Path
) -> dict[str, Any]:
    prior_binding = contract["parents"]["p4he_contract"]
    prior_path = root / prior_binding["path"]
    if not prior_path.is_file() or sha256_file(prior_path) != prior_binding["sha256"]:
        raise ValueError("P4HF prior contract drift")
    prior_contract = json.loads(prior_path.read_text(encoding="utf-8"))
    current_manifest_path = root / contract["source"]["manifest"]
    prior_manifest_path = root / prior_contract["source"]["manifest"]
    if (
        sha256_file(current_manifest_path) != contract["source"]["manifest_sha256"]
        or sha256_file(prior_manifest_path)
        != prior_contract["source"]["manifest_sha256"]
    ):
        raise ValueError("P4HF source manifest drift")
    current_rows = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    prior_rows = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    current_ids = {row["decoded_sha256"] for row in current_rows}
    prior_ids = {row["decoded_sha256"] for row in prior_rows}
    if (
        len(current_ids & prior_ids)
        != contract["source"]["required_identity_overlap_with_p4he"]
    ):
        raise ValueError("P4HF source identities overlap P4HE")
    return _evaluate_paired(
        contract,
        root=root,
        contact_sheet_path=contact_sheet_path,
        result_parent_name="p4he_result",
    )


__all__ = ["evaluate", "load_contract"]
