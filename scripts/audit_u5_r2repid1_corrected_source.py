"""Prospective API-semantics correction for the REPID aggregate source audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_u5_r2repid0_source import ROOT, _fetch
from scripts.audit_u5_r2repid0_source import evaluate as evaluate_base


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate(
    correction: dict[str, Any],
    *,
    fetch: Callable[[str, int], bytes] = _fetch,
    base_contract_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent = correction["parent"]
    parent_contract_bytes = (ROOT / parent["contract_path"]).read_bytes()
    parent_evidence_bytes = (ROOT / parent["evidence_path"]).read_bytes()
    parent_contract_exact = _sha256(parent_contract_bytes) == parent["contract_sha256"]
    parent_evidence_exact = _sha256(parent_evidence_bytes) == parent["evidence_sha256"]
    parent_evidence = json.loads(parent_evidence_bytes)
    parent_decision_exact = parent_evidence["decision"] == parent["required_decision"]
    contract = copy.deepcopy(
        json.loads(parent_contract_bytes)
        if base_contract_override is None
        else base_contract_override
    )
    contract["experiment_id"] = correction["experiment_id"]
    contract["claim_ceiling"] = correction["claim_ceiling"]
    contract["official_sources"]["expected_repository"]["last_modified"] = correction[
        "corrections"
    ]["expected_last_modified"]
    report = evaluate_base(contract, fetch=fetch)
    root_tree_exact = (
        report["root_tree_file_facts"]
        == correction["corrections"]["root_tree_file_facts"]
    )
    report["schema"] = "neuro_film.u5_r2repid1_corrected_source_audit_report.v1"
    report["parent_bindings"] = {
        "contract_sha256": _sha256(parent_contract_bytes),
        "evidence_sha256": _sha256(parent_evidence_bytes),
        "parent_decision": parent_evidence["decision"],
    }
    report["corrections"] = correction["corrections"]
    report["requests"]["image_tree_requests"] = 0
    report["gates"]["parent_contract_exact"] = parent_contract_exact
    report["gates"]["parent_failure_evidence_exact"] = parent_evidence_exact
    report["gates"]["parent_decision_exact"] = parent_decision_exact
    report["gates"]["exact_root_file_inventory"] = root_tree_exact
    report["automatic_pass"] = all(
        value
        for key, value in report["gates"].items()
        if key != "product_dependency_allowed"
    )
    report["decision"] = (
        correction["decision_if_pass"]
        if report["automatic_pass"]
        else correction["decision_if_fail"]
    )
    report["claim_ceiling"] = correction["claim_ceiling"]
    report.pop("stable_evidence_id", None)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["stable_evidence_id"] = f"sha256:{_sha256(canonical)}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u5_r2repid1_corrected_source_audit_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correction = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate(correction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["decision"],
                "stable_evidence_id": report["stable_evidence_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
