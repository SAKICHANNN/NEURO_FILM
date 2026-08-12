#!/usr/bin/env python3
"""Evaluate the P4GO exact scan-domain endpoint anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_u6_p4gn_monotone_scan_inverse import evaluate as evaluate_inverse


P4GN = ROOT / "configs/u6_p4gn_monotone_scan_inverse_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(contract_path: Path, visual: Path | None = None) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent_path = ROOT / contract["parent"]["path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        _sha(parent_path) != contract["parent"]["sha256"]
        or parent["decision"] != contract["parent"]["required_decision"]
    ):
        raise RuntimeError("P4GO parent drift")
    result = evaluate_inverse(P4GN, visual, scan_domain_anchors=True)
    stable = result["stable"]
    stable["contract_sha256"] = _sha(contract_path)
    stable["scan_domain_anchors"] = [[0.0, 0.0], [1.0, 1.0]]
    stable["decision"] = (
        contract["decision_if_pass"]
        if result["automatic_pass"]
        else contract["decision_if_fail"]
    )
    stable["claim_ceiling"] = contract["claim_ceiling"]
    result["schema"] = contract["schema"].replace("-contract", "-result")
    result["stable_evidence_id"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual", type=Path)
    arguments = parser.parse_args()
    result = evaluate(arguments.contract, arguments.visual)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if arguments.output:
        arguments.output.write_text(raw, encoding="utf-8")
    else:
        print(raw)


if __name__ == "__main__":
    main()
