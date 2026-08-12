from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.paired_scanner_mtf_source_disjoint_confirmation import (
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs/u6_p4hf_paired_scanner_mtf_source_disjoint_confirmation_v1.json"
)


def test_contract_sources_are_exactly_disjoint() -> None:
    payload = load_contract(CONTRACT)
    current = json.loads((ROOT / payload["source"]["manifest"]).read_text("utf-8"))
    prior_contract = json.loads(
        (ROOT / payload["parents"]["p4he_contract"]["path"]).read_text("utf-8")
    )
    prior = json.loads(
        (ROOT / prior_contract["source"]["manifest"]).read_text("utf-8")
    )
    assert len(current) == 18
    assert len({row["make"] for row in current}) == 9
    assert not (
        {row["decoded_sha256"] for row in current}
        & {row["decoded_sha256"] for row in prior}
    )


def test_nonzero_overlap_requirement_fails_before_pixel_decode(tmp_path: Path) -> None:
    payload = copy.deepcopy(load_contract(CONTRACT))
    payload["source"]["required_identity_overlap_with_p4he"] = 1
    with pytest.raises(ValueError, match="overlap"):
        evaluate(payload, root=ROOT, contact_sheet_path=tmp_path / "contact.png")
    assert not (tmp_path / "contact.png").exists()
