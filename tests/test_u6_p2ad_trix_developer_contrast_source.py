from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.trix_developer_contrast_source import (
    TrixDeveloperContrastError,
    load_contract,
    run_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2ad_trix_developer_contrast_source_v1.json"


def test_p2ad_first_party_vector_source_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = run_audit(root=ROOT, contract=contract)
    second = run_audit(root=ROOT, contract=contract)
    assert first == second
    assert first["automatic_pass"] is True
    assert len(first["signatures"]) == 7
    assert first["measurements"]["rgb_image_transform_count_zero"] is True


def test_p2ad_rejects_trace_identity_drift() -> None:
    contract = deepcopy(load_contract(CONTRACT))
    contract["source"]["vector_extraction"]["paths"]["d76"]["path_sha256"] = "0" * 64
    with pytest.raises(TrixDeveloperContrastError, match="path identity mismatch"):
        run_audit(root=ROOT, contract=contract)


def test_p2ad_rejects_uncovered_shared_contrast_level() -> None:
    contract = deepcopy(load_contract(CONTRACT))
    contract["analysis"]["shared_contrast_indices"][-1] = 1.1
    with pytest.raises(TrixDeveloperContrastError, match="does not cover"):
        run_audit(root=ROOT, contract=contract)
