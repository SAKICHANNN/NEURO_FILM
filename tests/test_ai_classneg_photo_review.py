import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_ai_classneg_photo_review import (
    diagnostics,
    simple_contrast,
    validate_rows,
)


def test_prior_transfer_overlap_and_rights_reject():
    row = {
        "id": "a",
        "raw_sha256": "hash",
        "rights_scope": "CC0_public_domain_internal_evaluation",
        "decoded_path": "outputs/a.png",
    }
    validate_rows([row], [{"sha256": "other"}], 1)
    with pytest.raises(ValueError, match="overlap"):
        validate_rows([row], [{"sha256": "hash"}], 1)
    with pytest.raises(ValueError, match="rights"):
        validate_rows([{**row, "rights_scope": "unknown"}], [], 1)


def test_simple_and_diagnostics():
    x = np.array([[[0, 0.5, 1], [0.01, 0.5, 0.99]]], dtype=np.float32)
    y = simple_contrast(x)
    assert np.array_equal(y, [[[0, 0.5, 1], [0, 0.5, 1]]])
    assert diagnostics(x, y)["new_exact_boundary_component_fraction"] == pytest.approx(
        1 / 3
    )
    assert diagnostics(x, x)["mean_absolute_rgb_change"] == 0
    with pytest.raises(ValueError):
        diagnostics(x, x + 1)


def test_review_counts_preserve_negative_decision():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "docs/evidence/AI_CLASSNEG_PHOTOGRAPHIC_REVIEW_20260907.json").read_text()
    )
    rows = evidence["rows"]
    assert len(rows) == len({row["id"] for row in rows}) == 17
    for key in (
        "preferred_over_identity",
        "preferred_over_safe_rich_portra",
        "preferred_over_simple_contrast",
    ):
        count = sum(row[key] for row in rows)
        assert count == evidence["aggregate"][key]
        assert count < evidence["aggregate"]["required_wins_per_control"]
    assert evidence["decision"] == "NO_PROMOTION_PHOTOGRAPHIC_VALUE_GATE_FAILED"
    assert not evidence["review_coverage"]["native_resolution_review"]
    assert not evidence["review_coverage"]["control_full_resolution_artifact_clearance"]
