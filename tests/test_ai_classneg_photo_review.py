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
