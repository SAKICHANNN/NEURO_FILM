from __future__ import annotations

import numpy as np
import pytest

from scripts import audit_p288_dng_profile_stage_composition as p288


def test_canonical_serialization_is_sorted_and_terminated() -> None:
    assert p288._canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_romm_domain_is_strict_but_tolerant_to_matrix_roundoff() -> None:
    tolerance = 16.0 * np.finfo(np.float64).eps
    valid = np.asarray([[[-tolerance, 0.5, 1.0 + tolerance]]])
    invalid = valid.copy()
    invalid[0, 0, 2] += tolerance

    assert p288._romm_domain(valid, tolerance)
    assert not p288._romm_domain(invalid, tolerance)


@pytest.mark.parametrize("exposure", [0.0, -1.0, np.inf, np.nan])
def test_compose_rejects_nonpositive_or_nonfinite_exposure(exposure: float) -> None:
    source = np.ones((1, 1, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="exposure"):
        p288._compose(source, exposure, {}, {}, direct=False)


def test_frozen_contract_names_all_stages() -> None:
    text = (
        p288.ROOT / "docs/planning/P288_DNG_PROFILE_STAGE_COMPOSITION_CONTRACT.md"
    ).read_text(encoding="utf-8")

    assert "ProfileHueSatMap" in text
    assert "ProfileGainTableMap v1" in text
    assert "ProfileLookTable" in text
    assert "ProfileToneCurve" in text
    assert "Candidate-3" not in text
