from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.bw_grain_external_confirmation import (
    BWGrainExternalConfirmationError,
    _normalized_median,
    evaluate_external_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p4z2_bw_grain_external_confirmation_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_normalized_median_is_unit_length_and_repeatable() -> None:
    rows = [
        np.array([0.2, 0.3, 0.5]),
        np.array([0.4, 0.2, 0.4]),
        np.array([0.3, 0.4, 0.3]),
    ]
    first = _normalized_median(rows)
    second = _normalized_median(rows)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert np.array_equal(first, second)
    assert first.flags.writeable is False


def test_contract_rejects_fit_authority_and_parent_drift() -> None:
    contract = deepcopy(_config())
    contract["operator_fitting_allowed"] = True
    with pytest.raises(BWGrainExternalConfirmationError, match="unsupported"):
        evaluate_external_confirmation(root=ROOT, contract=contract)
    contract = deepcopy(_config())
    contract["parents"]["p4x_decision"]["sha256"] = "0" * 64
    with pytest.raises(BWGrainExternalConfirmationError, match="parent hash"):
        evaluate_external_confirmation(root=ROOT, contract=contract)


@pytest.mark.skipif(
    not (ROOT / "data/external/wikimedia_bw_uniform_grain_v1").is_dir(),
    reason="ignored external B&W source is absent",
)
def test_frozen_external_confirmation_repeats_without_fit() -> None:
    contract = _config()
    first = evaluate_external_confirmation(root=ROOT, contract=contract)
    second = evaluate_external_confirmation(root=ROOT, contract=contract)
    assert first == second
    assert first["source_count"] == 3
    assert first["crop_count_per_source"] == [9, 9, 9]
    assert first["stock_labels_used_for_fit"] is False
    assert first["operator_fitting_executed"] is False
