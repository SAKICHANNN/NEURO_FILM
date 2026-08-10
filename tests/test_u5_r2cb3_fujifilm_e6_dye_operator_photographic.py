from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.fujifilm_e6_bounded_dye_operator import hash_file
from src.eval.fujifilm_e6_dye_operator_photographic import (
    CONTRACT_SHA256,
    FujifilmDyePhotographicError,
    _gradient_p999_ratio,
    _sample_indices,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cb3_fujifilm_e6_dye_operator_photographic_v1.json"


def test_contract_is_frozen() -> None:
    assert hash_file(CONFIG) == CONTRACT_SHA256
    assert load_contract(CONFIG)["experiment_id"] == "U5.R2CB3"


def test_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(CONFIG.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FujifilmDyePhotographicError, match="contract hash drift"):
        load_contract(changed)


def test_even_sample_includes_endpoints_and_is_repeat_exact() -> None:
    first = _sample_indices(1000, 17)
    second = _sample_indices(1000, 17)
    assert np.array_equal(first, second)
    assert first[0] == 0
    assert first[-1] == 999
    assert np.all(np.diff(first) > 0)


def test_identity_gradient_ratio_is_one() -> None:
    values = np.arange(8 * 7 * 3, dtype=np.float64).reshape(8, 7, 3) / 200.0
    assert _gradient_p999_ratio(values, values) == pytest.approx(1.0)
