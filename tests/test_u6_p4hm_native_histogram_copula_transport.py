from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.native_histogram_copula_transport import python_reference

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4hm_native_histogram_copula_transport_v1.json"


def test_p4hm_contract_freezes_partial_native_scope() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["candidate"]["rank_bins"] == 65536
    assert payload["candidate"]["gamma_density_inverse_in_scope"] is False
    assert payload["candidate"]["thomas_field_generation_in_scope"] is False


def test_python_reference_is_repeat_exact_and_uniform() -> None:
    rng = np.random.default_rng(7)
    fields = rng.normal(size=(1000, 3)).astype(np.float32)
    target = np.asarray(
        [[1.0, 0.27, 0.22], [0.27, 1.0, 0.21], [0.22, 0.21, 1.0]],
        dtype=np.float64,
    )
    first = python_reference(fields, target, 65536)
    second = python_reference(fields.copy(), target.copy(), 65536)
    assert np.array_equal(first, second)
    assert np.max(np.abs(np.mean(first, axis=0) - 0.5)) < 1.0e-5
    assert np.all(first > 0.0)
    assert np.all(first < 1.0)
