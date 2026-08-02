from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.multi_aperture_reference_sampler import (
    MultiApertureReferenceSamplerError,
    evaluate_sampler,
    load_contract,
)
from src.film_physics.aperture_cell_sampler import (
    counter_aperture_scaled_poisson_region,
    counter_high_rate_poisson_region,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bh_multi_aperture_reference_sampler_v1.json"


def test_contract_rejects_realized_renormalization(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["sampler"]["realized_field_renormalization_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MultiApertureReferenceSamplerError, match="contract drift"):
        load_contract(path)


def test_extended_sampler_preserves_frozen_range_and_bytes() -> None:
    shape = (17, 19)
    seed = 2608021100
    rate = 186330.23718236017
    legacy = counter_high_rate_poisson_region(
        shape, origin_yx=(0, 0), shape=shape, rate=rate, seed=seed
    )
    extended = counter_aperture_scaled_poisson_region(
        shape, origin_yx=(0, 0), shape=shape, rate=rate, seed=seed
    )
    assert np.array_equal(legacy, extended)
    high = counter_aperture_scaled_poisson_region(
        shape, origin_yx=(0, 0), shape=shape, rate=11_925_135.0, seed=seed
    )
    assert high.dtype == np.uint32
    with pytest.raises(ValueError, match="supported sampler range"):
        counter_aperture_scaled_poisson_region(
            shape, origin_yx=(0, 0), shape=shape, rate=16_000_001.0, seed=seed
        )


def test_multi_aperture_sampler_repeats() -> None:
    contract = load_contract(CONTRACT)
    report = evaluate_sampler(contract, ROOT)
    assert report["automatic_pass"] is True
    assert report["scaled_row_count"] == 105
    assert all(report["gate_results"].values())
