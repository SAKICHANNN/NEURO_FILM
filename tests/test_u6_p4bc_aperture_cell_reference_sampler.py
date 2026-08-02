from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from src.eval.aperture_cell_reference_sampler import (
    ApertureCellReferenceSamplerError,
    evaluate_sampler,
    load_contract,
)
from src.film_physics.aperture_cell_sampler import (
    counter_high_rate_poisson_region,
)
from src.film_physics.structure_compiler import counter_uniform_region

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bc_aperture_cell_reference_sampler_v1.json"


def test_contract_rejects_measured_nps_claim(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["sampler"]["independence_status"] = "measured_nps"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ApertureCellReferenceSamplerError, match="contract drift"):
        load_contract(path)


def test_high_rate_inverse_matches_reference_ppf_and_partitions() -> None:
    shape = (17, 19)
    rate = 186330.23718236017
    seed = 2608021100
    uniform = counter_uniform_region(
        shape, origin_yx=(0, 0), shape=shape, seed=seed
    )
    expected = poisson.ppf(uniform, rate).astype(np.uint32)
    full = counter_high_rate_poisson_region(
        shape, origin_yx=(0, 0), shape=shape, rate=rate, seed=seed
    )
    top = counter_high_rate_poisson_region(
        shape, origin_yx=(0, 0), shape=(7, 19), rate=rate, seed=seed
    )
    bottom = counter_high_rate_poisson_region(
        shape, origin_yx=(7, 0), shape=(10, 19), rate=rate, seed=seed
    )
    assert np.array_equal(full, expected)
    assert np.array_equal(np.concatenate([top, bottom]), full)


def test_invalid_rate_fails_before_output() -> None:
    with pytest.raises(ValueError, match="high-rate"):
        counter_high_rate_poisson_region(
            (3, 3), origin_yx=(0, 0), shape=(3, 3), rate=0.5, seed=1
        )


def test_frozen_sampler_repeats_and_passes() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_sampler(contract, ROOT)
    second = evaluate_sampler(contract, ROOT)
    assert first == second
    assert first["automatic_pass"]
    assert first["decision"] == "retain_generic_aperture_cell_reference_sampler"
    assert first["probe_count"] == 15
    assert all(row["repeat_exact"] for row in first["rows"])
    assert all(row["odd_partition_exact"] for row in first["rows"])
