from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.physical_developed_structure import load_contract as load_parent
from src.eval.physical_structure_compiler import (
    evaluate_structure_compiler,
    load_contract,
)
from src.film_physics import (
    MarginalProfile,
    counter_normal_region,
    render_marginal,
    render_marginal_region,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs" / "u6_p1b_developed_structure_reference_v1.json"
CONTRACT = ROOT / "configs" / "u6_p4_structure_compiler_v1.json"


def test_counter_field_has_coordinate_partition_identity() -> None:
    full = counter_normal_region((23, 19), origin_yx=(0, 0), shape=(23, 19), seed=4)
    top = counter_normal_region((23, 19), origin_yx=(0, 0), shape=(7, 19), seed=4)
    bottom = counter_normal_region(
        (23, 19), origin_yx=(7, 0), shape=(16, 19), seed=4
    )
    assert np.array_equal(np.concatenate([top, bottom]), full)


def test_gamma_and_beta_marginals_remain_in_physical_domains() -> None:
    gamma = render_marginal(MarginalProfile("gamma-density", 0.5, 0.1, 1.0, 5), (31, 29))
    beta = render_marginal(
        MarginalProfile("beta-transmittance", 0.2, 0.05, 1.0, 6), (31, 29)
    )
    assert np.all(gamma > 0.0)
    assert np.all((beta > 0.0) & (beta < 1.0))


def test_correlated_marginal_partition_is_exact() -> None:
    profile = MarginalProfile("gamma-density", 0.5, 0.1, 1.5, 7)
    full = render_marginal(profile, (37, 41))
    top = render_marginal_region(
        profile, full.shape, origin_yx=(0, 0), shape=(11, 41)
    )
    bottom = render_marginal_region(
        profile, full.shape, origin_yx=(11, 0), shape=(26, 41)
    )
    assert np.array_equal(np.concatenate([top, bottom]), full)


def test_frozen_structure_compiler_report_is_repeatable() -> None:
    parent = load_parent(PARENT)
    contract = load_contract(CONTRACT)
    first = evaluate_structure_compiler(parent, contract)
    second = evaluate_structure_compiler(parent, contract)
    assert first == second
    assert first["stable_evidence_id"] == (
        "1527acecc3567b4e058d9c9f516526b70ce5a0625f30842882bb874081126905"
    )
    assert first["automatic_pass"] is False
    assert first["decisions"] == {
        "acf": False,
        "domain": True,
        "mean": True,
        "nps": True,
        "partition": True,
        "repeat": True,
        "variance": True,
    }
