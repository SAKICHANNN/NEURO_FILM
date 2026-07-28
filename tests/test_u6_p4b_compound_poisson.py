from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eval.physical_compound_poisson import (
    evaluate_compound_poisson,
    load_contract,
)
from src.eval.physical_developed_structure import load_contract as load_parent
from src.film_physics import (
    CompoundPoissonProfile,
    counter_poisson_region,
    render_compound_poisson,
    render_compound_poisson_region,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "u6_p4b_compound_poisson_compiler_v1.json"
PARENT = ROOT / "configs" / "u6_p1b_developed_structure_reference_v1.json"


def test_contract_loads() -> None:
    assert load_contract(CONTRACT)["node"] == "U6.P4B"


def test_counter_poisson_partition_is_exact() -> None:
    full = counter_poisson_region(
        (31, 29), origin_yx=(0, 0), shape=(31, 29), rate=1.5, seed=8
    )
    top = counter_poisson_region(
        full.shape, origin_yx=(0, 0), shape=(9, 29), rate=1.5, seed=8
    )
    bottom = counter_poisson_region(
        full.shape, origin_yx=(9, 0), shape=(22, 29), rate=1.5, seed=8
    )
    assert np.array_equal(np.concatenate([top, bottom]), full)


def test_density_and_transmittance_stay_in_physical_domains() -> None:
    density = CompoundPoissonProfile("density-shot", 0.5, 0.6, 0.1, 0.8, 9)
    transmission = CompoundPoissonProfile(
        "transmittance-shot", 1.0, 0.4, 0.1, 2.0, 10
    )
    density_field = render_compound_poisson(density, (37, 41))
    transmission_field = render_compound_poisson(transmission, (37, 41))
    assert np.all(density_field > 0.0)
    assert np.all((transmission_field > 0.0) & (transmission_field <= 1.0))


def test_filtered_compound_poisson_partition_is_exact() -> None:
    profile = CompoundPoissonProfile("density-shot", 0.5, 0.6, 0.1, 0.8, 11)
    full = render_compound_poisson(profile, (37, 41))
    top = render_compound_poisson_region(
        profile, full.shape, origin_yx=(0, 0), shape=(11, 41)
    )
    bottom = render_compound_poisson_region(
        profile, full.shape, origin_yx=(11, 0), shape=(26, 41)
    )
    assert np.array_equal(np.concatenate([top, bottom]), full)
    assert np.array_equal(
        render_compound_poisson(replace(profile), full.shape), full
    )


def test_frozen_compound_poisson_report_passes_and_is_repeatable() -> None:
    parent = load_parent(PARENT)
    contract = load_contract(CONTRACT)
    first = evaluate_compound_poisson(parent, contract)
    second = evaluate_compound_poisson(parent, contract)
    assert first == second
    assert first["stable_evidence_id"] == (
        "67973ca0cb07902f379a0c29e115f28d7e8b53ce260bbadee2b3278563067edc"
    )
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
