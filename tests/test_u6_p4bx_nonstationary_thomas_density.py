from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.nonstationary_thomas_density import (
    NonstationaryThomasDensityError,
    _validate_contract,
    evaluate_nonstationary_thomas_density,
)
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4bx_nonstationary_thomas_density_v1.json"


def test_contract_rejects_clipping() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _validate_contract(contract)
    contract["candidate"]["clipping_allowed"] = True
    with pytest.raises(NonstationaryThomasDensityError):
        _validate_contract(contract)


def test_nonstationary_region_matches_rows() -> None:
    bundle = json.loads(
        (
            ROOT
            / "outputs/experiments/u6_p4bw_density_conditioned_thomas_profile_v1/run_a/bundle.json"
        ).read_text(encoding="utf-8")
    )
    prior_payload = json.loads(
        (ROOT / "outputs/u6_p2q_kodak_250d_characteristic_prior/bundle_run1.json").read_text(
            encoding="utf-8"
        )
    )
    profile = DensityConditionedThomasProfile.from_dict(bundle)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    shape = (41, 47)
    receipt = build_thomas_dc_receipt(
        shape,
        profile_id=profile.spatial_profile_id,
        particle_sigma_pixels=profile.particle_sigma_samples,
        cluster_sigma_pixels=profile.cluster_sigma_samples,
        mean_offspring=profile.mean_offspring,
        component_seeds=profile.component_seeds,
        realization_seed=101,
        truncate=profile.truncate,
        canonical_row_block_height=11,
    )
    lower, upper = prior.curves[0].domain
    exposure = np.linspace(lower, upper, shape[1], dtype=np.float64)[None, :]
    exposure = np.repeat(exposure, shape[0], axis=0)
    full = profile.render_nonstationary_developed_density_region(
        receipt,
        prior,
        channel="red",
        full_relative_log_exposure=exposure,
        origin_yx=(0, 0),
        shape=shape,
    )
    assembled = np.empty_like(full)
    for y0 in range(0, shape[0], 7):
        height = min(7, shape[0] - y0)
        assembled[y0 : y0 + height] = (
            profile.render_nonstationary_developed_density_region(
                receipt,
                prior,
                channel="red",
                full_relative_log_exposure=exposure,
                origin_yx=(y0, 0),
                shape=(height, shape[1]),
            )
        )
    assert np.array_equal(full, assembled)
    assert np.all(full > 0.0)


def test_formal_evaluation_is_exact_and_passes() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    first = evaluate_nonstationary_thomas_density(contract, ROOT)
    second = evaluate_nonstationary_thomas_density(contract, ROOT)
    assert first == second
    assert first["automatic_pass"]
    assert first["local_group_count"] == 42
    assert all(first["gate_results"].values())
