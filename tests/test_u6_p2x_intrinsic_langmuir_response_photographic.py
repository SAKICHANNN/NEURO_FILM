from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_intrinsic_langmuir_photographic import (
    _develop_density,
    _donor,
    _operator,
    load_contract,
)
from src.eval.physical_neutral_gauged_chain import validate_contract as validate_p7f

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2x_intrinsic_langmuir_response_photographic_v1.json"


def _runtime(config: dict[str, object]):
    parents = config["parents"]
    assert isinstance(parents, dict)
    p7f = json.loads(
        (ROOT / str(parents["p7f_contract_path"])).read_text(encoding="utf-8")
    )
    return validate_p7f(ROOT, p7f)[0]


def test_intrinsic_linear_and_langmuir_responses_are_bounded_and_distinct() -> None:
    config = load_contract(CONTRACT)
    runtime = _runtime(config)
    exposure = np.linspace(0.0, 1.0, 75).reshape(5, 5, 3)
    operator = _operator(config)
    encoder = runtime.print_operator.sensitometry.encoder
    linear = _develop_density(exposure, encoder, operator, _donor(config, linear=True))
    langmuir = _develop_density(
        exposure, encoder, operator, _donor(config, linear=False)
    )
    minimum = np.asarray(operator.density_min)
    maximum = np.asarray(operator.density_max)
    assert np.all(linear >= minimum) and np.all(linear <= maximum)
    assert np.all(langmuir >= minimum) and np.all(langmuir <= maximum)
    assert not np.array_equal(linear, langmuir)


def test_zero_exposure_is_finite_without_post_hoc_scale() -> None:
    config = load_contract(CONTRACT)
    runtime = _runtime(config)
    operator = _operator(config)
    density = _develop_density(
        np.zeros((3, 4, 3), dtype=np.float64),
        runtime.print_operator.sensitometry.encoder,
        operator,
        _donor(config, linear=False),
    )
    assert np.all(np.isfinite(density))
    assert not config["fixed_operator"]["per_pixel_post_hoc_scale_allowed"]


def test_contract_freezes_development_population_and_no_fitting() -> None:
    config = load_contract(CONTRACT)
    assert config["population"]["expected_rows"] == 16
    assert config["population"]["expected_makes"] == 9
    assert len(config["population"]["expected_ids"]) == 16
    assert not config["population"]["photograph_fitting_allowed"]
    assert config["fixed_operator"]["source_witness"].startswith(
        "U6.P2V symmetric-moderate"
    )
