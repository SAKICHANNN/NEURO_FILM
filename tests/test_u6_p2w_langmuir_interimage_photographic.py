from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.physical_langmuir_photographic import (
    _develop_density,
    load_contract,
)
from src.eval.physical_neutral_gauged_chain import validate_contract as validate_p7f
from src.film_physics import LangmuirDonorProfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2w_langmuir_interimage_photographic_v1.json"


def test_linear_and_langmuir_donors_are_distinct_and_bounded() -> None:
    config = load_contract(CONTRACT)
    p7f = json.loads(
        (ROOT / config["parents"]["p7f_contract_path"]).read_text(encoding="utf-8")
    )
    runtime, _ = validate_p7f(ROOT, p7f)
    exposure = np.linspace(0.001, 1.0, 75).reshape(5, 5, 3)
    coupling = np.asarray(config["fixed_operator"]["coupling"])
    linear = _develop_density(
        runtime.print_operator.sensitometry,
        exposure,
        coupling,
        LangmuirDonorProfile((1.0,) * 3, (float("inf"),) * 3, (0.5,) * 3),
    )
    langmuir = _develop_density(
        runtime.print_operator.sensitometry,
        exposure,
        coupling,
        LangmuirDonorProfile((1.0,) * 3, (1.0,) * 3, (0.5,) * 3),
    )
    assert np.all(np.isfinite(linear))
    assert np.all(np.isfinite(langmuir))
    assert not np.array_equal(linear, langmuir)


def test_analytical_safety_preserves_zero_exposure_boundary() -> None:
    config = load_contract(CONTRACT)
    p7f = json.loads(
        (ROOT / config["parents"]["p7f_contract_path"]).read_text(encoding="utf-8")
    )
    runtime, _ = validate_p7f(ROOT, p7f)
    exposure = np.zeros((3, 4, 3), dtype=np.float64)
    exposure[1:, :, :] = np.linspace(0.0, 0.01, 24).reshape(2, 4, 3)
    density = _develop_density(
        runtime.print_operator.sensitometry,
        exposure,
        np.asarray(config["fixed_operator"]["coupling"]),
        LangmuirDonorProfile((1.0,) * 3, (1.0,) * 3, (0.5,) * 3),
    )
    black = runtime.print_operator.interpretation.black_reference_density
    white = runtime.print_operator.interpretation.white_reference_density
    assert np.all(density >= black - 1e-12)
    assert np.all(density <= white + 1e-12)


def test_contract_freezes_exact_population_and_no_fitting() -> None:
    config = load_contract(CONTRACT)
    assert config["population"]["expected_rows"] == 16
    assert config["population"]["expected_makes"] == 9
    assert len(config["population"]["expected_ids"]) == 16
    assert not config["population"]["photograph_fitting_allowed"]
