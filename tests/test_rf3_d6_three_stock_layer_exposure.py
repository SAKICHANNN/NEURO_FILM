from __future__ import annotations

from pathlib import Path

from src.real_film.three_stock_layer_exposure import evaluate, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_rf3_d6_real_population_is_reproducible() -> None:
    contract = load_contract(ROOT / "configs/rf3_d6_three_stock_layer_exposure_observability_v1.json")
    left = evaluate(contract, ROOT)
    right = evaluate(contract, ROOT)
    assert left == right
    assert left["population"]["eligible_records"] == 1732
    assert len(left["pairwise_comparisons"]) == 3


def test_rf3_d6_has_no_render_or_fit_escape() -> None:
    contract = load_contract(ROOT / "configs/rf3_d6_three_stock_layer_exposure_observability_v1.json")
    assert contract["response"]["cohort_fitting_allowed"] is False
    assert contract["response"]["rgb_render_allowed"] is False
