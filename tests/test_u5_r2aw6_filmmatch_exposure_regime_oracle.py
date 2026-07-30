from src.eval.filmmatch_condition_oracle import exposure_regime


def test_exposure_regime_has_three_fixed_states() -> None:
    assert exposure_regime(-5) == "negative"
    assert exposure_regime(-1) == "negative"
    assert exposure_regime(0) == "zero"
    assert exposure_regime(1) == "positive"
    assert exposure_regime(5) == "positive"
