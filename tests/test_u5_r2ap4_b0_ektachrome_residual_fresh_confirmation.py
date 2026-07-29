import json
from pathlib import Path

from src.eval.b0_real_film_residual_fresh_confirmation import (
    SUPPORTED_EXPERIMENTS,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ap4_b0_ektachrome_residual_fresh_confirmation_v1.json"


def test_ap4_contract_is_supported_and_valid() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert SUPPORTED_EXPERIMENTS[config["experiment_id"]] == (0.10, 0.25)
    validated = validate_contract(ROOT, config)
    assert len(validated["eligible_ids"]) == 16
    assert validated["operator"].capture_matrix.shape == (3, 3)
    assert validated["operator"].scan_matrix.shape == (3, 3)


def test_ap4_contract_rejects_strength_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fixed_candidate"]["tone_strength"] = 0.15
    try:
        validate_contract(ROOT, config)
    except ValueError as exc:
        assert "fixed AO6 candidate drift" in str(exc)
    else:
        raise AssertionError("strength drift was accepted")
