from __future__ import annotations

import json
from pathlib import Path

from src.eval import fivek_factorized_safe_residual_run as runner
from src.eval.fivek_factorized_safe_residual_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq4_fivek_factorized_safe_residual_v1.json"


def test_bq4_contract_is_factorized_explicit_and_no_clip() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["model"]["family"] == (
        "factorized_identity_centered_parameter_direction_and_log_magnitude"
    )
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["direct_final_rgb_prediction"] is False
    assert payload["operator"]["residual_origin"] == "source_rgb"
    assert payload["operator"]["hard_clipping_allowed"] is False
    assert payload["development_gates"][
        "minimum_median_style_retention_ratio"
    ] == 0.7
    assert payload["production_integration_allowed"] is False


def test_bq4_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["decision"]["threshold_or_capacity_rescue_allowed"] is False
    assert validated["direct"]["development_pass"] is False
    assert validated["direct"]["confirmation_executed"] is False
    assert validated["oracle"]["automatic_pass"] is True


def test_bq4_development_failure_leaves_confirmation_unread(
    monkeypatch, tmp_path: Path
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    loaded: list[str] = []

    def fake_load(*args, split: str, **kwargs):
        loaded.append(split)
        if split == "confirmation":
            raise AssertionError("confirmation must remain unread")
        return []

    monkeypatch.setattr(runner, "load_split_population", fake_load)
    monkeypatch.setattr(
        runner,
        "evaluate_development",
        lambda **kwargs: {"automatic_pass": False},
    )
    report = runner.run_factorized_safe_residual(
        root=ROOT,
        config=payload,
        config_path=CONFIG,
        output_path=tmp_path / "report.json",
        software_commit="0" * 40,
    )
    assert report["development_pass"] is False
    assert report["confirmation_executed"] is False
    assert loaded == ["development", "development"]
