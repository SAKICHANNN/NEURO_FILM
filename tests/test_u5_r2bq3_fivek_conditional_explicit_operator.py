from __future__ import annotations

import json
from pathlib import Path

from src.eval import fivek_conditional_explicit_operator_run as runner
from src.eval.fivek_conditional_explicit_operator_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq3_fivek_conditional_explicit_operator_v1.json"


def test_bq3_contract_predicts_only_bounded_explicit_parameters() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["model"]["family"] == (
        "ridge_prediction_of_effective_triangular_logit_parameters"
    )
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["direct_final_rgb_prediction"] is False
    assert payload["operator"]["family"] == (
        "monotone_triangular_logit_transport"
    )
    assert payload["development_gates"][
        "minimum_mean_style_magnitude_ratio_to_global"
    ] == 0.9
    assert payload["confirmation_model_selection_allowed"] is False
    assert payload["production_integration_allowed"] is False
    assert payload["evaluation"]["ood_distance_quantile"] == 0.95


def test_bq3_real_parent_contract_validates_without_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["decision"]["ranker_capacity_rescue_allowed"] is False
    assert validated["oracle"]["automatic_pass"] is True
    assert validated["nearest"]["selected_family"] == "tone_layout"


def test_bq3_development_failure_leaves_confirmation_unread(
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
    report = runner.run_conditional_operator(
        root=ROOT,
        config=payload,
        config_path=CONFIG,
        output_path=tmp_path / "report.json",
        software_commit="0" * 40,
    )
    assert report["development_pass"] is False
    assert report["confirmation_executed"] is False
    assert loaded == ["development", "development"]
