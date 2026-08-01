from __future__ import annotations

import json
from pathlib import Path

from src.eval import fivek_pairwise_compatibility_run as runner
from src.eval.fivek_pairwise_compatibility_run import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bq2b_fivek_pairwise_compatibility_v1.json"


def test_bq2b_contract_is_low_capacity_hard_and_source_only_at_inference() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["model"]["family"] == (
        "closed_form_ridge_ranker_of_within_query_case_error_rank"
    )
    assert payload["model"]["pca_components"] == 32
    assert payload["model"]["development_samples_per_image"] == 512
    assert payload["model"]["confirmation_samples_per_image"] == 1024
    assert payload["model"]["query_target_used_at_inference"] is False
    assert payload["model"]["operator_parameters_used_at_inference"] is False
    assert payload["selector"]["dense_blending_allowed"] is False
    assert payload["confirmation_model_selection_allowed"] is False
    assert payload["learned_final_rgb_allowed"] is False
    assert payload["production_integration_allowed"] is False


def test_bq2b_real_parent_contract_validates_without_loading_images() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, payload)
    assert validated["decision"]["reports"]["automatic_pass"] is False
    assert validated["nearest"]["selected_family"] == "tone_layout"
    assert validated["oracle"]["automatic_pass"] is True


def test_bq2b_development_failure_never_loads_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    loaded_splits: list[str] = []

    def fake_load(*args, split: str, **kwargs):
        loaded_splits.append(split)
        if split == "confirmation":
            raise AssertionError("confirmation must remain unread")
        return []

    monkeypatch.setattr(runner, "load_split_population", fake_load)
    monkeypatch.setattr(runner, "prepare_development_evidence", lambda *a, **k: {})
    monkeypatch.setattr(
        runner,
        "evaluate_development",
        lambda **kwargs: {"automatic_pass": False},
    )
    report = runner.run_pairwise(
        root=ROOT,
        config=payload,
        config_path=CONFIG,
        output_path=tmp_path / "report.json",
        software_commit="0" * 40,
    )
    assert report["development_pass"] is False
    assert report["confirmation_executed"] is False
    assert loaded_splits == ["development", "development"]
