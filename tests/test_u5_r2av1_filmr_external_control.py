from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.filmr_external_control import (
    FilmrExternalControlError,
    build_deterministic_preset,
    canonical_sha256,
)


def _preset() -> dict:
    return {
        "film_type": "ColorSlide",
        "iso": 50.0,
        "manufacturer": "Fujifilm",
        "name": "Velvia 50",
        "color_matrix": [[1.3, -0.15, -0.15]] * 3,
        "grain_model": {
            "alpha": 0.000081,
            "sigma_read": 0.003,
            "monochrome": False,
            "blur_radius": 0.5,
            "roughness": 0.2,
            "color_correlation": 0.8,
            "shadow_noise": 0.001,
            "highlight_coarseness": 0.05,
        },
    }


def test_deterministic_preset_changes_only_random_amplitudes() -> None:
    original = _preset()
    result = build_deterministic_preset(
        original,
        expected_identity={
            "film_type": "ColorSlide",
            "iso": 50.0,
            "manufacturer": "Fujifilm",
            "name": "Velvia 50",
        },
        zero_grain_fields=(
            "alpha",
            "sigma_read",
            "shadow_noise",
            "highlight_coarseness",
        ),
    )
    expected = copy.deepcopy(original)
    for field in (
        "alpha",
        "sigma_read",
        "shadow_noise",
        "highlight_coarseness",
    ):
        expected["grain_model"][field] = 0.0
    assert result == expected
    assert original["grain_model"]["alpha"] == 0.000081
    assert canonical_sha256(result) == canonical_sha256(expected)


def test_deterministic_preset_rejects_identity_or_override_drift() -> None:
    with pytest.raises(FilmrExternalControlError, match="identity mismatch"):
        build_deterministic_preset(
            _preset(),
            expected_identity={"name": "Not Velvia"},
            zero_grain_fields=(
                "alpha",
                "sigma_read",
                "shadow_noise",
                "highlight_coarseness",
            ),
        )
    with pytest.raises(FilmrExternalControlError, match="field set drifted"):
        build_deterministic_preset(
            _preset(),
            expected_identity={"name": "Velvia 50"},
            zero_grain_fields=("alpha",),
        )


def test_formal_decision_closes_before_visual_review() -> None:
    root = Path(__file__).resolve().parents[1]
    decision = json.loads(
        (
            root
            / "configs/u5_r2av1_filmr_external_control_decision_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["formal_report_byte_exact"] is True
    assert decision["gate_checks"] == {
        "exact_repeat": True,
        "style": True,
        "non_basic": True,
        "gold_clipping": False,
    }
    assert decision["decision"] == "close_automatic_gate_failure"
    assert decision["visual_review_allowed"] is False
