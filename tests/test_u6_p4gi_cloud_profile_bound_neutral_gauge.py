import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.profile_bound_neutral_gauge import (
    analyze_profile_bound_neutral_response,
    apply_profile_bound_neutral_gauge,
    compile_profile_bound_neutral_gauge,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gi_cloud_profile_bound_neutral_gauge_v1.json"


def test_p4gi_contract_keeps_chart_out_of_fit_and_freezes_stop_rules():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["execution"]["chart_pixels_used_for_fit"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["calibration"]["layout"] == "latin-balanced-neutral-levels-step-17"
    assert contract["confirmation"]["layout"] == "latin-balanced-neutral-midpoints-step-17"
    assert "spatial confounding" in contract["execution"]["operational_amendment"]
    assert contract["gates"]["maximum_confirmation_neutral_chroma_p99"] == 0.09
    assert contract["decision_if_fail"] == "close_cloud_profile_bound_neutral_gauge_v1"


def test_p4gi_compiler_inverts_strict_profile_response():
    levels = np.linspace(0.0, 1.0, 17)
    response = np.stack(
        [0.03 + 0.92 * levels, 0.02 + 0.94 * levels, 0.01 + 0.96 * levels],
        axis=-1,
    )
    payload, metrics = compile_profile_bound_neutral_gauge(
        levels,
        response,
        base_component_sha256="a" * 64,
    )
    restored = apply_profile_bound_neutral_gauge(payload, response[1:-1])
    assert metrics["payload_sha256"]
    assert np.max(np.abs(restored - levels[1:-1, None])) < 1e-12


def test_p4gi_compiler_does_not_repair_nonmonotone_response():
    levels = np.linspace(0.0, 1.0, 5)
    response = np.repeat(levels[:, None], 3, axis=1)
    response[3, 1] = response[2, 1] - 0.01
    analysis = analyze_profile_bound_neutral_response(levels, response)
    assert analysis["channels"][1]["nonpositive_step_count"] == 1
    with pytest.raises(ValueError, match="not strictly increasing"):
        compile_profile_bound_neutral_gauge(
            levels,
            response,
            base_component_sha256="b" * 64,
        )


def test_p4gi_formal_result_closes_before_inverse_fit():
    result = json.loads(
        (ROOT / "docs/evidence/U6_P4GI_CLOUD_PROFILE_BOUND_NEUTRAL_GAUGE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["automatic_pass"] is False
    assert result["stable"]["compiled_gauge"] is None
    assert result["stable"]["confirmation_metrics"] is None
    assert result["stable"]["decision"] == "close_cloud_profile_bound_neutral_gauge_v1"
    channels = result["stable"]["compile_metrics"]["channels"]
    assert max(row["response_span"] for row in channels) < 0.00032
    assert min(row["nonpositive_step_count"] for row in channels) >= 73
    assert result["stable_evidence_id"] == "0936371e1b8cfd0082470e3a436fd3c59cff531ea14048bbceb5e01251543733"
