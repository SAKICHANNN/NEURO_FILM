import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.profile_bound_neutral_gauge import (
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
    with pytest.raises(ValueError, match="not strictly increasing"):
        compile_profile_bound_neutral_gauge(
            levels,
            response,
            base_component_sha256="b" * 64,
        )
