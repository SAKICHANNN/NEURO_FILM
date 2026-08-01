from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf_minimax_envelope import (
    MeasuredMtfEnvelopeError,
    evaluate_minimax_envelope,
    validate_contract,
)
from src.film_physics.measured_mtf_envelope import (
    PLACEMENT_ARMS,
    build_minimax_transmittance_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p5p_measured_mtf_minimax_envelope_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_minimax_envelope_exact_scalar_solution_and_guards() -> None:
    arms = {
        PLACEMENT_ARMS[0]: np.asarray([[[0.2, 0.4, 0.8]]]),
        PLACEMENT_ARMS[1]: np.asarray([[[0.4, 0.3, 0.6]]]),
        PLACEMENT_ARMS[2]: np.asarray([[[0.3, 0.5, 0.7]]]),
    }
    result = build_minimax_transmittance_envelope(arms)
    assert np.array_equal(result.lower, np.asarray([[[0.2, 0.3, 0.6]]]))
    assert np.array_equal(result.upper, np.asarray([[[0.4, 0.5, 0.8]]]))
    assert np.allclose(result.transmittance, np.asarray([[[0.3, 0.4, 0.7]]]))
    assert np.allclose(result.uncertainty_half_range, 0.1)
    with pytest.raises(ValueError, match="ordered fixed arms"):
        build_minimax_transmittance_envelope(dict(reversed(list(arms.items()))))
    bad = dict(arms)
    bad[PLACEMENT_ARMS[2]] = np.asarray([[[0.0, 0.5, 0.7]]])
    with pytest.raises(ValueError, match="inside"):
        build_minimax_transmittance_envelope(bad)


def test_contract_drift_rejected() -> None:
    config = _config()
    validate_contract(config)
    config["algorithm"]["fitted_parameters"] = 1
    with pytest.raises(MeasuredMtfEnvelopeError, match="contract drift"):
        validate_contract(config)


def test_formal_envelope_passes_exact_bounds(tmp_path: Path) -> None:
    first = evaluate_minimax_envelope(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "first.png"
    )
    second = evaluate_minimax_envelope(
        root=ROOT, config=_config(), diagnostic_path=tmp_path / "second.png"
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert first["decision"] == "retain_minimax_envelope_open_photographic_challenger"
    assert all(first["gate_results"].values())
    assert first["row_partition_exact"] is True
    assert (tmp_path / "first.png").read_bytes() == (
        tmp_path / "second.png"
    ).read_bytes()


def test_parent_drift_fails_closed(tmp_path: Path) -> None:
    config = _config()
    config["parents"]["p5o_decision_sha256"] = "0" * 64
    with pytest.raises(MeasuredMtfEnvelopeError, match="parent hash mismatch"):
        evaluate_minimax_envelope(
            root=ROOT, config=config, diagnostic_path=tmp_path / "x.png"
        )
