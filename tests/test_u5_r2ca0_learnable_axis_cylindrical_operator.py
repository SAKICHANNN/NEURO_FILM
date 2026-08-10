from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval.learnable_axis_cylindrical_operator import (
    LearnableAxisOperatorError,
    apply_cylindrical_operator,
    evaluate_contract,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2ca0_learnable_axis_cylindrical_operator_v1.json"


def test_analytical_operator_is_deterministic_and_cube_safe() -> None:
    source = np.asarray([[0.0, 0.4, 1.0], [0.2, 0.7, 0.9], [0.5, 0.5, 0.5]])
    kwargs = {
        "tone_amplitude": 0.045,
        "chroma_gain": 0.12,
        "maximum_rotation_radians": 0.11,
        "boundary_epsilon": 1e-12,
    }
    first, first_scale = apply_cylindrical_operator(
        source, np.asarray([1.32, 0.82, 0.94]), **kwargs
    )
    second, second_scale = apply_cylindrical_operator(
        source, np.asarray([1.32, 0.82, 0.94]), **kwargs
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_scale, second_scale)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)


def test_ca0_frozen_gate_closes_on_gamut_pressure() -> None:
    report = evaluate_contract(CONTRACT)
    assert not report["passed"]
    assert report["decision"] == "close_exact_learnable_axis_cylindrical_representation"
    assert len(report["rows"]) == 6
    assert report["gates"]["axis_recovery"]
    assert report["gates"]["confirmation_error"]
    assert not report["gates"]["safe_scale"]
    assert not report["gates"]["limited_fraction"]
    assert not report["gates"]["boundary"]


def test_ca0_contract_drift_rejected(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["fixture"]["wrong_axis_offset"] = 1
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LearnableAxisOperatorError, match="boundary drift"):
        load_contract(path)


def test_ca0_runner_is_byte_deterministic(tmp_path: Path) -> None:
    script = ROOT / "scripts/run_u5_r2ca0_learnable_axis_cylindrical_operator.py"
    outputs = [tmp_path / "a.json", tmp_path / "b.json"]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(script), "--output", str(output)],
            cwd=tmp_path,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
