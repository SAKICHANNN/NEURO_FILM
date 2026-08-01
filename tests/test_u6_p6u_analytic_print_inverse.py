from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.analytic_print_inverse import load_contract, run_audit
from src.eval.digital_lad_print_compatibility import _build_print_operator
from src.film_physics.print_sensitivity import invert_density_to_print

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p6u_analytic_print_inverse_v1.json"


def _operator():
    p6s = json.loads(
        (ROOT / "configs/u6_p6s_digital_lad_print_compatibility_v1.json").read_text(
            encoding="utf-8"
        )
    )
    sensitometry = json.loads(
        (ROOT / p6s["parents"]["sensitometry_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(
        (ROOT / p6s["parents"]["print_source_contract"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    return _build_print_operator(sensitometry, source, p6s)


def test_analytic_inverse_roundtrips_density() -> None:
    operator = _operator()
    density = np.asarray([[0.89, 1.04, 0.71], [1.31, 0.68, 1.12]])
    output = operator.apply(density)
    recovered = invert_density_to_print(operator, output)
    assert np.max(np.abs(recovered - density)) <= 1e-10
    assert np.max(np.abs(operator.apply(recovered) - output)) <= 1e-12


def test_inverse_rejects_nonphysical_outputs() -> None:
    operator = _operator()
    for output in (
        [[-0.01, 0.5, 0.5]],
        [[1.01, 0.5, 0.5]],
        [[0.0, 1.0, 0.0]],
        [[1.0, 0.0, 1.0]],
    ):
        with pytest.raises(ValueError):
            invert_density_to_print(operator, np.asarray(output))


def test_formal_inverse_audit_matches_frozen_gate_vocabulary() -> None:
    contract = load_contract(CONTRACT)
    report = run_audit(root=ROOT, contract=contract)
    assert set(report["gate_results"]) == set(contract["automatic_gates"])
    assert report["grid_rows"] == 343
    assert report["gate_results"]["optimization_iterations_zero"]
