from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_derivative_conditioned_structure import (
    DerivativeConditionedStructureError,
    evaluate_structure,
    load_contract,
    profile_from_contract,
)
from src.eval.sensitometry_primitive import build_operator
from src.film_physics.derivative_conditioned_structure import (
    render_derivative_conditioned_structure,
    render_derivative_conditioned_structure_region,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4af_derivative_conditioned_structure_v1.json"
SENSITOMETRY = ROOT / "configs/u2_2a_sensitometry_primitive_v1.json"


def test_contract_rejects_amplitude_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["peak_density_variance"] = 0.0002
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DerivativeConditionedStructureError, match="contract drift"):
        load_contract(path)


def test_renderer_rejects_out_of_domain_and_preserves_zero_noise() -> None:
    contract = load_contract(CONTRACT)
    profile = profile_from_contract(contract)
    operator = build_operator(json.loads(SENSITOMETRY.read_text(encoding="utf-8")))
    zero = np.zeros((17, 19, 3), dtype=np.float64)
    result = render_derivative_conditioned_structure(zero, operator, profile)
    target = operator.apply(zero).astype(np.float32)
    assert np.array_equal(result.density, target)
    expected_transmittance = np.power(
        10.0, -target.astype(np.float64)
    ).astype(np.float32)
    assert np.array_equal(result.transmittance, expected_transmittance)
    with pytest.raises(ValueError, match="profile domain"):
        render_derivative_conditioned_structure(
            np.full((2, 2, 3), 16.01), operator, profile
        )


def test_renderer_is_repeat_and_partition_exact() -> None:
    contract = load_contract(CONTRACT)
    profile = profile_from_contract(contract)
    operator = build_operator(json.loads(SENSITOMETRY.read_text(encoding="utf-8")))
    values = np.geomspace(1e-4, 16.0, 53)
    source = np.broadcast_to(values[None, :, None], (61, 53, 3)).copy()
    full = render_derivative_conditioned_structure(source, operator, profile)
    repeat = render_derivative_conditioned_structure(source, operator, profile)
    assert np.array_equal(full.density, repeat.density)
    assembled = np.empty_like(full.density)
    for y0 in range(0, source.shape[0], 13):
        y1 = min(source.shape[0], y0 + 13)
        region = render_derivative_conditioned_structure_region(
            source,
            operator,
            profile,
            origin_yx=(y0, 0),
            shape=(y1 - y0, source.shape[1]),
        )
        assembled[y0:y1] = region.density
    assert np.array_equal(full.density, assembled)


def test_frozen_structure_evaluation_is_repeat_exact() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate_structure(contract, ROOT)
    second = evaluate_structure(contract, ROOT)
    assert first == second
    assert first["passed"] is all(first["gate_results"].values())
