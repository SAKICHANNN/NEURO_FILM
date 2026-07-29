from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    EmulsionFamily,
    GenericReversalDevelopment,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    develop_reversal_layer_exposure,
    prepare_interpretation_medium,
    reversal_development_contract,
    reversal_development_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _operator() -> GenericReversalDevelopment:
    payload = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return GenericReversalDevelopment(build_operator(payload), 16.0)


def _exposure(values: np.ndarray) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.asarray(values, dtype=np.float32),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )


def test_reversal_density_decreases_and_direct_medium_brightens() -> None:
    operator = _operator()
    ramp = np.linspace(0.0, 16.0, 4097, dtype=np.float32)
    exposure = _exposure(np.repeat(ramp[:, None], 3, axis=1))
    result = develop_reversal_layer_exposure(
        exposure, operator, reversal_development_contract(operator)
    )
    assert np.max(np.diff(result.density.values, axis=0)) <= 0.0
    medium = prepare_interpretation_medium(result)
    assert medium.interpretation_route is InterpretationRoute.SLIDE_DIRECT_SCAN
    assert np.all(medium.values[-1] > medium.values[0])


def test_reversal_roundtrip_serialization_and_partition_are_exact() -> None:
    operator = _operator()
    replay = GenericReversalDevelopment.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    values = np.random.default_rng(20260729).uniform(
        0.0, 16.0, size=(401, 7, 3)
    )
    full = replay.apply(values)
    restored = replay.inverse(full)
    np.testing.assert_allclose(restored, values, rtol=0.0, atol=1e-9)
    parts = np.concatenate(
        [replay.apply(values[:103]), replay.apply(values[103:])]
    )
    np.testing.assert_array_equal(parts, full)
    assert replay.to_dict() == operator.to_dict()
    assert reversal_development_identity(
        replay
    ) == reversal_development_identity(operator)


def test_reversal_contract_and_domain_fail_closed() -> None:
    operator = _operator()
    contract = reversal_development_contract(operator)
    assert contract.emulsion_family is EmulsionFamily.SLIDE
    assert contract.interpretation_route is InterpretationRoute.SLIDE_DIRECT_SCAN
    with pytest.raises(ValueError, match="identity"):
        develop_reversal_layer_exposure(
            _exposure(np.full((2, 3), 0.18)),
            operator,
            replace(contract, sensitometry_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="maximum"):
        operator.apply(np.full((2, 3), 16.01))
    wrong = PhysicalDomainArray(
        np.full((2, 3), 0.18, np.float32),
        PhysicalDomain.SCENE_LINEAR,
        PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
        ("red", "green", "blue"),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        develop_reversal_layer_exposure(wrong, operator, contract)


def test_serialized_endpoint_identity_cannot_be_forged() -> None:
    payload = _operator().to_dict()
    payload["endpoint_density_sum"][0] += 1e-6
    with pytest.raises(ValueError, match="endpoint"):
        GenericReversalDevelopment.from_dict(payload)
