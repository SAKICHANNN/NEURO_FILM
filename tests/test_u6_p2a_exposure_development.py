from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    DevelopedExposureResult,
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationEvidence,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    ProcessCondition,
    develop_layer_exposure,
    sensitometry_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "configs"
    / "u6_p2a_exposure_development_interpretation_contract_v1.json"
)


def _operator():
    payload = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return build_operator(payload)


def _exposure(rows: int = 41) -> PhysicalDomainArray:
    values = np.random.default_rng(20260729).uniform(
        0.0, 16.0, size=(rows, 19, 3)
    ).astype(np.float32)
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(6.35),
    )


@pytest.mark.parametrize(
    ("family", "route"),
    [
        (
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        ),
        (
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_PRINT,
        ),
        (EmulsionFamily.SLIDE, InterpretationRoute.SLIDE_DIRECT_SCAN),
        (
            EmulsionFamily.BLACK_AND_WHITE,
            InterpretationRoute.BW_DEVELOPER_SCAN,
        ),
    ],
)
def test_routes_reuse_u2_sensitometry_exactly(
    family: EmulsionFamily, route: InterpretationRoute
) -> None:
    operator = _operator()
    contract = DevelopmentInterpretationContract.for_operator(
        operator, emulsion_family=family, interpretation_route=route
    )
    source = _exposure()
    before = source.values.tobytes()
    result = develop_layer_exposure(source, operator, contract)
    assert isinstance(result, DevelopedExposureResult)
    assert result.density.domain is PhysicalDomain.DEVELOPED_DENSITY
    assert result.density.unit is PhysicalUnit.OPTICAL_DENSITY
    assert result.density.values.dtype == np.float32
    np.testing.assert_array_equal(
        result.density.values,
        operator.apply(source.values).astype(np.float32),
    )
    assert source.values.tobytes() == before


def test_partition_serialization_and_replay_are_exact() -> None:
    operator = _operator()
    contract = DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.SLIDE,
        interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
    )
    replay = DevelopmentInterpretationContract.from_dict(
        json.loads(json.dumps(contract.to_dict()))
    )
    source = _exposure()
    full = develop_layer_exposure(source, operator, replay).density.values
    parts = []
    for start, stop in ((0, 7), (7, 29), (29, 41)):
        section = PhysicalDomainArray(
            source.values[start:stop],
            source.domain,
            source.unit,
            source.channels,
            source.scale,
        )
        parts.append(develop_layer_exposure(section, operator, replay).density.values)
    np.testing.assert_array_equal(np.concatenate(parts), full)
    assert replay == contract
    assert replay.sensitometry_sha256 == sensitometry_identity(operator)


def test_process_state_is_unknown_or_hypothesis_only() -> None:
    operator = _operator()
    DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
        interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
    )
    DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
        interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
        process_condition=ProcessCondition.PUSH,
        interpretation_evidence=InterpretationEvidence.HYPOTHESIS_ONLY,
    )
    with pytest.raises(ValueError, match="unknown process"):
        DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
            interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
            interpretation_evidence=InterpretationEvidence.HYPOTHESIS_ONLY,
        )
    with pytest.raises(ValueError, match="hypothesis_only"):
        DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.COLOR_NEGATIVE,
            interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
            process_condition=ProcessCondition.PULL,
        )


def test_wrong_route_domain_identity_and_bounds_fail_closed() -> None:
    operator = _operator()
    with pytest.raises(ValueError, match="does not belong"):
        DevelopmentInterpretationContract.for_operator(
            operator,
            emulsion_family=EmulsionFamily.SLIDE,
            interpretation_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
        )
    contract = DevelopmentInterpretationContract.for_operator(
        operator,
        emulsion_family=EmulsionFamily.SLIDE,
        interpretation_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
    )
    wrong_domain = PhysicalDomainArray(
        np.full((2, 3, 3), 0.5, np.float32),
        PhysicalDomain.SCENE_LINEAR,
        PhysicalUnit.RELATIVE_SCENE_EXPOSURE,
        ("red", "green", "blue"),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        develop_layer_exposure(wrong_domain, operator, contract)
    with pytest.raises(ValueError, match="identity"):
        develop_layer_exposure(
            _exposure(2),
            operator,
            replace(contract, sensitometry_sha256="0" * 64),
        )
    too_high = PhysicalDomainArray(
        np.full((2, 3, 3), 16.01, np.float32),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    with pytest.raises(ValueError, match="maximum"):
        develop_layer_exposure(too_high, operator, contract)


def test_contract_forbids_duplicate_tone_curve_implementation() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert payload["reused_sensitometry"]["duplicate_curve_implementation_allowed"] is False
    source = (
        ROOT / "src/film_physics/exposure_development.py"
    ).read_text(encoding="utf-8")
    assert "class AnchoredCharacteristicCurve" not in source
    assert "class LogExposureEncoder" not in source
    assert "class RationalQuadraticSpline" not in source
