from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    DevelopedExposureResult,
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationMediumKind,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    PostScanPolarity,
    develop_layer_exposure,
    prepare_interpretation_medium,
)
from src.roll2film.sensitometry_print import DensityToPrintInterpretation


ROOT = Path(__file__).resolve().parents[1]


def _operator():
    payload = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    return build_operator(payload)


def _exposure() -> PhysicalDomainArray:
    values = np.random.default_rng(20260729).uniform(
        0.0, 16.0, size=(37, 23, 3)
    ).astype(np.float32)
    values[0] = 0.0
    values[-1] = 16.0
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )


def _developed(
    family: EmulsionFamily, route: InterpretationRoute
) -> DevelopedExposureResult:
    operator = _operator()
    contract = DevelopmentInterpretationContract.for_operator(
        operator, emulsion_family=family, interpretation_route=route
    )
    return develop_layer_exposure(_exposure(), operator, contract)


def _print_operator() -> DensityToPrintInterpretation:
    sensitometry = _operator()
    references = sensitometry.apply(
        np.asarray([[0.0] * 3, [16.0] * 3], dtype=np.float64)
    )
    # P2A preserves the caller dtype.  Enclose the exact float64 endpoints by
    # one float32 ULP so a valid compiled float32 density cannot be rejected
    # solely by endpoint representation roundoff.
    black = np.nextafter(
        references[0].astype(np.float32), np.float32(-np.inf)
    ).astype(np.float64)
    white = np.nextafter(
        references[1].astype(np.float32), np.float32(np.inf)
    ).astype(np.float64)
    witness = json.loads(
        (
            ROOT / "configs/u5_r2e0_density_domain_operator_v1.json"
        ).read_text(encoding="utf-8")
    )["witnesses"]["cyan_shadow_warm_highlight_like"]
    return DensityToPrintInterpretation(
        np.asarray(witness["dye_absorption_matrix"]),
        np.asarray(witness["print_matrix"]),
        np.asarray(witness["paper_midpoints"]),
        np.asarray(witness["paper_slopes"]),
        np.asarray(witness["paper_maximum_densities"]),
        black,
        white,
    )


def test_negative_and_slide_share_medium_bytes_but_not_polarity() -> None:
    negative = prepare_interpretation_medium(
        _developed(
            EmulsionFamily.COLOR_NEGATIVE,
            InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN,
        )
    )
    slide = prepare_interpretation_medium(
        _developed(EmulsionFamily.SLIDE, InterpretationRoute.SLIDE_DIRECT_SCAN)
    )
    np.testing.assert_array_equal(negative.values, slide.values)
    assert negative.kind is InterpretationMediumKind.FILM_TRANSMITTANCE
    assert slide.kind is InterpretationMediumKind.FILM_TRANSMITTANCE
    assert negative.post_scan_polarity is PostScanPolarity.INVERT_TO_POSITIVE
    assert slide.post_scan_polarity is PostScanPolarity.IDENTITY


def test_print_route_produces_distinct_bounded_reflectance() -> None:
    developed = _developed(
        EmulsionFamily.COLOR_NEGATIVE,
        InterpretationRoute.COLOR_NEGATIVE_PRINT,
    )
    medium = prepare_interpretation_medium(
        developed, print_interpretation=_print_operator()
    )
    assert medium.kind is InterpretationMediumKind.PRINT_REFLECTANCE
    assert medium.post_scan_polarity is PostScanPolarity.IDENTITY
    assert medium.values.dtype == np.float32
    assert float(np.min(medium.values)) >= 0.0
    assert float(np.max(medium.values)) <= 1.0
    assert not np.array_equal(
        medium.values,
        np.power(10.0, -developed.density.values).astype(np.float32),
    )


def test_colour_density_cannot_masquerade_as_bw_silver() -> None:
    developed = _developed(
        EmulsionFamily.BLACK_AND_WHITE,
        InterpretationRoute.BW_DEVELOPER_SCAN,
    )
    with pytest.raises(ValueError, match="neutral axis"):
        prepare_interpretation_medium(developed)
    neutral = np.repeat(
        developed.density.values[..., 1:2], 3, axis=-1
    ).astype(np.float32)
    compatible = DevelopedExposureResult(
        PhysicalDomainArray(
            neutral,
            PhysicalDomain.DEVELOPED_DENSITY,
            PhysicalUnit.OPTICAL_DENSITY,
            ("red", "green", "blue"),
        ),
        developed.contract,
    )
    medium = prepare_interpretation_medium(compatible)
    assert medium.kind is InterpretationMediumKind.FILM_TRANSMITTANCE
    assert medium.post_scan_polarity is PostScanPolarity.INVERT_TO_POSITIVE
    np.testing.assert_array_equal(medium.values[..., 0], medium.values[..., 1])
    np.testing.assert_array_equal(medium.values[..., 1], medium.values[..., 2])


def test_missing_or_foreign_print_operator_fails_closed() -> None:
    print_developed = _developed(
        EmulsionFamily.COLOR_NEGATIVE,
        InterpretationRoute.COLOR_NEGATIVE_PRINT,
    )
    with pytest.raises(TypeError, match="requires"):
        prepare_interpretation_medium(print_developed)
    slide = _developed(
        EmulsionFamily.SLIDE, InterpretationRoute.SLIDE_DIRECT_SCAN
    )
    with pytest.raises(ValueError, match="cannot consume"):
        prepare_interpretation_medium(
            slide, print_interpretation=_print_operator()
        )


def test_medium_replay_partition_and_input_preservation() -> None:
    developed = _developed(
        EmulsionFamily.COLOR_NEGATIVE,
        InterpretationRoute.COLOR_NEGATIVE_PRINT,
    )
    before = developed.density.values.tobytes()
    operator = _print_operator()
    first = prepare_interpretation_medium(
        developed, print_interpretation=operator
    )
    second = prepare_interpretation_medium(
        developed, print_interpretation=operator
    )
    assert first.descriptor() == second.descriptor()
    assert first.values.tobytes() == second.values.tobytes()
    assert developed.density.values.tobytes() == before
