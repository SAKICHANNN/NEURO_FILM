from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_scanner_profile import _profile
from src.eval.sensitometry_primitive import build_operator
from src.film_physics import (
    DevelopmentInterpretationContract,
    EmulsionFamily,
    InterpretationRoute,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalUnit,
    apply_post_scan_polarity,
    compile_print_interpretation,
    develop_layer_exposure,
    prepare_interpretation_medium,
    scan_interpretation_medium,
)


ROOT = Path(__file__).resolve().parents[1]


def _setup(route: InterpretationRoute):
    u2 = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(
            encoding="utf-8"
        )
    )
    operator = build_operator(u2)
    family = (
        EmulsionFamily.SLIDE
        if route is InterpretationRoute.SLIDE_DIRECT_SCAN
        else EmulsionFamily.COLOR_NEGATIVE
    )
    contract = DevelopmentInterpretationContract.for_operator(
        operator, emulsion_family=family, interpretation_route=route
    )
    y, x = np.mgrid[:67, :103]
    values = np.stack(
        (
            16.0 * x / 102.0,
            16.0 * y / 66.0,
            8.0 * (x / 102.0 + y / 66.0),
        ),
        axis=-1,
    ).astype(np.float32)
    exposure = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
    )
    developed = develop_layer_exposure(exposure, operator, contract)
    print_operator = None
    if route is InterpretationRoute.COLOR_NEGATIVE_PRINT:
        witness = json.loads(
            (
                ROOT / "configs/u5_r2e0_density_domain_operator_v1.json"
            ).read_text(encoding="utf-8")
        )["witnesses"]["cyan_shadow_warm_highlight_like"]
        print_operator = compile_print_interpretation(
            operator,
            dye_absorption_matrix=np.asarray(
                witness["dye_absorption_matrix"]
            ),
            print_matrix=np.asarray(witness["print_matrix"]),
            paper_midpoints=np.asarray(witness["paper_midpoints"]),
            paper_slopes=np.asarray(witness["paper_slopes"]),
            paper_maximum_densities=np.asarray(
                witness["paper_maximum_densities"]
            ),
            maximum_relative_layer_exposure=16.0,
        )
    medium = prepare_interpretation_medium(
        developed, print_interpretation=print_operator
    )
    scanner_contract = json.loads(
        (
            ROOT / "configs/u6_p6a_scanner_profile_boundary_v1.json"
        ).read_text(encoding="utf-8")
    )
    return medium, _profile(scanner_contract["profiles"]["scanner_a"])


def test_negative_and_slide_diverge_only_after_explicit_polarity() -> None:
    negative, scanner = _setup(
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN
    )
    slide, _ = _setup(InterpretationRoute.SLIDE_DIRECT_SCAN)
    np.testing.assert_array_equal(negative.values, slide.values)
    neg = scan_interpretation_medium(
        negative, scanner, pixel_pitch_um=1.0
    )
    pos = scan_interpretation_medium(slide, scanner, pixel_pitch_um=1.0)
    np.testing.assert_allclose(
        neg.scan_linear.values + pos.scan_linear.values, 1.0, atol=1e-12
    )
    assert neg.order[-2:] == ("scanner_profile", "post_scan_polarity")


def test_polarity_before_scanner_is_detectably_wrong() -> None:
    negative, scanner = _setup(
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN
    )
    correct = scan_interpretation_medium(
        negative, scanner, pixel_pitch_um=1.0
    ).scan_linear.values
    from src.film_physics import apply_scanner_profile

    wrong = apply_scanner_profile(
        apply_post_scan_polarity(
            negative.values, negative.post_scan_polarity
        ),
        scanner,
        pixel_pitch_um=1.0,
    )
    assert float(np.max(np.abs(correct - wrong))) > 0.001


def test_print_route_is_bound_and_distinct() -> None:
    print_medium, scanner = _setup(
        InterpretationRoute.COLOR_NEGATIVE_PRINT
    )
    negative, _ = _setup(
        InterpretationRoute.COLOR_NEGATIVE_NEUTRAL_SCAN
    )
    printed = scan_interpretation_medium(
        print_medium,
        scanner,
        pixel_pitch_um=1.0,
        expected_route=InterpretationRoute.COLOR_NEGATIVE_PRINT,
    )
    neg = scan_interpretation_medium(
        negative, scanner, pixel_pitch_um=1.0
    )
    assert (
        float(
            np.max(
                np.abs(
                    printed.scan_linear.values - neg.scan_linear.values
                )
            )
        )
        > 0.01
    )
    with pytest.raises(ValueError, match="route binding"):
        scan_interpretation_medium(
            print_medium,
            scanner,
            pixel_pitch_um=1.0,
            expected_route=InterpretationRoute.SLIDE_DIRECT_SCAN,
        )


def test_scan_replay_and_medium_preservation_are_exact() -> None:
    medium, scanner = _setup(InterpretationRoute.SLIDE_DIRECT_SCAN)
    before = medium.values.tobytes()
    first = scan_interpretation_medium(
        medium, scanner, pixel_pitch_um=1.0
    )
    second = scan_interpretation_medium(
        medium, scanner, pixel_pitch_um=1.0
    )
    assert first.scan_linear.values.tobytes() == second.scan_linear.values.tobytes()
    assert first.descriptor() == second.descriptor()
    assert medium.values.tobytes() == before
