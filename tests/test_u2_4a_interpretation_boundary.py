from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.inference import (
    INTERPRETATION_CONTRACT_VERSION,
    InterpretationContractError,
    InterpretationPlugin,
    InterpretationRequest,
    execute_interpretation,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CLAIM = "synthetic test witness only; no physical or production claim"


def _source() -> np.ndarray:
    y, x = np.mgrid[:17, :23]
    return np.stack(
        (
            (x + y * 2) / 56.0,
            (x * 2 + y) / 62.0,
            (x * 3 + y * 4) / 130.0,
        ),
        axis=-1,
    ).astype(np.float32)


def _slide(rgb: np.ndarray) -> tuple[np.ndarray, object]:
    return rgb.copy(), (("witness", "identity"),)


def _bw(rgb: np.ndarray) -> tuple[np.ndarray, object]:
    luma = (
        rgb[..., 0] * np.float32(0.2126)
        + rgb[..., 1] * np.float32(0.7152)
        + rgb[..., 2] * np.float32(0.0722)
    ).astype(np.float32)
    return np.repeat(luma[..., None], 3, axis=2), (("witness", "neutral_axis"),)


def _negative(rgb: np.ndarray) -> tuple[np.ndarray, object]:
    return (np.float32(1.0) - rgb).astype(np.float32), (("witness", "synthetic_inversion"),)


PLUGINS = (
    InterpretationPlugin(
        "color_negative_neutral_scan",
        "synthetic_negative_v1",
        "negative_scan_linear_rgb",
        "display_linear_rgb",
        "synthetic_test_only",
        False,
        SYNTHETIC_CLAIM,
        _negative,
    ),
    InterpretationPlugin(
        "slide_direct_scan",
        "synthetic_slide_v1",
        "display_linear_rgb",
        "display_linear_rgb",
        "synthetic_test_only",
        False,
        SYNTHETIC_CLAIM,
        _slide,
    ),
    InterpretationPlugin(
        "bw_developer_scan",
        "synthetic_bw_v1",
        "display_linear_rgb",
        "display_linear_rgb",
        "synthetic_test_only",
        False,
        SYNTHETIC_CLAIM,
        _bw,
    ),
)


def _request(plugin: InterpretationPlugin, **changes) -> InterpretationRequest:
    values = {
        "interpretation_id": plugin.interpretation_id,
        "operator_id": plugin.operator_id,
        "input_domain": plugin.input_domain,
        "output_domain": plugin.output_domain,
        "production": False,
    }
    values.update(changes)
    return InterpretationRequest(**values)


@pytest.mark.parametrize("plugin", PLUGINS, ids=lambda value: value.operator_id)
def test_synthetic_witnesses_are_deterministic_bounded_and_non_mutating(
    plugin: InterpretationPlugin,
) -> None:
    source = _source()
    original = source.copy()
    first = execute_interpretation(source, _request(plugin), PLUGINS)
    second = execute_interpretation(source, _request(plugin), PLUGINS)
    assert first.rgb.dtype == np.float32
    assert first.rgb.shape == source.shape
    assert np.isfinite(first.rgb).all()
    assert 0.0 <= float(first.rgb.min()) <= float(first.rgb.max()) <= 1.0
    assert first.rgb.tobytes() == second.rgb.tobytes()
    assert first.metadata == second.metadata
    assert first.metadata.contract_version == INTERPRETATION_CONTRACT_VERSION
    assert first.metadata.evidence_scope == "synthetic_test_only"
    assert first.metadata.production_eligible is False
    assert source.tobytes() == original.tobytes()
    assert json.loads(json.dumps(asdict(first.metadata)))["evidence_scope"] == "synthetic_test_only"


def test_slide_is_exact_identity_and_bw_is_exact_neutral_axis() -> None:
    source = _source()
    slide = execute_interpretation(source, _request(PLUGINS[1]), PLUGINS).rgb
    bw = execute_interpretation(source, _request(PLUGINS[2]), PLUGINS).rgb
    np.testing.assert_array_equal(slide, source)
    np.testing.assert_array_equal(bw[..., 0], bw[..., 1])
    np.testing.assert_array_equal(bw[..., 1], bw[..., 2])


def test_synthetic_negative_is_invertible_within_frozen_tolerance() -> None:
    source = _source()
    first, _ = _negative(source)
    second, _ = _negative(first)
    assert float(np.max(np.abs(second - source))) <= 1e-6


@pytest.mark.parametrize(
    "case",
    [
        InterpretationRequest(
            "slide_direct_scan", "missing", "display_linear_rgb", "display_linear_rgb"
        ),
        InterpretationRequest(
            "bw_developer_scan",
            "synthetic_slide_v1",
            "display_linear_rgb",
            "display_linear_rgb",
        ),
        InterpretationRequest(
            "color_negative_neutral_scan",
            "synthetic_negative_v1",
            "display_linear_rgb",
            "display_linear_rgb",
        ),
        _request(PLUGINS[0], production=True),
    ],
)
def test_unknown_mismatched_domain_and_production_requests_fail_closed(
    case: InterpretationRequest,
) -> None:
    with pytest.raises(InterpretationContractError):
        execute_interpretation(_source(), case, PLUGINS)


@pytest.mark.parametrize(
    "source",
    [
        _source().astype(np.float64),
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((0, 4, 3), dtype=np.float32),
        np.full((3, 4, 3), np.nan, dtype=np.float32),
        np.full((3, 4, 3), 1.1, dtype=np.float32),
    ],
)
def test_invalid_input_arrays_fail_closed(source: np.ndarray) -> None:
    with pytest.raises(InterpretationContractError):
        execute_interpretation(source, _request(PLUGINS[1]), PLUGINS)


@pytest.mark.parametrize("failure", ["mutate", "alias", "shape", "nan", "bounds", "metadata_dict", "metadata_array"])
def test_invalid_plugin_outputs_and_metadata_fail_closed(failure: str) -> None:
    def invalid(rgb: np.ndarray) -> tuple[np.ndarray, object]:
        if failure == "mutate":
            rgb[0, 0, 0] = 0.0
        output = rgb.copy()
        metadata: object = ()
        if failure == "alias":
            output = rgb
        elif failure == "shape":
            output = output[:-1]
        elif failure == "nan":
            output[0, 0, 0] = np.nan
        elif failure == "bounds":
            output[0, 0, 0] = 2.0
        elif failure == "metadata_dict":
            metadata = {"bad": "mutable"}
        elif failure == "metadata_array":
            metadata = np.zeros(1)
        return output, metadata

    plugin = copy.copy(PLUGINS[1])
    plugin = InterpretationPlugin(
        plugin.interpretation_id,
        plugin.operator_id,
        plugin.input_domain,
        plugin.output_domain,
        plugin.evidence_scope,
        plugin.production_eligible,
        plugin.claim_ceiling,
        invalid,
    )
    with pytest.raises(InterpretationContractError):
        execute_interpretation(_source(), _request(plugin), (plugin,))


def test_duplicate_registration_and_synthetic_escalation_fail_closed() -> None:
    with pytest.raises(InterpretationContractError, match="duplicate"):
        execute_interpretation(_source(), _request(PLUGINS[1]), (PLUGINS[1], PLUGINS[1]))
    escalated = InterpretationPlugin(
        PLUGINS[1].interpretation_id,
        PLUGINS[1].operator_id,
        PLUGINS[1].input_domain,
        PLUGINS[1].output_domain,
        PLUGINS[1].evidence_scope,
        True,
        PLUGINS[1].claim_ceiling,
        PLUGINS[1].apply,
    )
    with pytest.raises(InterpretationContractError, match="synthetic"):
        execute_interpretation(_source(), _request(escalated), (escalated,))


def test_boundary_has_zero_renderer_or_schema_integration() -> None:
    assert "interpretation" not in (ROOT / "scripts/render_film.py").read_text(
        encoding="utf-8"
    )
    for path in (
        ROOT / "configs/schemas/render_profile_v1.schema.json",
        ROOT / "configs/schemas/render_recipe_v1.schema.json",
        ROOT / "configs/render_profiles/safe_rich_v1.json",
    ):
        assert "interpretation-plugin-boundary-v1" not in path.read_text(encoding="utf-8")
