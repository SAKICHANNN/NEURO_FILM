from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.sensitometry_primitive import build_operator
from src.roll2film.sensitometry_print import (
    DensityToPrintInterpretation,
    SensitometryPrintOperator,
)


ROOT = Path(__file__).resolve().parents[1]


def _operator() -> SensitometryPrintOperator:
    parent = json.loads(
        (ROOT / "configs/u2_2a_sensitometry_primitive_v1.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        (ROOT / "configs/u5_r2e0_density_domain_operator_v1.json").read_text(
            encoding="utf-8"
        )
    )["witnesses"]["cyan_shadow_warm_highlight_like"]
    sensitometry = build_operator(parent)
    references = sensitometry.apply(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    interpretation = DensityToPrintInterpretation(
        np.asarray(source["dye_absorption_matrix"]),
        np.asarray(source["print_matrix"]),
        np.asarray(source["paper_midpoints"]),
        np.asarray(source["paper_slopes"]),
        np.asarray(source["paper_maximum_densities"]),
        references[0],
        references[1],
    )
    return SensitometryPrintOperator(sensitometry, interpretation)


def test_composition_endpoints_partition_replay_and_input_preservation() -> None:
    operator = _operator()
    source = np.random.default_rng(61).random((307, 3))
    before = source.copy()
    full = operator.apply(source)
    partitioned = np.concatenate(
        (operator.apply(source[:101]), operator.apply(source[101:219]), operator.apply(source[219:])),
        axis=0,
    )
    replay = SensitometryPrintOperator.from_dict(json.loads(json.dumps(operator.to_dict())))
    assert np.max(np.abs(operator.apply(np.array([[0.0] * 3, [1.0] * 3])) - np.array([[0.0] * 3, [1.0] * 3]))) <= 1e-12
    assert np.array_equal(full, partitioned)
    assert np.array_equal(replay.apply(source), full)
    assert np.array_equal(source, before)
    assert np.min(full) >= -1e-12 and np.max(full) <= 1.0 + 1e-12


def test_composition_and_density_reference_guards() -> None:
    operator = _operator()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        operator.apply(np.array([[1.01, 0.5, 0.5]]))
    bad = operator.interpretation.black_reference_density.copy()
    bad[0] -= 1e-6
    with pytest.raises(ValueError, match="outside"):
        operator.interpretation.apply(bad[None, :])
