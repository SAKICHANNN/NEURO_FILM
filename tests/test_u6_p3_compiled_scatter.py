from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_compiled_scatter import (
    P1_SCHEMA,
    P3_SCHEMA,
    evaluate_compiled_scatter,
    load_json,
)
from src.film_physics import (
    CompiledScatterKernel,
    CompiledScatterProfile,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_scatter,
    apply_compiled_scatter_row_tiled,
    compile_scatter_profile,
)
from src.film_physics.reference_scatter import profile_from_contract


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "configs" / "u6_p1_reference_scatter_simulator_v1.json"
P3 = ROOT / "configs" / "u6_p3_compiled_scatter_challenger_v1.json"


def _input(values: np.ndarray) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )


def test_compiler_is_positive_and_energy_bounded() -> None:
    reference = profile_from_contract(load_json(P1, P1_SCHEMA))
    compiled = compile_scatter_profile(reference)
    assert compiled.parent_profile_sha256 == reference.profile_sha256
    assert compiled.required_halo == 180
    for kernel in compiled.kernels:
        assert kernel.weights.dtype == np.float32
        assert np.all(kernel.weights >= 0.0)
        assert np.sum(kernel.weights, dtype=np.float32) == np.float32(1.0)


def test_row_tiles_reconstruct_full_candidate_exactly() -> None:
    reference = profile_from_contract(load_json(P1, P1_SCHEMA))
    compiled = compile_scatter_profile(reference)
    values = np.random.default_rng(44).random((31, 29, 3), dtype=np.float32)
    source = _input(values)
    full = apply_compiled_scatter(source, compiled).values
    tiled = apply_compiled_scatter_row_tiled(source, compiled, tile_rows=7).values
    assert np.array_equal(full, tiled)
    with pytest.raises(ValueError, match="positive integer"):
        apply_compiled_scatter_row_tiled(source, compiled, tile_rows=0)


def test_candidate_requires_float32() -> None:
    reference = profile_from_contract(load_json(P1, P1_SCHEMA))
    compiled = compile_scatter_profile(reference)
    with pytest.raises(TypeError, match="float32"):
        apply_compiled_scatter(
            _input(np.ones((3, 3, 3), dtype=np.float64)), compiled
        )


def test_compiled_profile_rejects_invalid_energy_partition() -> None:
    kernel = CompiledScatterKernel(
        "test", np.asarray([0.25, 0.5, 0.25], dtype=np.float32), (0.1, 0.1, 0.1)
    )
    with pytest.raises(ValueError, match="sum to one"):
        CompiledScatterProfile("1" * 64, 8.0, (0.8, 0.8, 0.8), (kernel,))


def test_frozen_compiled_report_passes_and_repeats() -> None:
    first = evaluate_compiled_scatter(
        load_json(P1, P1_SCHEMA), load_json(P3, P3_SCHEMA)
    )
    second = evaluate_compiled_scatter(
        load_json(P1, P1_SCHEMA), load_json(P3, P3_SCHEMA)
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["claim_ceiling"].endswith("development-challenger")
