from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_compiled_backing_return import (
    P3D_SCHEMA,
    P3E_SCHEMA,
    evaluate_compiled_backing_return,
    load_json,
)
from src.film_physics import (
    CompiledBackingReturnKernel,
    CompiledBackingReturnProfile,
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    apply_compiled_backing_return,
    apply_compiled_backing_return_row_tiled,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
)


ROOT = Path(__file__).resolve().parents[1]
P3D = ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json"
P3E = ROOT / "configs" / "u6_p3e_backing_return_standard_v1.json"


def _input(values: np.ndarray, pitch: float = 8.0) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pitch),
    )


def test_compiler_preserves_positive_return_profile() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    assert compiled.parent_profile_sha256 == reference.profile_sha256
    assert compiled.required_halo == 180
    for kernel in compiled.kernels:
        assert kernel.weights.dtype == np.float32
        assert kernel.return_weights.dtype == np.float32
        assert np.all(kernel.weights >= 0.0)
        assert np.all(kernel.return_weights >= 0.0)
        assert np.sum(kernel.weights, dtype=np.float32) == np.float32(1.0)


def test_row_tiles_reconstruct_full_candidate_exactly() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    values = np.random.default_rng(81).random((31, 29, 3), dtype=np.float32)
    source = _input(values)
    full = apply_compiled_backing_return(source, compiled).values
    tiled = apply_compiled_backing_return_row_tiled(
        source, compiled, tile_rows=7
    ).values
    assert np.array_equal(full, tiled)
    with pytest.raises(ValueError, match="positive integer"):
        apply_compiled_backing_return_row_tiled(source, compiled, tile_rows=0)


def test_candidate_rejects_wrong_dtype_domain_and_scale() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    with pytest.raises(TypeError, match="float32"):
        apply_compiled_backing_return(
            _input(np.ones((3, 3, 3), dtype=np.float64)), compiled
        )
    wrong_domain = PhysicalDomainArray(
        np.ones((3, 3, 3), dtype=np.float32),
        PhysicalDomain.SCAN_LINEAR,
        PhysicalUnit.RELATIVE_SCAN_SIGNAL,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    with pytest.raises(ValueError, match="domain mismatch"):
        apply_compiled_backing_return(wrong_domain, compiled)
    with pytest.raises(ValueError, match="scale"):
        apply_compiled_backing_return(
            _input(np.ones((3, 3, 3), dtype=np.float32), 10.0), compiled
        )


def test_compiled_profile_rejects_invalid_hash_and_energy() -> None:
    kernel = CompiledBackingReturnKernel(
        "test",
        np.asarray([0.25, 0.5, 0.25], dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="sha256"):
        CompiledBackingReturnProfile("bad", 8.0, (kernel,))
    second = CompiledBackingReturnKernel(
        "second",
        np.asarray([0.25, 0.5, 0.25], dtype=np.float32),
        np.eye(3, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="aggregate"):
        CompiledBackingReturnProfile("1" * 64, 8.0, (kernel, second))


def test_frozen_standard_report_passes_and_repeats() -> None:
    first = evaluate_compiled_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3E, P3E_SCHEMA)
    )
    second = evaluate_compiled_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3E, P3E_SCHEMA)
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
    assert first["metrics"]["zero_return_identity_exact"] is True
