from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_streamed_fft_backing_return import (
    P3D_SCHEMA,
    P3J_SCHEMA,
    evaluate_streamed_fft_backing_return,
    load_json,
)
from src.film_physics import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
    backing_return_profile_from_contract,
    compile_backing_return_profile,
    iter_fft_backing_return_row_cores,
)


ROOT = Path(__file__).resolve().parents[1]
P3D = ROOT / "configs" / "u6_p3d_backing_return_reference_v1.json"
P3J = ROOT / "configs" / "u6_p3j_streamed_fft_backing_return_v1.json"


def _input(values: np.ndarray, pitch: float = 8.0) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pitch),
    )


def test_iterator_yields_owned_bounded_cores_in_requested_order() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    values = np.random.default_rng(13).random((31, 37, 3), dtype=np.float32)
    cores = list(
        iter_fft_backing_return_row_cores(
            _input(values), compiled, tile_rows=11, order="reverse"
        )
    )
    assert [(y0, y1) for y0, y1, _ in cores] == [
        (22, 31),
        (11, 22),
        (0, 11),
    ]
    assert all(core.flags.owndata for _, _, core in cores)
    assert all(core.shape[0] <= 11 for _, _, core in cores)


def test_iterator_rejects_invalid_order_dtype_and_rows() -> None:
    reference = backing_return_profile_from_contract(load_json(P3D, P3D_SCHEMA))
    compiled = compile_backing_return_profile(reference)
    source = _input(np.ones((7, 9, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="order"):
        list(
            iter_fft_backing_return_row_cores(
                source, compiled, tile_rows=3, order="sideways"
            )
        )
    with pytest.raises(ValueError, match="positive integer"):
        list(
            iter_fft_backing_return_row_cores(
                source, compiled, tile_rows=0
            )
        )
    with pytest.raises(TypeError, match="float32"):
        list(
            iter_fft_backing_return_row_cores(
                _input(np.ones((7, 9, 3), dtype=np.float64)),
                compiled,
                tile_rows=3,
            )
        )


def test_frozen_streamed_report_passes_and_repeats() -> None:
    first = evaluate_streamed_fft_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3J, P3J_SCHEMA)
    )
    second = evaluate_streamed_fft_backing_return(
        load_json(P3D, P3D_SCHEMA), load_json(P3J, P3J_SCHEMA)
    )
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["decisions"].values())
