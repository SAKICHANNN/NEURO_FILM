from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.native_thomas_input import RelativeLayerLogExposure
from src.film_physics.native_thomas_spatial_chain import (
    apply_native_thomas_spatial_chain_fft,
    compile_native_thomas_spatial_chain,
)
from src.film_physics.native_thomas_streaming_input import (
    map_streamed_log_exposure,
    stream_fft_spatial_log_exposure_chw,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_streamed_spatial_log_input_is_complete_hash_bound_and_mapped(
    tmp_path: Path,
) -> None:
    chain = compile_native_thomas_spatial_chain(
        _json("configs/u6_p1_reference_scatter_simulator_v1.json"),
        _json("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    rng = np.random.default_rng(20260813)
    exposure = PhysicalDomainArray(
        rng.uniform(1.5, 12.0, size=(521, 547, 3)).astype(np.float32),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )
    path = tmp_path / "input.f32"
    receipt = stream_fft_spatial_log_exposure_chw(
        exposure, chain, destination=path, tile_rows=257
    )
    mapped = map_streamed_log_exposure(receipt)
    assert isinstance(mapped.values_chw, np.memmap)
    assert mapped.values_chw.flags.writeable is False
    assert receipt["bytes"] == 3 * 521 * 547 * 4
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]

    full = apply_native_thomas_spatial_chain_fft(exposure, chain)
    expected = RelativeLayerLogExposure.from_layer_exposure(full).values_chw
    error = np.abs(expected.astype(np.float64) - mapped.values_chw.astype(np.float64))
    assert float(np.max(error)) <= 5e-7


def test_streamed_spatial_log_input_rejects_tamper(tmp_path: Path) -> None:
    values = np.ones((3, 17, 19), dtype="<f4")
    path = tmp_path / "mapped.f32"
    path.write_bytes(values.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mapped = RelativeLayerLogExposure.map_chw(
        path, height=17, width=19, expected_sha256=digest
    )
    assert isinstance(mapped.values_chw, np.memmap)
    del mapped
    gc.collect()
    path.write_bytes(path.read_bytes()[:-4] + b"\0\0\0\0")
    try:
        RelativeLayerLogExposure.map_chw(
            path, height=17, width=19, expected_sha256=digest
        )
    except ValueError as exc:
        assert "hash drift" in str(exc)
    else:
        raise AssertionError("tampered mapped native input was accepted")
