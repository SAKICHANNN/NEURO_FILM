from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.native_thomas_rgb16_png_conformance import build_msvc
from src.film_physics.contracts import (
    PhysicalDomain,
    PhysicalDomainArray,
    PhysicalScale,
    PhysicalUnit,
)
from src.film_physics.native_thomas_package import resolve_native_thomas_package
from src.film_physics.native_thomas_runtime import NativeThomasExportRuntime
from src.film_physics.native_thomas_spatial_chain import (
    apply_native_thomas_spatial_chain,
    compile_native_thomas_spatial_chain,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_native_thomas_spatial_chain_reuses_fixed_physical_order() -> None:
    chain = compile_native_thomas_spatial_chain(
        _json("configs/u6_p1_reference_scatter_simulator_v1.json"),
        _json("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    rng = np.random.default_rng(20260812)
    values = rng.uniform(1.5, 12.0, size=(129, 131, 3)).astype(np.float32)
    exposure = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )
    first = apply_native_thomas_spatial_chain(exposure, chain)
    second = apply_native_thomas_spatial_chain(exposure, chain)
    assert first.values.tobytes() == second.values.tobytes()
    assert first.domain is PhysicalDomain.LAYER_EXPOSURE
    assert first.channels == exposure.channels
    assert np.all(first.values > 0.0)
    assert not np.array_equal(first.values, exposure.values)


def test_native_thomas_spatial_chain_rejects_semantic_or_scale_drift() -> None:
    chain = compile_native_thomas_spatial_chain(
        _json("configs/u6_p1_reference_scatter_simulator_v1.json"),
        _json("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    wrong_channels = PhysicalDomainArray(
        np.ones((17, 19, 3), dtype=np.float32),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(8.0),
    )
    try:
        apply_native_thomas_spatial_chain(wrong_channels, chain)
    except ValueError as exc:
        assert "sensitive-layer" in str(exc)
    else:
        raise AssertionError("non-layer channel labels were accepted")

    wrong_scale = PhysicalDomainArray(
        np.ones((17, 19, 3), dtype=np.float32),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(7.0),
    )
    try:
        apply_native_thomas_spatial_chain(wrong_scale, chain)
    except ValueError as exc:
        assert "pixel scale" in str(exc)
    else:
        raise AssertionError("wrong physical scale was accepted")


def test_native_thomas_runtime_spatial_entry_matches_explicit_composition(
    tmp_path: Path,
) -> None:
    chain = compile_native_thomas_spatial_chain(
        _json("configs/u6_p1_reference_scatter_simulator_v1.json"),
        _json("configs/u6_p3d_backing_return_reference_v1.json"),
    )
    package = _json("configs/native_thomas_export_package_win_x64_v1.json")
    build = build_msvc(ROOT, tmp_path / "build")
    resolved = resolve_native_thomas_package(
        package,
        profile_path=ROOT
        / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json",
        library_path=Path(build["dll_path"]),
    )
    runtime = NativeThomasExportRuntime(package=package, resolved=resolved)
    rows = np.linspace(1.5, 8.0, 129, dtype=np.float32)[:, None, None]
    columns = np.linspace(0.0, 1.0, 67, dtype=np.float32)[None, :, None]
    channels = np.asarray([0.0, 0.2, 0.4], dtype=np.float32)[None, None, :]
    values = np.ascontiguousarray(rows + columns + channels)
    exposure = PhysicalDomainArray(
        values,
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red-sensitive", "green-sensitive", "blue-sensitive"),
        PhysicalScale(8.0),
    )
    explicit = apply_native_thomas_spatial_chain(exposure, chain)
    explicit_receipt = runtime.publish_layer_exposure(
        explicit, destination=tmp_path / "explicit.png"
    )
    composed_receipt = runtime.publish_spatial_layer_exposure(
        exposure, chain, destination=tmp_path / "composed.png"
    )
    assert explicit_receipt["output"]["sha256"] == composed_receipt["output"][
        "sha256"
    ]
    assert (tmp_path / "explicit.png").read_bytes() == (
        tmp_path / "composed.png"
    ).read_bytes()
