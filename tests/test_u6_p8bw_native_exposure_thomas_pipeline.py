from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_thomas_density_conformance import (
    evaluate_conformance as evaluate_thomas_conformance,
)
from src.eval.native_thomas_field_conformance import profile_from_contract
from src.film_physics.native_exposure_thomas_pipeline import (
    render_native_exposure_thomas_rgb,
)
from src.film_physics.native_granularity_amplitude import (
    apply_native_granularity_amplitude,
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_density import (
    load_native_thomas_density_library,
    render_native_thomas_rgb_transmittance,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _profiles() -> tuple[object, object, object]:
    contract = _json(ROOT / "configs/u6_p8bs_native_thomas_field_v1.json")
    template = profile_from_contract(contract)
    seeds = _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")[
        "candidate"
    ]["layer_realization_seeds"]  # type: ignore[index]
    rows = [
        type(template)(
            particle_sigma_pixels=template.particle_sigma_pixels,
            cluster_sigma_pixels=template.cluster_sigma_pixels,
            mean_offspring=template.mean_offspring,
            truncate=template.truncate,
            component_seeds=template.component_seeds,
            realization_seed=int(seed),
        )
        for seed in seeds
    ]
    return rows[0], rows[1], rows[2]


def test_contract_binds_retained_primitives() -> None:
    contract = _json(CONFIG)
    for binding in contract["parents"].values():  # type: ignore[union-attr]
        payload = _json(ROOT / binding["path"])
        assert payload["decision"] == binding["required_decision"]
    assert contract["candidate"]["model_or_profile_change_allowed"] is False  # type: ignore[index]


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_serial_pipeline_is_exact_separated_execution(tmp_path: Path) -> None:
    amplitude_contract = _json(
        ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
    )
    thomas_contract = _json(
        ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
    )
    amplitude_build = evaluate_amplitude_conformance(
        ROOT, amplitude_contract, output_dir=tmp_path / "amplitude", clang=CLANG
    )
    thomas_build = evaluate_thomas_conformance(
        ROOT, thomas_contract, output_dir=tmp_path / "thomas", clang=CLANG
    )
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(row["dll_path"]))
        for name, row in amplitude_build["toolchains"].items()
    }
    thomas_library = load_native_thomas_density_library(
        Path(thomas_build["toolchains"]["msvc"]["dll_path"])
    )
    p4bw = _json(
        ROOT
        / amplitude_contract["parents"]["p4bw_bundle"]["path"]  # type: ignore[index]
    )
    prior = _json(
        ROOT / amplitude_contract["parents"]["p2q_bundle"]["path"]  # type: ignore[index]
    )
    amplitude_profile = compile_native_granularity_amplitude_profile(p4bw, prior)
    exposure = np.empty((3, 193, 257), dtype=np.float32)
    for channel in range(3):
        count = int(amplitude_profile.knot_count[channel])
        lower = float(amplitude_profile.log_exposure_knots[channel][0])
        upper = float(amplitude_profile.log_exposure_knots[channel][count - 1])
        lower32 = np.float32(lower)
        if float(lower32) < lower:
            lower32 = np.nextafter(lower32, np.float32(np.inf))
        upper32 = np.float32(upper)
        if float(upper32) > upper:
            upper32 = np.nextafter(upper32, np.float32(-np.inf))
        exposure[channel] = np.linspace(
            lower32, upper32, exposure[channel].size,
            dtype=np.float32,
        ).reshape(exposure[channel].shape)
    density, sigma = apply_native_granularity_amplitude(
        amplitude_libraries["msvc"], amplitude_profile, exposure
    )
    expected, expected_means, separated_bytes = render_native_thomas_rgb_transmittance(
        thomas_library, _profiles(), density, sigma
    )
    for amplitude_library in amplitude_libraries.values():
        actual, actual_means, reused_bytes = render_native_exposure_thomas_rgb(
            amplitude_library, thomas_library, amplitude_profile, _profiles(), exposure
        )
        assert np.array_equal(actual, expected)
        assert actual_means == expected_means
        assert reused_bytes < separated_bytes + density.nbytes + sigma.nbytes
        assert np.all(actual > 0.0)
        assert np.all(actual <= 1.0)
