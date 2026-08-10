from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_thomas_density_conformance import (
    evaluate_conformance,
    fixture_arrays,
)
from src.eval.native_thomas_field_conformance import profile_from_contract
from src.film_physics.native_thomas_density import (
    load_native_thomas_density_library,
    render_native_thomas_rgb_transmittance,
    render_native_thomas_transmittance,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bu_native_rgb_thomas_serial_v1.json"
P8BT = ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
P8BS = ROOT / "configs/u6_p8bs_native_thomas_field_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_preserves_independent_layer_seeds() -> None:
    contract = _json(CONFIG)
    assert contract["candidate"]["layer_realization_seeds"] == [  # type: ignore[index]
        2608024301,
        11400714821016565496,
        15111065704576690158,
    ]
    assert contract["candidate"]["parallel_full_frame_workspaces_allowed"] is False  # type: ignore[index]


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_rgb_serial_executor_matches_three_independent_calls(tmp_path: Path) -> None:
    build = evaluate_conformance(
        ROOT, _json(P8BT), output_dir=tmp_path, clang=CLANG
    )
    library = load_native_thomas_density_library(
        Path(build["toolchains"]["msvc"]["dll_path"])
    )
    base, sigma = fixture_arrays((31, 47))
    base_chw = np.ascontiguousarray(
        np.stack((base, base + np.float32(0.1), base + np.float32(0.2)))
    )
    sigma_chw = np.ascontiguousarray(np.stack((sigma, sigma * 0.9, sigma * 1.1)))
    base_before = base_chw.tobytes()
    sigma_before = sigma_chw.tobytes()
    template = profile_from_contract(_json(P8BS))
    seeds = _json(CONFIG)["candidate"]["layer_realization_seeds"]  # type: ignore[index]
    profiles = tuple(
        type(template)(
            particle_sigma_pixels=template.particle_sigma_pixels,
            cluster_sigma_pixels=template.cluster_sigma_pixels,
            mean_offspring=template.mean_offspring,
            truncate=template.truncate,
            component_seeds=template.component_seeds,
            realization_seed=int(seed),
        )
        for seed in seeds
    )
    rgb, raw_means, reused_bytes = render_native_thomas_rgb_transmittance(
        library, profiles, base_chw, sigma_chw
    )
    independent = np.stack(
        [
            render_native_thomas_transmittance(
                library, profiles[channel], base_chw[channel], sigma_chw[channel]
            )[0]
            for channel in range(3)
        ]
    )
    repeat, repeat_means, _ = render_native_thomas_rgb_transmittance(
        library, profiles, base_chw, sigma_chw
    )
    assert np.array_equal(rgb, independent)
    assert np.array_equal(rgb, repeat)
    assert raw_means == repeat_means
    assert len({layer.tobytes() for layer in rgb}) == 3
    assert reused_bytes == 5 * 31 * 47 * 4
    assert base_chw.tobytes() == base_before
    assert sigma_chw.tobytes() == sigma_before
    assert float(rgb.min()) > 0.0
    assert float(rgb.max()) <= 1.0
