from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
    _profiles,
)
from src.eval.native_granularity_amplitude_conformance import (
    evaluate_conformance as evaluate_amplitude_conformance,
)
from src.eval.native_thomas_density_conformance import (
    evaluate_conformance as evaluate_density_conformance,
)
from src.eval.native_thomas_rows_conformance import (
    build_and_load,
    failure_output_atomic,
    validate_contract,
)
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_exposure_thomas_pipeline import (
    render_native_exposure_thomas_rgb,
)
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
    load_native_granularity_amplitude_library,
)
from src.film_physics.native_thomas_density import load_native_thomas_density_library
from src.film_physics.native_thomas_rows import (
    render_native_exposure_thomas_rgb_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p8bx_native_thomas_row_stream_v1.json"
CLANG = ROOT / "outputs/tmp/tools/llvm-mingw-20260616-ucrt-x86_64/bin/clang.exe"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_binds_exact_p8bw_parent() -> None:
    validate_contract(ROOT, _json(CONFIG))


@pytest.mark.skipif(not CLANG.is_file(), reason="pinned LLVM-MinGW unavailable")
def test_row_partitions_are_bit_exact_and_failure_atomic(tmp_path: Path) -> None:
    contract = _json(CONFIG)
    amplitude_contract = _json(
        ROOT / "configs/u6_p8bv_native_granularity_amplitude_v1.json"
    )
    density_contract = _json(
        ROOT / "configs/u6_p8bt_native_density_thomas_transmittance_v1.json"
    )
    amplitude_build = evaluate_amplitude_conformance(
        ROOT, amplitude_contract, output_dir=tmp_path / "amplitude", clang=CLANG
    )
    density_build = evaluate_density_conformance(
        ROOT, density_contract, output_dir=tmp_path / "density", clang=CLANG
    )
    _, row_libraries = build_and_load(ROOT, tmp_path / "rows", CLANG)
    amplitude_libraries = {
        name: load_native_granularity_amplitude_library(Path(row["dll_path"]))
        for name, row in amplitude_build["toolchains"].items()
    }
    density_library = load_native_thomas_density_library(
        Path(density_build["toolchains"]["msvc"]["dll_path"])
    )
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude_profile = compile_native_granularity_amplitude_profile(
        p4bw, prior_payload
    )
    shape = tuple(int(value) for value in contract["conformance"]["shape_chw"])
    exposure = _exposure_fixture(prior, (shape[1], shape[2]))
    input_copy = exposure.copy()
    profiles = _profiles(
        _json(ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
    )
    expected, expected_means, _ = render_native_exposure_thomas_rgb(
        amplitude_libraries["msvc"],
        density_library,
        amplitude_profile,
        profiles,
        exposure,
    )
    for name, rows_library in row_libraries.items():
        assert failure_output_atomic(rows_library, ROOT), name
        for partition in contract["candidate"]["row_partitions"]:
            actual, actual_means, _ = render_native_exposure_thomas_rgb_rows(
                amplitude_libraries[name],
                rows_library,
                amplitude_profile,
                profiles,
                exposure,
                row_partition=int(partition),
            )
            assert np.array_equal(actual, expected), (name, partition)
            assert actual_means == expected_means
            assert np.array_equal(exposure, input_copy)
            assert np.all(actual > 0.0)
            assert np.all(actual <= 1.0)
