from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _exposure_fixture,
    _parent_payloads,
)
from src.eval.native_thomas_rgb16_png_conformance import build_msvc
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.native_thomas_input import RelativeLayerLogExposure
from src.film_physics.native_thomas_package import (
    resolve_native_thomas_package,
    validate_native_thomas_package,
)
from src.film_physics.native_thomas_runtime import NativeThomasExportRuntime

ROOT = Path(__file__).resolve().parents[1]


def _package() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/native_thomas_export_package_win_x64_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_native_thomas_package_resolves_and_renders(tmp_path: Path) -> None:
    package = _package()
    validate_native_thomas_package(package)
    build = build_msvc(ROOT, tmp_path / "build")
    resolved = resolve_native_thomas_package(
        package,
        profile_path=ROOT
        / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json",
        library_path=Path(build["dll_path"]),
    )
    runtime = NativeThomasExportRuntime(package=package, resolved=resolved)
    prior = ManufacturerCharacteristicPrior.from_dict(_parent_payloads()[1]["prior"])
    exposure = _exposure_fixture(prior, (129, 67))
    input_sha256 = hashlib.sha256(exposure.tobytes()).hexdigest()
    destination = (tmp_path / "runtime.png").resolve()
    typed = RelativeLayerLogExposure.from_chw(exposure)
    receipt = runtime.publish(typed, destination=destination)
    assert destination.is_file()
    assert receipt["input_sha256"] == input_sha256
    assert receipt["input_contract"] == typed.descriptor()
    assert receipt["profile_sha256"] == package["profile_asset"]["profile_sha256"]
    assert receipt["output"]["sha256"] == hashlib.sha256(
        destination.read_bytes()
    ).hexdigest()
    assert receipt["production_default_changed"] is False


def test_native_thomas_runtime_rejects_untyped_or_mislabeled_input(
    tmp_path: Path,
) -> None:
    package = _package()
    build = build_msvc(ROOT, tmp_path / "build")
    resolved = resolve_native_thomas_package(
        package,
        profile_path=ROOT
        / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json",
        library_path=Path(build["dll_path"]),
    )
    runtime = NativeThomasExportRuntime(package=package, resolved=resolved)
    values = np.zeros((3, 2, 2), dtype=np.float32)
    try:
        runtime.publish(values, destination=tmp_path / "untyped.png")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "RelativeLayerLogExposure" in str(exc)
    else:
        raise AssertionError("untyped exposure was accepted")
    assert not (tmp_path / "untyped.png").exists()

    try:
        RelativeLayerLogExposure(
            np.zeros((3, 2, 2), dtype=np.float32),
            channels=("blue", "green", "red"),
        )
    except ValueError as exc:
        assert "red/green/blue" in str(exc)
    else:
        raise AssertionError("wrong layer order was accepted")

    adopted_source = np.zeros((3, 2, 2), dtype=np.float32)
    adopted = RelativeLayerLogExposure.adopt_chw(adopted_source)
    assert adopted.values_chw is adopted_source
    assert not adopted_source.flags.writeable


def test_native_thomas_package_rejects_wrong_library(tmp_path: Path) -> None:
    fake = tmp_path / "fake.dll"
    fake.write_bytes(b"not-the-export-core")
    try:
        resolve_native_thomas_package(
            _package(),
            profile_path=ROOT
            / "configs/render_profiles/generic_physical_thomas_p8cr_v1.json",
            library_path=fake,
        )
    except ValueError as exc:
        assert "export core hash mismatch" in str(exc)
    else:
        raise AssertionError("wrong export core accepted")
