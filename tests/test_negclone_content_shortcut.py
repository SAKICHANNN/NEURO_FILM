from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from src.roll2film.negclone_content_shortcut import (
    build_synthetic_probes,
    evaluate_shortcut_gates,
    inspect_source_contract,
    run_exact_module_probes,
    validate_sdist,
    validate_source_files,
)


class _FakeFingerprintModule:
    @staticmethod
    def fingerprint_stock(image_paths, stock, sample_size=20, verbose=False):
        del image_paths, stock, sample_size, verbose

    @staticmethod
    def _analyze_color_bias(arr):
        mean = np.mean(arr, axis=(0, 1))
        bias = tuple(float(value - np.mean(mean)) for value in mean)
        return SimpleNamespace(midtones=bias)

    @staticmethod
    def _analyze_tonal_rolloff(arr):
        is_dark = float(np.mean(arr)) < 0.5
        inputs = np.linspace(0.0, 255.0, 17)
        outputs = (
            np.linspace(0.0, 255.0, 17)
            if is_dark
            else np.linspace(255.0, 0.0, 17)
        )
        return SimpleNamespace(
            shadow_lift=0.48 if is_dark else 0.8,
            highlight_compression=0.5 if is_dark else 0.48,
            midtone_contrast=0.998,
            curve_points=list(zip(inputs, outputs, strict=True)),
        )

    @staticmethod
    def _analyze_grain(arr):
        textured = float(np.std(arr)) > 0.1
        return SimpleNamespace(
            mean_intensity=0.4 if textured else 0.0,
            size_estimate=2.0 if textured else 1.0,
            clumping_factor=1.0 if textured else 0.0,
            peak_frequency=0.5 if textured else 0.0,
            spectral_slope=5.811 if textured else 0.0,
            spectral_centroid=0.5 if textured else 0.0,
        )


def test_synthetic_probes_isolate_colour_exposure_and_texture() -> None:
    probes = build_synthetic_probes()
    assert set(probes) == {
        "red_flat",
        "blue_flat",
        "dark_gradient",
        "bright_gradient",
        "flat_grey",
        "checker_texture",
    }
    assert all(value.shape == (256, 256, 3) for value in probes.values())
    assert np.std(probes["flat_grey"]) == 0.0
    assert np.std(probes["checker_texture"]) > 0.1


def test_probe_adapter_detects_all_three_shortcuts() -> None:
    observed = run_exact_module_probes(
        _FakeFingerprintModule, random_seed=27070
    )
    gates = evaluate_shortcut_gates(
        observed,
        {
            "minimum_colour_bias_l2_shortcut": 0.5,
            "minimum_tone_curve_output_difference": 100.0,
            "maximum_flat_grain_intensity": 1e-12,
            "minimum_checker_grain_intensity": 0.2,
            "minimum_checker_clumping": 0.9,
        },
    )
    assert all(gates.values())
    assert observed["colour"]["midtone_bias_l2_distance"] > 0.8
    assert observed["tone"]["curve_output_maximum_absolute_difference"] == 255


def test_sdist_validation_is_hash_bound_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("bounded", encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="package/payload.txt")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert validate_sdist(
        archive, expected_sha256=digest, expected_member_count=1
    ) == ("package/payload.txt",)
    with pytest.raises(ValueError):
        validate_sdist(
            archive, expected_sha256="0" * 64, expected_member_count=1
        )
    traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as handle:
        handle.add(payload, arcname="../payload.txt")
    traversal_hash = hashlib.sha256(traversal.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_sdist(
            traversal,
            expected_sha256=traversal_hash,
            expected_member_count=1,
        )


def test_source_hash_validation_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert validate_source_files(tmp_path, {"source.py": digest}) == {
        "source.py": digest
    }
    with pytest.raises(ValueError):
        validate_source_files(tmp_path, {"source.py": "0" * 64})
    with pytest.raises(ValueError):
        validate_source_files(tmp_path, {"../outside.py": digest})


def test_static_contract_reads_only_declared_source_files(
    tmp_path: Path,
) -> None:
    package = tmp_path / "negclone"
    presets = package / "presets"
    presets.mkdir(parents=True)
    (package / "fingerprint.py").write_text(
        "\n".join(
            (
                "random.sample(image_paths, sample_size)",
                "random.randint(",
                "np.percentile(luminance, SHADOW_PERCENTILE)",
                "np.percentile(luminance, HIGHLIGHT_PERCENTILE)",
                "np.histogram(flat, bins=256",
                "PchipInterpolator(x_unique, y_unique)",
                "local_std = float(np.std(patch))",
                "np.fft.fft2(windowed)",
                "_aggregate_grain(grain_profiles)",
                "_aggregate_color(color_biases)",
                "_aggregate_tone(tonal_rolloffs)",
            )
        ),
        encoding="utf-8",
    )
    (package / "scanner_profiles.py").write_text(
        "Values are approximate compensation offsets", encoding="utf-8"
    )
    (presets / "lightroom.py").write_text(
        "ToneCurvePV2012 ColorGradeShadowHue GrainAmount", encoding="utf-8"
    )
    observed = inspect_source_contract(tmp_path)
    assert all(
        value
        for key, value in observed.items()
        if key != "preset_has_3d_lut"
    )
    assert observed["preset_has_3d_lut"] is False
