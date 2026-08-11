from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_u6_p8bw_native_exposure_thomas_pipeline import (
    _parent_payloads,
    _profiles,
)
from scripts.evaluate_u6_p8ca_native_thomas_gauged_sink import _gauge_payload
from src.eval.native_thomas_export_profile import _configure_parallel, _run
from src.eval.native_thomas_rgb16_png_conformance import build_msvc, load_library
from src.film_physics.atomic_native_output import (
    AtomicNativeOutputError,
    AtomicNativeOutputSink,
    publish_native_thomas_profile_rgb16_png,
    publish_native_thomas_rgb16_png,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.native_gauge_profile import native_gauge_profile_struct
from src.film_physics.native_granularity_amplitude import (
    compile_native_granularity_amplitude_profile,
)
from src.film_physics.native_thomas_export_profile import (
    compile_native_thomas_export_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def test_atomic_native_output_publishes_exact_bytes(tmp_path: Path) -> None:
    destination = (tmp_path / "result.png").resolve()
    chunks = [b"native-", b"png-", b"bytes"]
    with AtomicNativeOutputSink(destination, maximum_bytes=64) as sink:
        assert all(sink.write(chunk) for chunk in chunks)
        result = sink.finish()
    expected = b"".join(chunks)
    assert destination.read_bytes() == expected
    assert result == {
        "path": str(destination),
        "sha256": hashlib.sha256(expected).hexdigest(),
        "bytes": len(expected),
    }
    assert not list(tmp_path.glob("*.stage"))


def test_atomic_native_output_aborts_partial_stream(tmp_path: Path) -> None:
    destination = (tmp_path / "result.png").resolve()
    with AtomicNativeOutputSink(destination, maximum_bytes=4) as sink:
        assert sink.write(b"1234")
        assert not sink.write(b"5")
    assert not destination.exists()
    assert not list(tmp_path.glob("*.stage"))


def test_atomic_native_output_does_not_overwrite_concurrent_claim(
    tmp_path: Path,
) -> None:
    destination = (tmp_path / "result.png").resolve()
    sink = AtomicNativeOutputSink(destination, maximum_bytes=64)
    assert sink.write(b"candidate")
    destination.write_bytes(b"foreign")
    with pytest.raises(AtomicNativeOutputError, match="publication failed"):
        sink.finish()
    assert destination.read_bytes() == b"foreign"
    assert not list(tmp_path.glob("*.stage"))


@pytest.mark.parametrize("maximum_bytes", [0, -1, True])
def test_atomic_native_output_rejects_invalid_preflight(
    tmp_path: Path, maximum_bytes: int
) -> None:
    with pytest.raises(AtomicNativeOutputError, match="preflight"):
        AtomicNativeOutputSink(
            (tmp_path / "result.png").resolve(), maximum_bytes=maximum_bytes
        )


def test_p8cs_native_png_stream_is_exact_and_failure_atomic(tmp_path: Path) -> None:
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    fields = tuple(
        row.as_abi()
        for row in _profiles(
            __import__("json").loads(
                (ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
                .read_text(encoding="utf-8")
            )
        )
    )
    gauge = native_gauge_profile_struct(_gauge_payload())
    rng = np.random.default_rng(2026081201)
    exposure = np.empty((3, 17, 19), dtype=np.float32)
    for channel, curve in enumerate(prior.curves):
        lower, upper = curve.domain
        exposure[channel] = rng.uniform(lower, upper, size=(17, 19)).astype(
            np.float32
        )
    build = build_msvc(ROOT, tmp_path / "build")
    library = load_library(Path(build["dll_path"]))
    _configure_parallel(library)
    expected = _run(
        library, amplitude, fields, gauge, exposure, row_partition=7
    )
    destination = (tmp_path / "native.png").resolve()
    actual = publish_native_thomas_rgb16_png(
        library,
        amplitude,
        fields,
        gauge,
        exposure,
        row_partition=7,
        destination=destination,
        maximum_output_bytes=len(expected["png"]) + 1024,
    )
    assert destination.read_bytes() == expected["png"]
    assert actual["sha256"] == expected["png_sha256"]
    assert actual["raw_field_means"] == expected["raw_field_means"]
    assert actual["workspace_bytes"] == expected["workspace_bytes"]

    failed = (tmp_path / "failed.png").resolve()
    with pytest.raises(AtomicNativeOutputError, match="stream failed"):
        publish_native_thomas_rgb16_png(
            library,
            amplitude,
            fields,
            gauge,
            exposure,
            row_partition=7,
            destination=failed,
            maximum_output_bytes=16,
        )
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.png.*.stage"))


def test_profile_bound_atomic_publication_validates_before_output(
    tmp_path: Path,
) -> None:
    p4bw, prior_payload = _parent_payloads()
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    amplitude = compile_native_granularity_amplitude_profile(p4bw, prior_payload)
    fields = _profiles(
        json.loads(
            (ROOT / "configs/u6_p8bw_native_exposure_to_thomas_pipeline_v1.json")
            .read_text(encoding="utf-8")
        )
    )
    contract = json.loads(
        (ROOT / "configs/u6_p8cs_thomas_atomic_publication_v1.json").read_text(
            encoding="utf-8"
        )
    )
    profile = compile_native_thomas_export_profile(
        amplitude,
        fields,
        _gauge_payload(),
        source_bindings=contract["profile_source_bindings"],
    )
    exposure = np.empty((3, 17, 19), dtype=np.float32)
    rng = np.random.default_rng(2026081202)
    for channel, curve in enumerate(prior.curves):
        exposure[channel] = rng.uniform(*curve.domain, size=(17, 19)).astype(
            np.float32
        )
    build = build_msvc(ROOT, tmp_path / "profile-build")
    library = load_library(Path(build["dll_path"]))
    _configure_parallel(library)
    destination = (tmp_path / "profile.png").resolve()
    result = publish_native_thomas_profile_rgb16_png(
        library,
        profile,
        exposure,
        expected_profile_sha256=profile["profile_sha256"],
        row_partition=7,
        destination=destination,
        maximum_output_bytes=1048576,
    )
    assert destination.is_file()
    assert result["profile_sha256"] == profile["profile_sha256"]

    rejected = dict(profile)
    rejected["profile_sha256"] = "0" * 64
    rejected_path = (tmp_path / "rejected.png").resolve()
    with pytest.raises(AtomicNativeOutputError, match="profile rejected"):
        publish_native_thomas_profile_rgb16_png(
            library,
            rejected,
            exposure,
            expected_profile_sha256=profile["profile_sha256"],
            row_partition=7,
            destination=rejected_path,
            maximum_output_bytes=1048576,
        )
    assert not rejected_path.exists()
