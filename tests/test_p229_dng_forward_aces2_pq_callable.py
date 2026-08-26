from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocess import dng_forward_aces2_pq as module
from src.preprocess.dng_forward_raster import DngForwardRasterError
from src.preprocess.types import SourceProfile, WorkingImage


def _working() -> WorkingImage:
    return WorkingImage(
        pixels=np.full((2, 3, 3), 0.18, dtype=np.float32),
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("bound.dng"),
    )


def test_callable_forwards_exact_source_identity_and_fixed_target(monkeypatch) -> None:
    calls: dict[str, object] = {}
    working = _working()
    rendered = np.full((2, 3, 3), 0.5, dtype=np.float32)

    def load(path, *, expected_source_bytes, expected_source_sha256):
        calls["load"] = (path, expected_source_bytes, expected_source_sha256)
        return working

    def apply(value, target):
        calls["apply"] = (value, target)
        return rendered

    monkeypatch.setattr(module, "load_dng_forward_working_image", load)
    monkeypatch.setattr(module, "apply_working_image_aces2_output", apply)
    output = module.render_dng_forward_to_aces2_p3_pq(
        "bound.dng",
        expected_source_bytes=123,
        expected_source_sha256="a" * 64,
    )

    assert calls["load"] == ("bound.dng", 123, "a" * 64)
    assert calls["apply"] == (working, module.TARGET)
    assert output.dtype == np.float32
    assert output.flags.c_contiguous and output.flags.owndata
    assert np.array_equal(output, rendered)
    output[0, 0, 0] = 0.25
    assert rendered[0, 0, 0] == np.float32(0.5)


@pytest.mark.parametrize(
    ("source_bytes", "source_sha"),
    [(0, "a" * 64), (True, "a" * 64), (1, "A" * 64), (1, "0" * 63)],
)
def test_callable_rejects_invalid_identity_arguments(source_bytes, source_sha) -> None:
    with pytest.raises(module.DngForwardAces2PqError):
        module.render_dng_forward_to_aces2_p3_pq(
            "bound.dng",
            expected_source_bytes=source_bytes,
            expected_source_sha256=source_sha,
        )


def test_callable_wraps_parent_decode_failure(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise DngForwardRasterError("source SHA-256 mismatch")

    monkeypatch.setattr(module, "load_dng_forward_working_image", fail)
    with pytest.raises(module.DngForwardAces2PqError, match="source SHA-256 mismatch"):
        module.render_dng_forward_to_aces2_p3_pq(
            "bound.dng",
            expected_source_bytes=1,
            expected_source_sha256="a" * 64,
        )


@pytest.mark.parametrize("bad_value", [-0.001, 1.001, np.nan])
def test_callable_rejects_invalid_official_output(monkeypatch, bad_value) -> None:
    monkeypatch.setattr(module, "load_dng_forward_working_image", lambda *_a, **_k: _working())
    rendered = np.full((1, 1, 3), 0.5, dtype=np.float32)
    rendered[0, 0, 0] = bad_value
    monkeypatch.setattr(
        module,
        "apply_working_image_aces2_output",
        lambda *_args: rendered,
    )
    with pytest.raises(module.DngForwardAces2PqError):
        module.render_dng_forward_to_aces2_p3_pq(
            "bound.dng",
            expected_source_bytes=1,
            expected_source_sha256="a" * 64,
        )
