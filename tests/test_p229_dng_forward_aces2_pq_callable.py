from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import dng_forward_aces2_pq as module
from src.preprocess.dng_forward_raster import DngForwardRasterError
from src.preprocess.types import SourceProfile, WorkingImage

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P229_DNG_FORWARD_ACES2_P3_PQ_CALLABLE_RESULT.json"


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


def test_callable_rejects_existing_non_dng_before_decode(tmp_path: Path) -> None:
    payload = b"P229 non-DNG predecode control\n"
    source = tmp_path / "not-a-dng.bin"
    source.write_bytes(payload)
    with pytest.raises(module.DngForwardAces2PqError, match="existing \\.dng"):
        module.render_dng_forward_to_aces2_p3_pq(
            source,
            expected_source_bytes=len(payload),
            expected_source_sha256=hashlib.sha256(payload).hexdigest(),
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


def test_formal_evidence_closes_exact_callable_without_mapping() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL_CLOSED_DNG_FORWARD_ACES2_P3_PQ_CALLABLE"
    assert len(evidence["rows"]["returned_exact"]) == 3
    assert len(evidence["rows"]["rejected_before_output"]) == 2
    assert evidence["gates"]["outputs_float32_contiguous_finite_in_unit"] is False
    assert evidence["consumer_mapping"] is False
