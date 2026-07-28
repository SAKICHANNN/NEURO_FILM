from __future__ import annotations

import hashlib

import pytest

from scripts.audit_srgb_oetf_quantize_streaming_v1 import (
    DEFAULT_LLVM,
    PROTOCOL,
    StreamingQuantizeAuditError,
    _canonical_bytes,
    audit_streaming_quantize,
)
from scripts.build_reference_chain_msvc import VSWHERE


def test_streaming_quantize_is_exact_and_bounded() -> None:
    if not DEFAULT_LLVM.is_dir() or not VSWHERE.is_file():
        pytest.skip("both Windows C toolchains are required")
    report = audit_streaming_quantize(
        pixels=65_537,
        chunk_samples=4_099,
    )
    assert report["protocol"] == PROTOCOL
    assert report["stable_evidence_id"] == (
        "sha256:"
        + hashlib.sha256(
            _canonical_bytes(report["evidence"])
        ).hexdigest()
    )
    assert report["evidence"]["cross_compiler_exact"] is True
    assert report["evidence"]["two_replays_exact"] is True
    assert report["evidence"]["sample_count"] == 196_611
    for bit_depth, bytes_per_sample in (("8", 6), ("16", 8)):
        msvc = report["evidence"]["runs"]["msvc"][bit_depth]
        llvm = report["evidence"]["runs"]["llvm-mingw"][bit_depth]
        assert msvc == llvm
        assert msvc["chunks"] == 48
        assert (
            msvc["max_live_array_payload_bytes"]
            <= bytes_per_sample * 4_099
        )


@pytest.mark.parametrize(
    ("pixels", "chunk_samples"),
    [
        (0, 1),
        (-1, 1),
        (1, 0),
        (1, -1),
        ((2**63 - 1) // 3 + 1, 1),
    ],
)
def test_streaming_quantize_rejects_invalid_bounds(
    pixels: int,
    chunk_samples: int,
) -> None:
    with pytest.raises(StreamingQuantizeAuditError):
        audit_streaming_quantize(
            pixels=pixels,
            chunk_samples=chunk_samples,
        )


def test_streaming_quantize_unloads_dlls_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DEFAULT_LLVM.is_dir() or not VSWHERE.is_file():
        pytest.skip("both Windows C toolchains are required")

    def fail_after_load(*args, **kwargs):
        raise StreamingQuantizeAuditError("injected runtime failure")

    monkeypatch.setattr(
        "scripts.audit_srgb_oetf_quantize_streaming_v1._run_once",
        fail_after_load,
    )
    with pytest.raises(StreamingQuantizeAuditError, match="injected"):
        audit_streaming_quantize(
            pixels=1,
            chunk_samples=1,
        )
