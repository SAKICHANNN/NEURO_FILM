from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.audit_srgb_eotf_streaming_v1 import (
    DEFAULT_LLVM,
    PROTOCOL,
    StreamingAuditError,
    _canonical_bytes,
    audit_streaming,
)
from scripts.build_reference_chain_msvc import VSWHERE


def test_streaming_audit_is_exact_and_bounded() -> None:
    if not DEFAULT_LLVM.is_dir() or not VSWHERE.is_file():
        pytest.skip("both Windows C toolchains are required")
    report = audit_streaming(
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
    for bit_depth in ("8", "16"):
        msvc = report["evidence"]["runs"]["msvc"][bit_depth]
        llvm = report["evidence"]["runs"]["llvm-mingw"][bit_depth]
        assert msvc == llvm
        assert msvc["chunks"] == 48
        assert msvc["max_live_array_payload_bytes"] <= 10 * 4_099


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
def test_streaming_audit_rejects_invalid_bounds(
    pixels: int,
    chunk_samples: int,
) -> None:
    with pytest.raises(StreamingAuditError):
        audit_streaming(
            pixels=pixels,
            chunk_samples=chunk_samples,
        )


def test_streaming_audit_unloads_dlls_after_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not DEFAULT_LLVM.is_dir() or not VSWHERE.is_file():
        pytest.skip("both Windows C toolchains are required")

    def fail_after_load(*args, **kwargs):
        raise StreamingAuditError("injected runtime failure")

    monkeypatch.setattr(
        "scripts.audit_srgb_eotf_streaming_v1._run_once",
        fail_after_load,
    )
    with pytest.raises(StreamingAuditError, match="injected"):
        audit_streaming(
            pixels=1,
            chunk_samples=1,
        )
