from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_p283_libavif_gainmap_p87_bridge import (
    parse_chosen_cicp,
    roundtrip_rgb16,
)
from src.color_match.contracts import ReferenceMatchContractError
from src.color_match.core_contracts import MATCH_PROFILE_ABSOLUTE_REC2020
from src.color_match.libavif_gainmap_ingress import (
    LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS,
    LIBAVIF_GAINMAP_RENDER_BRIDGE_ID,
    prepare_libavif_gainmap_match_view_v1,
    validate_prepared_libavif_gainmap_match_view_v1,
)

ROOT = Path(__file__).resolve().parents[1]


def _prepare(samples: np.ndarray | None = None):
    source = b"p283-source"
    if samples is None:
        samples = np.asarray(
            [
                [[0, 0, 0], [32768, 32768, 32768]],
                [[65535, 65535, 65535], [10000, 20000, 30000]],
            ],
            dtype=np.uint16,
        )
    return prepare_libavif_gainmap_match_view_v1(
        source_asset=source,
        expected_source_sha256=hashlib.sha256(source).hexdigest(),
        encoded_rgb16=samples,
        decoder_version="1.4.2",
        source_color_primaries=1,
        source_transfer_characteristics=16,
        source_full_range=True,
        higher_rendition_role="alternate",
    )


def test_p283_bridge_returns_owned_absolute_match_view() -> None:
    samples = np.asarray([[[0, 0, 0], [65535, 65535, 65535]]], dtype=np.uint16)
    original = samples.copy()
    result = _prepare(samples)
    validate_prepared_libavif_gainmap_match_view_v1(result)
    assert np.array_equal(samples, original)
    assert result.pixels.dtype == np.float32
    assert result.pixels.flags.c_contiguous
    assert result.pixels.flags.owndata
    assert not result.pixels.flags.writeable
    assert result.descriptor.profile_id == MATCH_PROFILE_ABSOLUTE_REC2020
    assert (
        result.descriptor.reference_white_nits == LIBAVIF_GAINMAP_REFERENCE_WHITE_NITS
    )
    assert result.descriptor.render_bridge_id == LIBAVIF_GAINMAP_RENDER_BRIDGE_ID
    assert np.array_equal(result.pixels[0, 0], np.zeros(3, dtype=np.float32))
    assert np.max(np.abs(result.pixels[0, 1] - np.float32(10000.0))) < 0.002


@pytest.mark.parametrize(
    "override,match",
    [
        ({"expected_source_sha256": "0" * 64}, "identity"),
        ({"decoder_version": "1.4.1"}, "decoder"),
        ({"source_color_primaries": 9}, "primaries"),
        ({"source_transfer_characteristics": 13}, "transfer"),
        ({"source_full_range": False}, "full range"),
        ({"higher_rendition_role": "lower"}, "role"),
    ],
)
def test_p283_bridge_rejects_wrong_source_semantics(
    override: dict[str, object], match: str
) -> None:
    source = b"p283-source"
    arguments: dict[str, object] = {
        "source_asset": source,
        "expected_source_sha256": hashlib.sha256(source).hexdigest(),
        "encoded_rgb16": np.zeros((1, 1, 3), dtype=np.uint16),
        "decoder_version": "1.4.2",
        "source_color_primaries": 1,
        "source_transfer_characteristics": 16,
        "source_full_range": True,
        "higher_rendition_role": "base",
    }
    arguments.update(override)
    with pytest.raises(ReferenceMatchContractError, match=match):
        prepare_libavif_gainmap_match_view_v1(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "samples,match",
    [
        (np.zeros((1, 1, 3), dtype=np.float32), "uint16"),
        (np.zeros((1, 3), dtype=np.uint16), "HxWx3"),
        (np.zeros((1, 1, 4), dtype=np.uint16), "HxWx3"),
        (np.zeros((0, 1, 3), dtype=np.uint16), "positive"),
    ],
)
def test_p283_bridge_rejects_invalid_samples(samples: np.ndarray, match: str) -> None:
    with pytest.raises(ReferenceMatchContractError, match=match):
        _prepare(samples)


def test_p283_info_parser_selects_exact_base_and_alternate_cicp() -> None:
    output = """\
 * Range          : Full
 * Color Primaries: 1
 * Transfer Char. : 16
 * Matrix Coeffs. : 6
 * Alternate image:
    * Color Primaries: 1
    * Transfer Char. : 13
    * Matrix Coeffs. : 6
"""
    assert parse_chosen_cicp(output, "base") == (1, 16, 6)
    assert parse_chosen_cicp(output, "alternate") == (1, 13, 6)


def test_p283_roundtrip_helper_is_exact_on_neutral_codes() -> None:
    samples = np.asarray(
        [[[0, 0, 0], [32768, 32768, 32768], [65535, 65535, 65535]]],
        dtype=np.uint16,
    )
    prepared = _prepare(samples)
    replay = roundtrip_rgb16(prepared.pixels)
    assert np.max(np.abs(replay.astype(np.int32) - samples.astype(np.int32))) <= 1


def test_p283_evidence_preserves_strict_roundtrip_failure() -> None:
    evidence = json.loads(
        (ROOT / "docs/evidence/P283_LIBAVIF_GAINMAP_P87_BRIDGE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "FAIL_CLOSED_LIBAVIF_GAINMAP_P87_BRIDGE"
    result = evidence["formal_result"]
    assert result["report_sha256"] == (
        "c59bf3126077d4131f78433cd111e51446be81e79f27dcfd1ffaa957014dcd56"
    )
    assert result["gates"]["all_ranges_valid"] is True
    assert result["gates"]["all_roundtrip_domains_valid"] is False
    assert result["gates"]["all_roundtrips_within_bound"] is False
    assert [row["inverse_out_of_domain_components"] for row in result["records"]] == [
        133,
        4,
        33,
    ]
