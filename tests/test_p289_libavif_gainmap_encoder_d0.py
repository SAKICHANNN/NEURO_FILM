from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts import audit_p289_libavif_gainmap_encoder_d0 as p289


def test_canonical_is_sorted_and_terminated() -> None:
    assert p289._canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'


def test_to_12bit_maps_uint16_endpoints_exactly() -> None:
    source = np.asarray([0, 65535], dtype=np.uint16)
    assert p289._to_12bit(source).tolist() == [0, 4095]


def test_error_reports_exact_code_statistics() -> None:
    reference = np.asarray([[[0, 0, 0], [65535, 65535, 65535]]], dtype=np.uint16)
    candidate = reference.copy()
    candidate[0, 1, 0] = np.uint16(65519)
    result = p289._error(reference, candidate)
    assert result["maximum"] == 1.0
    assert result["median"] == 0.0


def test_metadata_headrooms_parse() -> None:
    text = " * Base headroom: 1.3 (as fraction: 13/10)\n * Alternate headroom: 0"
    assert p289._headroom(text, "Base") == 1.3
    assert p289._headroom(text, "Alternate") == 0.0


def test_info_cicp_parse() -> None:
    text = """
 * Color Primaries: 1
 * Transfer Char. : 16
 * Matrix Coeffs. : 0
 * Alternate image:
    * Color Primaries: 1
    * Transfer Char. : 13
    * Matrix Coeffs. : 0
"""
    assert p289._cicp(text, alternate=False) == (1, 16, 0)
    assert p289._cicp(text, alternate=True) == (1, 13, 0)


def test_create_only_rejects_foreign_destination(tmp_path: Path) -> None:
    destination = tmp_path / "foreign.avif"
    destination.write_bytes(b"foreign\n")
    with pytest.raises(FileExistsError):
        p289._publish_combine_create_only(["unused"], destination)
    assert destination.read_bytes() == b"foreign\n"


def test_contract_forbids_rescue_and_candidate3() -> None:
    contract = (
        p289.ROOT / "docs/planning/P289_LIBAVIF_GAINMAP_ENCODER_D0_CONTRACT.md"
    ).read_text(encoding="utf-8")
    assert "No threshold" in contract
    assert "candidate 3" in contract
