from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/P306_POLYHAVEN_RADIANCE_RGBE_WRITER_RESULT.json"


def test_p306_evidence_binds_exact_source_writer_only() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["decision"] == "PASS_PRIVATE_POLYHAVEN_RADIANCE_RGBE_WRITER"
    assert evidence["formal_reports"]["byte_exact"]
    assert evidence["formal_reports"]["sha256_each"] == (
        "620f13a2d0b1a549729c077173164b625d2fafe7a7650ba58b6f8c618238346c"
    )
    result = evidence["result"]
    assert result["full_container_exact_to_source"]
    assert result["python_exact_to_compiled_official_c"]
    assert result["source_codes_exact"] and result["decoded_float_exact"]
    assert result["formal_media_residue_zero"]
    claim = evidence["claim_ceiling"].casefold()
    assert "workingimage" in claim and "arbitrary hdr" in claim
    assert "candidate 3" in claim
