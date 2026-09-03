from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_u7_21a_strict_sdr_heic_ingress as audit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_21a_strict_sdr_heic_ingress_v1.json"


def test_source_and_decode_only_wheel_identities_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(
        row["bytes"] == row["expected_bytes"]
        and row["sha256"] == row["expected_sha256"]
        for row in audit._source_rows(config)
    )
    wheel = audit._wheel_record(config)
    assert wheel["bytes"] == wheel["expected_bytes"]
    assert wheel["sha256"] == wheel["expected_sha256"]
    assert wheel["declares_libheif_lgplv3"] is True
    assert wheel["declares_libde265_lgplv3"] is True
    assert wheel["contains_x265"] is False


def test_canonical_preexisting_product_oracle_is_unchanged(tmp_path: Path) -> None:
    row = audit._canonical_product_probe(tmp_path)
    assert row["returncode"] == 0
    assert row["output_sha256"] == row["expected_output_sha256"]
