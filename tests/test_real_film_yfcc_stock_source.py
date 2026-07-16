from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from src.real_film.yfcc_stock_source import (
    YfccStockSourceError,
    scan_stock_candidates,
    validate_parquet_index,
)


def _config() -> dict:
    return {
        "dataset_id": "test",
        "claim_ceiling": "test only",
        "metadata_freeze": {
            "split": "train", "expected_shards": 1,
            "expected_filenames": ["0000.parquet"], "expected_total_bytes": 10,
        },
        "rights_filter": {"allowed_license_urls": ["https://creativecommons.org/licenses/by/2.0/"]},
        "stock_patterns": [
            {"film_stock_id": "kodak_portra_400", "exact_regex": "(^|[^a-z0-9])(?:kodak )?portra 400([^a-z0-9]|$)"}
        ],
        "metadata_gates": {"minimum_rows": 2, "minimum_author_uids": 2, "maximum_largest_author_share": 0.6},
    }


def test_parquet_index_fails_closed_on_size_drift() -> None:
    payload = {"pending": [], "failed": [], "parquet_files": [
        {"filename": "0000.parquet", "url": "https://example.test/0", "size": 10, "split": "train"}
    ]}
    assert validate_parquet_index(payload, _config())[0]["filename"] == "0000.parquet"
    payload["parquet_files"][0]["size"] = 11
    with pytest.raises(YfccStockSourceError, match="byte total"):
        validate_parquet_index(payload, _config())


def test_local_scan_filters_licence_and_matches_exact_phrase(tmp_path: Path) -> None:
    path = tmp_path / "0000.parquet"
    connection = duckdb.connect()
    connection.execute("""CREATE TABLE rows AS SELECT * FROM (VALUES
      (1, 'a', 'A', 'Kodak Portra 400 portrait', '', '', 'p1', 'd1', 'Attribution', 'https://creativecommons.org/licenses/by/2.0/', 0),
      (2, 'b', 'B', '', 'shot on portra-400', '', 'p2', 'd2', 'Attribution', 'https://creativecommons.org/licenses/by/2.0/', 0),
      (3, 'c', 'C', 'Portra 400', '', '', 'p3', 'd3', 'ShareAlike', 'https://creativecommons.org/licenses/by-sa/2.0/', 0),
      (4, 'd', 'D', 'Portra 4000', '', '', 'p4', 'd4', 'Attribution', 'https://creativecommons.org/licenses/by/2.0/', 0)
    ) t(photoid,uid,unickname,title,description,usertags,pageurl,downloadurl,licensename,licenseurl,marker)""")
    connection.execute("COPY rows TO ? (FORMAT PARQUET)", [str(path)])
    report = scan_stock_candidates([path], _config())
    result = report["candidate_results"]["kodak_portra_400"]
    assert result["rows"] == 2
    assert result["author_uids"] == 2
    assert result["metadata_gate_passed"] is True
    assert {row["photoid"] for row in report["matches"]} == {1, 2}


def test_local_scan_reports_missing_parquet_as_contract_error(tmp_path: Path) -> None:
    with pytest.raises(YfccStockSourceError, match="missing or invalid"):
        scan_stock_candidates([tmp_path / "missing.parquet"], _config())
