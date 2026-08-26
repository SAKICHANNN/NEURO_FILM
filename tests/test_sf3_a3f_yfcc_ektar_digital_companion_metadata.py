from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.real_film.yfcc_ektar_companion import (
    _candidate_evaluation,
    audit,
    canonical_json,
    sha256_file,
)


def _query() -> dict:
    return {
        "maximum_capture_time_delta_seconds": 86400,
        "maximum_upload_time_delta_seconds": 86400,
        "allowed_license_urls": ["http://creativecommons.org/licenses/by/2.0/"],
        "scanner_device_terms": ["scanner", "noritsu"],
        "film_text_terms": ["film", "ektar"],
        "same_scene_evidence": {
            "maximum_capture_time_delta_seconds": 1800,
            "maximum_geodesic_distance_metres": 250.0,
            "minimum_location_accuracy": 11,
            "minimum_shared_title_tokens": 1,
            "minimum_title_token_length": 4,
            "allowed_routes": ["time_and_geo", "time_and_title"],
        },
    }


def _row(**updates) -> dict:
    row = {
        "photoid": 101,
        "uid": "owner",
        "datetaken": "2020-01-01 12:05:00.0",
        "dateuploaded": "1577880300",
        "capturedevice": "Nikon D850",
        "title": "Harbour sunset digital",
        "description": "",
        "usertags": "harbour",
        "machinetags": "",
        "longitude": "1.0001",
        "latitude": "51.0001",
        "accuracy": 16,
        "pageurl": "https://example/101",
        "downloadurl": "https://example/101.jpg",
        "licensename": "Attribution License",
        "licenseurl": "http://creativecommons.org/licenses/by/2.0/",
    }
    row.update(updates)
    return row


def test_candidate_accepts_time_and_geo_digital_companion() -> None:
    target = _row(
        photoid=100,
        datetaken="2020-01-01 12:00:00.0",
        dateuploaded="1577880000",
        capturedevice="Epson scanner",
        title="Ektar harbour",
        usertags="film ektar",
        longitude="1.0",
        latitude="51.0",
    )
    reason, result = _candidate_evaluation(target, _row(), _query())
    assert reason == "eligible"
    assert result is not None
    assert "time_and_geo" in result["same_scene_routes"]


def test_candidate_rejects_scanner_film_and_wrong_scene() -> None:
    target = _row(photoid=100, datetaken="2020-01-01 12:00:00.0", dateuploaded="1577880000")
    assert _candidate_evaluation(target, _row(capturedevice="Noritsu QSS"), _query())[0] == "scanner_device"
    assert _candidate_evaluation(target, _row(usertags="Kodak Ektar film"), _query())[0] == "film_text"
    assert (
        _candidate_evaluation(
            target,
            _row(title="Unrelated", longitude="8.0", latitude="40.0", accuracy=16),
            _query(),
        )[0]
        == "same_scene"
    )


def _synthetic_root(tmp_path: Path) -> tuple[Path, Path]:
    database = tmp_path / "data.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "create table yfcc100m_dataset (photoid integer primary key, uid text, unickname text, "
        "datetaken text, dateuploaded text, capturedevice text, title text, description text, usertags text, "
        "machinetags text, longitude text, latitude text, accuracy integer, pageurl text, downloadurl text, "
        "licensename text, licenseurl text, serverid integer, farmid integer, secret text, secretoriginal text, "
        "ext text, marker integer)"
    )
    columns = (
        "photoid,uid,datetaken,dateuploaded,capturedevice,title,description,usertags,machinetags,"
        "longitude,latitude,accuracy,pageurl,downloadurl,licensename,licenseurl"
    )
    values = (
        ":photoid,:uid,:datetaken,:dateuploaded,:capturedevice,:title,:description,:usertags,:machinetags,"
        ":longitude,:latitude,:accuracy,:pageurl,:downloadurl,:licensename,:licenseurl"
    )
    target = _row(
        photoid=100,
        datetaken="2020-01-01 12:00:00.0",
        dateuploaded="1577880000",
        capturedevice="Epson scanner",
        title="Ektar harbour",
        usertags="film ektar",
    )
    connection.execute(f"insert into yfcc100m_dataset ({columns}) values ({values})", target)
    connection.execute(f"insert into yfcc100m_dataset ({columns}) values ({values})", _row())
    connection.commit()
    connection.close()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        canonical_json(
            {
                "attempts": [
                    {"photoid": 100, "uid": "owner", "decision": "retain", "film_stock_id": "kodak_ektar_100"}
                ]
            }
        )
    )
    source_config = tmp_path / "source.json"
    source_config.write_text("{}\n", encoding="utf-8")
    contract = {
        "experiment_id": "sf3.a3f-yfcc-ektar-digital-companion-metadata-v1",
        "sources": {
            "yfcc_sqlite": database.name,
            "yfcc_sqlite_sha256": sha256_file(database),
            "yfcc_table": "yfcc100m_dataset",
            "yfcc_config": source_config.name,
            "yfcc_config_sha256": sha256_file(source_config),
            "ektar_target_manifest": manifest.name,
            "ektar_target_manifest_sha256": sha256_file(manifest),
            "target_stock_id": "kodak_ektar_100",
            "target_decision": "retain",
        },
        "query": {**_query(), "photoid_radius": 1000},
        "gates": {
            "minimum_target_rows": 1,
            "minimum_target_uids": 1,
            "minimum_eligible_pairs": 1,
            "minimum_eligible_uids": 1,
            "maximum_sqlite_rows_examined": 10,
            "maximum_network_requests": 0,
            "maximum_image_requests": 0,
            "maximum_pixel_decodes": 0,
        },
        "claim_ceiling": "test",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(canonical_json(contract))
    return tmp_path, contract_path


def test_audit_is_order_invariant_and_network_free(tmp_path: Path) -> None:
    root, contract = _synthetic_root(tmp_path)
    forward = audit(root, contract)
    reverse = audit(root, contract, reverse=True)
    assert forward == reverse
    assert forward["decision"] == "PASS_OPEN_BOUNDED_LIVE_RIGHTS_PREFLIGHT"
    assert forward["network_requests"] == forward["image_requests"] == forward["pixel_decodes"] == 0


def test_project_contract_is_valid_json_and_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/sf3_a3f_yfcc_ektar_digital_companion_metadata_v1.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["gates"]["maximum_network_requests"] == 0
    assert contract["claim_ceiling"].startswith("metadata-only Ektar")
