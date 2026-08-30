from __future__ import annotations

import json
from pathlib import Path

from src.real_film.film2paint_reversal_target_source import run_source_audit


def _config(tmp_path: Path, article: bytes, *, with_dataset: bool) -> Path:
    bundles = [
        {"name": "ORIGINAL", "uuid": "bundle-original"},
        {"name": "DATA", "uuid": "bundle-data"},
    ]
    bitstreams = [
        {
            "bundle": "ORIGINAL",
            "name": "thesis.pdf",
            "uuid": "thesis",
            "bytes": 10,
            "checksum_algorithm": "MD5",
            "checksum": "a" * 32,
        }
    ]
    if with_dataset:
        bitstreams.extend(
            [
                {
                    "bundle": "DATA",
                    "name": "film_targets.zip",
                    "uuid": "data",
                    "bytes": 20,
                    "checksum_algorithm": "SHA-256",
                    "checksum": "b" * 64,
                },
                {
                    "bundle": "DATA",
                    "name": "manifest_sha256.json",
                    "uuid": "manifest",
                    "bytes": 30,
                    "checksum_algorithm": "SHA-256",
                    "checksum": "c" * 64,
                },
            ]
        )
    payload = {
        "experiment_id": "test",
        "article": {
            "url": "https://example/article",
            "bytes": len(article),
            "sha256": __import__("hashlib").sha256(article).hexdigest(),
            "required_text_fragments": ["Velvia 50", "hyperspectral"],
        },
        "repository": {
            "base_url": "https://example/api/core",
            "item_uuid": "item",
            "doi": "10.test/item",
            "expected_bundles": bundles,
            "expected_bitstreams": bitstreams,
            "document_bitstream_licence": "CC_BY_NC",
        },
        "target_stock": {"film_stock_id": "velvia", "public_label": "Velvia 50"},
        "required_dataset_properties": ["dataset"],
        "admission_markers": {
            "dataset_extensions": [".zip"],
            "manifest_name_fragments": ["manifest"],
            "dataset_licence_fragments": ["dataset license cc by 4.0"],
            "group_role_fragments": ["roll_id", "process_id", "scanner_id", "source_group", "confirmation"],
        },
        "operation_limits": {
            "article_document_requests": 1,
            "repository_metadata_requests": 4,
            "repository_bitstream_content_requests": 0,
            "film_target_scan_requests": 0,
            "image_derivative_requests": 0,
            "account_or_copy_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reader(config: dict, article: bytes, *, with_dataset: bool):
    base = config["repository"]["base_url"]
    item_text = "dataset license CC BY 4.0 roll_id process_id scanner_id source_group confirmation"
    item = {
        "uuid": "item",
        "metadata": {
            "dc.identifier.doi": [{"value": "10.test/item"}],
            "dc.description": [{"value": item_text if with_dataset else "document only"}],
        },
    }
    bundles = {"_embedded": {"bundles": config["repository"]["expected_bundles"]}}
    by_bundle = {}
    for bundle in config["repository"]["expected_bundles"]:
        rows = []
        for row in config["repository"]["expected_bitstreams"]:
            if row["bundle"] != bundle["name"]:
                continue
            rows.append(
                {
                    "name": row["name"],
                    "uuid": row["uuid"],
                    "sizeBytes": row["bytes"],
                    "checkSum": {
                        "checkSumAlgorithm": row["checksum_algorithm"],
                        "value": row["checksum"],
                    },
                }
            )
        by_bundle[bundle["uuid"]] = {"_embedded": {"bitstreams": rows}}
    payloads = {
        config["article"]["url"]: article,
        f"{base}/items/item": json.dumps(item).encode(),
        f"{base}/items/item/bundles?size=100": json.dumps(bundles).encode(),
    }
    for uuid, payload in by_bundle.items():
        payloads[f"{base}/bundles/{uuid}/bitstreams?size=100"] = json.dumps(payload).encode()
    return lambda url, _max_bytes: payloads[url]


def test_document_only_repository_fails_before_dataset_reads(tmp_path: Path) -> None:
    article = b"article"
    config_path = _config(tmp_path, article, with_dataset=False)
    config = json.loads(config_path.read_text())
    report = run_source_audit(
        config_path,
        binary_reader=_reader(config, article, with_dataset=False),
        article_text_reader=lambda _payload: "Velvia 50 hyperspectral",
    )
    assert report["decision"] == "FAIL"
    assert report["admission"]["dataset_bitstreams"] == []
    assert report["gates"]["official_repository_inventory_matches"]
    assert not report["gates"]["public_film_target_dataset_payload_present"]
    assert report["operation_counts"]["repository_bitstream_content_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_complete_mock_dataset_can_only_open_next_audit(tmp_path: Path) -> None:
    article = b"article"
    config_path = _config(tmp_path, article, with_dataset=True)
    config = json.loads(config_path.read_text())
    kwargs = {
        "binary_reader": _reader(config, article, with_dataset=True),
        "article_text_reader": lambda _payload: "Velvia 50 hyperspectral",
    }
    forward = run_source_audit(config_path, **kwargs)
    reverse = run_source_audit(config_path, reverse_request_order=True, **kwargs)
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["gates"].values())


def test_project_contract_forbids_dataset_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3p_film2paint_reversal_target_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["target_stock"]["film_stock_id"] == "fujifilm_velvia_50"
    for key in (
        "repository_bitstream_content_requests",
        "film_target_scan_requests",
        "image_derivative_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
