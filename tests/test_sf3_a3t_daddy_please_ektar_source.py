from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.real_film.daddy_please_ektar_source import (
    DaddyPleaseEktarSourceError,
    run_daddy_please_ektar_source_audit,
)


def _cbor_text(value: str) -> bytes:
    payload = value.encode("utf-8")
    if len(payload) < 24:
        return bytes([0x60 + len(payload)]) + payload
    return bytes([0x78, len(payload)]) + payload


def _metadata(model: str, mood: str) -> bytes:
    values = {
        "MODEL": model,
        "MOOD": mood,
        "SIGN": "ARIES",
        "PROP": "NONE",
        "BACKGROUND": "BLUE",
    }
    raw = bytes([0xA0 + len(values)]) + b"".join(
        _cbor_text(key) + _cbor_text(value) for key, value in values.items()
    )
    return json.dumps(raw.hex()).encode("utf-8")


def _parent_html(fields: dict[str, str]) -> bytes:
    rows = "".join(f"<dt>{key}</dt><dd>{value}</dd>" for key, value in fields.items())
    return f"<html><dl>{rows}</dl></html>".encode()


def _child_html(parent_id: str, *, length: int) -> bytes:
    return (
        f'<a href="/inscription/{parent_id}">parent</a>'
        f"<dt>content length</dt><dd>{length} bytes</dd>"
        "<dt>content type</dt><dd>image/avif</dd>"
    ).encode()


def _fixture(
    tmp_path: Path, *, complete_groups: bool = False
) -> tuple[Path, dict[str, bytes]]:
    ids = [f"{'a' * 63}{index}i0" for index in range(4)]
    ranked = sorted(ids, key=lambda item: hashlib.sha256(item.encode()).hexdigest())
    parent_id = f"{'b' * 64}i0"
    parent_fields = {
        "PHOTOGRAPHER": "PARKER DAY",
        "CAMERA": "CANON EOS-1V",
        "FILM": "KODAK EKTAR 100",
        "LICENSE": "CC0",
    }
    ordered_sha = hashlib.sha256((("\n".join(ids)) + "\n").encode()).hexdigest()
    group_evidence = {
        "explicit_content_group_count": 2,
        "independent_author_source_count": 1,
        "explicit_roll_group_count": 0,
        "explicit_process_group_count": 0,
        "explicit_scanner_group_count": 0,
        "same_source_cross_stock_control_count": 0,
        "same_scene_neutral_film_pair_count": 0,
        "sealed_confirmation_group_count": 0,
    }
    minimums = {key: value for key, value in group_evidence.items()}
    if not complete_groups:
        minimums["independent_author_source_count"] = 2
    config = {
        "experiment_id": "test",
        "source": {
            "base_url": "https://example.test",
            "parent_id": parent_id,
            "parent_slug_url": "https://example.test/inscription/project",
            "child_page_count": 1,
            "expected_ids_per_page": 4,
            "expected_child_count": 4,
            "expected_ordered_child_id_sha256": ordered_sha,
            "required_parent_fields": parent_fields,
            "required_child_metadata_keys": [
                "MODEL",
                "MOOD",
                "SIGN",
                "PROP",
                "BACKGROUND",
            ],
            "expected_model_count": 2,
            "expected_photos_per_model": 2,
            "metadata_workers": 2,
        },
        "sample": {
            "selection": "test",
            "count": 2,
            "expected_ids": ranked[:2],
            "expected_total_content_bytes": 201,
            "required_content_type": "image/avif",
        },
        "group_evidence": group_evidence,
        "admission_minimums": minimums,
        "operation_limits": {
            "parent_html_get_requests": 1,
            "child_index_json_get_requests": 1,
            "child_metadata_json_get_requests": 4,
            "child_html_get_requests": 2,
            "image_body_requests": 0,
            "image_range_requests": 0,
            "pixel_decodes": 0,
            "fit_calls": 0,
            "render_calls": 0,
            "score_calls": 0,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    payloads = {
        config["source"]["parent_slug_url"]: _parent_html(parent_fields),
        f"https://example.test/r/children/{parent_id}": json.dumps(
            {"ids": ids, "more": False, "page": 0}
        ).encode(),
    }
    for index, child_id in enumerate(ids):
        payloads[f"https://example.test/r/metadata/{child_id}"] = _metadata(
            "MODEL_A" if index < 2 else "MODEL_B", f"MOOD_{index}"
        )
    payloads[f"https://example.test/inscription/{ranked[0]}"] = _child_html(
        parent_id, length=100
    )
    payloads[f"https://example.test/inscription/{ranked[1]}"] = _child_html(
        parent_id, length=101
    )
    return config_path, payloads


def test_complete_mock_source_is_exact_across_request_order(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path, complete_groups=True)
    requests: list[str] = []

    def fetcher(url: str) -> bytes:
        requests.append(url)
        assert "/content/" not in url and "/preview/" not in url
        return payloads[url]

    forward = run_daddy_please_ektar_source_audit(config, fetcher=fetcher)
    reverse = run_daddy_please_ektar_source_audit(config, reverse=True, fetcher=fetcher)
    assert forward == reverse
    assert forward["decision"] == "PASS"
    assert all(forward["audit_gates"].values())
    assert forward["metadata_summary"]["model_count"] == 2
    assert requests


def test_single_source_groups_fail_admission_without_media(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    report = run_daddy_please_ektar_source_audit(
        config, fetcher=lambda url: payloads[url]
    )
    assert report["decision"] == "FAIL"
    assert all(report["audit_gates"].values())
    assert not report["admission_gates"]["independent_author_source_count"]
    assert report["operation_counts"]["image_body_requests"] == 0
    assert report["operation_counts"]["pixel_decodes"] == 0


def test_malformed_child_metadata_fails_closed(tmp_path: Path) -> None:
    config, payloads = _fixture(tmp_path)
    metadata_url = next(url for url in payloads if "/r/metadata/" in url)
    payloads[metadata_url] = b'"ff"'
    with pytest.raises(DaddyPleaseEktarSourceError):
        run_daddy_please_ektar_source_audit(config, fetcher=lambda url: payloads[url])


def test_project_contract_forbids_media_and_pixel_reads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/sf3_a3t_daddy_please_ektar_source_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["source"]["required_parent_fields"]["FILM"] == "KODAK EKTAR 100"
    assert config["source"]["required_parent_fields"]["LICENSE"] == "CC0"
    for key in (
        "image_body_requests",
        "image_range_requests",
        "pixel_decodes",
        "fit_calls",
        "render_calls",
        "score_calls",
    ):
        assert config["operation_limits"][key] == 0
