from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path

from PIL import Image

from src.real_film.commons_stock_pilot import (
    CommonsStockPilotError,
    audit_download_manifest,
    build_selection_manifest,
    download_selected_rows,
    merge_metadata_snapshots,
    normalize_author,
    resolve_download_url,
)


def _config() -> dict:
    return {
        "pilot_id": "test", "metadata_snapshot_sha256": "abc",
        "allowed_label_scope": "exact", "allowed_stock_ids": ["a", "b", "c"],
        "rights_filter": {
            "allowed_licenses_with_explicit_url": ["CC BY 4.0"],
            "allow_public_domain_with_usage_terms_and_file_page": True,
        },
        "selection": {
            "algorithm": "test", "maximum_files_per_stock": 3,
            "maximum_files_per_normalized_author_per_stock": 2,
            "maximum_files_per_uploader_per_stock": 2,
            "preflight_expected_selected": {"a": 3, "b": 3, "c": 3},
            "maximum_files_total": 9,
        },
        "download_limits": {
            "maximum_bytes_total": 100000, "maximum_bytes_per_file": 50000,
            "temporary_suffix": ".part",
        },
        "pixel_audit": {
            "minimum_short_dimension": 8, "near_duplicate_hamming_threshold": 4,
            "minimum_retained_files_per_stock": 2,
            "minimum_normalized_author_groups_for_learning": 2,
            "maximum_largest_normalized_author_share_for_learning": 0.7,
        },
        "claim_ceiling": "test",
    }


def _row(index: int, author: str, uploader: str) -> dict:
    return {
        "page_id": index, "title": f"File:{index}.jpg", "uploader": uploader,
        "author_raw_html": f"<b>{author}</b>", "credit_raw_html": "credit",
        "license_short_name": "CC BY 4.0", "license_url": "https://license",
        "usage_terms": "", "file_page_url": "https://page", "original_url": "https://original",
        "derivative_1600_url": f"https://pixel/{index}", "api_sha1_base36": f"sha{index}",
    }


def _snapshot() -> dict:
    categories = []
    for offset, stock in enumerate(("a", "b", "c")):
        categories.append({
            "film_stock_id": stock, "label_scope": "exact",
            "files": [_row(offset * 10 + i, f"author{i % 2}", f"user{i % 3}") for i in range(4)],
        })
    return {"categories": categories}


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(buffer, "JPEG")
    return buffer.getvalue()


class _Response:
    def __init__(self, payload: bytes):
        self.content = payload
        self.headers = {"content-type": "image/jpeg"}

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None


class _Session:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: int, stream: bool = False) -> _Response:
        return _Response(self.payloads[url])


def test_author_normalization_and_dual_cap_selection_are_stable() -> None:
    assert normalize_author("<a>Alice&nbsp; Smith</a>") == "alice smith"
    first = build_selection_manifest(_snapshot(), _config())
    second = build_selection_manifest(deepcopy(_snapshot()), _config())
    assert first == second
    assert first["selected_by_stock"] == {"a": 3, "b": 3, "c": 3}
    assert max(row["selection_index"] for row in first["rows"]) == 8


def test_missing_license_url_is_excluded_fail_closed() -> None:
    snapshot = _snapshot()
    snapshot["categories"][0]["files"][0]["license_url"] = ""
    snapshot["categories"][0]["files"][1]["license_url"] = ""
    try:
        build_selection_manifest(snapshot, _config())
    except CommonsStockPilotError as exc:
        assert "selection drift" in str(exc)
    else:
        raise AssertionError("expected selection drift")


def test_download_and_offline_audit_verify_pixels(tmp_path: Path) -> None:
    config = _config()
    selection = build_selection_manifest(_snapshot(), config)
    payloads = {
        row["derivative_1600_url"]: _jpeg((20 + row["selection_index"] * 20, 40, 80))
        for row in selection["rows"]
    }
    manifest = download_selected_rows(
        selection["rows"], root=tmp_path, config=config, session=_Session(payloads)
    )
    report = audit_download_manifest(manifest, root=tmp_path, config=config)
    assert manifest["files"] == 9
    assert report["files"] == 9
    assert set(report["stock_source_gates"]) == {"a", "b", "c"}


def test_untracked_resume_fails_closed(tmp_path: Path) -> None:
    config = _config()
    row = build_selection_manifest(_snapshot(), config)["rows"][0]
    path = tmp_path / row["film_stock_id"] / f"{row['page_id']}.img"
    path.parent.mkdir(parents=True)
    path.write_bytes(_jpeg((1, 2, 3)))
    try:
        download_selected_rows([row], root=tmp_path, config=config, session=_Session({}))
    except CommonsStockPilotError as exc:
        assert "untracked resumed file" in str(exc)
    else:
        raise AssertionError("expected untracked resume failure")


def test_merge_metadata_snapshots_keeps_only_allowed_stocks() -> None:
    snapshot = _snapshot()
    merged = merge_metadata_snapshots(
        [{"categories": snapshot["categories"][:2]}, {"categories": snapshot["categories"][2:]}],
        allowed_stock_ids=["a", "c"],
    )
    assert [row["film_stock_id"] for row in merged["categories"]] == ["a", "c"]
    assert merged["image_payloads_downloaded_or_decoded"] is False


def test_unscaled_commons_tracking_url_becomes_real_bounded_thumbnail() -> None:
    config = _config()
    config["download_limits"]["thumbnail_unscaled_max_width"] = 960
    row = _row(1, "author", "user")
    row.update(
        {
            "mime": "image/jpeg",
            "width": 1058,
            "derivative_1600_url": (
                "https://upload.wikimedia.org/wikipedia/commons/1/14/example.jpg"
                "?utm_source=commons.wikimedia.org&utm_content=thumbnail_unscaled"
            ),
        }
    )
    assert resolve_download_url(row, config) == (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/example.jpg/"
        "960px-example.jpg"
    )
