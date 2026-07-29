from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.apollo_step_chart_source import (
    ApolloStepChartSourceError,
    acquire_archive,
    inspect_archive,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u6_p2k_apollo_step_chart_source_v1.json").read_text(
        encoding="utf-8"
    )
)


def _zip_bytes(name: str = "AS07-03-StepChart/raw_scan.tif") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"raw-scan-placeholder")
    return output.getvalue()


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.url = CONFIG["source"]["download_url"]
        self.headers = {
            "Content-Type": "application/zip",
            "Content-Disposition": (
                'attachment; filename="AS07-03-StepChart.zip"'
            ),
        }
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get(self, *_args, **_kwargs):
        return self.response


def test_contract_binds_one_so368_archive_without_fitting() -> None:
    validate_config(CONFIG)
    assert CONFIG["magazine_evidence"]["film_code"] == "SO368"
    assert CONFIG["acquisition"]["resume_allowed"] is False
    assert CONFIG["operator_fitting_allowed"] is False


def test_acquisition_is_bounded_atomic_and_audited(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["acquisition"]["destination"] = "archive.zip"
    response = _Response(_zip_bytes())
    manifest = acquire_archive(
        tmp_path, config, session=_Session(response)
    )
    assert response.closed
    assert manifest["zip_crc_pass"] is True
    assert manifest["member_count"] == 1
    assert manifest["analysis_allowed"] is False
    assert (tmp_path / "archive.zip").is_file()
    assert not (tmp_path / "archive.zip.part").exists()
    assert inspect_archive(tmp_path / "archive.zip", config) == manifest


def test_unsafe_member_and_off_host_redirect_fail_closed(
    tmp_path: Path,
) -> None:
    config = deepcopy(CONFIG)
    config["acquisition"]["destination"] = "archive.zip"
    response = _Response(_zip_bytes("../escape.tif"))
    with pytest.raises(ApolloStepChartSourceError, match="unsafe member"):
        acquire_archive(tmp_path, config, session=_Session(response))
    assert not (tmp_path / "archive.zip").exists()
    assert not (tmp_path / "archive.zip.part").exists()

    response = _Response(_zip_bytes())
    response.url = "https://example.com/archive.zip"
    with pytest.raises(ApolloStepChartSourceError, match="redirected"):
        acquire_archive(tmp_path, config, session=_Session(response))
    assert not (tmp_path / "archive.zip").exists()


def test_overflow_removes_owned_partial(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["acquisition"]["destination"] = "archive.zip"
    config["acquisition"]["maximum_compressed_bytes"] = 8
    response = _Response(_zip_bytes())
    with pytest.raises(ApolloStepChartSourceError, match="exceeded"):
        acquire_archive(tmp_path, config, session=_Session(response))
    assert not (tmp_path / "archive.zip.part").exists()
