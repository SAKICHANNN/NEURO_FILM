from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.apollo16_bw_step_chart_source import (
    Apollo16BWStepChartSourceError,
    acquire_archive,
    inspect_archive,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/u6_p2l_apollo16_bw_step_chart_source_v1.json").read_text(
        encoding="utf-8"
    )
)


def _zip_bytes(name: str = "AS16-111-stepchart05.tif") -> bytes:
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
                'attachment; filename="AS16-111-stepchart05.zip"'
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


def test_contract_binds_only_as16_111_kodak_3401_without_fitting() -> None:
    validate_config(CONFIG)
    assert CONFIG["magazine_evidence"]["film_code"] == "3401"
    assert CONFIG["magazine_evidence"]["physical_magazine"] == "J"
    assert CONFIG["operator_fitting_allowed"] is False


def test_atomic_bounded_acquisition_and_crc(tmp_path: Path) -> None:
    config = deepcopy(CONFIG)
    config["acquisition"]["destination"] = "segment.zip"
    response = _Response(_zip_bytes())
    manifest = acquire_archive(tmp_path, config, session=_Session(response))
    assert response.closed
    assert manifest["zip_crc_pass"] is True
    assert manifest["member_count"] == 1
    assert manifest["analysis_allowed"] is False
    assert inspect_archive(tmp_path / "segment.zip", config) == manifest
    assert not (tmp_path / "segment.zip.part").exists()


def test_redirect_unsafe_member_and_learning_drift_fail_closed(
    tmp_path: Path,
) -> None:
    config = deepcopy(CONFIG)
    config["acquisition"]["destination"] = "segment.zip"
    response = _Response(_zip_bytes("../escape.tif"))
    with pytest.raises(RuntimeError, match="unsafe member"):
        acquire_archive(tmp_path, config, session=_Session(response))
    assert not (tmp_path / "segment.zip").exists()

    response = _Response(_zip_bytes())
    response.url = "https://example.com/segment.zip"
    with pytest.raises(Apollo16BWStepChartSourceError, match="redirected"):
        acquire_archive(tmp_path, config, session=_Session(response))

    config = deepcopy(CONFIG)
    config["training_allowed"] = True
    with pytest.raises(Apollo16BWStepChartSourceError, match="closed"):
        validate_config(config)
