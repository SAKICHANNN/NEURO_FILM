from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from scripts.audit_u5_r2ppsd0_source import evaluate

ROOT = Path(__file__).resolve().parents[1]


def _archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("responses.csv", "user,choice\nprivate,0\n")
    return output.getvalue()


def _fetcher(contract: dict, *, first_size_delta: int = 0):
    source = contract["official_sources"]
    expected = source["expected_drive_files"]
    project = (
        b"767 users ~60,000 valid preference judgments 1,192 unique scenes "
        b"7,972 unique image style pairs across five source categories"
    )
    drive_text = "".join(
        f'[[["{row["id"]}",["folder"],"{row["name"]}","application/zip",'
        f'0,null,0,0,0,1,1,null,null,{row["size_bytes"] + (first_size_delta if index == 0 else 0)},'
        for index, row in enumerate(expected)
    )
    drive = (
        drive_text.replace('"', r"\x22").replace("[", r"\x5b").replace("]", r"\x5d").encode()
    )
    archive = _archive()

    def fetch(url: str, maximum_bytes: int) -> bytes:
        payload = archive if "usercontent" in url else drive if "drive.google.com" in url else project
        assert len(payload) <= maximum_bytes
        return payload

    return fetch


def test_audit_retains_only_annotation_structure() -> None:
    contract = json.loads((ROOT / "configs/u5_r2ppsd0_source_audit_v1.json").read_text())
    report = evaluate(contract, fetch=_fetcher(contract))
    assert report["automatic_pass"] is True
    assert report["gates"]["pixels_or_training_allowed"] is False
    assert report["requests"]["image_archives"] == 0
    assert report["annotation_structure"]["members"][0]["path"] == "responses.csv"
    assert "private" not in json.dumps(report)


def test_drive_size_drift_fails_closed() -> None:
    contract = json.loads((ROOT / "configs/u5_r2ppsd0_source_audit_v1.json").read_text())
    report = evaluate(contract, fetch=_fetcher(contract, first_size_delta=1))
    assert report["automatic_pass"] is False


def test_project_fact_drift_fails_closed() -> None:
    contract = json.loads((ROOT / "configs/u5_r2ppsd0_source_audit_v1.json").read_text())
    contract["official_sources"]["expected_project_facts"]["retained_users"] = 768
    report = evaluate(contract, fetch=_fetcher(contract))
    assert report["automatic_pass"] is False
