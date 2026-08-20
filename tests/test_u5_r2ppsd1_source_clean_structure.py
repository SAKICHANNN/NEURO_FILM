from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import scripts.audit_u5_r2ppsd1_source_clean_structure as module

ROOT = Path(__file__).resolve().parents[1]


def _archive() -> bytes:
    output = io.BytesIO()
    contract = json.loads((ROOT / "configs/u5_r2ppsd1_source_clean_structure_v1.json").read_text())
    with zipfile.ZipFile(output, "w") as bundle:
        rows = []
        for collection in contract["source_roles"]["included_collections"]:
            rows.append(
                {
                    "collection": collection,
                    "scene_id": f"{collection}-001",
                    "left_style": "a",
                    "right_style": "b",
                    "choice": 0,
                    "user_id": f"participant-{collection}",
                }
            )
        rows.append(
            {
                "collection": "C1",
                "scene_id": "C1-001",
                "left_style": "a",
                "right_style": "b",
                "choice": 1,
                "user_id": "excluded-participant",
            }
        )
        for collection in contract["source_roles"]["excluded_collections"][1:]:
            rows.append(
                {
                    "collection": collection,
                    "scene_id": f"{collection}-001",
                    "left_style": "a",
                    "right_style": "b",
                    "choice": 1,
                    "user_id": f"excluded-{collection}",
                }
            )
        bundle.writestr(
            "responses/raw/votes_items.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
        processed = {}
        for collection, count in contract["source_roles"]["expected_processed_scene_counts"].items():
            processed.update({f"{collection}-{index:03d}": {} for index in range(count)})
        bundle.writestr("responses/processed/opaque_desktop.json", json.dumps(processed))
    return output.getvalue()


def _test_contract(tmp_path: Path, monkeypatch) -> tuple[dict, bytes]:
    contract = json.loads((ROOT / "configs/u5_r2ppsd1_source_clean_structure_v1.json").read_text())
    archive = _archive()
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"decision": contract["parents"]["ppsd0_evidence"]["required_decision"]}))
    paper = tmp_path / "paper.pdf"
    supplement = tmp_path / "supplement.pdf"
    paper.write_bytes(b"paper")
    supplement.write_bytes(b"supplement")
    contract["parents"]["ppsd0_evidence"] |= {"path": parent.name, "sha256": module._sha256(parent.read_bytes())}
    contract["parents"]["paper"] |= {"path": paper.name, "sha256": module._sha256(paper.read_bytes())}
    contract["parents"]["supplemental"] |= {"path": supplement.name, "sha256": module._sha256(supplement.read_bytes())}
    contract["parents"]["responses_archive_sha256"] = module._sha256(archive)
    return contract, archive


def test_source_clean_structure_passes_without_persisting_identifiers(tmp_path: Path, monkeypatch) -> None:
    contract, archive = _test_contract(tmp_path, monkeypatch)
    report = module.evaluate(contract, root=tmp_path, fetch=lambda _url, _maximum: archive)
    assert report["automatic_pass"] is True
    assert report["gates"]["pixels_or_training_allowed"] is False
    assert report["annotation_structure"]["retained_processed_scene_keys"] == 457
    assert report["annotation_structure"]["retained_vote_rows"] == 7
    encoded = json.dumps(report)
    assert "participant-A1" not in encoded
    assert "A1-001" not in encoded
    assert report["requests"]["image_archives"] == 0


def test_scene_count_drift_fails_closed(tmp_path: Path, monkeypatch) -> None:
    contract, archive = _test_contract(tmp_path, monkeypatch)
    contract["source_roles"]["expected_processed_scene_counts"]["A1"] += 1
    report = module.evaluate(contract, root=tmp_path, fetch=lambda _url, _maximum: archive)
    assert report["automatic_pass"] is False


def test_invalid_retained_vote_fails_closed(tmp_path: Path, monkeypatch) -> None:
    contract, archive = _test_contract(tmp_path, monkeypatch)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as destination:
            for item in source.infolist():
                data = source.read(item)
                if item.filename.endswith("votes_items.jsonl"):
                    rows = data.splitlines()
                    payload = json.loads(rows[0])
                    payload.pop("choice")
                    rows[0] = json.dumps(payload).encode()
                    data = b"\n".join(rows) + b"\n"
                destination.writestr(item, data)
    drifted = output.getvalue()
    contract["parents"]["responses_archive_sha256"] = module._sha256(drifted)
    report = module.evaluate(contract, root=tmp_path, fetch=lambda _url, _maximum: drifted)
    assert report["automatic_pass"] is False
