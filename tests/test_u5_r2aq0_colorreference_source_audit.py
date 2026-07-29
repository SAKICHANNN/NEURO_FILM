from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pytest

from scripts.run_u5_r2aq0_colorreference_velvia100f_source_audit import (
    CONFIG_SHA256,
    load_config,
    reference_member_record,
    safe_zip_member_names,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2aq0_colorreference_velvia100f_source_audit_v1.json"
)


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def test_frozen_contract_is_bounded_and_fit_closed() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["acquisition"]["expected_total_bytes_per_pass"] == 3147395
    assert config["acquisition"]["maximum_total_bytes_per_pass"] == 8388608
    assert len(config["acquisition"]["source_tiffs"]) == 5
    assert len(config["acquisition"]["measured_reference_archives"]) == 12
    assert config["fit_allowed"] is False
    assert config["rights_contract"]["redistribution_allowed"] is False


def test_zip_member_policy_rejects_traversal_and_duplicates() -> None:
    safe = _zip_bytes({"testscan1_1.txt": b"x"})
    assert safe_zip_member_names(safe) == ["testscan1_1.txt"]
    with pytest.raises(ValueError, match="unsafe"):
        safe_zip_member_names(_zip_bytes({"../escape.txt": b"x"}))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("same.txt", b"a")
        archive.writestr("same.txt", b"b")
    with pytest.raises(ValueError, match="duplicate"):
        safe_zip_member_names(stream.getvalue())


def test_reference_member_parser_checks_declared_shape() -> None:
    text = b"\n".join(
        [
            b"CGATS.5",
            b'MATERIAL "Fujichrome Velvia 100F (RVP 100F)"',
            b'NUMBER_OF_FIELDS "3"',
            b"BEGIN_DATA_FORMAT",
            b"SAMPLE_ID XYZ_X XYZ_Y",
            b"END_DATA_FORMAT",
            b'NUMBER_OF_SETS "2"',
            b"BEGIN_DATA",
            b"A1 1.0 2.0",
            b"A2 3.0 4.0",
            b"END_DATA",
        ]
    )
    payload = _zip_bytes({"testscan1_1.txt": text})
    record = reference_member_record(payload, "testscan1_1.txt")
    assert record["number_of_fields"] == 3
    assert record["number_of_sets"] == 2
    assert record["data_row_count"] == 2
    assert record["data_token_count"] == 6
    assert (
        record["header_fields"]["MATERIAL"]
        == "Fujichrome Velvia 100F (RVP 100F)"
    )
    bad = text.replace(b'NUMBER_OF_SETS "2"', b'NUMBER_OF_SETS "3"')
    with pytest.raises(ValueError, match="inconsistent"):
        reference_member_record(
            _zip_bytes({"testscan1_1.txt": bad}), "testscan1_1.txt"
        )


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["fit_allowed"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_config(tampered, expected_sha256=CONFIG_SHA256)
