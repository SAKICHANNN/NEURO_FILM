from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_u7_19b_multi_vendor_raw_product_ingress as audit
from src.preprocess.raw_decode import RAW_SUFFIXES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_19b_multi_vendor_raw_product_ingress_v1.json"


def test_u7_19b_contract_is_predecode_and_extensions_are_public() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    assert config["status"] == "FROZEN_PREDECODE"
    assert config["predecode_fact"] == {
        "pixel_decodes": 0,
        "product_renders": 0,
        "media_outputs": 0,
        "reports": 0,
        "all_extensions_already_declared": True,
    }
    assert {row["extension"] for row in config["strata"]} == {
        ".3fr",
        ".fff",
        ".mos",
        ".nef",
        ".raf",
        ".x3f",
    }
    assert all(row["extension"] in RAW_SUFFIXES for row in config["strata"])
    assert sum(len(row["sources"]) for row in config["strata"]) == 12


def test_u7_19b_adapters_preserve_frozen_roles() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    for stratum in config["strata"]:
        adapted = audit._adapt_stratum(stratum)
        assert adapted["extension"] == stratum["extension"]
        assert adapted["representative_source_id"] == stratum["representative"]
        assert adapted["required_members"] == len(stratum["sources"])
        assert [row["source_id"] for row in adapted["sources"]] == [
            row["id"] for row in stratum["sources"]
        ]


def test_u7_19b_resource_gate_is_frozen() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    passing = {
        "worker_ok": True,
        "resource": {"wall_seconds": 1.0, "peak_process_rss_bytes": 1024},
    }
    assert audit._resource_gate(passing, config, product=False)
    assert audit._resource_gate(passing, config, product=True)
    failing = {
        "worker_ok": True,
        "resource": {
            "wall_seconds": config["limits"]["source_worker_seconds"] + 1,
            "peak_process_rss_bytes": 1024,
        },
    }
    assert not audit._resource_gate(failing, config, product=False)


def test_u7_19b_scientific_view_excludes_only_resource_measurements() -> None:
    report = {
        "results": [{"records": [{"source_id": "x", "resource": {"wall_seconds": 1}}]}],
        "product_records": [{"extension": ".x", "resource": {"wall_seconds": 2}}],
        "scientific_identity": "sha256:old",
    }
    view = audit._scientific_view(report)
    assert "scientific_identity" not in view
    assert view["results"][0]["records"][0] == {"source_id": "x"}
    assert view["product_records"][0] == {"extension": ".x"}


def test_u7_19b_worker_failure_is_fail_closed() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    stratum = config["strata"][0]
    gates = audit._source_gates(
        [{"worker_ok": False, "key": stratum["sources"][0]["id"]}],
        stratum,
        config,
    )
    assert not any(gates.values())


def test_u7_19b_product_worker_requires_scratch_root(tmp_path: Path) -> None:
    with pytest.raises(audit.U719BError, match="requires --key and --scratch-root"):
        # Exercise the exact parser invariant without decoding a source.
        parser_args = [
            "audit",
            "--stage",
            "product-worker",
            "--config",
            str(CONFIG),
            "--producer-repo",
            str(tmp_path),
            "--key",
            ".nef",
            "--output",
            str(tmp_path / "out.json"),
        ]
        old = audit.sys.argv
        try:
            audit.sys.argv = parser_args
            audit.main()
        finally:
            audit.sys.argv = old
