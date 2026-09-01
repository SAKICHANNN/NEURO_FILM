from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_u7_9b_private_runtime_canon_raw_compatibility as audit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_9b_private_runtime_canon_raw_compatibility_v1.json"


def test_contract_freezes_exact_six_file_oracle_and_claim_ceiling() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    p313 = json.loads(
        (ROOT / config["bindings"]["p313_config"]["path"]).read_text("utf-8")
    )
    assert len(p313["rows"]) == config["gates"]["required_rows"] == 6
    assert set(config["p314_output_oracles"]) == {
        row["source_id"] for row in p313["rows"]
    }
    assert config["execution"] == {
        "effects": {"dust": 0.0, "grain": 0.0, "halation": 0.0},
        "formal_orders": ["forward", "reverse"],
        "look_amount": 1.0,
        "look_id": "ektar_100",
        "network_requests": 0,
        "output_bit_depth": 8,
        "profile_path": "configs/render_profiles/safe_rich_product_v1.json",
    }
    assert config["claim"] == {
        "calibrated_stock_response": False,
        "evidence_grade": "look-approximation",
        "general_raw_support": False,
        "mode": "film-inspired",
        "physical_film_reproduction": False,
        "public_release": False,
        "standalone_or_public_installer": False,
    }


def test_render_arguments_use_explicit_product_selector_and_same_output() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    source = Path("source.cr2")
    output = Path("product.png")
    arguments = audit._render_arguments(source, output, config)
    assert arguments[0] == str(source)
    assert arguments[arguments.index("--product-look") + 1] == "ektar_100"
    assert arguments[arguments.index("--look-amount") + 1] == "1.0"
    assert arguments[arguments.index("--output") + 1] == str(output)
    assert "--write-recipe" in arguments
    assert "--style" not in arguments


def test_wheelhouse_validator_rejects_missing_and_drift(tmp_path: Path) -> None:
    first = tmp_path / "a-1-py3-none-any.whl"
    second = tmp_path / "b-1-py3-none-any.whl"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    expected = [
        [first.name, first.stat().st_size, audit._sha256(first)],
        [second.name, second.stat().st_size, audit._sha256(second)],
    ]
    assert audit._wheelhouse_rows(tmp_path, expected) == expected
    second.write_bytes(b"drift")
    with pytest.raises(audit.U79BError, match="wheelhouse identity"):
        audit._wheelhouse_rows(tmp_path, expected)
    second.unlink()
    with pytest.raises(audit.U79BError, match="wheelhouse identity"):
        audit._wheelhouse_rows(tmp_path, expected)


def test_source_identity_requires_size_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.cr2"
    source.write_bytes(b"exact")
    row = {
        "bytes": source.stat().st_size,
        "sha256": audit._sha256(source),
    }
    assert audit._source_identity(source, row)
    source.write_bytes(b"drift")
    assert not audit._source_identity(source, row)
