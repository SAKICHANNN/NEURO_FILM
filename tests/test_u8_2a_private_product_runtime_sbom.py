from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u8_2a_private_product_runtime_sbom_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text("utf-8"))


def test_u8_2a_contract_freezes_private_claim_and_exact_runtime() -> None:
    config = _config()
    claim = config["claim_ceiling"]
    assert config["formats"] == {"cyclonedx": "1.7", "spdx": "SPDX-2.3"}
    assert config["official_schemas"]["cyclonedx"]["sha256"] == (
        "71152f97948eeeca2fd4a1434a9d29aab35d377be11828b504d029dfeeb1925a"
    )
    assert config["official_schemas"]["spdx"]["sha256"] == (
        "239208b7ac287b3cf5d9a9af23f9d69863971102a5e1587a27a398b43490b89b"
    )
    assert config["input"]["receipt_sha256"] == (
        "aa3a97686f9172de139f07edf0be784a23d86d9da85042d0cdda9bbbdb21f9c9"
    )
    assert claim["mode"] == "film-inspired"
    assert claim["evidence_grade"] == "look-approximation"
    assert claim["public_release"] is False
    assert claim["legal_clearance"] is False
    assert all(config["gates"].values())


def test_u8_2a_builder_surface_exists() -> None:
    from src.inference import product_runtime_sbom as sbom

    assert callable(sbom.collect_installed_inventory)
    assert callable(sbom.build_sbom_documents)
    assert callable(sbom.publish_sbom_documents)
    assert callable(sbom.validate_sbom_documents)


def test_u8_2a_formal_audit_binds_all_owned_sources() -> None:
    from scripts import audit_u8_2a_private_product_runtime_sbom as audit

    assert audit.CONFIG == CONFIG
    assert {path.as_posix() for path in audit.SOURCE_PATHS} == {
        "configs/u8_2a_private_product_runtime_sbom_v1.json",
        "docs/planning/U8_2A_PRIVATE_PRODUCT_RUNTIME_SBOM_CONTRACT.md",
        "scripts/audit_u8_2a_private_product_runtime_sbom.py",
        "scripts/build_private_product_runtime_sbom.py",
        "src/inference/product_runtime_sbom.py",
        "tests/test_u8_2a_private_product_runtime_sbom.py",
    }


def test_u8_2a_cross_format_documents_are_deterministic() -> None:
    from src.inference import product_runtime_sbom as sbom

    receipt = {
        "schema": "kmcfm.private-product-runtime-receipt.v3",
        "source_commit": "1" * 40,
        "requirements": {"sha256": "2" * 64},
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "public_release": False,
        },
        "distributions": {"alpha-lib": "1.2.3"},
        "python": {"implementation": "CPython", "version": "3.12.10"},
    }
    inventory = [
        {
            "name": "alpha-lib",
            "normalized_name": "alpha-lib",
            "version": "1.2.3",
            "metadata_version": "2.4",
            "license_expression": "MIT",
            "license_raw": None,
            "classifiers": [],
            "metadata_sha256": "3" * 64,
            "license_files": [],
            "dependencies": [],
        },
        {
            "name": "pip",
            "normalized_name": "pip",
            "version": "25.0.1",
            "metadata_version": "2.2",
            "license_expression": None,
            "license_raw": "MIT",
            "classifiers": ["License :: OSI Approved :: MIT License"],
            "metadata_sha256": "4" * 64,
            "license_files": [],
            "dependencies": [],
        },
    ]
    first = sbom.build_sbom_documents(
        receipt=receipt,
        receipt_sha256="5" * 64,
        inventory=inventory,
        creation_time="1980-01-01T00:00:00Z",
    )
    second = sbom.build_sbom_documents(
        receipt=receipt,
        receipt_sha256="5" * 64,
        inventory=list(reversed(inventory)),
        creation_time="1980-01-01T00:00:00Z",
    )
    assert first == second
    sbom.validate_sbom_documents(*first)
    cdx = json.loads(first[0])
    spdx = json.loads(first[1])
    assert cdx["metadata"]["component"]["licenses"] == []
    root = next(p for p in spdx["packages"] if p["name"] == "K-MCFM")
    assert root["licenseDeclared"] == "NOASSERTION"
    pip = next(p for p in spdx["packages"] if p["name"] == "pip")
    assert pip["licenseDeclared"] == "NOASSERTION"


def test_u8_2a_builder_accepts_validated_v4_repository_binding() -> None:
    from src.inference import product_runtime_sbom as sbom

    receipt = {
        "schema": "kmcfm.private-product-runtime-receipt.v4",
        "source_commit": "1" * 40,
        "requirements": {"sha256": "2" * 64},
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "public_release": False,
        },
        "distributions": {"alpha-lib": "1.2.3"},
        "python": {"implementation": "CPython", "version": "3.12.10"},
        "repository_binding": {
            "installed_source_commit": "1" * 40,
            "head_policy": "descendant",
            "runtime_scope": ["src", "configs", "scripts/render_film.py"],
            "runtime_scope_policy": "exact-to-installed-source-commit",
            "tracked_repository_policy": "clean",
            "runtime_scope_untracked_policy": "reject",
            "committed_non_runtime_drift_allowed": True,
        },
    }
    inventory = [
        {
            "name": "alpha-lib",
            "normalized_name": "alpha-lib",
            "version": "1.2.3",
            "metadata_version": "2.4",
            "license_expression": "MIT",
            "license_raw": None,
            "classifiers": [],
            "metadata_sha256": "3" * 64,
            "license_files": [],
            "dependencies": [],
        }
    ]
    documents = sbom.build_sbom_documents(
        receipt=receipt,
        receipt_sha256="5" * 64,
        inventory=inventory,
        creation_time="1980-01-01T00:00:00Z",
    )
    sbom.validate_sbom_documents(*documents)
    receipt["repository_binding"]["runtime_scope"] = ["../escape"]
    with pytest.raises(sbom.SbomValidationError, match="repository binding"):
        sbom.build_sbom_documents(
            receipt=receipt,
            receipt_sha256="5" * 64,
            inventory=inventory,
            creation_time="1980-01-01T00:00:00Z",
        )


def test_u8_2a_publish_is_create_only_and_rolls_back_owned_first_file(
    tmp_path: Path,
) -> None:
    from src.inference import product_runtime_sbom as sbom

    receipt = {
        "schema": "kmcfm.private-product-runtime-receipt.v3",
        "source_commit": "1" * 40,
        "requirements": {"sha256": "2" * 64},
        "claim": {
            "mode": "film-inspired",
            "evidence_grade": "look-approximation",
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "public_release": False,
        },
        "distributions": {"alpha-lib": "1.2.3"},
        "python": {"implementation": "CPython", "version": "3.12.10"},
    }
    inventory = [
        {
            "name": "alpha-lib",
            "normalized_name": "alpha-lib",
            "version": "1.2.3",
            "metadata_version": "2.4",
            "license_expression": "MIT",
            "license_raw": None,
            "classifiers": [],
            "metadata_sha256": "3" * 64,
            "license_files": [],
            "dependencies": [],
        }
    ]
    cdx, spdx = sbom.build_sbom_documents(
        receipt=receipt,
        receipt_sha256="5" * 64,
        inventory=inventory,
        creation_time="1980-01-01T00:00:00Z",
    )
    output = tmp_path / "output"
    output.mkdir()
    foreign = output / "runtime.spdx.json"
    foreign.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        sbom.publish_sbom_documents(
            output_directory=output,
            cyclonedx_name="runtime.cdx.json",
            spdx_name="runtime.spdx.json",
            cyclonedx_bytes=cdx,
            spdx_bytes=spdx,
        )
    assert foreign.read_bytes() == b"foreign"
    assert not (output / "runtime.cdx.json").exists()
    assert not list(output.glob(".u8-2a-*.stage"))
