from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_reference_chain_conformance import build_fixture
from build_reference_chain_msvc import VSWHERE, build


FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "reference_product_chain_conformance_v1.json"
)
SOURCE = ROOT / "native" / "reference_product_chain_conformance.cpp"
SCHEMA = (
    ROOT
    / "configs"
    / "schemas"
    / "reference_product_chain_conformance_v1.schema.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_frozen_fixture_rebuilds_exactly() -> None:
    fixture = _fixture()
    assert build_fixture() == fixture
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(fixture)


def test_python_recomputes_every_canonical_identity() -> None:
    fixture = _fixture()
    labels = []
    for case in fixture["cases"]:
        for identity in case["identities"]:
            labels.append(f"{case['case_id']}:{identity['label']}")
            assert (
                hashlib.sha256(
                    bytes.fromhex(identity["canonical_hex"])
                ).hexdigest()
                == identity["sha256"]
            )
    assert len(labels) == 10
    assert len(set(labels)) == 10


@pytest.fixture(scope="module")
def cpp_verifier(tmp_path_factory) -> Path:
    if not VSWHERE.is_file():
        pytest.skip("MSVC discovery is unavailable")
    output = (
        tmp_path_factory.mktemp("product-chain-cpp")
        / "reference_product_chain_conformance.exe"
    )
    try:
        build(SOURCE, output)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"MSVC C++17 toolchain is unavailable: {exc}")
    return output


def test_msvc_cpp17_recomputes_all_ten_identities(
    cpp_verifier: Path,
) -> None:
    for case in _fixture()["cases"]:
        for identity in case["identities"]:
            completed = subprocess.run(
                [
                    cpp_verifier,
                    "hash",
                    identity["canonical_hex"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            assert completed.stdout.strip() == identity["sha256"]


def test_cpp17_state_rule_distinguishes_research_override(
    cpp_verifier: Path,
) -> None:
    fixture = _fixture()
    for case in fixture["cases"]:
        promoted = case["case_id"] == "promoted-product"
        research = case["case_id"] == "research-override"
        completed = subprocess.run(
            [
                cpp_verifier,
                "state",
                "1",
                "1" if promoted else "0",
                "1" if research else "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert (
            completed.stdout.strip()
            == case["expected_authorization_state"]
        )


def test_cpp17_rejects_noncanonical_hex(cpp_verifier: Path) -> None:
    completed = subprocess.run(
        [cpp_verifier, "hash", "AA"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "lowercase" in completed.stderr
