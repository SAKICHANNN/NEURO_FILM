from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_reference_chain_llvm_mingw import build


TOOLCHAIN = Path(
    r"C:\Users\hhvrf\Documents\追色\outputs\tmp\tools\llvm-mingw-20260616-ucrt-x86_64"
)
FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "reference_product_chain_conformance_v1.json"
)


def test_pinned_llvm_mingw_executes_all_product_vectors(
    tmp_path: Path,
) -> None:
    if not TOOLCHAIN.is_dir():
        pytest.skip("pinned local LLVM-MinGW is unavailable")
    executable = tmp_path / "reference_product_chain.exe"
    report = build(TOOLCHAIN, executable)
    assert report["toolchain"]["llvm_version"] == "22.1.8"
    assert report["toolchain"]["target"] == "x86_64-w64-windows-gnu"
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        for identity in case["identities"]:
            completed = subprocess.run(
                [
                    executable,
                    "hash",
                    identity["canonical_hex"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            assert completed.stdout.strip() == identity["sha256"]
    states = {
        "promoted-product": ("1", "0"),
        "research-override": ("0", "1"),
    }
    for case in fixture["cases"]:
        promoted, research = states[case["case_id"]]
        completed = subprocess.run(
            [executable, "state", "1", promoted, research],
            check=True,
            capture_output=True,
            text=True,
        )
        assert (
            completed.stdout.strip()
            == case["expected_authorization_state"]
        )
