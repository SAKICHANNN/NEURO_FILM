from __future__ import annotations

import json
from pathlib import Path

from src.eval.native_density_photographic_integration import (
    _stable_toolchain,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p4ht_contract_changes_only_the_gamma_runtime_mechanism() -> None:
    slow = json.loads(
        (
            ROOT / "configs/u6_p4hr_native_density_photographic_integration_v1.json"
        ).read_text("utf-8")
    )
    fast_path = (
        ROOT / "configs/u6_p4ht_fast_native_density_photographic_integration_v1.json"
    )
    fast = load_contract(fast_path)
    assert fast["candidate"]["gamma_backend"] == "fast-hybrid-v1"
    assert fast["candidate"]["gamma_newton_iterations"] == 6
    for key in (
        "p4hq_evidence",
        "p4he_contract",
        "p4he_evidence",
        "p4hj_evidence",
        "p4hj_report",
        "profile",
    ):
        assert fast["parents"][key] == slow["parents"][key]
    assert fast["automatic_gates"] == slow["automatic_gates"]
    assert fast["measurement"] == slow["measurement"]


def test_p4ht_requires_p4hs_pass_and_p4hr_runtime_close() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p4ht_fast_native_density_photographic_integration_v1.json"
    )
    assert contract["parents"]["p4hs_evidence"]["required_decision"].startswith(
        "retain_fast"
    )
    assert contract["parents"]["p4hr_evidence"]["required_decision"].startswith(
        "close_native"
    )
    assert contract["candidate"]["cohort_refit_allowed"] is False


def test_scientific_toolchain_identity_excludes_run_location() -> None:
    first = {
        "toolchain": "msvc-x64",
        "source_sha256": "a" * 64,
        "dll_sha256": "b" * 64,
        "dll_path": "D:/run-1/kernel.dll",
        "compiler_output": "created D:/run-1/kernel.lib",
    }
    second = {
        **first,
        "dll_path": "D:/run-2/kernel.dll",
        "compiler_output": "created D:/run-2/kernel.lib",
    }
    assert _stable_toolchain(first) == _stable_toolchain(second)
    assert "dll_path" not in _stable_toolchain(first)
    assert "compiler_output" not in _stable_toolchain(first)
