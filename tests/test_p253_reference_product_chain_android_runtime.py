from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.audit_p253_reference_product_chain_android_runtime as audit
from scripts.audit_p253_reference_product_chain_android_runtime import (
    P253RuntimeError,
    _canonical_bytes,
    _create_avd,
    _fixture_rows,
    _truth_table_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "configs/p253_reference_product_chain_android_runtime_v1.json").read_text(
        encoding="utf-8"
    )
)
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/reference_product_chain_conformance_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_orders_are_exact_reversals_with_same_canonical_rows() -> None:
    normal_fixture = _fixture_rows(FIXTURE, "normal")
    reverse_fixture = _fixture_rows(FIXTURE, "reverse")
    assert normal_fixture == list(reversed(reverse_fixture))
    assert sorted(normal_fixture) == sorted(reverse_fixture)
    normal_truth = _truth_table_rows(CONTRACT, "normal")
    reverse_truth = _truth_table_rows(CONTRACT, "reverse")
    assert normal_truth == list(reversed(reverse_truth))
    assert sorted(normal_truth, key=json.dumps) == sorted(reverse_truth, key=json.dumps)


def test_contract_freezes_complete_truth_table_and_private_ceiling() -> None:
    assert len(CONTRACT["truth_table"]) == 8
    assert (
        sum(
            row["expected"] == "authorized-for-staging"
            for row in CONTRACT["truth_table"]
        )
        == 1
    )
    assert CONTRACT["runtime"]["api_level"] == 34
    assert CONTRACT["runtime"]["abi"] == "x86_64"
    assert CONTRACT["runtime"]["runtime_library"] == {
        "ndk_relative_path": (
            "toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/"
            "x86_64-linux-android/libc++_shared.so"
        ),
        "bytes": 1_617_608,
        "sha256": ("3d8817f3a50515665b9152c620dedfd8509f84b68307f5f7ef20a75d9ae08f69"),
    }
    assert "arm64-v8a link-only" in CONTRACT["claim_ceiling"]
    assert CONTRACT["decision_if_pass"].startswith("PASS_PRIVATE_")


def test_unknown_order_fails_closed() -> None:
    with pytest.raises(P253RuntimeError, match="unsupported enumeration order"):
        _fixture_rows(FIXTURE, "random")
    with pytest.raises(P253RuntimeError, match="unsupported enumeration order"):
        _truth_table_rows(CONTRACT, "random")


def test_owned_minimal_avd_is_project_isolated(tmp_path: Path) -> None:
    sdk = tmp_path / "sdk"
    avd_home = tmp_path / "owned-avd"
    _create_avd(avd_home, sdk, CONTRACT)
    name = CONTRACT["runtime"]["avd_name"]
    pointer = (avd_home / f"{name}.ini").read_text(encoding="utf-8")
    config = (avd_home / f"{name}.avd/config.ini").read_text(encoding="utf-8")
    assert f"path={avd_home.resolve()}\\{name}.avd" in pointer
    assert "target=android-34" in pointer
    assert "abi.type = x86_64" in config
    assert "image.sysdir.1 = system-images\\android-34\\google_apis\\x86_64\\" in config


def test_large_canonical_hex_is_transferred_outside_host_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_command(arguments, **_kwargs):
        calls.append([str(value) for value in arguments])
        return subprocess.CompletedProcess(calls[-1], 0, "digest\n", "")

    monkeypatch.setattr(audit, "_command", fake_command)
    canonical_hex = "ab" * 20_000
    result = audit._adb_hash_file(
        Path("adb"),
        "emulator-5584",
        "/data/local/tmp/probe",
        canonical_hex,
        environment={},
        work=tmp_path,
    )
    assert result.stdout == "digest\n"
    assert (tmp_path / "canonical.hex").read_text(encoding="ascii") == canonical_hex
    assert all(canonical_hex not in " ".join(call) for call in calls)
    assert "$(cat /data/local/tmp/nf_p253_canonical.hex)" in calls[-1][-1]


def test_report_serialization_is_order_independent_after_sorting() -> None:
    normal = {
        "identities": sorted(_fixture_rows(FIXTURE, "normal")),
        "truth": sorted(_truth_table_rows(CONTRACT, "normal"), key=json.dumps),
    }
    reverse = {
        "truth": sorted(_truth_table_rows(CONTRACT, "reverse"), key=json.dumps),
        "identities": sorted(_fixture_rows(FIXTURE, "reverse")),
    }
    assert _canonical_bytes(normal) == _canonical_bytes(reverse)
