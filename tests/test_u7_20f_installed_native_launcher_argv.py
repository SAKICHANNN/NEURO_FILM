from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_20f_installed_native_launcher_v1.json"


def test_native_launcher_contract_freezes_primary_and_compatibility_names() -> None:
    contract = json.loads(CONTRACT.read_text("utf-8"))
    assert contract["receipt_schema"] == "kmcfm.private-product-runtime-receipt.v3"
    assert contract["source_date_epoch"] == 315532800
    assert contract["launchers"] == {
        "cli": {
            "native_name": "kmcfm-look.exe",
            "native_source_name": "kmcfm-look.py",
            "compatibility_name": "kmcfm-look.cmd",
        },
        "desktop": {
            "native_name": "kmcfm-desktop.exe",
            "native_source_name": "kmcfm-desktop.py",
            "compatibility_name": "kmcfm-desktop.cmd",
        },
    }
    assert all(contract["gates"].values())


@pytest.mark.skipif(os.name != "nt", reason="Windows native launcher semantics")
def test_native_console_launcher_preserves_literal_windows_argv(tmp_path: Path) -> None:
    writer = getattr(installer, "_write_native_console_script", None)
    assert writer is not None, "parent installer has no native console launcher"
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = source_root / "argv-probe.py"
    source.write_text(
        "#!python -I\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:], ensure_ascii=False))\n",
        encoding="utf-8",
        newline="\n",
    )
    argument = "%TEMP% & (film) 100%.png"
    launcher = writer(
        python=Path(sys.executable),
        source=source,
        destination=target_root / "argv-probe.exe",
        source_date_epoch=315532800,
    )
    completed = subprocess.run(
        [str(launcher), argument],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [argument]


def test_runtime_receipt_promotes_native_commands_and_binds_cmd_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "installed"
    expected_commit = "2" * 40
    monkeypatch.setattr(installer.sys, "version_info", (3, 12, 10))

    def checked(runner, command, *, cwd, env):  # type: ignore[no-untyped-def]
        del runner, cwd, env
        arguments = tuple(map(str, command))
        if "rev-parse" in arguments:
            return expected_commit + "\n"
        if "status" in arguments:
            return ""
        if "venv" in arguments:
            python = destination / "runtime" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python-fixture")
            return ""
        if "-c" in arguments:
            return json.dumps(
                installer._pinned_versions(ROOT / "requirements-product-v2.txt")
            )
        return ""

    def native_writer(  # type: ignore[no-untyped-def]
        *, python, source, destination, source_date_epoch, **_kwargs
    ):
        assert Path(python).name == "python.exe"
        assert Path(source).read_text("utf-8").startswith("#!python -I\n")
        assert source_date_epoch == 315532800
        Path(destination).write_bytes(b"native-launcher-fixture")
        return Path(destination)

    monkeypatch.setattr(installer, "_checked", checked)
    monkeypatch.setattr(
        installer, "_write_native_console_script", native_writer, raising=False
    )
    receipt = installer.install_product_runtime(destination)

    assert receipt["schema"] == "kmcfm.private-product-runtime-receipt.v4"
    assert receipt["repository_binding"] == {
        "installed_source_commit": expected_commit,
        "head_policy": "descendant",
        "runtime_scope": [
            "src",
            "configs",
            "scripts/render_film.py",
            "scripts/open_product_desktop.py",
            "requirements-product-v2.txt",
        ],
        "runtime_scope_policy": "exact-to-installed-source-commit",
        "tracked_repository_policy": "clean",
        "runtime_scope_untracked_policy": "reject",
        "committed_non_runtime_drift_allowed": True,
    }
    assert receipt["launcher"] == receipt["launchers"]["cli"]["command"]
    for role, prefix in (("cli", "kmcfm-look"), ("desktop", "kmcfm-desktop")):
        row = receipt["launchers"][role]
        assert Path(row["command"]) == destination / f"{prefix}.exe"
        assert Path(row["compatibility_command"]) == destination / f"{prefix}.cmd"
        assert Path(row["native_source"]).parent.name == "native-launcher-sources"
        for key in (
            "command_sha256",
            "compatibility_command_sha256",
            "native_source_sha256",
            "python_sha256",
        ):
            assert len(row[key]) == 64
