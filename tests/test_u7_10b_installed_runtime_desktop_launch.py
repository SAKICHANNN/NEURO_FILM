from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u7_10b_installed_runtime_desktop_launch_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_desktop_contract_binds_four_generated_launcher_files() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    assert config["receipt_schema"] == "kmcfm.private-product-runtime-receipt.v2"
    assert config["launchers"] == {
        "cli": {
            "command_name": "kmcfm-look.cmd",
            "python_name": "product-launch.py",
            "entrypoint": "scripts/render_film.py",
        },
        "desktop": {
            "command_name": "kmcfm-desktop.cmd",
            "python_name": "product-desktop-launch.py",
            "entrypoint": "scripts/open_product_desktop.py",
        },
    }
    assert config["gates"]["all_four_launcher_files_receipt_bound"] is True


def test_launcher_source_binds_desktop_entrypoint_and_rejects_escape() -> None:
    requirements = ROOT / "requirements-product-v2.txt"
    source = installer._launcher_source(
        project_root=ROOT,
        source_commit="0" * 40,
        requirements_path=requirements,
        requirements_sha256=installer._sha256(requirements),
        entrypoint=Path("scripts/open_product_desktop.py"),
    )
    assert '"entrypoint": "scripts/open_product_desktop.py"' in source
    assert 'entry = root / BOUND["entrypoint"]' in source
    assert 'fail("bound entrypoint is missing")' in source
    assert '[sys.executable, "-I", str(entry)' in source
    assert 'key.upper().startswith("PYTHON")' in source
    with pytest.raises(ValueError, match="repository-relative"):
        installer._launcher_source(
            project_root=ROOT,
            source_commit="0" * 40,
            requirements_path=requirements,
            requirements_sha256=installer._sha256(requirements),
            entrypoint=Path("../foreign.py"),
        )


def test_cli_launcher_source_is_exact_historical_u7_9a_bytes(
    tmp_path: Path,
) -> None:
    historical_source = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            "5018362a17962c0c84b25484b5171fd3ccc77ed4:scripts/install_product_runtime.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    historical_path = tmp_path / "historical_installer.py"
    historical_path.write_bytes(historical_source)
    requirements = ROOT / "requirements-product-v2.txt"
    arguments = {
        "project_root": ROOT,
        "source_commit": "1" * 40,
        "requirements_path": requirements,
        "requirements_sha256": installer._sha256(requirements),
    }
    expected = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib.util,sys;"
                "p=sys.argv[1];s=importlib.util.spec_from_file_location('old',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "from pathlib import Path;"
                "sys.stdout.write(m._launcher_source(project_root=Path(sys.argv[2]),"
                "source_commit=sys.argv[3],requirements_path=Path(sys.argv[4]),"
                "requirements_sha256=sys.argv[5]))"
            ),
            str(historical_path),
            str(arguments["project_root"]),
            str(arguments["source_commit"]),
            str(arguments["requirements_path"]),
            str(arguments["requirements_sha256"]),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    assert installer._launcher_source(**arguments) == expected


def test_success_receipt_binds_both_command_and_python_launchers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "installed"
    expected_commit = "1" * 40

    monkeypatch.setattr(installer.sys, "version_info", (3, 12, 9))

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

    monkeypatch.setattr(installer, "_checked", checked)
    receipt = installer.install_product_runtime(destination)

    assert receipt["schema"] == "kmcfm.private-product-runtime-receipt.v2"
    assert receipt["launcher"] == receipt["launchers"]["cli"]["command"]
    assert set(receipt["launchers"]) == {"cli", "desktop"}
    expected = {
        "cli": ("kmcfm-look.cmd", "product-launch.py", "scripts/render_film.py"),
        "desktop": (
            "kmcfm-desktop.cmd",
            "product-desktop-launch.py",
            "scripts/open_product_desktop.py",
        ),
    }
    for role, (command_name, python_name, entrypoint) in expected.items():
        row = receipt["launchers"][role]
        command = Path(row["command"])
        python = Path(row["python"])
        assert command == destination / command_name
        assert python == destination / python_name
        assert row["command_sha256"] == _sha256(command)
        assert row["python_sha256"] == _sha256(python)
        assert row["entrypoint"] == entrypoint
        python_source = python.read_text("utf-8")
        if role == "desktop":
            assert f'"entrypoint": "{entrypoint}"' in python_source
        else:
            assert 'entry = root / "scripts" / "render_film.py"' in python_source
            assert 'BOUND["entrypoint"]' not in python_source
        command_text = command.read_text("utf-8")
        assert " -I " in command_text
        assert str(python) in command_text

    published = json.loads((destination / "product-runtime.json").read_text("utf-8"))
    assert published == receipt


def test_historical_u7_9a_installer_remains_v1_in_recorded_git_object() -> None:
    evidence = json.loads(
        (
            ROOT
            / "docs"
            / "evidence"
            / "U7_9A_PRIVATE_WINDOWS_PRODUCT_RUNTIME_INSTALL_RESULT.json"
        ).read_text("utf-8")
    )
    binding = evidence["source_bindings"]["scripts/install_product_runtime.py"]
    commit = evidence["formal_execution_commit"]
    blob = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            f"{commit}:scripts/install_product_runtime.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert blob == binding["git_blob"]
    source = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{commit}:scripts/install_product_runtime.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(source).hexdigest() == binding["sha256"]
    assert b'"schema": "kmcfm.private-product-runtime-receipt.v1"' in source
    assert b"product-desktop-launch.py" not in source
