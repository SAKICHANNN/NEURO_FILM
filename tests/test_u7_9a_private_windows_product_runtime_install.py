from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]


def test_existing_destination_rejects_without_change(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    destination.mkdir()
    marker = destination / "foreign.txt"
    marker.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        installer.install_product_runtime(destination)
    assert marker.read_bytes() == b"foreign"


def test_requirements_drift_rejects_before_destination(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config_target = project / installer.CONFIG_PATH.relative_to(ROOT)
    config_target.parent.mkdir(parents=True)
    config = json.loads(installer.CONFIG_PATH.read_text("utf-8"))
    config_target.write_text(json.dumps(config), encoding="utf-8")
    (project / "requirements-product-v2.txt").write_text("drift\n", encoding="utf-8")
    destination = tmp_path / "runtime"
    with pytest.raises(RuntimeError, match="requirements identity"):
        installer.install_product_runtime(destination, project_root=project)
    assert not destination.exists()


def test_failure_removes_only_still_owned_destination(tmp_path: Path) -> None:
    destination = tmp_path / "runtime"
    calls = 0

    def fail_venv(command, cwd, env):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        if calls == 2:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "injected")

    with pytest.raises(RuntimeError, match="injected"):
        installer.install_product_runtime(destination, runner=fail_venv)
    assert not destination.exists()

    displaced = tmp_path / "displaced"
    calls = 0

    def replace_then_fail(command, cwd, env):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, "abc123\n", "")
        if calls == 2:
            return subprocess.CompletedProcess(command, 0, "", "")
        if calls == 3:
            python = destination / "runtime" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fixture")
            return subprocess.CompletedProcess(command, 0, "", "")
        destination.rename(displaced)
        destination.mkdir()
        (destination / "foreign.txt").write_bytes(b"foreign")
        return subprocess.CompletedProcess(command, 1, "", "injected")

    with pytest.raises(RuntimeError, match="injected"):
        installer.install_product_runtime(destination, runner=replace_then_fail)
    assert (destination / "foreign.txt").read_bytes() == b"foreign"
    assert displaced.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher syntax")
def test_launcher_source_rejects_requirements_and_commit_drift(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_bytes(b"exact\n")
    source = installer._launcher_source(
        project_root=ROOT,
        source_commit="0" * 40,
        requirements_path=requirements,
        requirements_sha256=installer._sha256(requirements),
    )
    assert "repository commit drift" in source
    assert "product requirements identity drift" in source
    assert "--untracked-files=no" in source


def test_config_preserves_claim_ceiling() -> None:
    config = json.loads(installer.CONFIG_PATH.read_text("utf-8"))
    assert config["claim"] == {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
        "public_release": False,
    }


def test_pinned_versions_are_exact() -> None:
    pins = installer._pinned_versions(ROOT / "requirements-product-v2.txt")
    assert pins["numpy"] == "2.4.4"
    assert set(pins) == set(installer._IMPORTS)
