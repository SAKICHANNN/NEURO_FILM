from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import install_product_runtime as installer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "u7_9c_product_runtime_startup_isolation_v1.json"


def _launcher(tmp_path: Path) -> tuple[Path, Path]:
    requirements = ROOT / "requirements-product-v2.txt"
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    path = tmp_path / "product-launch.py"
    path.write_text(
        installer._launcher_source(
            project_root=ROOT,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=installer._sha256(requirements),
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path, requirements


def test_launcher_and_renderer_are_explicitly_isolated(tmp_path: Path) -> None:
    launcher, _ = _launcher(tmp_path)
    source = launcher.read_text("utf-8")
    assert '[sys.executable, "-I", str(entry)' in source
    assert 'key.upper().startswith("PYTHON")' in source

    command = installer._launcher_source(
        project_root=ROOT,
        source_commit="0" * 40,
        requirements_path=ROOT / "requirements-product-v2.txt",
        requirements_sha256="0" * 64,
    )
    assert '[sys.executable, "-I", str(entry)' in command


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime contract")
def test_hostile_pythonpath_sitecustomize_cannot_run(tmp_path: Path) -> None:
    launcher, _ = _launcher(tmp_path)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "sitecustomize-ran.txt"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile)
    environment["PYTHONUSERBASE"] = str(hostile / "user-base")
    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), "--list-product-looks"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_id"] == "kmcfm.product-look-catalog.v1"
    assert not marker.exists()


def test_contract_preserves_claim_and_parent_runtime() -> None:
    config = json.loads(CONFIG.read_text("utf-8"))
    assert config["runtime"]["isolated_flag"] == "-I"
    assert config["runtime"]["environment_prefix_removed_case_insensitive"] == (
        "PYTHON"
    )
    assert config["unchanged"]["receipt_schema"] == (
        "kmcfm.private-product-runtime-receipt.v1"
    )
    assert config["claim"] == {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "private_windows_runtime_only": True,
        "hostile_host_guarantee": False,
        "public_release": False,
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
    }
