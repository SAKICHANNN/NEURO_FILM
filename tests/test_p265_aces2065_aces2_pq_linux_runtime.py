from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_p265_aces2065_aces2_pq_linux_runtime.py"


def _load_module():
    specification = importlib.util.spec_from_file_location("p265_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_p265_contract_and_local_wheel_identity() -> None:
    config = json.loads(
        (ROOT / "configs" / "p265_aces2065_aces2_pq_linux_runtime_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment_id"] == "P265"
    assert config["gates"]["require_exact_windows_linux_rgb16"] is True
    assert config["gates"]["require_exact_windows_linux_png"] is True
    wheel = ROOT / config["linux_wheels"]["opencolorio"][0]
    expected = config["linux_wheels"]["opencolorio"]
    assert wheel.stat().st_size == expected[1]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected[2]


def test_windows_to_wsl_uses_resolved_drive() -> None:
    module = _load_module()
    converted = module._windows_to_wsl(ROOT)
    assert converted.startswith("/mnt/")
    assert converted.endswith("/neuro_film")


def test_wheel_extraction_is_local_and_complete(tmp_path: Path) -> None:
    module = _load_module()
    wheel = tmp_path / "fixture.whl"
    import zipfile

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture/__init__.py", "VALUE = 1\n")
    destination = tmp_path / "site"
    module._extract_wheels([wheel], destination)
    assert (destination / "fixture" / "__init__.py").read_text() == "VALUE = 1\n"
