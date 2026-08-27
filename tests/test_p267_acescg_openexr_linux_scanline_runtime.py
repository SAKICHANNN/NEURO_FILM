from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_p267_acescg_openexr_linux_scanline_runtime import (
    _cmake_project,
    _stable_controller_payload,
    _wsl_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p267_acescg_openexr_linux_scanline_runtime_v1.json"
SOURCE = ROOT / "src/eval/p267_acescg_openexr_linux_scanline_writer.cpp"


def test_p267_config_freezes_linux_only_runtime_change() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] in {
        "METHOD_FROZEN_SOURCE_IDENTITY_PENDING",
        "FORMAL_EXECUTION_LOCKED",
    }
    assert config["probe"]["write_row_block"] == 16
    assert config["probe"]["compression"] == "ZIP_COMPRESSION"
    assert config["bindings"]["p248_expected_pixel_f32le_sha256"] == (
        "2b4890bd8fdc9af7caadccbdee56960300803a4ca7f5851fd00280ee0ea86adf"
    )
    assert config["platform"]["wsl_distribution"] == "Ubuntu-22.04"
    assert config["platform"]["formal_network_allowed"] is False
    assert config["diagnostics_not_gates"]


def test_p267_source_changes_only_platform_publication_boundary() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "writePixels(count)" in text
    assert "ZIP_COMPRESSION" in text
    assert "std::filesystem::rename" in text
    assert "MoveFileExW" not in text
    assert "windows.h" not in text
    assert "args.row_block != 16" in text


def test_p267_cmake_is_offline_and_uses_frozen_static_sources() -> None:
    text = _cmake_project("/tmp/imath", "/tmp/openexr", "/mnt/c/native.cpp")
    assert "FETCHCONTENT_FULLY_DISCONNECTED ON" in text
    assert "OPENEXR_ENABLE_THREADING OFF" in text
    assert "OpenEXR::OpenEXR" in text
    assert "/tmp/imath" in text
    assert "/tmp/openexr" in text


def test_p267_windows_paths_translate_without_shell_interpolation() -> None:
    translated = _wsl_path(
        ROOT / "configs" / "p267_acescg_openexr_linux_scanline_runtime_v1.json"
    )
    assert translated.startswith("/mnt/c/")
    assert "\\" not in translated
    assert translated.endswith(
        "/configs/p267_acescg_openexr_linux_scanline_runtime_v1.json"
    )


def test_p267_stable_payload_excludes_only_declared_runtime_diagnostics() -> None:
    base = {
        "build": {
            "binary_bytes": 1,
            "binary_sha256": "a",
            "cmake_sha256": "fixed",
        },
        "controller_order": ["a", "b"],
        "controls": {"exact": True},
        "workers": [
            {
                "file_bytes": 10,
                "file_sha256": "container-a",
                "inspection": {"decoded_pixel_f32le_sha256": "pixels"},
                "resource": {"wall_seconds": 1.0},
            }
        ],
    }
    changed = json.loads(json.dumps(base))
    changed["build"]["binary_bytes"] = 2
    changed["build"]["binary_sha256"] = "b"
    changed["controller_order"] = ["b", "a"]
    changed["workers"][0]["file_sha256"] = "container-b"
    changed["workers"][0]["resource"]["wall_seconds"] = 2.0
    assert _stable_controller_payload(base) == _stable_controller_payload(changed)
