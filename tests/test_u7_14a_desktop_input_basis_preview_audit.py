from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_u7_14a_desktop_input_basis_preview import (
    U714AError,
    _portable_manifest,
)


def test_portable_manifest_normalizes_windows_paths_structurally(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "candidate").resolve()
    manifest = {
        "rows": [
            {
                "style_id": "velvia_50",
                "output_path": str(root / "velvia_50.preview.png"),
            }
        ],
        "input_preview": {
            "role": "input_basis",
            "output_path": str(root / "input.preview.png"),
        },
    }
    portable = _portable_manifest(manifest, root)
    assert portable["rows"][0]["output_path"] == "<root>/velvia_50.preview.png"
    assert portable["input_preview"]["output_path"] == "<root>/input.preview.png"
    assert manifest["rows"][0]["output_path"] == str(root / "velvia_50.preview.png")


def test_portable_manifest_rejects_foreign_output_path(tmp_path: Path) -> None:
    root = (tmp_path / "candidate").resolve()
    manifest = {
        "rows": [
            {
                "style_id": "velvia_50",
                "output_path": str(tmp_path / "foreign.preview.png"),
            }
        ]
    }
    with pytest.raises(U714AError, match="escaped"):
        _portable_manifest(manifest, root)
