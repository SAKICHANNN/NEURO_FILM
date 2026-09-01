from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import audit_u7_10c_installed_desktop_export as audit

ROOT = Path(__file__).resolve().parents[1]


def test_contract_and_parent_bindings_are_exact() -> None:
    config = json.loads(audit.CONFIG.read_text("utf-8"))
    assert config["source_parent_commit"] == (
        "4fe46002015b7b63097b2bb310cee178f74221e2"
    )
    for row in config["parent_bindings"].values():
        payload = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{config['source_parent_commit']}:{row['path']}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        assert audit.hashlib.sha256(payload).hexdigest() == row["sha256"]
    assert config["visual_lock"]["expected_screenshot_sha256"] == (
        "bddfe2ea290e00ff5d56e8a81484e3b75ba92bc27ac28304d2fc5252736cbf44"
    )
    assert config["visual_lock"]["manual_preflight_review"][
        "final_recipe_name_visible"
    ] == "installed-desktop-velvia.recipe.json"


def test_invalid_order_rejects_before_root_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sideways"):
        audit.audit(tmp_path, tmp_path, tmp_path / "visual.png", "sideways")


def test_formal_roots_must_remain_repo_relative(tmp_path: Path) -> None:
    output = ROOT / "outputs" / "eval" / "u7_10c" / "visual.png"
    with pytest.raises(ValueError, match="repo-relative tmp"):
        audit.audit(tmp_path / "root", tmp_path, output, "forward")


def test_visual_output_must_remain_repo_relative(tmp_path: Path) -> None:
    root = ROOT / "tmp" / "u7_10c_invalid_visual"
    wheelhouse = ROOT / "tmp" / "u7_9a_wheelhouse_v1"
    with pytest.raises(ValueError, match="repo-relative outputs"):
        audit.audit(root, wheelhouse, tmp_path / "visual.png", "forward")


def test_formal_root_and_visual_are_create_only() -> None:
    wheelhouse = ROOT / "tmp" / "u7_9a_wheelhouse_v1"
    output_root = ROOT / "outputs" / "eval" / "u7_10c_create_only_test"
    output_root.mkdir(parents=True, exist_ok=True)
    visual = output_root / "visual.png"
    visual.write_bytes(b"foreign")
    root = ROOT / "tmp" / "u7_10c_create_only_test"
    with pytest.raises(FileExistsError, match="visual output"):
        audit.audit(root, wheelhouse, visual, "forward")
    assert visual.read_bytes() == b"foreign"
    visual.unlink()
    output_root.rmdir()
    root.mkdir()
    marker = root / "foreign.bin"
    marker.write_bytes(b"foreign")
    visual = ROOT / "outputs" / "eval" / "u7_10c_create_only_visual.png"
    with pytest.raises(FileExistsError, match="formal root"):
        audit.audit(root, wheelhouse, visual, "forward")
    assert marker.read_bytes() == b"foreign"
    marker.unlink()
    root.rmdir()


def test_png16_rgb_header_is_strict(tmp_path: Path) -> None:
    valid = bytearray(29)
    valid[:8] = b"\x89PNG\r\n\x1a\n"
    valid[12:16] = b"IHDR"
    valid[24] = 16
    valid[25] = 2
    path = tmp_path / "valid.png"
    path.write_bytes(valid)
    assert audit._png16_rgb(path)
    valid[24] = 8
    path.write_bytes(valid)
    assert not audit._png16_rgb(path)


def test_recipe_semantics_require_bounded_product_claim(tmp_path: Path) -> None:
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    source.write_bytes(b"input")
    output.write_bytes(b"output")
    recipe = {
        "input": {
            "path": str(source.resolve(strict=False)),
            "sha256": audit.parent._sha256(source),
        },
        "output": {
            "path": str(output.resolve(strict=False)),
            "sha256": audit.parent._sha256(output),
            "bit_depth": 16,
        },
        "render": {"style": "velvia_50", "look_amount": 1.0},
        "claim": {
            "evidence_grade": "look-approximation",
            "calibrated_reference_allowed": False,
        },
        "software": {"commit": "a" * 40},
    }
    profile = json.loads(audit.PROFILE.read_text("utf-8"))
    recipe["profile"] = {
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "sha256": audit.parent._sha256(audit.PROFILE),
    }
    assert audit._recipe_semantics(
        recipe, source=source, output=output, source_commit="a" * 40
    )
    recipe["claim"]["calibrated_reference_allowed"] = True
    assert not audit._recipe_semantics(
        recipe, source=source, output=output, source_commit="a" * 40
    )


def test_formal_binds_real_dialog_to_owner_and_process() -> None:
    source = Path(audit.__file__).read_text("utf-8")
    assert "GetWindowThreadProcessId" in source
    assert "GW_OWNER" in source
    assert "owner in _owner_chain(hwnd)" in source
    assert "foreground_alt_n_unicode_sendinput" in source
    assert "SendInput" in source
    assert "_physical_dialog_button_click" in source
    assert "native_save_dialog_accept" in source
    assert "replay_style_safe_recipe_to_file" in source
    assert "build_product_desktop_app" not in source


def test_source_git_objects_reject_uncommitted_materialization() -> None:
    commit = audit.parent._source_commit()
    tracked = audit.parent._source_git_objects(commit, [audit.CONFIG])
    assert tracked["configs/u7_10c_installed_desktop_export_v1.json"]["exact"]
    untracked = ROOT / "tmp" / "u7_10c_untracked_source_probe.py"
    untracked.write_text("probe\n", encoding="utf-8", newline="\n")
    try:
        facts = audit.parent._source_git_objects(commit, [untracked])
        assert facts["tmp/u7_10c_untracked_source_probe.py"]["exact"] is False
    finally:
        untracked.unlink()
