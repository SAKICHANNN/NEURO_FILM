from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import audit_u7_10b_installed_runtime_desktop_launch as audit

ROOT = Path(__file__).resolve().parents[1]


def test_contract_and_parent_bindings_are_exact() -> None:
    config = json.loads(audit.CONFIG.read_text("utf-8"))
    assert config["source_parent_commit"] == (
        "5018362a17962c0c84b25484b5171fd3ccc77ed4"
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


def test_formal_roots_must_remain_repo_relative(tmp_path: Path) -> None:
    output = ROOT / "outputs" / "eval" / "u7_10b" / "visual.png"
    with pytest.raises(ValueError, match="repo-relative tmp"):
        audit.audit(tmp_path / "root", tmp_path, output, "forward")


def test_visual_output_must_remain_repo_relative(tmp_path: Path) -> None:
    root = ROOT / "tmp" / "u7_10b_invalid_visual"
    wheelhouse = ROOT / "tmp" / "u7_9a_wheelhouse_v1"
    with pytest.raises(ValueError, match="repo-relative outputs"):
        audit.audit(root, wheelhouse, tmp_path / "visual.png", "forward")


def test_formal_root_and_visual_are_create_only() -> None:
    wheelhouse = ROOT / "tmp" / "u7_9a_wheelhouse_v1"
    output_root = ROOT / "outputs" / "eval" / "u7_10b_create_only_test"
    output_root.mkdir(parents=True, exist_ok=True)
    visual = output_root / "visual.png"
    visual.write_bytes(b"foreign")
    root = ROOT / "tmp" / "u7_10b_create_only_test"
    with pytest.raises(FileExistsError, match="visual output"):
        audit.audit(root, wheelhouse, visual, "forward")
    assert visual.read_bytes() == b"foreign"
    visual.unlink()
    output_root.rmdir()
    root.mkdir()
    marker = root / "foreign.bin"
    marker.write_bytes(b"foreign")
    visual = ROOT / "outputs" / "eval" / "u7_10b_create_only_visual.png"
    with pytest.raises(FileExistsError, match="formal root"):
        audit.audit(root, wheelhouse, visual, "forward")
    assert marker.read_bytes() == b"foreign"
    marker.unlink()
    root.rmdir()


def test_report_publication_is_create_only_and_repo_bound() -> None:
    output_root = ROOT / "outputs" / "eval" / "u7_10b_report_test"
    output_root.mkdir(parents=True, exist_ok=True)
    report = output_root / "report.json"
    report.write_bytes(b"foreign")
    with pytest.raises(FileExistsError, match="report output"):
        audit._publish_report(report, {"status": "pass"})
    assert report.read_bytes() == b"foreign"
    report.unlink()
    output_root.rmdir()
    with pytest.raises(ValueError, match="repo-relative outputs"):
        audit._preflight_report_output(ROOT / "tmp" / "report.json")


def test_invalid_order_rejects_before_root_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sideways"):
        audit.audit(tmp_path, tmp_path, tmp_path / "visual.png", "sideways")


def test_source_git_objects_reject_uncommitted_materialization() -> None:
    commit = audit._source_commit()
    tracked = audit._source_git_objects(commit, [audit.CONFIG])
    assert (
        tracked["configs/u7_10b_installed_runtime_desktop_launch_v1.json"]["exact"]
        is True
    )
    untracked = ROOT / "tmp" / "u7_10b_untracked_source_probe.py"
    untracked.write_text("probe\n", encoding="utf-8", newline="\n")
    try:
        facts = audit._source_git_objects(commit, [untracked])
        assert facts["tmp/u7_10b_untracked_source_probe.py"]["exact"] is False
    finally:
        untracked.unlink()


def test_git_safe_environment_is_scoped_and_additive() -> None:
    source = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.autocrlf",
        "GIT_CONFIG_VALUE_0": "false",
    }
    repository = ROOT / "tmp" / "owned-fixture"
    result = audit._git_safe_environment(source, repository)
    assert source["GIT_CONFIG_COUNT"] == "1"
    assert result["GIT_CONFIG_COUNT"] == "2"
    assert result["GIT_CONFIG_KEY_0"] == "core.autocrlf"
    assert result["GIT_CONFIG_VALUE_0"] == "false"
    assert result["GIT_CONFIG_KEY_1"] == "safe.directory"
    assert result["GIT_CONFIG_VALUE_1"] == repository.as_posix()


def test_formal_uses_actual_installed_desktop_launcher_and_native_input() -> None:
    source = Path(audit.__file__).read_text("utf-8")
    assert "_desktop_interaction(" in source
    assert "cmd.exe" in source
    assert "physical_velvia_radio_click" in source
    assert "mouse_event" in source
    assert "audit_u7_10a_product_desktop_input_workflow.py" not in source
    assert "--smoke-exit-ms" in source
    assert "environment_probe.py" in source
    assert "source_commit_stable" in source
    assert "ImageGrab.grab(window=hwnd)" in source
