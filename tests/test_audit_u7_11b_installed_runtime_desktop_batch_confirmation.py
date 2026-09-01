from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_u7_11b_installed_runtime_desktop_batch_confirmation.py"


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("audit_u7_11b", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u7_11b_bootstrap_is_deterministic_and_repository_bound(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first_sha = module._bootstrap(first)
    second_sha = module._bootstrap(second)
    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    assert repr(str(ROOT)).encode("utf-8") in first.read_bytes()


def test_u7_11b_artifact_projection_excludes_only_runtime_identity() -> None:
    module = _module()
    payload = {
        "child_argv0": ["python.exe"],
        "child_python_environment_keys": [[]],
        "worker_python": "python.exe",
        "members": {"batch.json": {"bytes": 1, "sha256": "0" * 64}},
        "canonical_receipt_identity_sha256": "1" * 64,
    }
    assert module._artifact_projection(payload) == {
        "members": payload["members"],
        "canonical_receipt_identity_sha256": "1" * 64,
    }


def test_u7_11b_config_separates_run_and_outer_evidence_gates() -> None:
    module = _module()
    config = module.json.loads(module.CONFIG.read_text(encoding="utf-8"))
    assert "forward_reverse_report_exact" not in config["run_gates"]
    assert config["outer_evidence_gates"] == {
        "forward_reverse_report_exact": True
    }
    for gate in (
        "direct_installed_png16_exact",
        "direct_installed_raw_recipes_exact",
        "direct_installed_raw_batch_receipt_exact",
        "installed_child_python_exact",
        "python_environment_removed",
        "source_and_owned_residue_exact",
    ):
        assert config["run_gates"][gate] is True


def test_u7_11b_frozen_run_gates_are_computed_by_runner_source() -> None:
    module = _module()
    config = module.json.loads(module.CONFIG.read_text(encoding="utf-8"))
    source = module.AUDIT.read_text(encoding="utf-8")
    for gate in config["run_gates"]:
        assert f'"{gate}"' in source


def test_u7_11b_runtime_identity_compares_resolved_paths(tmp_path: Path) -> None:
    module = _module()
    runtime = tmp_path / "runtime.exe"
    runtime.write_bytes(b"fixture")
    assert module._same_resolved_path(runtime, runtime.resolve(strict=True))


def test_u7_11b_parent_bindings_match_frozen_files() -> None:
    module = _module()
    config = module.json.loads(module.CONFIG.read_text(encoding="utf-8"))
    bindings = module._source_bindings(config)
    assert bindings
    assert all(row["exact"] for row in bindings.values())


def test_u7_11b_invalid_order_rejects_before_install() -> None:
    with pytest.raises(ValueError, match="forward or reverse"):
        _module().build_report("random")
