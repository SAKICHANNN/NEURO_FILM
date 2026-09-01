from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_13a_linux_product_cli_runtime_v1.json"
SCRIPT = ROOT / "scripts/audit_u7_13a_linux_product_cli_runtime.py"
CONTRACT = ROOT / "docs/planning/U7_13A_LINUX_PRODUCT_CLI_RUNTIME_CONTRACT.md"


def _load_module():
    specification = importlib.util.spec_from_file_location("u7_13a_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_freezes_twelve_exact_linux_wheels_and_claim_ceiling() -> None:
    config = _config()
    contract = CONTRACT.read_text(encoding="utf-8")
    assert config["status"].startswith("IMPLEMENTATION_LOCKED_")
    assert "exact twelve names and versions" in contract
    assert len(config["wheels"]) == 12
    assert config["candidate"] == {
        "style_ids": ["velvia_50", "portra_400", "ektar_100"],
        "look_amount": 1.0,
        "output_format_id": "png16",
        "output_extension": ".png",
        "profile_id": "safe-rich-product-v1",
    }
    assert "Look Approximation" in config["claim_ceiling"]
    assert "calibrated stock response" in config["claim_ceiling"]
    assert "physical-film reproduction" in config["claim_ceiling"]


def test_wheel_lock_exactly_matches_product_requirements() -> None:
    module = _load_module()
    config = _config()
    rows = module._wheel_rows(config)
    expected = module._requirements(config)
    actual = {str(row[0]).casefold().replace("_", "-"): str(row[1]) for row in rows}
    assert actual == expected
    assert sum(int(row[3]) for row in rows) == 138_274_320
    assert all(
        str(row[5]).startswith("https://files.pythonhosted.org/") for row in rows
    )


def test_exact_real_source_identity_and_all_binding_paths_exist() -> None:
    config = _config()
    source = ROOT / config["source"]["path"]
    assert source.stat().st_size == config["source"]["bytes"]
    assert _sha256(source) == config["source"]["sha256"]
    for name, value in config["bindings"].items():
        if name.endswith("_path"):
            assert (ROOT / value).is_file(), name


def test_windows_to_wsl_uses_resolved_drive_and_preserves_suffix() -> None:
    module = _load_module()
    converted = module._windows_to_wsl(ROOT)
    assert converted.startswith("/mnt/")
    assert converted.endswith("/neuro_film")
    with pytest.raises(module.U713AError, match="unsupported Windows path"):
        module._windows_to_wsl(Path("//server/share/file"))


def test_recipe_normalization_removes_only_platform_specific_identity(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first = {
        "input": {"path": "C:/one/input.jpg", "sha256": "input"},
        "output": {"path": "C:/one/output.png", "sha256": "windows"},
        "render": {"look_amount": 1.0},
    }
    second = {
        "input": {"path": "/mnt/c/two/input.jpg", "sha256": "input"},
        "output": {"path": "/mnt/c/two/output.png", "sha256": "linux"},
        "render": {"look_amount": 1.0},
    }
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert module._normalized_recipe(first_path) == module._normalized_recipe(
        second_path
    )
    second["render"]["look_amount"] = 0.5
    second_path.write_text(json.dumps(second), encoding="utf-8")
    assert module._normalized_recipe(first_path) != module._normalized_recipe(
        second_path
    )


def test_wheel_verifier_rejects_size_or_hash_drift(tmp_path: Path) -> None:
    module = _load_module()
    wheel = tmp_path / "fixture.whl"
    wheel.write_bytes(b"exact wheel")
    row = (
        "fixture",
        "1.0",
        wheel.name,
        wheel.stat().st_size,
        _sha256(wheel),
        "https://files.pythonhosted.org/fixture.whl",
    )
    assert module._verify_wheel(wheel, row)
    wheel.write_bytes(b"drift")
    assert not module._verify_wheel(wheel, row)


def test_implementation_lock_matches_every_bound_git_blob() -> None:
    module = _load_module()
    identities = module._bound_identities(_config())
    assert identities
    assert all(identities.values()), identities
