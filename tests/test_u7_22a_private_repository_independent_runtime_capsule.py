from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_private_runtime_capsule as builder
from src.inference import product_desktop
from src.inference.runtime_source_capsule import (
    CAPSULE_MANIFEST_ENV,
    CAPSULE_MANIFEST_SHA256_ENV,
    RuntimeSourceCapsuleError,
    verify_runtime_source_capsule,
)


def _identity(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _capsule(tmp_path: Path) -> tuple[Path, dict[str, str], str]:
    runtime = tmp_path / "runtime-root"
    root = runtime / "capsule"
    (root / "configs").mkdir(parents=True)
    receipt = runtime / "product-runtime.json"
    python = runtime / "runtime/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    receipt.write_bytes(b"receipt")
    python.write_bytes(b"python")
    archive = root / "product-source.zip"
    archive.write_bytes(b"archive")
    config = root / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"
    config.write_text(
        json.dumps(
            {
                "schema": "kmcfm.u7-9d-private-runtime-source-scope-binding.v1",
                "runtime_scope": ["src", "configs"],
            }
        ),
        "utf-8",
    )
    commit = "1" * 40
    manifest = {
        "schema": "kmcfm.private-runtime-source-capsule.v1",
        "source_commit": commit,
        "runtime_scope": ["src", "configs"],
        "archive": {
            "path": archive.name,
            **_identity(archive),
            "member_count": 1,
        },
        "external_files": {
            "configs/u7_9d_private_runtime_source_scope_binding_v1.json": _identity(
                config
            )
        },
        "parent_runtime": {
            "receipt": {"path": receipt.name, **_identity(receipt)},
            "python": {
                "path": "runtime/Scripts/python.exe",
                **_identity(python),
            },
        },
        "claim": {
            "calibrated_stock_response": False,
            "evidence_grade": "look-approximation",
            "mode": "film-inspired",
            "physical_film_reproduction": False,
            "public_release": False,
            "redistribution_authorized": False,
            "repository_independent_private_runtime": True,
        },
    }
    manifest_path = root / "runtime-source-capsule.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
    )
    environment = {
        CAPSULE_MANIFEST_ENV: str(manifest_path),
        CAPSULE_MANIFEST_SHA256_ENV: hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    return root, environment, commit


def test_capsule_identity_drives_product_source_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, environment, commit = _capsule(tmp_path)
    monkeypatch.setenv(CAPSULE_MANIFEST_ENV, environment[CAPSULE_MANIFEST_ENV])
    monkeypatch.setenv(
        CAPSULE_MANIFEST_SHA256_ENV, environment[CAPSULE_MANIFEST_SHA256_ENV]
    )
    assert product_desktop._source_commit(root) == commit
    scope = product_desktop._load_product_runtime_scope(root)
    assert scope == ("src", "configs")
    product_desktop._validate_runtime_source_scope(root, commit, scope)

    (root / "configs/u7_9d_private_runtime_source_scope_binding_v1.json").write_text(
        "drift", "utf-8"
    )
    with pytest.raises(product_desktop.ProductDesktopError, match="scope changed"):
        product_desktop._validate_runtime_source_scope(root, commit, scope)


def test_git_checkout_cannot_use_capsule_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CAPSULE_MANIFEST_ENV, "C:/foreign/runtime-source-capsule.json")
    monkeypatch.setenv(CAPSULE_MANIFEST_SHA256_ENV, "0" * 64)
    with pytest.raises(RuntimeSourceCapsuleError, match="Git checkout"):
        verify_runtime_source_capsule(Path(__file__).resolve().parents[1])
    assert (
        len(product_desktop._source_commit(Path(__file__).resolve().parents[1])) == 40
    )


def test_launcher_is_root_relative_and_revalidates_before_import() -> None:
    source = builder._launcher_source(
        manifest_sha256="2" * 64, entrypoint="scripts/render_film.py"
    )
    assert "C:\\Users" not in source
    assert source.index("source archive identity changed") < source.index(
        "sys.path.insert"
    )
    assert source.index("external source identity changed") < source.index(
        "runpy.run_path"
    )
    assert "root = Path(sys.argv[0]).resolve().parent" in source
    assert 'os.environ["KMCFM_PRIVATE_CAPSULE_MANIFEST"]' in source


def test_config_keeps_private_look_approximation_ceiling() -> None:
    config = json.loads(builder.CONFIG.read_text("utf-8"))
    assert config["claim"] == {
        "mode": "film-inspired",
        "evidence_grade": "look-approximation",
        "repository_independent_private_runtime": True,
        "public_release": False,
        "redistribution_authorized": False,
        "calibrated_stock_response": False,
        "physical_film_reproduction": False,
    }
    assert config["capsule"]["maximum_materialized_files"] < 100
    assert config["capsule"]["maximum_logical_bytes"] == 64 * 1024 * 1024


def test_external_git_text_materializes_with_frozen_windows_line_endings() -> None:
    assert builder._windows_worktree_text(b"a\nb\n") == b"a\r\nb\r\n"
    assert builder._windows_worktree_text(b"a\r\nb\r\n") == b"a\r\nb\r\n"
    with pytest.raises(RuntimeError, match="must be text"):
        builder._windows_worktree_text(b"a\x00b")
