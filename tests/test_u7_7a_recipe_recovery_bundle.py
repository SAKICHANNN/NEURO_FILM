from __future__ import annotations

import io
import json
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

import src.inference.recipe_recovery_bundle as recovery
from src.inference import (
    RecipeRecoveryBundleError,
    build_recipe_recovery_bundle,
    inspect_recipe_recovery_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
RECIPE = ROOT / "outputs/eval/u7_2_three_stock_24mp_smoke/velvia_50.recipe.json"


def _build(path: Path) -> dict:
    return build_recipe_recovery_bundle(
        recipe_path=RECIPE,
        profile_path=PROFILE,
        root=ROOT,
        bundle_path=path,
    )


def _rewrite(source: Path, destination: Path, transform) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        rows = [
            (info.filename, archive.read(info.filename)) for info in archive.infolist()
        ]
    transformed = transform(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name, payload in transformed:
                archive.writestr(name, payload)


def test_bundle_is_deterministic_strict_and_contains_no_pixels(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    one = _build(first)
    two = _build(second)
    assert first.read_bytes() == second.read_bytes()
    assert one == two
    assert one["privacy"] == {
        "includes_input_bytes": False,
        "includes_output_bytes": False,
        "network_required": False,
        "telemetry": False,
    }
    with zipfile.ZipFile(first, "r") as archive:
        names = set(archive.namelist())
    assert names == {
        "manifest.json",
        "recipe.json",
        "payload/configs/render_profiles/safe_rich_v1.json",
        "payload/configs/color_rendering_profiles.yaml",
        "payload/configs/film_color_stats.json",
        "payload/configs/color_guardrails.json",
    }
    assert inspect_recipe_recovery_bundle(first) == one


def test_builder_does_not_read_recipe_input_or_output(
    tmp_path: Path, monkeypatch
) -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    forbidden = {
        Path(recipe["input"]["path"]),
        Path(recipe["output"]["path"]),
    }
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path in forbidden:
            raise AssertionError("pixel payload was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    result = _build(tmp_path / "bundle.zip")
    assert result["style"] == "velvia_50"


def test_existing_destination_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "bundle.zip"
    path.write_bytes(b"foreign")
    with pytest.raises(RecipeRecoveryBundleError, match="already exists"):
        _build(path)
    assert path.read_bytes() == b"foreign"


def test_failed_publication_removes_partial_bundle(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bundle.zip"
    original = recovery.os.write
    called = False

    def fail_after_partial(descriptor: int, payload) -> int:
        nonlocal called
        if called:
            raise OSError("injected write failure")
        called = True
        return original(descriptor, payload[: max(1, len(payload) // 2)])

    monkeypatch.setattr(recovery.os, "write", fail_after_partial)
    with pytest.raises(OSError, match="injected"):
        _build(path)
    assert not path.exists()


@pytest.mark.parametrize("case", ["payload", "unexpected", "traversal", "duplicate"])
def test_inspector_rejects_tamper_and_archive_ambiguity(
    tmp_path: Path, case: str
) -> None:
    source = tmp_path / "source.zip"
    _build(source)
    target = tmp_path / f"{case}.zip"

    def transform(rows: list[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
        if case == "payload":
            return [
                (
                    name,
                    payload + b"x"
                    if name == "payload/configs/color_guardrails.json"
                    else payload,
                )
                for name, payload in rows
            ]
        if case == "unexpected":
            return rows + [("payload/extra.bin", b"unexpected")]
        if case == "traversal":
            return rows + [("../escape", b"forbidden")]
        return rows + [("recipe.json", b"{}")]

    _rewrite(source, target, transform)
    with pytest.raises(RecipeRecoveryBundleError):
        inspect_recipe_recovery_bundle(target)


def test_inspector_rejects_invalid_zip(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    path.write_bytes(io.BytesIO(b"not a zip").getvalue())
    with pytest.raises(RecipeRecoveryBundleError, match="valid ZIP"):
        inspect_recipe_recovery_bundle(path)


def test_cli_build_and_inspect_are_exact(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    command = [
        sys.executable,
        str(ROOT / "scripts/package_film_recipe.py"),
        "build",
        "--recipe",
        str(RECIPE),
        "--profile",
        str(PROFILE),
        "--output",
        str(bundle),
    ]
    built = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert built.returncode == 0, built.stderr
    inspected = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/package_film_recipe.py"),
            "inspect",
            "--bundle",
            str(bundle),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(built.stdout) == json.loads(inspected.stdout)
