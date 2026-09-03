from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference import product_render_transaction as transaction_module
from src.inference.product_render_transaction import (
    ProductRenderTransactionError,
    prepare_product_image_recipe_transaction,
)
from src.inference.render_contract import verify_render_recipe_files

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2m_product_image_recipe_transaction_v1.json"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> None:
    y, x = np.mgrid[:47, :61]
    pixels = np.stack(
        (
            (x * 13 + y * 7) % 251,
            (x * 3 + y * 17 + 19) % 251,
            (x * 11 + y * 5 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _command(source: Path, output: Path, *, style: str = "velvia_50") -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--write-recipe",
        "--output",
        str(output),
    ]


def _run(source: Path, output: Path, *, style: str = "velvia_50"):
    return subprocess.run(
        _command(source, output, style=style),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_current_recipe(path: Path, *, style: str) -> None:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    verify_render_recipe_files(recipe, profile_path=PRODUCT_PROFILE, root=ROOT)
    assert recipe["render"]["style"] == style
    assert recipe["claim"]["output_label"] == "film-inspired"
    assert recipe["claim"]["calibrated_reference_allowed"] is False
    assert recipe["input"]["warnings"] == [
        {
            "code": "assumed_srgb",
            "message": "no embedded ICC profile; assuming sRGB",
        }
    ]


def _stages(parent: Path) -> list[Path]:
    return [path for path in parent.iterdir() if path.name.startswith(".")]


def test_existing_recipe_rejects_before_decode_and_is_preserved(tmp_path: Path) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    recipe.write_bytes(b"foreign-recipe")
    completed = _run(missing, output)
    assert completed.returncode != 0
    assert "product recipe destination must not already exist" in completed.stderr
    assert "must_not_decode" not in completed.stderr
    assert recipe.read_bytes() == b"foreign-recipe"
    assert not output.exists()


@pytest.mark.parametrize("broken", [False, True])
def test_existing_recipe_symlink_rejects_before_decode(
    tmp_path: Path,
    broken: bool,
) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    target = tmp_path / "foreign.json"
    if not broken:
        target.write_bytes(b"foreign")
    try:
        recipe.symlink_to(target.name)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    completed = _run(missing, output)
    assert completed.returncode != 0
    assert "product recipe destination must not already exist" in completed.stderr
    assert recipe.is_symlink()
    assert not output.exists()


def test_existing_recipe_hardlink_rejects_before_decode(tmp_path: Path) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    foreign = tmp_path / "foreign.json"
    foreign.write_bytes(b"foreign")
    os.link(foreign, recipe)
    completed = _run(missing, output)
    assert completed.returncode != 0
    assert "product recipe destination must not already exist" in completed.stderr
    assert recipe.read_bytes() == b"foreign"
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point contract")
def test_existing_recipe_reparse_rejects_before_decode(tmp_path: Path) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    target = tmp_path / "junction-target"
    target.mkdir()
    command = (
        "$null=New-Item -ItemType Junction "
        f"-Path '{str(recipe).replace("'", "''")}' "
        f"-Target '{str(target).replace("'", "''")}'"
    )
    created = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr}")
    try:
        completed = _run(missing, output)
        assert completed.returncode != 0
        assert "product recipe destination must not already exist" in completed.stderr
        assert recipe.is_dir()
    finally:
        recipe.rmdir()


@pytest.mark.parametrize("style", ["velvia_50", "portra_400", "ektar_100"])
def test_pair_output_and_recipe_oracles_are_exact(
    tmp_path: Path,
    style: str,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / f"{style}.png"
    recipe = output.with_suffix(".recipe.json")
    _source(source)
    before = source.read_bytes()
    completed = _run(source, output, style=style)
    assert completed.returncode == 0, completed.stderr
    oracle = config["prechange_oracle"][style]
    assert _sha(output) == oracle["output_sha256"]
    _assert_current_recipe(recipe, style=style)
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    assert payload["output"]["path"] == str(output.resolve())
    assert payload["output"]["sha256"] == _sha(output)
    assert source.read_bytes() == before
    assert _stages(tmp_path) == []


def test_recipe_stage_failure_cleans_owned_image_stage(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    with pytest.raises(TypeError), transaction:
        transaction.image_stage.write_bytes(b"image-stage")
        transaction.bind_image_stage()
        transaction.stage_recipe({"not-json": object()})
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert _stages(tmp_path) == []


def test_late_foreign_recipe_is_preserved_and_owned_image_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == recipe:
            final.write_bytes(b"foreign-recipe")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.image_stage.write_bytes(b"image")
        transaction.bind_image_stage()
        transaction.stage_recipe({"recipe": 1})
        transaction.publish()
    assert not output.exists()
    assert recipe.read_bytes() == b"foreign-recipe"
    assert _stages(tmp_path) == []


def test_foreign_image_replacement_before_rollback_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    recipe = output.with_suffix(".recipe.json")
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == recipe:
            output.unlink()
            output.write_bytes(b"foreign-image")
            final.write_bytes(b"foreign-recipe")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.image_stage.write_bytes(b"owned-image")
        transaction.bind_image_stage()
        transaction.stage_recipe({"recipe": 1})
        transaction.publish()
    assert output.read_bytes() == b"foreign-image"
    assert recipe.read_bytes() == b"foreign-recipe"


def test_foreign_stage_replacement_is_preserved_and_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    with (
        pytest.raises(
            ProductRenderTransactionError,
            match="image stage identity",
        ),
        transaction,
    ):
        transaction.image_stage.write_bytes(b"owned-image")
        transaction.bind_image_stage()
        transaction.stage_recipe({"recipe": 1})
        transaction.image_stage.unlink()
        transaction.image_stage.write_bytes(b"foreign-stage")
        transaction.publish()
    assert transaction.image_stage.read_bytes() == b"foreign-stage"
    assert not output.exists()


@pytest.mark.parametrize("stage_name", ["image", "recipe"])
def test_in_place_stage_mutation_fails_before_publication(
    tmp_path: Path,
    stage_name: str,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    before = source.read_bytes()
    transaction = prepare_product_image_recipe_transaction(source, output)
    with (
        pytest.raises(
            ProductRenderTransactionError,
            match="content changed",
        ),
        transaction,
    ):
        transaction.image_stage.write_bytes(b"owned-image")
        transaction.bind_image_stage()
        transaction.stage_recipe({"recipe": 1})
        selected = (
            transaction.image_stage
            if stage_name == "image"
            else transaction.recipe_stage
        )
        selected.write_bytes(b"mutated-in-place")
        transaction.publish()
    assert source.read_bytes() == before
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert _stages(tmp_path) == []


def test_concurrent_product_pair_has_one_complete_winner(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / "shared.png"
    recipe = output.with_suffix(".recipe.json")
    _source(source)
    before = source.read_bytes()
    processes = [
        subprocess.Popen(
            _command(source, output),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=60) for process in processes]
    assert sorted(process.returncode for process in processes) == [0, 1]
    oracle = config["prechange_oracle"]["velvia_50"]
    assert _sha(output) == oracle["output_sha256"]
    _assert_current_recipe(recipe, style="velvia_50")
    assert source.read_bytes() == before
    assert _stages(tmp_path) == []
    failed = next(
        stderr
        for process, (_, stderr) in zip(processes, results, strict=True)
        if process.returncode != 0
    )
    assert failed
