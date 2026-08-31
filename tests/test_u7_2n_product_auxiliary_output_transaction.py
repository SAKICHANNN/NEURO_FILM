from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import render_film as render_script
from src.filmfx.layers import FilmLayer
from src.inference import product_render_transaction as transaction_module
from src.inference.product_render_transaction import (
    ProductRenderTransactionError,
    prepare_product_render_bundle_transaction,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_film.py"
CONFIG = ROOT / "configs/u7_2n_product_auxiliary_output_transaction_v1.json"
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


def _command(
    source: Path,
    output: Path,
    *,
    style: str = "ektar_100",
    effects: bool = True,
    recipe: bool = True,
    layers: bool = True,
    metrics: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE),
        "--output",
        str(output),
    ]
    if effects:
        command.extend(["--grain", "0.05", "--halation", "0.15", "--dust", "0.02"])
    if recipe:
        command.append("--write-recipe")
    if layers:
        command.append("--write-layers")
    if metrics:
        command.append("--write-metrics")
    return command


def _run(source: Path, output: Path, **kwargs):
    return subprocess.run(
        _command(source, output, **kwargs),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _normalized_recipe_sha(path: Path) -> str:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    recipe["software"]["commit"] = "0" * 40
    recipe["input"]["path"] = "<INPUT>"
    recipe["output"]["path"] = "<OUTPUT>"
    encoded = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_metrics_sha(path: Path) -> str:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["input"] = "<INPUT>"
    metrics["output"] = "<OUTPUT>"
    metrics["render_recipe"]["path"] = "<RECIPE>"
    metrics["render_recipe"]["sha256"] = "<RECIPE_SHA256>"
    encoded = json.dumps(
        metrics,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _stages(parent: Path) -> list[Path]:
    return sorted(path for path in parent.iterdir() if path.name.startswith("."))


def _populate_transaction(source: Path, output: Path):
    transaction = prepare_product_render_bundle_transaction(
        source,
        output,
        include_recipe=True,
        include_layers=True,
        include_metrics=True,
    )
    transaction.image_stage.write_bytes(b"image")
    transaction.bind_image_stage()
    transaction.stage_recipe({"schema_id": "recipe", "output": {"path": str(output)}})
    transaction.prepare_layer_stage()
    assert transaction.layer_stage is not None
    layer = transaction.layer_stage / "grain.png"
    layer.write_bytes(b"layer")
    transaction.bind_layer_file(layer)
    transaction.bind_layer_stage()
    transaction.stage_metrics({"metric": 1})
    return transaction


@pytest.mark.parametrize("kind", ["metrics", "layers"])
def test_existing_auxiliary_rejects_before_decode_and_is_preserved(
    tmp_path: Path,
    kind: str,
) -> None:
    missing = tmp_path / "must_not_decode.png"
    output = tmp_path / "out.png"
    if kind == "metrics":
        foreign = output.with_suffix(".metrics.json")
        foreign.write_bytes(b"foreign-metrics")
    else:
        foreign = output.parent / f"{output.stem}_layers"
        foreign.mkdir()
        (foreign / "foreign.bin").write_bytes(b"foreign-layer")
    completed = _run(missing, output)
    assert completed.returncode != 0
    assert (
        f"product {'metrics' if kind == 'metrics' else 'layer root'} destination"
        in (completed.stderr)
    )
    assert "must_not_decode" not in completed.stderr
    assert not output.exists()
    if kind == "metrics":
        assert foreign.read_bytes() == b"foreign-metrics"
    else:
        assert (foreign / "foreign.bin").read_bytes() == b"foreign-layer"


def test_zero_effect_write_layers_remains_noop(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    _source(source)
    completed = _run(
        source,
        output,
        effects=False,
        recipe=False,
        layers=True,
        metrics=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert not (tmp_path / "out_layers").exists()


def test_full_bundle_oracles_are_exact(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    oracle = config["effects_oracle"]
    source = tmp_path / "source.png"
    output = tmp_path / "ektar.png"
    _source(source)
    before = source.read_bytes()
    completed = _run(source, output)
    assert completed.returncode == 0, completed.stderr
    assert _sha(output) == oracle["output_sha256"]
    assert (
        _normalized_recipe_sha(output.with_suffix(".recipe.json"))
        == oracle["normalized_recipe_sha256"]
    )
    assert (
        _normalized_metrics_sha(output.with_suffix(".metrics.json"))
        == oracle["normalized_metrics_sha256"]
    )
    layer_dir = tmp_path / "ektar_layers"
    assert {path.name: _sha(path) for path in sorted(layer_dir.iterdir())} == oracle[
        "layers"
    ]
    assert source.read_bytes() == before
    assert _stages(tmp_path) == []


@pytest.mark.parametrize("mode", ["metrics-only", "layers-only"])
def test_individual_auxiliary_modes_publish_complete(
    tmp_path: Path,
    mode: str,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "out.png"
    _source(source)
    completed = _run(
        source,
        output,
        effects=mode == "layers-only",
        recipe=False,
        layers=mode == "layers-only",
        metrics=mode == "metrics-only",
    )
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    if mode == "metrics-only":
        metrics = json.loads(
            output.with_suffix(".metrics.json").read_text(encoding="utf-8")
        )
        assert "render_recipe" not in metrics
        assert not (tmp_path / "out_layers").exists()
    else:
        assert not output.with_suffix(".metrics.json").exists()
        assert sorted(path.name for path in (tmp_path / "out_layers").iterdir()) == [
            "dust_scratch.png",
            "grain.png",
            "halation.png",
            "halation_on_black.png",
            "halation_on_white.png",
        ]
    assert _stages(tmp_path) == []


@pytest.mark.parametrize("failure_role", ["recipe", "layer", "metrics"])
def test_publication_failure_rolls_back_complete_owned_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_role: str,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    finals = {
        "recipe": output.with_suffix(".recipe.json"),
        "layer": tmp_path / "out_layers" / "grain.png",
        "metrics": output.with_suffix(".metrics.json"),
    }
    real_publish = transaction_module.publish_create_only

    def fail_selected(stage: Path, final: Path):
        if final == finals[failure_role]:
            raise OSError(f"injected {failure_role} publication failure")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", fail_selected)
    with pytest.raises(OSError, match=f"injected {failure_role}"), transaction:
        transaction.publish()
    assert source.read_bytes() == b"source"
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert not output.with_suffix(".metrics.json").exists()
    assert not (tmp_path / "out_layers").exists()
    assert _stages(tmp_path) == []


def test_late_foreign_metrics_preserved_and_earlier_bundle_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    metrics = output.with_suffix(".metrics.json")
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == metrics:
            metrics.write_bytes(b"foreign-metrics")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.publish()
    assert metrics.read_bytes() == b"foreign-metrics"
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert not (tmp_path / "out_layers").exists()


def test_foreign_image_replacement_survives_late_metrics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    metrics = output.with_suffix(".metrics.json")
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == metrics:
            output.unlink()
            output.write_bytes(b"foreign-image")
            metrics.write_bytes(b"foreign-metrics")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.publish()
    assert output.read_bytes() == b"foreign-image"
    assert metrics.read_bytes() == b"foreign-metrics"
    assert not output.with_suffix(".recipe.json").exists()
    assert not (tmp_path / "out_layers").exists()


def test_foreign_layer_addition_survives_metrics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    metrics = output.with_suffix(".metrics.json")
    layer_root = tmp_path / "out_layers"
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == metrics:
            (layer_root / "foreign.bin").write_bytes(b"foreign")
            metrics.write_bytes(b"foreign-metrics")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.publish()
    assert (layer_root / "foreign.bin").read_bytes() == b"foreign"
    assert not (layer_root / "grain.png").exists()
    assert metrics.read_bytes() == b"foreign-metrics"
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()


def test_foreign_layer_replacement_survives_metrics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    metrics = output.with_suffix(".metrics.json")
    layer_root = tmp_path / "out_layers"
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == metrics:
            owned_layer = layer_root / "grain.png"
            owned_layer.unlink()
            owned_layer.write_bytes(b"foreign-layer")
            metrics.write_bytes(b"foreign-metrics")
        return real_publish(stage, final)

    monkeypatch.setattr(transaction_module, "publish_create_only", inject)
    with pytest.raises(FileExistsError), transaction:
        transaction.publish()
    assert (layer_root / "grain.png").read_bytes() == b"foreign-layer"
    assert metrics.read_bytes() == b"foreign-metrics"
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()


def test_second_layer_encode_failure_cleans_first_bound_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = prepare_product_render_bundle_transaction(
        source,
        output,
        include_recipe=False,
        include_layers=True,
        include_metrics=False,
    )
    pixels = np.zeros((5, 7, 3), dtype=np.float32)
    layers = [
        FilmLayer(name="first", mode="residual", residual=pixels),
        FilmLayer(name="second", mode="residual", residual=pixels),
    ]
    real_save = render_script.save_rgb
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second layer encode failure")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(render_script, "save_rgb", fail_second)
    with pytest.raises(OSError, match="second layer encode"), transaction:
        transaction.prepare_layer_stage()
        assert transaction.layer_stage is not None
        render_script._write_layer_outputs(
            layers,
            transaction.layer_stage,
            create_only=True,
            on_written=transaction.bind_layer_file,
        )
    assert _stages(tmp_path) == []
    assert not output.exists()


def test_layer_manifest_drift_cleans_only_bound_owned_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = prepare_product_render_bundle_transaction(
        source,
        output,
        include_recipe=False,
        include_layers=True,
        include_metrics=False,
    )
    with (
        pytest.raises(ProductRenderTransactionError, match="manifest changed"),
        transaction,
    ):
        transaction.image_stage.write_bytes(b"image")
        transaction.bind_image_stage()
        transaction.prepare_layer_stage()
        assert transaction.layer_stage is not None
        owned = transaction.layer_stage / "grain.png"
        owned.write_bytes(b"owned")
        transaction.bind_layer_file(owned)
        foreign = transaction.layer_stage / "foreign"
        foreign.mkdir()
        transaction.bind_layer_stage()
    assert not owned.exists()
    assert foreign.is_dir()
    assert transaction.layer_stage.is_dir()
    assert not output.exists()


@pytest.mark.parametrize("role", ["metrics", "layer"])
def test_auxiliary_stage_content_drift_rejects_before_publication(
    tmp_path: Path,
    role: str,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = _populate_transaction(source, output)
    if role == "metrics":
        assert transaction.metrics_stage is not None
        selected = transaction.metrics_stage
    else:
        assert transaction.layer_stage is not None
        selected = transaction.layer_stage / "grain.png"
    with (
        pytest.raises(ProductRenderTransactionError, match="content changed"),
        transaction,
    ):
        selected.write_bytes(b"mutated-in-place")
        transaction.publish()
    assert not output.exists()
    assert not output.with_suffix(".recipe.json").exists()
    assert not output.with_suffix(".metrics.json").exists()
    assert not (tmp_path / "out_layers").exists()
    assert _stages(tmp_path) == []


def test_input_metrics_alias_rejects_before_decode(tmp_path: Path) -> None:
    output = tmp_path / "out.png"
    source = output.with_suffix(".metrics.json")
    with pytest.raises(ProductRenderTransactionError, match="must be distinct"):
        prepare_product_render_bundle_transaction(
            source,
            output,
            include_recipe=False,
            include_layers=False,
            include_metrics=True,
        )


@pytest.mark.parametrize("role", ["recipe", "metrics"])
def test_json_stage_fsync_failure_leaves_no_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out.png"
    source.write_bytes(b"source")
    transaction = prepare_product_render_bundle_transaction(
        source,
        output,
        include_recipe=role == "recipe",
        include_layers=False,
        include_metrics=role == "metrics",
    )

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(f"injected {role} fsync failure")

    monkeypatch.setattr(transaction_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match=f"injected {role}"), transaction:
        if role == "recipe":
            transaction.stage_recipe({"recipe": 1})
        else:
            transaction.stage_metrics({"metric": 1})
    assert _stages(tmp_path) == []
    assert not output.exists()


def test_concurrent_full_bundle_has_one_complete_winner(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = tmp_path / "source.png"
    output = tmp_path / "shared.png"
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
    oracle = config["effects_oracle"]
    assert _sha(output) == oracle["output_sha256"]
    assert (
        _normalized_recipe_sha(output.with_suffix(".recipe.json"))
        == oracle["normalized_recipe_sha256"]
    )
    assert (
        _normalized_metrics_sha(output.with_suffix(".metrics.json"))
        == oracle["normalized_metrics_sha256"]
    )
    assert {
        path.name: _sha(path) for path in sorted((tmp_path / "shared_layers").iterdir())
    } == oracle["layers"]
    assert source.read_bytes() == before
    assert _stages(tmp_path) == []
    failed = next(
        stderr
        for process, (_stdout, stderr) in zip(processes, results, strict=True)
        if process.returncode != 0
    )
    assert "destination must not already exist" in failed or "FileExistsError" in failed
