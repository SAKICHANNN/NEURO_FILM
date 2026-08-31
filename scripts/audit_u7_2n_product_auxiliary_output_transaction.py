"""Formal audit for the U7.2N product auxiliary-output transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_film as render_script
from src.filmfx.layers import FilmLayer
from src.inference import product_render_transaction as transaction_module
from src.inference.product_render_transaction import (
    ProductRenderTransactionError,
    prepare_product_render_bundle_transaction,
)

SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT_COMMIT = "21e1ca45"
IMPLEMENTATION_COMMIT = "996faf4c"
IMPLEMENTATION_PATHS = (
    "scripts/render_film.py",
    "src/inference/product_render_transaction.py",
    "tests/test_u7_2n_product_auxiliary_output_transaction.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


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
    effects: bool = False,
    recipe: bool = False,
    layers: bool = False,
    metrics: bool = False,
    product: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        str(source),
        "--style",
        style,
        "--use-render-profile",
        "--render-profile",
        str(PRODUCT_PROFILE if product else LEGACY_PROFILE),
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


def _run(source: Path, output: Path, **kwargs) -> subprocess.CompletedProcess[str]:
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


def _hidden(directory: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob(".*")
    )


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


def _publication_failures(work: Path) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for role in ("recipe", "layer", "metrics"):
        source = work / f"publish-{role}-source"
        output = work / f"publish-{role}.png"
        source.write_bytes(b"source")
        transaction = _populate_transaction(source, output)
        finals = {
            "recipe": output.with_suffix(".recipe.json"),
            "layer": work / f"publish-{role}_layers" / "grain.png",
            "metrics": output.with_suffix(".metrics.json"),
        }
        real_publish = transaction_module.publish_create_only

        def fail_selected(
            stage: Path,
            final: Path,
            *,
            target: Path = finals[role],
            label: str = role,
            publish=real_publish,
        ):
            if final == target:
                raise OSError(f"injected {label}")
            return publish(stage, final)

        passed = False
        with patch.object(transaction_module, "publish_create_only", fail_selected):
            try:
                with transaction:
                    transaction.publish()
            except OSError as exc:
                passed = str(exc) == f"injected {role}"
        gates[f"{role}_publication_failure_residue_zero"] = (
            passed
            and not output.exists()
            and not output.with_suffix(".recipe.json").exists()
            and not output.with_suffix(".metrics.json").exists()
            and not (work / f"publish-{role}_layers").exists()
        )
    return gates


def _json_stage_failures(work: Path) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for role in ("recipe", "metrics"):
        source = work / f"fsync-{role}-source"
        output = work / f"fsync-{role}.png"
        source.write_bytes(b"source")
        transaction = prepare_product_render_bundle_transaction(
            source,
            output,
            include_recipe=role == "recipe",
            include_layers=False,
            include_metrics=role == "metrics",
        )

        def fail_fsync(_descriptor: int, *, label: str = role) -> None:
            raise OSError(f"injected {label} fsync")

        passed = False
        with patch.object(transaction_module.os, "fsync", fail_fsync):
            try:
                with transaction:
                    if role == "recipe":
                        transaction.stage_recipe({"recipe": 1})
                    else:
                        transaction.stage_metrics({"metric": 1})
            except OSError as exc:
                passed = str(exc) == f"injected {role} fsync"
        gates[f"{role}_stage_failure_residue_zero"] = passed and not _hidden(work)
    return gates


def _partial_layer_failure(work: Path) -> bool:
    source = work / "partial-layer-source"
    output = work / "partial-layer.png"
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
            raise OSError("injected second layer encode")
        return real_save(*args, **kwargs)

    passed = False
    with patch.object(render_script, "save_rgb", fail_second):
        try:
            with transaction:
                transaction.prepare_layer_stage()
                assert transaction.layer_stage is not None
                render_script._write_layer_outputs(
                    layers,
                    transaction.layer_stage,
                    create_only=True,
                    on_written=transaction.bind_layer_file,
                )
        except OSError as exc:
            passed = str(exc) == "injected second layer encode"
    return passed and not _hidden(work) and not output.exists()


def _foreign_gates(work: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for mode in ("metrics", "image", "layer-addition", "layer-replacement"):
        source = work / f"foreign-{mode}-source"
        output = work / f"foreign-{mode}.png"
        recipe = output.with_suffix(".recipe.json")
        metrics = output.with_suffix(".metrics.json")
        layer_root = work / f"foreign-{mode}_layers"
        source.write_bytes(b"source")
        transaction = _populate_transaction(source, output)
        real_publish = transaction_module.publish_create_only

        def inject(
            stage: Path,
            final: Path,
            *,
            metrics_path: Path = metrics,
            mode_label: str = mode,
            output_path: Path = output,
            layers_path: Path = layer_root,
            publish=real_publish,
        ):
            if final == metrics_path:
                if mode_label == "image":
                    output_path.unlink()
                    output_path.write_bytes(b"foreign-image")
                elif mode_label == "layer-addition":
                    (layers_path / "foreign.bin").write_bytes(b"foreign-layer")
                elif mode_label == "layer-replacement":
                    owned = layers_path / "grain.png"
                    owned.unlink()
                    owned.write_bytes(b"foreign-layer")
                metrics_path.write_bytes(b"foreign-metrics")
            return publish(stage, final)

        failed = False
        with patch.object(transaction_module, "publish_create_only", inject):
            try:
                with transaction:
                    transaction.publish()
            except FileExistsError:
                failed = True
        if mode == "metrics":
            preserved = metrics.read_bytes() == b"foreign-metrics"
        elif mode == "image":
            preserved = (
                output.read_bytes() == b"foreign-image"
                and metrics.read_bytes() == b"foreign-metrics"
            )
        else:
            preserved = (
                (
                    layer_root
                    / ("foreign.bin" if mode == "layer-addition" else "grain.png")
                ).read_bytes()
                == b"foreign-layer"
                and metrics.read_bytes() == b"foreign-metrics"
            )
        results[f"foreign_{mode.replace('-', '_')}_preserved"] = (
            failed
            and preserved
            and (mode == "image" or not output.exists())
            and not recipe.exists()
        )
        if output.exists():
            output.unlink()
        if metrics.exists():
            metrics.unlink()
        if layer_root.exists():
            shutil.rmtree(layer_root)
    return results


def _stage_drift_gates(work: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for role in ("metrics", "layer"):
        source = work / f"mutation-{role}-source"
        output = work / f"mutation-{role}.png"
        source.write_bytes(b"source")
        transaction = _populate_transaction(source, output)
        selected = (
            transaction.metrics_stage
            if role == "metrics"
            else transaction.layer_stage / "grain.png"
        )
        assert selected is not None
        passed = False
        try:
            with transaction:
                selected.write_bytes(b"mutated")
                transaction.publish()
        except ProductRenderTransactionError as exc:
            passed = "content changed" in str(exc)
        results[f"{role}_stage_mutation_rejected"] = (
            passed
            and not output.exists()
            and not output.with_suffix(".recipe.json").exists()
            and not output.with_suffix(".metrics.json").exists()
            and not (work / f"mutation-{role}_layers").exists()
        )
    return results


def _existing_entry_gates(work: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for role in ("metrics", "layer root"):
        output = work / f"existing-{role.replace(' ', '-')}.png"
        missing = work / "must-not-decode.png"
        if role == "metrics":
            foreign = output.with_suffix(".metrics.json")
            foreign.write_bytes(b"foreign")
        else:
            foreign = work / f"{output.stem}_layers"
            foreign.mkdir()
            (foreign / "foreign.bin").write_bytes(b"foreign")
        completed = _run(
            missing,
            output,
            effects=role == "layer root",
            recipe=True,
            layers=role == "layer root",
            metrics=role == "metrics",
        )
        results[f"existing_{role.replace(' ', '_')}_predecode_preserved"] = (
            completed.returncode != 0
            and f"product {role} destination must not already exist" in completed.stderr
            and not output.exists()
            and (
                foreign.read_bytes() == b"foreign"
                if role == "metrics"
                else (foreign / "foreign.bin").read_bytes() == b"foreign"
            )
        )
    return results


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="neuro-film-u7-2n-",
        dir=scratch_root,
    ) as raw_work:
        work = Path(raw_work)
        source = work / "source.png"
        _source(source)
        source_before = source.read_bytes()
        pair_results: dict[str, dict[str, Any]] = {}
        parent = json.loads(
            (ROOT / "configs/u7_2m_product_image_recipe_transaction_v1.json").read_text(
                encoding="utf-8"
            )
        )
        for style in order:
            output = work / f"pair-{style}.png"
            completed = _run(source, output, style=style, recipe=True)
            pair_results[style] = {
                "returncode": completed.returncode,
                "output_sha256": _sha256(output) if output.is_file() else None,
                "normalized_recipe_sha256": (
                    _normalized_recipe_sha(output.with_suffix(".recipe.json"))
                    if output.with_suffix(".recipe.json").is_file()
                    else None
                ),
            }
        pair_exact = all(
            pair_results[style]["returncode"] == 0
            and pair_results[style]["output_sha256"]
            == parent["prechange_oracle"][style]["output_sha256"]
            and pair_results[style]["normalized_recipe_sha256"]
            == parent["prechange_oracle"][style]["normalized_recipe_sha256"]
            for style in config["available_product_looks"]
        )

        bundle = work / "bundle.png"
        bundle_run = _run(
            source,
            bundle,
            effects=True,
            recipe=True,
            layers=True,
            metrics=True,
        )
        oracle = config["effects_oracle"]
        layer_root = work / "bundle_layers"
        bundle_exact = (
            bundle_run.returncode == 0
            and _sha256(bundle) == oracle["output_sha256"]
            and _normalized_recipe_sha(bundle.with_suffix(".recipe.json"))
            == oracle["normalized_recipe_sha256"]
            and _normalized_metrics_sha(bundle.with_suffix(".metrics.json"))
            == oracle["normalized_metrics_sha256"]
            and {path.name: _sha256(path) for path in sorted(layer_root.iterdir())}
            == oracle["layers"]
        )

        no_effect = work / "no-effect.png"
        no_effect_run = _run(
            source,
            no_effect,
            layers=True,
        )
        metrics_only = work / "metrics-only.png"
        metrics_only_run = _run(source, metrics_only, metrics=True)
        layers_only = work / "layers-only.png"
        layers_only_run = _run(source, layers_only, effects=True, layers=True)

        concurrent = work / "concurrent.png"
        processes = [
            subprocess.Popen(
                _command(
                    source,
                    concurrent,
                    effects=True,
                    recipe=True,
                    layers=True,
                    metrics=True,
                ),
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        for process in processes:
            process.communicate(timeout=60)
        concurrent_codes = sorted(int(process.returncode) for process in processes)
        concurrent_exact = (
            concurrent_codes == [0, 1]
            and _sha256(concurrent) == oracle["output_sha256"]
            and _normalized_recipe_sha(concurrent.with_suffix(".recipe.json"))
            == oracle["normalized_recipe_sha256"]
            and _normalized_metrics_sha(concurrent.with_suffix(".metrics.json"))
            == oracle["normalized_metrics_sha256"]
            and {
                path.name: _sha256(path)
                for path in sorted((work / "concurrent_layers").iterdir())
            }
            == oracle["layers"]
        )

        image_only = work / "image-only.png"
        image_only_run = _run(source, image_only, style="velvia_50")
        legacy = work / "legacy.png"
        legacy.write_bytes(b"replace-me")
        legacy_run = _run(
            source,
            legacy,
            style="velvia_50",
            product=False,
        )

        prechange_commit = subprocess.check_output(
            ["git", "rev-parse", f"{CONTRACT_COMMIT}^"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        source_locks = config["source_locks"]
        source_locks_exact = all(
            hashlib.sha256(_git_blob(prechange_commit, path)).hexdigest()
            == source_locks[key]
            for key, path in (
                ("render_film_sha256", "scripts/render_film.py"),
                (
                    "product_render_transaction_sha256",
                    "src/inference/product_render_transaction.py",
                ),
                (
                    "create_only_file_sha256",
                    "src/film_physics/create_only_file.py",
                ),
                (
                    "product_profile_sha256",
                    "configs/render_profiles/safe_rich_product_v1.json",
                ),
                (
                    "u7_2m_config_sha256",
                    "configs/u7_2m_product_image_recipe_transaction_v1.json",
                ),
                (
                    "u7_2m_evidence_sha256",
                    "docs/evidence/U7_2M_PRODUCT_IMAGE_RECIPE_TRANSACTION_RESULT.json",
                ),
            )
        )
        implementation_exact = all(
            hashlib.sha256(_git_blob(IMPLEMENTATION_COMMIT, path)).hexdigest()
            == _sha256(ROOT / path)
            for path in IMPLEMENTATION_PATHS
        )
        gates = {
            "prechange_source_locks_exact": source_locks_exact,
            "implementation_files_exact": implementation_exact,
            "fixture_exact": hashlib.sha256(source_before).hexdigest()
            == config["fixture"]["sha256"],
            **_existing_entry_gates(work),
            "three_parent_pair_oracles_exact": pair_exact,
            "full_effects_bundle_oracles_exact": bundle_exact,
            "zero_effect_write_layers_noop": no_effect_run.returncode == 0
            and not (work / "no-effect_layers").exists(),
            "metrics_only_complete": metrics_only_run.returncode == 0
            and metrics_only.with_suffix(".metrics.json").is_file()
            and not metrics_only.with_suffix(".recipe.json").exists(),
            "layers_only_complete": layers_only_run.returncode == 0
            and (work / "layers-only_layers").is_dir()
            and not layers_only.with_suffix(".recipe.json").exists()
            and not layers_only.with_suffix(".metrics.json").exists(),
            **_publication_failures(work),
            **_json_stage_failures(work),
            "partial_layer_encode_failure_residue_zero": _partial_layer_failure(work),
            **_foreign_gates(work),
            **_stage_drift_gates(work),
            "concurrent_exactly_one_complete_bundle": concurrent_exact,
            "source_immutable": source.read_bytes() == source_before,
            "product_image_only_unchanged": image_only_run.returncode == 0
            and _sha256(image_only)
            == parent["prechange_oracle"]["velvia_50"]["output_sha256"],
            "legacy_replace_behavior_unchanged": legacy_run.returncode == 0
            and _sha256(legacy)
            == parent["prechange_oracle"]["velvia_50"]["output_sha256"],
            "owned_stage_residue_zero": not _hidden(work),
        }
        report = {
            "schema_id": (
                "neuro-film.u7-2n-product-auxiliary-output-transaction-result.v1"
            ),
            "experiment_id": "U7.2N",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "bindings": {
                "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "contract_commit": CONTRACT_COMMIT,
                "implementation_commit": IMPLEMENTATION_COMMIT,
            },
            "pair_results": dict(sorted(pair_results.items())),
            "full_bundle": {
                "output_sha256": _sha256(bundle),
                "normalized_recipe_sha256": _normalized_recipe_sha(
                    bundle.with_suffix(".recipe.json")
                ),
                "normalized_metrics_sha256": _normalized_metrics_sha(
                    bundle.with_suffix(".metrics.json")
                ),
                "layer_sha256": {
                    path.name: _sha256(path) for path in sorted(layer_root.iterdir())
                },
            },
            "concurrent_returncodes": concurrent_codes,
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Enable create-only process-level publication and identity-safe "
                "rollback for all explicitly requested safe-rich-product-v1 "
                "image, recipe, layer and metrics artifacts."
            ),
            "claim_ceiling": config["claim_ceiling"],
        }
    report["owned_runtime_residue_zero"] = not Path(raw_work).exists()
    if not report["owned_runtime_residue_zero"]:
        report["status"] = "FAIL_CLOSED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/u7_2n_product_auxiliary_output_transaction_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), default="forward")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    order = tuple(config["available_product_looks"])
    if args.order == "reverse":
        order = tuple(reversed(order))
    report = build_report(config_path=args.config, order=order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
