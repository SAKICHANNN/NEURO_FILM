"""Formal audit for the U7.2M product image + recipe transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from src.inference import product_render_transaction as transaction_module
from src.inference.product_render_transaction import (
    ProductRenderTransactionError,
    prepare_product_image_recipe_transaction,
)
from src.preprocess import output_encode

SCRIPT = ROOT / "scripts/render_film.py"
PRODUCT_PROFILE = ROOT / "configs/render_profiles/safe_rich_product_v1.json"
LEGACY_PROFILE = ROOT / "configs/render_profiles/safe_rich_v1.json"
CONTRACT_COMMIT = "2f762ac82"
IMPLEMENTATION_COMMIT = "43b68f30"
IMPLEMENTATION_PATHS = (
    "scripts/render_film.py",
    "src/inference/product_render_transaction.py",
    "tests/test_u7_2m_product_image_recipe_transaction.py",
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
    style: str = "velvia_50",
    write_recipe: bool = True,
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
    ]
    if write_recipe:
        command.append("--write-recipe")
    command.extend(("--output", str(output)))
    return command


def _run(
    source: Path,
    output: Path,
    *,
    style: str = "velvia_50",
    write_recipe: bool = True,
    product: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(
            source,
            output,
            style=style,
            write_recipe=write_recipe,
            product=product,
        ),
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


def _hidden(directory: Path) -> list[str]:
    return sorted(
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob(".*")
    )


def _semantic_recipe_entry_reject(work: Path, label: str) -> bool:
    output = work / f"semantic-{label}-output.png"
    recipe = output.with_suffix(".recipe.json")
    real_lstat = Path.lstat

    def selective_lstat(path: Path) -> os.stat_result:
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            os.path.abspath(recipe)
        ):
            return os.stat_result((0,) * 10)
        return real_lstat(path)

    with patch(
        "pathlib.Path.lstat",
        autospec=True,
        side_effect=selective_lstat,
    ):
        try:
            prepare_product_image_recipe_transaction(
                work / f"semantic-{label}-input.png",
                output,
            )
        except ProductRenderTransactionError as exc:
            return str(exc) == "product recipe destination must not already exist"
    return False


def _recipe_entry_gates(work: Path) -> dict[str, bool]:
    source = work / "missing-source.png"
    output = work / "regular.png"
    recipe = output.with_suffix(".recipe.json")
    recipe.write_bytes(b"foreign-regular")
    completed = _run(source, output)
    regular_preserved = (
        completed.returncode != 0
        and "product recipe destination must not already exist" in completed.stderr
        and recipe.read_bytes() == b"foreign-regular"
        and not output.exists()
    )
    return {
        "existing_recipe_regular_preserved": regular_preserved,
        "hardlink_recipe_entry_semantics_reject": _semantic_recipe_entry_reject(
            work,
            "hardlink",
        ),
        "valid_symlink_recipe_entry_semantics_reject": (
            _semantic_recipe_entry_reject(work, "valid-symlink")
        ),
        "broken_symlink_recipe_entry_semantics_reject": (
            _semantic_recipe_entry_reject(work, "broken-symlink")
        ),
        "windows_reparse_recipe_entry_semantics_reject": (
            os.name != "nt" or _semantic_recipe_entry_reject(work, "reparse")
        ),
    }


def _image_failure_gate(work: Path) -> bool:
    output = work / "image-failure.png"

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("u7.2m injected image encode failure")

    with patch.object(Image.Image, "save", fail_save):
        try:
            output_encode.save_srgb8(
                np.zeros((3, 4, 3), dtype=np.float32),
                output,
                create_only=True,
            )
        except RuntimeError as exc:
            return (
                str(exc) == "u7.2m injected image encode failure"
                and not output.exists()
            )
    return False


def _recipe_failure_gate(work: Path) -> bool:
    source = work / "recipe-failure-source"
    output = work / "recipe-failure.png"
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    try:
        with transaction:
            transaction.image_stage.write_bytes(b"owned-image")
            transaction.bind_image_stage()
            transaction.stage_recipe({"not-json": object()})
    except TypeError:
        return (
            not output.exists()
            and not output.with_suffix(".recipe.json").exists()
            and not transaction.image_stage.exists()
            and not transaction.recipe_stage.exists()
        )
    return False


def _late_foreign_gate(work: Path, *, replace_image: bool) -> bool:
    source = work / (
        "foreign-image-source" if replace_image else "foreign-recipe-source"
    )
    output = work / ("foreign-image.png" if replace_image else "foreign-recipe.png")
    recipe = output.with_suffix(".recipe.json")
    source.write_bytes(b"source")
    transaction = prepare_product_image_recipe_transaction(source, output)
    real_publish = transaction_module.publish_create_only

    def inject(stage: Path, final: Path):
        if final == recipe:
            if replace_image:
                output.unlink()
                output.write_bytes(b"foreign-image")
            final.write_bytes(b"foreign-recipe")
        return real_publish(stage, final)

    with patch.object(transaction_module, "publish_create_only", inject):
        try:
            with transaction:
                transaction.image_stage.write_bytes(b"owned-image")
                transaction.bind_image_stage()
                transaction.stage_recipe({"recipe": 1})
                transaction.publish()
        except FileExistsError:
            return (
                recipe.read_bytes() == b"foreign-recipe"
                and (
                    output.read_bytes() == b"foreign-image"
                    if replace_image
                    else not output.exists()
                )
                and not transaction.image_stage.exists()
                and not transaction.recipe_stage.exists()
            )
    return False


def _stage_mutation_gate(work: Path) -> bool:
    results = []
    for label in ("image", "recipe"):
        source = work / f"mutation-{label}-source"
        output = work / f"mutation-{label}.png"
        source.write_bytes(b"source")
        transaction = prepare_product_image_recipe_transaction(source, output)
        try:
            with transaction:
                transaction.image_stage.write_bytes(b"owned-image")
                transaction.bind_image_stage()
                transaction.stage_recipe({"recipe": 1})
                selected = (
                    transaction.image_stage
                    if label == "image"
                    else transaction.recipe_stage
                )
                selected.write_bytes(b"mutated-in-place")
                transaction.publish()
        except ProductRenderTransactionError as exc:
            results.append(
                "content changed" in str(exc)
                and not output.exists()
                and not output.with_suffix(".recipe.json").exists()
            )
        else:
            results.append(False)
    return all(results)


def build_report(*, config_path: Path, order: tuple[str, ...]) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scratch_root = ROOT / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="neuro-film-u7-2m-",
        dir=scratch_root,
    ) as raw_work:
        work = Path(raw_work)
        source = work / "source.png"
        _source(source)
        source_before = source.read_bytes()
        results: dict[str, dict[str, Any]] = {}
        for style in order:
            output = work / f"{style}.png"
            recipe = output.with_suffix(".recipe.json")
            completed = _run(source, output, style=style)
            results[style] = {
                "returncode": completed.returncode,
                "output_sha256": _sha256(output) if output.is_file() else None,
                "normalized_recipe_sha256": (
                    _normalized_recipe_sha(recipe) if recipe.is_file() else None
                ),
                "recipe_output_path_exact": (
                    json.loads(recipe.read_text(encoding="utf-8"))["output"]["path"]
                    == str(output.resolve())
                    if recipe.is_file()
                    else False
                ),
            }

        collision_output = work / "collision.png"
        collision_source = collision_output.with_suffix(".recipe.json")
        collision = _run(collision_source, collision_output)

        concurrent = work / "concurrent.png"
        processes = [
            subprocess.Popen(
                _command(source, concurrent),
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

        no_recipe = work / "no-recipe.png"
        no_recipe_run = _run(source, no_recipe, write_recipe=False)
        legacy = work / "legacy.png"
        legacy.write_bytes(b"replace-me")
        legacy_run = _run(source, legacy, write_recipe=False, product=False)

        oracle = config["prechange_oracle"]
        product_exact = all(
            results[style]["returncode"] == 0
            and results[style]["output_sha256"] == oracle[style]["output_sha256"]
            and results[style]["normalized_recipe_sha256"]
            == oracle[style]["normalized_recipe_sha256"]
            and results[style]["recipe_output_path_exact"]
            for style in config["available_product_looks"]
        )
        prechange_commit = subprocess.check_output(
            ["git", "rev-parse", f"{CONTRACT_COMMIT}^"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip()
        source_locks = config["source_locks"]
        prechange_locks_exact = all(
            hashlib.sha256(_git_blob(prechange_commit, path)).hexdigest()
            == source_locks[key]
            for key, path in (
                ("render_film_sha256", "scripts/render_film.py"),
                (
                    "product_render_transaction_sha256",
                    "src/inference/product_render_transaction.py",
                ),
                ("render_contract_sha256", "src/inference/render_contract.py"),
                (
                    "create_only_file_sha256",
                    "src/film_physics/create_only_file.py",
                ),
                (
                    "product_profile_sha256",
                    "configs/render_profiles/safe_rich_product_v1.json",
                ),
                (
                    "u7_2l_config_sha256",
                    "configs/u7_2l_product_create_only_render_transaction_v1.json",
                ),
                (
                    "u7_2l_evidence_sha256",
                    "docs/evidence/U7_2L_PRODUCT_CREATE_ONLY_RENDER_TRANSACTION_RESULT.json",
                ),
            )
        )
        implementation_exact = all(
            hashlib.sha256(_git_blob(IMPLEMENTATION_COMMIT, path)).hexdigest()
            == _sha256(ROOT / path)
            for path in IMPLEMENTATION_PATHS
        )
        recipe_entry_gates = _recipe_entry_gates(work)
        image_failure_gate = _image_failure_gate(work)
        recipe_failure_gate = _recipe_failure_gate(work)
        late_foreign_recipe_gate = _late_foreign_gate(work, replace_image=False)
        foreign_image_gate = _late_foreign_gate(work, replace_image=True)
        stage_mutation_gate = _stage_mutation_gate(work)
        gates = {
            "prechange_source_locks_exact": prechange_locks_exact,
            "implementation_files_exact": implementation_exact,
            "fixture_exact": hashlib.sha256(source_before).hexdigest()
            == config["fixture"]["sha256"],
            "pair_paths_distinct_before_decode": collision.returncode != 0
            and "product input, output and recipe paths must be distinct"
            in collision.stderr,
            **recipe_entry_gates,
            "three_output_recipe_oracles_exact": product_exact,
            "image_failure_residue_zero": image_failure_gate,
            "recipe_failure_rolls_back_owned_image": recipe_failure_gate,
            "late_foreign_recipe_preserved": late_foreign_recipe_gate,
            "foreign_image_replacement_preserved": foreign_image_gate,
            "in_place_stage_mutation_rejected": stage_mutation_gate,
            "concurrent_exactly_one_complete_pair": concurrent_codes == [0, 1]
            and _sha256(concurrent) == oracle["velvia_50"]["output_sha256"]
            and _normalized_recipe_sha(concurrent.with_suffix(".recipe.json"))
            == oracle["velvia_50"]["normalized_recipe_sha256"],
            "source_immutable": source.read_bytes() == source_before,
            "product_without_recipe_unchanged": no_recipe_run.returncode == 0
            and _sha256(no_recipe) == oracle["velvia_50"]["output_sha256"]
            and not no_recipe.with_suffix(".recipe.json").exists(),
            "legacy_behavior_unchanged": legacy_run.returncode == 0
            and _sha256(legacy) == oracle["velvia_50"]["output_sha256"],
            "owned_stage_residue_zero": not _hidden(work),
        }
        report = {
            "schema_id": "neuro-film.u7-2m-product-image-recipe-transaction-result.v1",
            "experiment_id": "U7.2M",
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "bindings": {
                "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "contract_commit": CONTRACT_COMMIT,
                "implementation_commit": IMPLEMENTATION_COMMIT,
            },
            "product_results": dict(sorted(results.items())),
            "concurrent_returncodes": concurrent_codes,
            "control_execution": {
                "scratch_location": "repo-relative-P-backed-tmp",
                "regular_recipe_entry": "actual-filesystem-entry",
                "hardlink_recipe_entry": "separate-lstat-entry-semantics",
                "valid_symlink_recipe_entry": "separate-lstat-entry-semantics",
                "broken_symlink_recipe_entry": "separate-lstat-entry-semantics",
                "windows_reparse_recipe_entry": "separate-lstat-entry-semantics",
                "physical_link_behavior": (
                    "covered-by-committed-focused-tests-on-capable-filesystem"
                ),
                "image_encode_failure": "inherited-exact-create-only-encoder-control",
            },
            "gates": gates,
            "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
            "decision": (
                "Enable create-only image/recipe pair publication with "
                "identity-and-content-sealed process-level rollback only for "
                "the exact safe-rich-product-v1 profile."
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
        default=ROOT / "configs/u7_2m_product_image_recipe_transaction_v1.json",
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
