#!/usr/bin/env python3
"""Formal audit for the private repository-independent product capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_private_runtime_capsule as builder
from src.inference.runtime_source_capsule import (
    CAPSULE_MANIFEST_ENV,
    CAPSULE_MANIFEST_SHA256_ENV,
    verify_runtime_source_capsule,
)

CONFIG = ROOT / "configs/u7_22a_private_repository_independent_runtime_capsule_v1.json"
RENDERER = ROOT / "scripts/render_film.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str], *, cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _require_success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[-4000:])
    return result.stdout


def _tiny_input(path: Path, width: int, height: int) -> None:
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + 5) % 251,
            (xx * 5 + yy * 13 + 19) % 251,
            (xx * 7 + yy * 17 + 43) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _gitless_environment(root: Path) -> tuple[dict[str, str], Path]:
    trap = root / "path-trap"
    trap.mkdir()
    marker = root / "git-invoked.txt"
    (trap / "git.cmd").write_text(
        f'@echo off\r\necho invoked>"{marker}"\r\nexit /b 91\r\n',
        "utf-8",
        newline="",
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
        and not key.startswith("KMCFM_PRIVATE_CAPSULE_")
    }
    environment.update(
        {
            "PATH": str(trap),
            "GIT_DIR": str(root / "missing-git-dir"),
            "GIT_WORK_TREE": str(root / "missing-git-work-tree"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMP": str(root / "process-temp"),
            "TEMP": str(root / "process-temp"),
        }
    )
    (root / "process-temp").mkdir()
    return environment, marker


def _capsule_environment(capsule: Path, environment: dict[str, str]) -> dict[str, str]:
    result = dict(environment)
    manifest = capsule / "runtime-source-capsule.json"
    result[CAPSULE_MANIFEST_ENV] = str(manifest)
    result[CAPSULE_MANIFEST_SHA256_ENV] = _sha256(manifest)
    return result


def _replay(
    *,
    python: Path,
    capsule: Path,
    recipe: Path,
    output: Path,
    environment: dict[str, str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    code = """import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file
root=Path(sys.argv[2]); recipe=json.loads(Path(sys.argv[3]).read_text('utf-8'))
print(replay_style_safe_recipe_to_file(recipe,profile_path=root/'configs/render_profiles/safe_rich_product_v1.json',output_path=Path(sys.argv[4]),root=root))
"""
    manifest = json.loads((capsule / "runtime-source-capsule.json").read_text("utf-8"))
    archive = capsule / manifest["archive"]["path"]
    return _run(
        [
            str(python),
            "-I",
            "-c",
            code,
            str(archive),
            str(capsule),
            str(recipe),
            str(output),
        ],
        cwd=cwd,
        environment=_capsule_environment(capsule, environment),
    )


def _render_rows(
    *,
    capsule: Path,
    python: Path,
    work: Path,
    looks: list[str],
    gitless: dict[str, str],
) -> list[dict[str, object]]:
    source = work / "input.png"
    rows: list[dict[str, object]] = []
    for look in looks:
        output = work / f"{look}.png"
        recipe = output.with_suffix(".recipe.json")
        command = [
            str(source),
            "--product-look",
            look,
            "--write-recipe",
            "--output",
            str(output),
        ]
        direct = _run(
            [str(python), "-I", str(RENDERER), *command],
            cwd=work,
            environment=dict(os.environ),
        )
        _require_success(direct)
        direct_output = output.read_bytes()
        direct_recipe = recipe.read_bytes()
        output.unlink()
        recipe.unlink()
        launched = _run(
            [str(capsule / "kmcfm-capsule-look.exe"), *command],
            cwd=work,
            environment=gitless,
        )
        _require_success(launched)
        replay = work / f"{look}-replay.png"
        replay_result = _replay(
            python=python,
            capsule=capsule,
            recipe=recipe,
            output=replay,
            environment=gitless,
            cwd=work,
        )
        replay_digest = _require_success(replay_result).strip()
        recipe_payload = json.loads(recipe.read_text("utf-8"))
        rows.append(
            {
                "look_id": look,
                "output_sha256": _sha256(output),
                "direct_output_exact": output.read_bytes() == direct_output,
                "direct_recipe_exact": recipe.read_bytes() == direct_recipe,
                "replay_sha256": replay_digest,
                "replay_exact": replay.read_bytes() == output.read_bytes(),
                "recipe_source_commit": recipe_payload["software"]["commit"],
                "claim": recipe_payload["claim"],
            }
        )
    rows.sort(key=lambda row: str(row["look_id"]))
    return rows


def _tamper_controls(
    capsule: Path, python: Path, work: Path, environment: dict[str, str]
) -> dict[str, bool]:
    fake_runtime = work / "fake-runtime"
    fake_capsule = fake_runtime / "capsule"
    shutil.copytree(capsule, fake_capsule)
    (fake_runtime / "runtime/Scripts").mkdir(parents=True)
    shutil.copy2(python, fake_runtime / "runtime/Scripts/python.exe")
    shutil.copy2(
        capsule.parent / "product-runtime.json", fake_runtime / "product-runtime.json"
    )
    launcher = fake_capsule / "kmcfm-capsule-look.exe"

    def rejected(path: Path) -> bool:
        original = path.read_bytes()
        path.write_bytes(original + b"drift")
        result = _run(
            [str(launcher), "--list-product-looks"],
            cwd=work,
            environment=environment,
        )
        path.write_bytes(original)
        return result.returncode == 2 and "rejected" in result.stderr

    baseline = _run(
        [str(launcher), "--list-product-looks"], cwd=work, environment=environment
    )
    return {
        "relocated_control_baseline": baseline.returncode == 0,
        "manifest_drift_rejected": rejected(
            fake_capsule / "runtime-source-capsule.json"
        ),
        "archive_drift_rejected": rejected(fake_capsule / "product-source.zip"),
        "external_drift_rejected": rejected(
            fake_capsule / "configs/color_guardrails.json"
        ),
        "parent_receipt_drift_rejected": rejected(
            fake_runtime / "product-runtime.json"
        ),
        "parent_python_drift_rejected": rejected(
            fake_runtime / "runtime/Scripts/python.exe"
        ),
    }


def _ownership_controls(runtime_root: Path, head: str) -> dict[str, bool]:
    existing = runtime_root / f"capsule-u7-22a-existing-{head[:9]}"
    existing.mkdir()
    marker = existing / "foreign.txt"
    marker.write_bytes(b"foreign")
    try:
        with mock.patch.object(builder, "_verify_parent", wraps=builder._verify_parent):
            try:
                builder.build_capsule(existing)
            except FileExistsError:
                existing_preserved = marker.read_bytes() == b"foreign"
            else:
                existing_preserved = False
    finally:
        marker.unlink()
        existing.rmdir()

    failed = runtime_root / f"capsule-u7-22a-failed-{head[:9]}"
    with mock.patch.object(
        builder, "_build_archive", side_effect=RuntimeError("injected")
    ):
        try:
            builder.build_capsule(failed)
        except RuntimeError as exc:
            failed_clean = str(exc) == "injected" and not failed.exists()
        else:
            failed_clean = False

    displaced = runtime_root / f"capsule-u7-22a-displaced-{head[:9]}"
    replacement = runtime_root / f"capsule-u7-22a-foreign-{head[:9]}"

    def replace_then_fail(path, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        Path(path).parent.rename(displaced)
        replacement_source = runtime_root / f"capsule-u7-22a-replacement-{head[:9]}"
        replacement_source.mkdir()
        (replacement_source / "foreign.txt").write_bytes(b"foreign")
        replacement_source.rename(Path(path).parent)
        raise RuntimeError("injected replacement")

    try:
        with mock.patch.object(
            builder, "_build_archive", side_effect=replace_then_fail
        ):
            try:
                builder.build_capsule(replacement)
            except RuntimeError as exc:
                foreign_preserved = (
                    str(exc) == "injected replacement"
                    and (replacement / "foreign.txt").read_bytes() == b"foreign"
                    and displaced.is_dir()
                )
            else:
                foreign_preserved = False
    finally:
        if replacement.exists():
            shutil.rmtree(replacement)
        if displaced.exists():
            shutil.rmtree(displaced)
    return {
        "existing_destination_preserved": existing_preserved,
        "injected_failure_residue_zero": failed_clean,
        "late_foreign_replacement_preserved": foreign_preserved,
    }


def audit(scratch_root: Path, order: str) -> dict[str, object]:
    config = json.loads(CONFIG.read_text("utf-8"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    runtime_root = (ROOT / config["parent_runtime"]["path"]).resolve(strict=True)
    destination = runtime_root / f"capsule-u7-22a-formal-{order}-{head[:9]}"
    receipt = builder.build_capsule(destination)
    manifest_path = destination / "runtime-source-capsule.json"
    capsule_environment = {
        CAPSULE_MANIFEST_ENV: str(manifest_path),
        CAPSULE_MANIFEST_SHA256_ENV: _sha256(manifest_path),
    }
    manifest = verify_runtime_source_capsule(
        destination, environment=capsule_environment
    )
    python = runtime_root / config["parent_runtime"]["python_path"]
    with tempfile.TemporaryDirectory(prefix="u7-22a-", dir=scratch_root) as temporary:
        work = Path(temporary)
        foreign = work / "foreign-cwd"
        foreign.mkdir()
        _tiny_input(
            work / "input.png",
            int(config["formal"]["input_width"]),
            int(config["formal"]["input_height"]),
        )
        gitless, git_marker = _gitless_environment(work)
        catalog_result = _run(
            [str(destination / "kmcfm-capsule-look.exe"), "--list-product-looks"],
            cwd=foreign,
            environment=gitless,
        )
        catalog = json.loads(_require_success(catalog_result))
        looks = list(config["formal"]["available_looks"])
        if order == "reverse":
            looks.reverse()
        render_rows = _render_rows(
            capsule=destination,
            python=python,
            work=work,
            looks=looks,
            gitless=gitless,
        )
        desktop = _run(
            [
                str(destination / "kmcfm-capsule-desktop.exe"),
                "--scratch-root",
                str(destination / "tmp"),
                "--smoke-exit-ms",
                str(config["formal"]["desktop_smoke_exit_ms"]),
            ],
            cwd=foreign,
            environment=gitless,
        )
        tamper = _tamper_controls(destination, python, work, gitless)
        git_invoked = git_marker.exists()
    ownership = _ownership_controls(runtime_root, head)
    file_rows = [path for path in destination.rglob("*") if path.is_file()]
    encoded_root = str(ROOT).encode("utf-8").lower()
    source_root_absent = True
    with zipfile.ZipFile(destination / manifest["archive"]["path"]) as archive:
        for name in archive.namelist():
            if not name.endswith("/") and encoded_root in archive.read(name).lower():
                source_root_absent = False
                break
    if source_root_absent:
        for relative in manifest["external_files"]:
            if (
                encoded_root
                in destination.joinpath(*relative.split("/")).read_bytes().lower()
            ):
                source_root_absent = False
                break
    available = [
        row["look_id"] for row in catalog["looks"] if row["availability"] == "available"
    ]
    expected_claim = {
        "calibrated_reference_allowed": False,
        "color_state_policy": "look_approximation_only",
        "evidence_grade": "look-approximation",
        "input_color_state": "display_referred",
        "output_label": "film-inspired",
        "render_mode": "Style-safe",
        "claim_ceiling": "evidence-bounded deterministic Look Approximation catalog; no stock response, calibration or authenticity claim",
    }
    gates = {
        "manifest_verified": manifest["source_commit"] == head,
        "receipt_source_exact": receipt["source_commit"] == head,
        "materialized_file_count_bounded": len(file_rows)
        <= config["capsule"]["maximum_materialized_files"],
        "logical_bytes_bounded": sum(path.stat().st_size for path in file_rows)
        <= config["capsule"]["maximum_logical_bytes"],
        "second_environment_absent": not (destination / "runtime").exists(),
        "source_checkout_path_absent": source_root_absent,
        "catalog_exact": available == config["formal"]["available_looks"],
        "git_unavailable_and_unused": not git_invoked,
        "three_outputs_direct_exact": all(
            row["direct_output_exact"] for row in render_rows
        ),
        "three_recipes_direct_exact": all(
            row["direct_recipe_exact"] for row in render_rows
        ),
        "three_replays_exact": all(row["replay_exact"] for row in render_rows),
        "recipe_commit_exact": all(
            row["recipe_source_commit"] == head for row in render_rows
        ),
        "claim_ceiling_exact": all(
            row["claim"] == expected_claim for row in render_rows
        ),
        "desktop_smoke": desktop.returncode == 0,
        "tamper_controls": all(tamper.values()),
        "ownership_controls": all(ownership.values()),
        "capsule_tmp_clean": not any((destination / "tmp").iterdir()),
        "public_release_false": receipt["claim"] == config["claim"],
    }
    return {
        "schema": "kmcfm.u7-22a-private-repository-independent-runtime-capsule-result.v1",
        "status": (
            "PASS_PRIVATE_U7_22A_REPOSITORY_INDEPENDENT_RUNTIME_CAPSULE"
            if all(gates.values())
            else "FAIL_CLOSED_U7_22A_REPOSITORY_INDEPENDENT_RUNTIME_CAPSULE"
        ),
        "source_commit": head,
        "capsule": {
            "manifest_sha256": _sha256(manifest_path),
            "receipt_sha256": _sha256(destination / "capsule-receipt.json"),
            "source_archive_sha256": _sha256(destination / manifest["archive"]["path"]),
            "materialized_file_count": len(file_rows),
            "logical_bytes": sum(path.stat().st_size for path in file_rows),
        },
        "catalog": catalog,
        "render_rows": render_rows,
        "tamper_controls": tamper,
        "ownership_controls": ownership,
        "gates": gates,
        "claim_ceiling": config["claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.scratch_root.resolve(strict=True), args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0 if str(report["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
