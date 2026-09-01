#!/usr/bin/env python3
"""Audit the U7.9A runtime on six exact Canon sRAW/mRAW product inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.install_product_runtime import (
    _cleanup_owned_directory,
    _entry_identity,
    _sha256,
    install_product_runtime,
)

CONFIG = ROOT / "configs" / "u7_9b_private_runtime_canon_raw_compatibility_v1.json"
RENDERER = ROOT / "scripts" / "render_film.py"
REPLAY_CLI = ROOT / "scripts" / "replay_film_recipe.py"


class U79BError(RuntimeError):
    """Raised when a frozen U7.9B identity or execution invariant differs."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise U79BError(f"expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_file(path: Path, expected: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected["bytes"])
        and _sha256(path) == str(expected["sha256"])
    )


def _run(
    command: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _require_success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed")[-4000:]
        raise U79BError(f"command failed ({result.returncode}): {detail}")
    return result.stdout


def _git_head() -> str:
    return _require_success(
        _run(["git", "rev-parse", "HEAD"], cwd=ROOT, env=dict(os.environ))
    ).strip()


def _tracked_clean() -> bool:
    result = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        env=dict(os.environ),
    )
    return result.returncode == 0 and not result.stdout.strip()


def _wheelhouse_rows(
    wheelhouse: Path, expected: list[list[object]]
) -> list[list[object]]:
    rows = [
        [path.name, path.stat().st_size, _sha256(path)]
        for path in sorted(
            wheelhouse.glob("*.whl"), key=lambda item: item.name.casefold()
        )
    ]
    if rows != expected:
        raise U79BError("frozen wheelhouse identity differs")
    return rows


def _source_identity(path: Path, row: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(row["bytes"])
        and _sha256(path) == str(row["sha256"])
    )


def _create_formal_root(
    root: Path, scratch_parent: Path
) -> tuple[tuple[int, int], tuple[int, int]]:
    if root.parent != scratch_parent:
        raise ValueError("formal root must be one direct child of frozen scratch root")
    if os.path.lexists(scratch_parent) or os.path.lexists(root):
        raise FileExistsError("formal scratch root must be absent")
    scratch_parent.mkdir()
    parent_identity = _entry_identity(scratch_parent)
    try:
        root.mkdir()
    except BaseException:
        _cleanup_owned_directory(scratch_parent, parent_identity)
        raise
    return _entry_identity(root), parent_identity


def _render_arguments(source: Path, output: Path, config: dict[str, Any]) -> list[str]:
    execution = config["execution"]
    return [
        str(source),
        "--product-look",
        str(execution["look_id"]),
        "--look-amount",
        str(execution["look_amount"]),
        "--output-bit-depth",
        str(execution["output_bit_depth"]),
        "--write-recipe",
        "--output",
        str(output),
    ]


def _png_facts(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {
            "bytes": path.stat().st_size,
            "format": image.format,
            "height": image.height,
            "icc_present": bool(image.info.get("icc_profile")),
            "mode": image.mode,
            "sha256": _sha256(path),
            "width": image.width,
        }


def _recipe_exact(
    recipe: dict[str, Any],
    *,
    source: Path,
    output: Path,
    row: dict[str, Any],
    output_sha256: str,
    profile_sha256: str,
    source_commit: str,
) -> dict[str, bool]:
    claim = recipe.get("claim", {})
    render = recipe.get("render", {})
    input_row = recipe.get("input", {})
    output_row = recipe.get("output", {})
    profile = recipe.get("profile", {})
    software = recipe.get("software", {})
    return {
        "claim_is_look_approximation": claim.get("render_mode") == "Style-safe"
        and claim.get("output_label") == "film-inspired"
        and claim.get("evidence_grade") == "look-approximation"
        and claim.get("calibrated_reference_allowed") is False,
        "explicit_product_look": render.get("style") == "ektar_100",
        "input_identity": Path(str(input_row.get("path", ""))).resolve()
        == source.resolve()
        and input_row.get("sha256") == row["sha256"],
        "output_identity": Path(str(output_row.get("path", ""))).resolve()
        == output.resolve()
        and output_row.get("sha256") == output_sha256
        and output_row.get("format") == "PNG"
        and output_row.get("bit_depth") == 8,
        "product_profile": profile.get("profile_id") == "safe-rich-product-v1"
        and profile.get("sha256") == profile_sha256,
        "software_commit": software.get("commit") == source_commit,
    }


def _existing_destination_control(
    path: Path, *, wheelhouse: Path, env: dict[str, str]
) -> bool:
    path.mkdir()
    marker = path / "foreign.bin"
    marker.write_bytes(b"foreign")
    try:
        try:
            install_product_runtime(path, wheelhouse=wheelhouse, environment=env)
        except FileExistsError:
            return marker.read_bytes() == b"foreign"
        return False
    finally:
        shutil.rmtree(path)


def _missing_wheel_control(
    root: Path,
    *,
    env: dict[str, str],
) -> bool:
    incomplete = root / "missing-wheelhouse"
    incomplete.mkdir()
    destination = root / "missing-wheel-install"
    try:
        try:
            install_product_runtime(destination, wheelhouse=incomplete, environment=env)
        except RuntimeError:
            return not os.path.lexists(destination)
        return False
    finally:
        shutil.rmtree(incomplete)


def _row_record(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    source_commit: str,
    direct_python: Path,
    launcher: Path,
    runtime_python: Path,
    profile: Path,
    root: Path,
    foreign_cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    source = ROOT / str(row["path"])
    if not _source_identity(source, row):
        raise U79BError(f"source identity differs: {row['source_id']}")
    source_before = _sha256(source)
    row_root = root / f"row-{row['source_id']}"
    row_root.mkdir()
    output = row_root / "product.png"
    recipe_path = output.with_suffix(".recipe.json")
    replay = row_root / "replay.png"
    args = _render_arguments(source, output, config)
    try:
        direct_result = _run(
            [str(direct_python), str(RENDERER), *args], cwd=foreign_cwd, env=env
        )
        _require_success(direct_result)
        if not output.is_file() or not recipe_path.is_file():
            raise U79BError(f"direct artifacts missing: {row['source_id']}")
        direct_output = output.read_bytes()
        direct_recipe = recipe_path.read_bytes()
        direct_facts = _png_facts(output)
        output.unlink()
        recipe_path.unlink()

        launcher_result = _run(
            ["cmd.exe", "/d", "/c", str(launcher), *args],
            cwd=foreign_cwd,
            env=env,
        )
        _require_success(launcher_result)
        if not output.is_file() or not recipe_path.is_file():
            raise U79BError(f"launcher artifacts missing: {row['source_id']}")
        launcher_output = output.read_bytes()
        launcher_recipe = recipe_path.read_bytes()
        launcher_facts = _png_facts(output)
        recipe = json.loads(launcher_recipe)
        recipe_checks = _recipe_exact(
            recipe,
            source=source,
            output=output,
            row=row,
            output_sha256=launcher_facts["sha256"],
            profile_sha256=_sha256(profile),
            source_commit=source_commit,
        )

        replay_result = _run(
            [
                str(runtime_python),
                str(REPLAY_CLI),
                "--recipe",
                str(recipe_path),
                "--profile",
                str(profile),
                "--output",
                str(replay),
                "--workers",
                "1",
            ],
            cwd=foreign_cwd,
            env=env,
        )
        replay_stdout = _require_success(replay_result).strip()
        if not replay.is_file():
            raise U79BError(f"replay artifact missing: {row['source_id']}")
        replay_facts = _png_facts(replay)
        expected = str(config["p314_output_oracles"][row["source_id"]])
        return {
            "direct_launcher_output_byte_exact": direct_output == launcher_output,
            "direct_launcher_recipe_byte_exact": direct_recipe == launcher_recipe,
            "direct_output": direct_facts,
            "launcher_output": launcher_facts,
            "model": row["model"],
            "mode": row["mode"],
            "oracle_sha256": expected,
            "oracle_exact": direct_facts["sha256"]
            == launcher_facts["sha256"]
            == expected,
            "recipe": {
                "bytes": len(launcher_recipe),
                "checks": recipe_checks,
                "sha256": _sha256_bytes(launcher_recipe),
            },
            "replay": {
                **replay_facts,
                "byte_exact": replay.read_bytes() == launcher_output,
                "stdout": replay_stdout,
            },
            "source_id": row["source_id"],
            "source_sha256": source_before,
            "source_unchanged": _sha256(source) == source_before,
        }
    finally:
        for path in (replay, recipe_path, output):
            path.unlink(missing_ok=True)
        if row_root.exists():
            row_root.rmdir()


def audit(root: Path, wheelhouse: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError(order)
    config = _load_json(CONFIG)
    bindings = {
        name: _verify_file(ROOT / str(item["path"]), item)
        for name, item in sorted(config["bindings"].items())
    }
    if not all(bindings.values()):
        raise U79BError("frozen product binding differs")
    if not _tracked_clean():
        raise U79BError("tracked worktree must be clean before U7.9B")

    tmp_root = (ROOT / "tmp").resolve(strict=True)
    root = root.resolve(strict=False)
    wheelhouse = wheelhouse.resolve(strict=True)
    if tmp_root not in root.parents or tmp_root not in wheelhouse.parents:
        raise ValueError("formal roots must remain below repo-relative tmp")
    scratch_parent = (ROOT / str(config["runtime"]["scratch_root"])).resolve(
        strict=False
    )
    expected_wheels = list(config["runtime"]["wheels"])
    wheel_rows = _wheelhouse_rows(wheelhouse, expected_wheels)

    p313 = _load_json(ROOT / config["bindings"]["p313_config"]["path"])
    rows = list(p313["rows"])
    if len(rows) != int(config["gates"]["required_rows"]):
        raise U79BError("frozen source cohort count differs")
    if set(config["p314_output_oracles"]) != {str(row["source_id"]) for row in rows}:
        raise U79BError("P314 oracle cohort differs")
    for row in rows:
        if not _source_identity(ROOT / str(row["path"]), row):
            raise U79BError(f"source identity differs: {row['source_id']}")
    if order == "reverse":
        rows.reverse()

    root_identity, parent_identity = _create_formal_root(root, scratch_parent)
    cleanup_ok = False
    parent_cleanup_ok = False
    report: dict[str, Any] | None = None
    try:
        cache = root / "pip-cache"
        process_temp = root / "process-temp"
        foreign_cwd = root / "foreign-cwd"
        cache.mkdir()
        process_temp.mkdir()
        foreign_cwd.mkdir()
        env = dict(os.environ)
        env.update(
            {
                "PIP_CACHE_DIR": str(cache),
                "PIP_NO_INDEX": "1",
                "TMP": str(process_temp),
                "TEMP": str(process_temp),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        source_commit = _git_head()
        installation = root / "installation"
        receipt = install_product_runtime(
            installation, wheelhouse=wheelhouse, environment=env
        )
        launcher = installation / "kmcfm-look.cmd"
        runtime_python = installation / "runtime" / "Scripts" / "python.exe"
        direct_python = ROOT / str(config["runtime"]["direct_python"])
        profile = ROOT / str(config["execution"]["profile_path"])
        if not all(
            path.is_file() for path in (launcher, runtime_python, direct_python)
        ):
            raise U79BError("required product runtime executable is missing")

        existing_destination = _existing_destination_control(
            root / "existing-destination", wheelhouse=wheelhouse, env=env
        )
        missing_wheel = _missing_wheel_control(
            root,
            env=env,
        )
        drift_file = root / "source-drift.cr2"
        drift_file.write_bytes(b"drift")
        source_drift = not _source_identity(drift_file, rows[0])
        drift_file.unlink()
        invalid_output = root / "invalid.png"
        invalid_result = _run(
            [
                "cmd.exe",
                "/d",
                "/c",
                str(launcher),
                str(ROOT / str(rows[0]["path"])),
                "--product-look",
                "not-a-product-look",
                "--output",
                str(invalid_output),
            ],
            cwd=foreign_cwd,
            env=env,
        )
        invalid_launcher = (
            invalid_result.returncode != 0 and not invalid_output.exists()
        )

        records = [
            _row_record(
                row,
                config=config,
                source_commit=source_commit,
                direct_python=direct_python,
                launcher=launcher,
                runtime_python=runtime_python,
                profile=profile,
                root=root,
                foreign_cwd=foreign_cwd,
                env=env,
            )
            for row in rows
        ]
        records.sort(key=lambda item: str(item["source_id"]))

        receipt_path = installation / "product-runtime.json"
        requirements_names = {
            line.split("==", 1)[0].casefold()
            for line in (ROOT / config["bindings"]["requirements"]["path"])
            .read_text("utf-8")
            .splitlines()
            if line and not line.startswith("#")
        }
        # The U7.9A receipt intentionally carries the narrower installer claim.
        receipt_claim_exact = (
            receipt["claim"]
            == _load_json(ROOT / config["bindings"]["u7_9a_config"]["path"])["claim"]
        )
        gates = {
            "bindings_exact": all(bindings.values()),
            "direct_launcher_outputs_exact": all(
                row["direct_launcher_output_byte_exact"] for row in records
            ),
            "direct_launcher_recipes_exact": all(
                row["direct_launcher_recipe_byte_exact"] for row in records
            ),
            "foreign_existing_destination_preserved": existing_destination,
            "installed_recipe_fields_exact": all(
                all(row["recipe"]["checks"].values()) for row in records
            ),
            "installed_replays_exact": all(
                row["replay"]["byte_exact"]
                and row["replay"]["sha256"] == row["launcher_output"]["sha256"]
                for row in records
            ),
            "invalid_launcher_rejected": invalid_launcher,
            "missing_wheel_rejected_and_cleaned": missing_wheel,
            "network_requests_zero": int(config["execution"]["network_requests"]) == 0,
            "p314_output_oracles_exact": all(row["oracle_exact"] for row in records),
            "receipt_claim_exact": receipt_claim_exact,
            "receipt_distribution_and_commit_exact": receipt["source_commit"]
            == source_commit
            and set(receipt["distributions"]) == requirements_names,
            "required_rows_complete": len(records)
            == int(config["gates"]["required_rows"]),
            "source_drift_rejected": source_drift,
            "sources_immutable": all(row["source_unchanged"] for row in records),
            "tracked_clean": _tracked_clean(),
            "wheelhouse_exact": wheel_rows == expected_wheels,
        }
        report = {
            "schema": "kmcfm.u7-9b-private-runtime-canon-raw-compatibility-result.v1",
            "source_commit": source_commit,
            "bindings": bindings,
            "claim": config["claim"],
            "controls": {
                "foreign_existing_destination_preserved": existing_destination,
                "invalid_launcher_rejected": invalid_launcher,
                "missing_wheel_rejected_and_cleaned": missing_wheel,
                "source_drift_rejected": source_drift,
            },
            "gates": gates,
            "receipt_sha256": _sha256(receipt_path),
            "records": records,
            "wheelhouse": {
                "bytes": sum(int(row[1]) for row in wheel_rows),
                "count": len(wheel_rows),
                "rows": wheel_rows,
            },
        }
        # Keep this assignment separate from reportability so failed gates still emit.
        report["automatic_pass"] = all(gates.values())
    finally:
        cleanup_ok = _cleanup_owned_directory(root, root_identity)
        parent_cleanup_ok = _cleanup_owned_directory(scratch_parent, parent_identity)

    if report is None:
        raise U79BError("formal report was not constructed")
    report["owned_residue_zero"] = (
        cleanup_ok
        and parent_cleanup_ok
        and not os.path.lexists(root)
        and not os.path.lexists(scratch_parent)
    )
    report["gates"]["owned_residue_zero"] = report["owned_residue_zero"]
    report["automatic_pass"] = all(report["gates"].values())
    report["status"] = (
        "PASS_PRIVATE_U7_9B_BINARY_RUNTIME_CANON_RAW_COMPATIBILITY"
        if report["automatic_pass"]
        else "FAIL_CLOSED_U7_9B_BINARY_RUNTIME_CANON_RAW_COMPATIBILITY"
    )
    identity = {
        "claim": report["claim"],
        "controls": report["controls"],
        "gates": report["gates"],
        "records": report["records"],
        "source_commit": report["source_commit"],
        "status": report["status"],
        "wheelhouse": report["wheelhouse"],
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.wheelhouse, args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
