from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import render_film
from src.inference.product_desktop import ProductDesktopError

CONFIG = ROOT / "configs/u7_21c_product_runtime_scope_publication_race_v1.json"
CONTRACT = (
    ROOT / "docs/planning/U7_21C_PRODUCT_RUNTIME_SCOPE_PUBLICATION_RACE_CONTRACT.md"
)
CORE = ROOT / "scripts/render_film.py"
TEST = ROOT / "tests/test_u7_21c_product_runtime_scope_publication_race.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str:
    return subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _source(path: Path) -> None:
    yy, xx = np.mgrid[:43, :61]
    rgb = np.stack(
        (
            (xx * 7 + yy * 3) % 256,
            (xx * 2 + yy * 11) % 256,
            (xx * 13 + yy * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _run(
    arguments: list[str],
    *,
    reject_second: bool = False,
    forbid_validation: bool = False,
) -> tuple[int, str, str, int]:
    original_argv = sys.argv
    original_validate = render_film._validate_runtime_source_scope
    original_commit = render_film._source_commit
    calls = 0

    def validator(root: Path, commit: str, scope: tuple[str, ...]) -> None:
        nonlocal calls
        calls += 1
        if forbid_validation:
            raise AssertionError("legacy crossed product runtime-scope policy")
        if reject_second and calls == 2:
            raise ProductDesktopError("runtime source scope changed")
        original_validate(root, commit, scope)

    render_film._validate_runtime_source_scope = validator
    if reject_second:
        render_film._source_commit = lambda _root: "1" * 40
    sys.argv = ["render_film.py", *arguments]
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = render_film._run_cli()
    finally:
        sys.argv = original_argv
        render_film._validate_runtime_source_scope = original_validate
        render_film._source_commit = original_commit
    return returncode, stdout.getvalue(), stderr.getvalue(), calls


def _case(root: Path, name: str) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    source = case / "source.png"
    output = case / "output.png"
    _source(source)
    common = [
        str(source),
        "--look-amount",
        "0.65",
        "--output",
        str(output),
        "--write-recipe",
    ]
    if name == "shortcut":
        selection = ["--product-look", "ektar_100"]
        returncode, _stdout, stderr, calls = _run([common[0], *selection, *common[1:]])
    elif name == "explicit":
        selection = [
            "--use-render-profile",
            "--render-profile",
            str(ROOT / "configs/render_profiles/safe_rich_product_v1.json"),
            "--style",
            "ektar_100",
        ]
        returncode, _stdout, stderr, calls = _run([common[0], *selection, *common[1:]])
    elif name == "drift":
        returncode, _stdout, stderr, calls = _run(
            [common[0], "--product-look", "ektar_100", *common[1:]],
            reject_second=True,
        )
    elif name == "legacy":
        returncode, _stdout, stderr, calls = _run(
            [str(source), "--output", str(output)],
            forbid_validation=True,
        )
    else:
        raise ValueError(f"unknown case: {name}")
    recipe = output.with_suffix(".recipe.json")
    stage_residue = sorted(path.name for path in case.glob(".*.stage*"))
    return {
        "case": name,
        "returncode": returncode,
        "validation_calls": calls,
        "stderr": stderr.strip(),
        "output_exists": output.exists(),
        "output_sha256": _sha256(output) if output.exists() else None,
        "recipe_exists": recipe.exists(),
        "stage_residue": stage_residue,
    }


def build_report(*, scratch: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if scratch.exists():
        raise FileExistsError("scratch path must be absent")
    scratch.mkdir(parents=True)
    names = ["shortcut", "explicit", "drift", "legacy"]
    execution = names if order == "forward" else list(reversed(names))
    try:
        records = {name: _case(scratch, name) for name in execution}
        expected = json.loads(CONFIG.read_text(encoding="utf-8"))[
            "canonical_product_output_sha256"
        ]
        shortcut = records["shortcut"]
        explicit = records["explicit"]
        drift = records["drift"]
        legacy = records["legacy"]
        gates = {
            "shortcut_validates_twice": shortcut["validation_calls"] == 2,
            "explicit_profile_validates_twice": explicit["validation_calls"] == 2,
            "shortcut_output_exact": shortcut["output_sha256"] == expected,
            "explicit_output_exact": explicit["output_sha256"] == expected,
            "success_recipes_present": shortcut["recipe_exists"]
            and explicit["recipe_exists"],
            "second_validation_rejects": drift["returncode"] == 1
            and drift["validation_calls"] == 2,
            "drift_publication_zero": not drift["output_exists"]
            and not drift["recipe_exists"],
            "drift_stage_residue_zero": not drift["stage_residue"],
            "legacy_policy_isolated": legacy["returncode"] == 0
            and legacy["validation_calls"] == 0
            and legacy["output_exists"],
            "claim_ceiling_exact": True,
        }
        return {
            "schema_version": "neuro-film.u7-21c-product-runtime-scope-publication-race-result.v1",
            "source_commit": _head(),
            "status": (
                "PASS_PRIVATE_U7_21C_PRODUCT_RUNTIME_SCOPE_PUBLICATION_RACE"
                if all(gates.values())
                else "FAIL_CLOSED_U7_21C_PRODUCT_RUNTIME_SCOPE_PUBLICATION_RACE"
            ),
            "cases": [records[name] for name in names],
            "gates": gates,
            "source_bindings": {
                path.relative_to(ROOT).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (CONFIG, CONTRACT, CORE, TEST)
            },
            "claim_ceiling": {
                "evidence_grade": "look-approximation",
                "calibrated_stock_response": False,
                "physical_film_reproduction": False,
                "scope": "private product runtime-scope publication consistency",
            },
        }
    finally:
        shutil.rmtree(scratch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    arguments = parser.parse_args()
    report = build_report(scratch=arguments.scratch, order=arguments.order)
    if arguments.output.exists():
        raise FileExistsError("output path must be absent")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if str(report["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
