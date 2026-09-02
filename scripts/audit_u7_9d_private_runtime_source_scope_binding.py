#!/usr/bin/env python3
"""Formal U7.9D private-runtime source-scope binding audit."""

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

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import install_product_runtime as installer

CONFIG = ROOT / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"
RUNTIME = ROOT / "outputs/private-product-runtime-u7-9d-e465886"
SOURCE_PATHS = (
    Path("configs/u7_9d_private_runtime_source_scope_binding_v1.json"),
    Path("docs/planning/U7_9D_PRIVATE_RUNTIME_SOURCE_SCOPE_BINDING_CONTRACT.md"),
    Path("scripts/install_product_runtime.py"),
    Path("scripts/audit_u7_9d_private_runtime_source_scope_binding.py"),
    Path("tests/test_u7_9d_private_runtime_source_scope_binding.py"),
    Path("tests/test_u7_20f_installed_native_launcher_argv.py"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    environment: dict[str, str] | None = None,
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
        timeout=timeout,
    )


def _git(root: Path, *arguments: str) -> str:
    completed = _run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "-C",
            str(root),
            *arguments,
        ],
        cwd=root,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _fixture_project(root: Path) -> tuple[str, Path, Path]:
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "u7-9d@example.invalid")
    _git(root, "config", "user.name", "U7.9D")
    requirements = root / "requirements-product-v2.txt"
    _write(requirements, "fixture==1.0\n")
    _write(root / "src/runtime.py", "VALUE = 1\n")
    _write(root / "configs/product.json", "{}\n")
    _write(root / "scripts/render_film.py", "print('fixture-ok')\n")
    _write(root / "scripts/open_product_desktop.py", "raise SystemExit(0)\n")
    _write(root / "docs/note.md", "base\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    commit = _git(root, "rev-parse", "HEAD")
    launcher = root.parent / f"{root.name}-launcher.py"
    launcher.write_text(
        installer._launcher_source(
            project_root=root,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=installer._sha256(requirements),
            runtime_scope=(
                "src",
                "configs",
                "scripts/render_film.py",
                "scripts/open_product_desktop.py",
                "requirements-product-v2.txt",
            ),
        ),
        encoding="utf-8",
        newline="\n",
    )
    return commit, requirements, launcher


def _control(control: str, root: Path) -> dict[str, Any]:
    commit, requirements, launcher = _fixture_project(root)
    if control == "docs_descendant":
        _write(root / "docs/note.md", "descendant\n")
        _git(root, "add", "docs/note.md")
        _git(root, "commit", "-m", "docs")
        expected = (0, "fixture-ok")
    elif control == "dirty_tracked_docs":
        _write(root / "docs/note.md", "dirty\n")
        expected = (2, "tracked repository drift")
    elif control == "scoped_untracked":
        _write(root / "src/untracked.py", "VALUE = 2\n")
        expected = (2, "untracked runtime source scope")
    elif control == "runtime_commit_drift":
        _write(root / "src/runtime.py", "VALUE = 2\n")
        _git(root, "add", "src/runtime.py")
        _git(root, "commit", "-m", "runtime")
        expected = (2, "runtime source scope drift")
    elif control == "requirements_commit_drift":
        _write(requirements, "fixture==2.0\n")
        _git(root, "add", requirements.name)
        _git(root, "commit", "-m", "requirements")
        expected = (2, "product requirements identity drift")
    elif control == "non_descendant":
        _git(root, "checkout", "--orphan", "unrelated")
        _git(root, "rm", "-rf", ".")
        _write(requirements, "fixture==1.0\n")
        _write(root / "src/runtime.py", "VALUE = 1\n")
        _write(root / "configs/product.json", "{}\n")
        _write(root / "scripts/render_film.py", "print('fixture-ok')\n")
        _write(root / "scripts/open_product_desktop.py", "raise SystemExit(0)\n")
        _git(root, "add", ".")
        _git(root, "commit", "-m", "unrelated")
        expected = (2, "repository commit drift")
    else:
        raise ValueError(control)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": str(root.resolve()),
        }
    )
    completed = _run(
        [sys.executable, "-I", str(launcher)],
        cwd=root.parent,
        environment=environment,
    )
    observed = completed.stdout if completed.returncode == 0 else completed.stderr
    passed = completed.returncode == expected[0] and expected[1] in observed
    return {
        "name": control,
        "installed_source_commit_valid": len(commit) == 40,
        "returncode": completed.returncode,
        "expected_message": expected[1],
        "passed": passed,
    }


def _runtime_smoke(root: Path, current_head: str) -> dict[str, Any]:
    receipt_path = RUNTIME / "product-runtime.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    cli = Path(receipt["launchers"]["cli"]["command"])
    desktop = Path(receipt["launchers"]["desktop"]["command"])
    catalog = _run([str(cli), "--list-product-looks"], cwd=ROOT)
    desktop_help = _run([str(desktop), "--help"], cwd=ROOT)
    source = root / "source.png"
    direct_output = root / "direct.png"
    installed_output = root / "installed.png"
    Image.new("RGB", (31, 23), (47, 101, 173)).save(source)
    common = [
        str(source),
        "--product-look",
        "ektar_100",
        "--look-amount",
        "0.65",
        "--write-recipe",
    ]
    direct = _run(
        [
            sys.executable,
            str(ROOT / "scripts/render_film.py"),
            *common,
            "--output",
            str(direct_output),
        ],
        cwd=ROOT,
    )
    installed = _run([str(cli), *common, "--output", str(installed_output)], cwd=ROOT)
    direct_recipe = json.loads(
        direct_output.with_suffix(".recipe.json").read_text("utf-8")
    )
    installed_recipe = json.loads(
        installed_output.with_suffix(".recipe.json").read_text("utf-8")
    )
    catalog_payload = json.loads(catalog.stdout) if catalog.returncode == 0 else {}
    return {
        "receipt": {
            "path": receipt_path.relative_to(ROOT).as_posix(),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256(receipt_path),
            "schema": receipt["schema"],
            "source_commit": receipt["source_commit"],
            "repository_binding": receipt["repository_binding"],
        },
        "catalog_returncode": catalog.returncode,
        "catalog_look_ids": [
            row["look_id"] for row in catalog_payload.get("looks", [])
        ],
        "desktop_help_returncode": desktop_help.returncode,
        "direct_returncode": direct.returncode,
        "installed_returncode": installed.returncode,
        "output_sha256": _sha256(installed_output),
        "direct_output_sha256": _sha256(direct_output),
        "recipe_current_head": installed_recipe["software"]["commit"],
        "direct_recipe_current_head": direct_recipe["software"]["commit"],
        "claim": installed_recipe["claim"],
        "source_commit_is_ancestor": _run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                receipt["source_commit"],
                current_head,
            ],
            cwd=ROOT,
            timeout=30,
        ).returncode
        == 0,
        "runtime_scope_diff_empty": _run(
            [
                "git",
                "-C",
                str(ROOT),
                "diff",
                "--quiet",
                receipt["source_commit"],
                "--",
                *receipt["repository_binding"]["runtime_scope"],
            ],
            cwd=ROOT,
            timeout=30,
        ).returncode
        == 0,
    }


def _report(order: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text("utf-8"))
    start_head = _git(ROOT, "rev-parse", "HEAD")
    preexisting_scratch = sorted(path.name for path in (ROOT / "tmp").glob("u7_9d_*"))
    if _git(ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("formal audit requires a tracked-clean worktree")
    source_objects = {
        path.as_posix(): _git(ROOT, "rev-parse", f"{start_head}:{path.as_posix()}")
        for path in SOURCE_PATHS
    }
    names = [
        "docs_descendant",
        "dirty_tracked_docs",
        "scoped_untracked",
        "runtime_commit_drift",
        "requirements_commit_drift",
        "non_descendant",
    ]
    if order == "reverse":
        names.reverse()
    with tempfile.TemporaryDirectory(prefix="u7_9d_", dir=ROOT / "tmp") as raw:
        scratch = Path(raw)
        controls = [_control(name, scratch / name) for name in names]
        smoke = _runtime_smoke(scratch, start_head)
    controls.sort(key=lambda row: row["name"])
    expected_binding = {
        "installed_source_commit": "e46588632959ee6ef4a150857b2d491b94514f99",
        "head_policy": "descendant",
        "runtime_scope": config["runtime_scope"],
        "runtime_scope_policy": "exact-to-installed-source-commit",
        "tracked_repository_policy": "clean",
        "runtime_scope_untracked_policy": "reject",
        "committed_non_runtime_drift_allowed": True,
    }
    expected_claim = {
        "calibrated_reference_allowed": False,
        "data_grade": "none",
        "evidence_grade": "look-approximation",
        "method": "heuristic",
        "output_label": "film-inspired",
    }
    gates = {
        "receipt_v4_exact": smoke["receipt"]["schema"] == config["receipt_schema"],
        "receipt_scope_disclosure_exact": smoke["receipt"]["repository_binding"]
        == expected_binding,
        "installed_source_commit_exact": smoke["receipt"]["source_commit"]
        == expected_binding["installed_source_commit"],
        "current_head_is_descendant": smoke["source_commit_is_ancestor"],
        "current_runtime_scope_exact": smoke["runtime_scope_diff_empty"],
        "docs_only_descendant_executes": smoke["catalog_returncode"] == 0,
        "catalog_exact": smoke["catalog_look_ids"]
        == ["velvia_50", "portra_400", "ektar_100", "generic_bw"],
        "desktop_launcher_policy_executes": smoke["desktop_help_returncode"] == 0,
        "installed_direct_output_exact": smoke["installed_returncode"] == 0
        and smoke["direct_returncode"] == 0
        and smoke["output_sha256"] == smoke["direct_output_sha256"],
        "recipe_records_current_head": smoke["recipe_current_head"]
        == smoke["direct_recipe_current_head"]
        == start_head,
        "look_approximation_claim_exact": smoke["claim"] == expected_claim,
        "all_negative_controls_exact": all(row["passed"] for row in controls),
        "source_stable": _git(ROOT, "rev-parse", "HEAD") == start_head
        and not _git(ROOT, "status", "--porcelain", "--untracked-files=no")
        and {
            path.as_posix(): _git(ROOT, "rev-parse", f"{start_head}:{path.as_posix()}")
            for path in SOURCE_PATHS
        }
        == source_objects,
        "owned_scratch_residue_unchanged": sorted(
            path.name for path in (ROOT / "tmp").glob("u7_9d_*")
        )
        == preexisting_scratch,
    }
    scientific = {
        "status": (
            "PASS_PRIVATE_U7_9D_RUNTIME_SOURCE_SCOPE_BINDING"
            if all(gates.values())
            else "FAIL_CLOSED_U7_9D_RUNTIME_SOURCE_SCOPE_BINDING"
        ),
        "installed_source_commit": smoke["receipt"]["source_commit"],
        "formal_source_commit": start_head,
        "runtime_scope": config["runtime_scope"],
        "controls": controls,
        "smoke": smoke,
        "gates": gates,
        "excluded_preformal_scratch": preexisting_scratch,
        "claim_ceiling": config["claim"],
    }
    return {
        "schema": "kmcfm.u7-9d-private-runtime-source-scope-binding-report.v1",
        "node_id": "U7.9D",
        "status": scientific["status"],
        "source_commit": start_head,
        "source_objects": source_objects,
        "scientific_identity": _canonical_sha256(scientific),
        "scientific": scientific,
    }


def main() -> int:
    args = _arguments()
    report = _report(args.order)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
