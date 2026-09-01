#!/usr/bin/env python3
"""Audit the U7.14B canonical product-desktop scratch boundary."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import types
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import open_product_desktop

CONFIG = ROOT / "configs/u7_14b_desktop_canonical_scratch_boundary_v1.json"
CONTRACT = ROOT / "docs/planning/U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY_CONTRACT.md"
ENTRYPOINT = ROOT / "scripts/open_product_desktop.py"
IMPLEMENTATION_TEST = ROOT / "tests/test_u7_14b_desktop_canonical_scratch_boundary.py"
PARENT_TEST = ROOT / "tests/test_u7_10a_product_desktop_input_workflow.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _run(
    command: Sequence[str], *, cwd: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def _git_text(*args: str) -> str:
    result = _run(("git", *args))
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "git command failed")
    return result.stdout


def _parent_defect(config: dict[str, Any], external: Path) -> dict[str, Any]:
    parent = str(config["parent_head"])
    source = _git_text("show", f"{parent}:scripts/open_product_desktop.py")
    observed: dict[str, Any] = {"constructed": False, "mainloop": False}

    product_module = types.ModuleType("src.inference.product_desktop")

    class StubWorkflow:
        def __init__(self, *, root: Path, scratch_root: Path) -> None:
            observed["constructed"] = True
            observed["root"] = str(root)
            observed["scratch_root"] = str(scratch_root)

    product_module.ProductDesktopWorkflow = StubWorkflow  # type: ignore[attr-defined]

    ui_module = types.ModuleType("src.inference.product_desktop_ui")

    class StubApp:
        def close(self) -> None:
            observed["closed"] = True

    def build_product_desktop_app(
        root: object, workflow: object, initial_input: Path | None = None
    ) -> StubApp:
        del root, workflow
        observed["input"] = None if initial_input is None else str(initial_input)
        return StubApp()

    ui_module.build_product_desktop_app = build_product_desktop_app  # type: ignore[attr-defined]

    tk_module = types.ModuleType("tkinter")

    class StubRoot:
        def after(self, delay: int, callback: object) -> None:
            observed["delay"] = int(delay)
            observed["callback"] = callable(callback)

        def mainloop(self) -> None:
            observed["mainloop"] = True

    tk_module.Tk = StubRoot  # type: ignore[attr-defined]
    replacements = {
        "src.inference.product_desktop": product_module,
        "src.inference.product_desktop_ui": ui_module,
        "tkinter": tk_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    argv = sys.argv
    try:
        sys.modules.update(replacements)
        namespace = {
            "__file__": str(ENTRYPOINT),
            "__name__": "u7_14b_parent_entrypoint",
        }
        exec(  # noqa: S102 - exact frozen Git blob in isolated stub modules
            compile(source, str(ENTRYPOINT), "exec"), namespace
        )
        sys.argv = [
            str(ENTRYPOINT),
            "--input",
            str(external / "unread.png"),
            "--scratch-root",
            str(external),
            "--smoke-exit-ms",
            "1",
        ]
        observed["returncode"] = int(namespace["main"]())
    finally:
        sys.argv = argv
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    observed["crossed_external_boundary"] = (
        observed.get("constructed") is True
        and Path(str(observed.get("scratch_root"))).resolve(strict=True)
        == external.resolve(strict=True)
        and observed.get("mainloop") is True
        and observed.get("returncode") == 0
    )
    return observed


def _current_preimport_rejection(external: Path) -> dict[str, Any]:
    imported: list[str] = []
    original_import = builtins.__import__
    argv = sys.argv

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tkinter" or name.startswith("tkinter."):
            imported.append(name)
            raise AssertionError("tkinter imported before scratch rejection")
        if name in {
            "src.inference.product_desktop",
            "src.inference.product_desktop_ui",
        }:
            imported.append(name)
            raise AssertionError("product module imported before scratch rejection")
        return original_import(name, *args, **kwargs)

    stderr = io.StringIO()
    try:
        builtins.__import__ = guarded_import
        sys.argv = [
            str(ENTRYPOINT),
            "--input",
            str(external / "must-not-be-read.png"),
            "--scratch-root",
            str(external),
        ]
        with contextlib.redirect_stderr(stderr):
            returncode = open_product_desktop.main()
    finally:
        builtins.__import__ = original_import
        sys.argv = argv
    return {
        "returncode": int(returncode),
        "imported_forbidden_modules": imported,
        "stderr_has_boundary": "repository tmp root or its descendant"
        in stderr.getvalue(),
        "input_exists": (external / "must-not-be-read.png").exists(),
    }


def _smoke(source: Path, scratch: Path | None) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        str(ENTRYPOINT),
        "--input",
        str(source),
        "--smoke-exit-ms",
        "200",
    ]
    if scratch is not None:
        command.extend(("--scratch-root", str(scratch)))
    result = _run(command)
    return {
        "returncode": result.returncode,
        "stderr": result.stderr[-1000:],
    }


def _make_source(path: Path) -> None:
    pixels = bytes((index * 17 + 13) % 256 for index in range(31 * 23 * 3))
    Image.frombytes("RGB", (31, 23), pixels).save(path)


def build_report(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    config = json.loads(CONFIG.read_text("utf-8"))
    head = _git_text("rev-parse", "HEAD").strip()
    if _git_text("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("tracked worktree must be clean before formal execution")

    source_paths = (CONFIG, CONTRACT, ENTRYPOINT, IMPLEMENTATION_TEST, PARENT_TEST)
    source_before = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    canonical = (ROOT / "tmp").resolve(strict=True)
    outputs = (ROOT / "outputs").resolve(strict=True)
    work = canonical / f"u7_14b_formal_{uuid.uuid4().hex}"
    external = outputs / f"u7_14b_external_{uuid.uuid4().hex}"
    work.mkdir()
    external.mkdir()
    sentinel = external / "foreign.bin"
    sentinel.write_bytes(b"u7-14b-foreign-sentinel")
    sentinel_sha = _sha256(sentinel)
    accepted = work / "accepted"
    audit_child = work / "u7_10b_formal_descendant"
    accepted.mkdir()
    audit_child.mkdir()
    source = work / "source.png"
    _make_source(source)
    regular = work / "regular.bin"
    regular.write_bytes(b"regular-file-control")

    rejected_candidates = {
        "project_root": ROOT,
        "data": ROOT / "data",
        "outputs": ROOT / "outputs",
        "drive_root": Path(ROOT.anchor),
        "normalized_escape": ROOT / "tmp" / ".." / "outputs",
        "external_project_path": external,
        "missing": work / "missing",
        "regular_file": regular,
    }
    ordered_ids = list(rejected_candidates)
    if order == "reverse":
        ordered_ids.reverse()
    rejected: dict[str, bool] = {}
    for control_id in ordered_ids:
        candidate = rejected_candidates[control_id]
        try:
            open_product_desktop.resolve_product_scratch_root(candidate)
        except (OSError, ValueError):
            rejected[control_id] = True
        else:
            rejected[control_id] = False

    link = work / "escape-link"
    link_supported = True
    link_rejected = False
    try:
        link.symlink_to(external, target_is_directory=True)
    except OSError:
        link_supported = False
    else:
        try:
            open_product_desktop.resolve_product_scratch_root(link)
        except (OSError, ValueError):
            link_rejected = True

    parent_defect = _parent_defect(config, external)
    preimport = _current_preimport_rejection(external)
    accepted_results = {
        "canonical": str(
            open_product_desktop.resolve_product_scratch_root(ROOT / "tmp") == canonical
        ).lower(),
        "descendant": str(
            open_product_desktop.resolve_product_scratch_root(accepted) == accepted
        ).lower(),
        "normalized_descendant": str(
            open_product_desktop.resolve_product_scratch_root(
                ROOT / "tmp" / "." / work.name / accepted.name
            )
            == accepted
        ).lower(),
        "parent_launcher_descendant": str(
            open_product_desktop.resolve_product_scratch_root(audit_child)
            == audit_child
        ).lower(),
    }
    smoke_default = _smoke(source, None)
    smoke_explicit = _smoke(source, accepted)

    source_after = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    foreign_exact = _sha256(sentinel) == sentinel_sha
    if link.exists() or link.is_symlink():
        link.unlink()
    for path in (regular, source):
        path.unlink()
    audit_child.rmdir()
    accepted.rmdir()
    work.rmdir()
    sentinel.unlink()
    external.rmdir()

    science = {
        "accepted": accepted_results,
        "rejected": dict(sorted(rejected.items())),
        "reparse_escape": {
            "supported": link_supported,
            "rejected_when_supported": link_rejected if link_supported else None,
        },
        "parent_defect_reproduced": bool(parent_defect["crossed_external_boundary"]),
        "current_preimport_rejection": preimport,
        "smoke": {
            "default_returncode": smoke_default["returncode"],
            "explicit_descendant_returncode": smoke_explicit["returncode"],
        },
        "foreign_sentinel_exact": foreign_exact,
        "source_immutable": source_before == source_after,
    }
    gates = {
        "parent_defect_reproduced": science["parent_defect_reproduced"],
        "canonical_and_descendants_accepted": all(
            value == "true" for value in accepted_results.values()
        ),
        "external_and_normalized_escapes_rejected": all(rejected.values()),
        "reparse_escape_rejected_when_supported": (
            link_rejected if link_supported else True
        ),
        "rejected_before_tk_input_and_scratch": (
            preimport["returncode"] == 2
            and preimport["imported_forbidden_modules"] == []
            and preimport["stderr_has_boundary"] is True
            and preimport["input_exists"] is False
        ),
        "foreign_sentinel_immutable": foreign_exact,
        "real_default_smoke_mainloop": smoke_default["returncode"] == 0,
        "real_explicit_descendant_smoke_mainloop": (smoke_explicit["returncode"] == 0),
        "source_immutable": source_before == source_after,
        "tracked_clean_before": True,
        "owned_residue_zero": not work.exists() and not external.exists(),
    }
    scientific_identity = hashlib.sha256(_canonical_bytes(science)).hexdigest()
    return {
        "schema": "kmcfm.u7-14b-desktop-canonical-scratch-boundary-report.v1",
        "experiment_id": "U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY",
        "status": (
            "PASS_PRIVATE_U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY"
            if all(gates.values())
            else "FAIL_CLOSED_U7_14B_DESKTOP_CANONICAL_SCRATCH_BOUNDARY"
        ),
        "source_commit": head,
        "scientific_identity": scientific_identity,
        "science": science,
        "gates": gates,
        "bindings": source_before,
        "claim": {
            "mode": "film-inspired / Look Approximation",
            "canonical_repo_storage_only": True,
            "filesystem_sandbox": False,
            "standalone_installer": False,
            "public_release": False,
            "cross_platform_gui": False,
            "calibrated_camera_rendering": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve(strict=False)
    output_root = (ROOT / "outputs").resolve(strict=True)
    if output_root not in output.parents:
        raise ValueError("formal report must be under repository outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(args.order)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
