#!/usr/bin/env python3
"""Audit the U7.15A explicit desktop batch-preview representative."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_desktop import (
    ProductDesktopError,
    ProductDesktopWorkflow,
)

CONFIG = ROOT / "configs/u7_15a_desktop_batch_representative_selection_v1.json"
CONTRACT = (
    ROOT / "docs/planning/U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION_CONTRACT.md"
)
CORE = ROOT / "src/inference/product_desktop.py"
UI = ROOT / "src/inference/product_desktop_ui.py"
IMPLEMENTATION_TEST = (
    ROOT / "tests/test_u7_15a_desktop_batch_representative_selection.py"
)
PARENT_TEST = ROOT / "tests/test_u7_11a_desktop_single_look_batch.py"


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


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )


def _git_text(*args: str) -> str:
    result = _run(("git", *args))
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "git command failed")
    return result.stdout


def _source(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:31, :47]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _workflow(scratch: Path, **kwargs: object) -> ProductDesktopWorkflow:
    scratch.mkdir()
    return ProductDesktopWorkflow(
        root=ROOT,
        scratch_root=scratch,
        python_executable=Path(sys.executable),
        max_preview_pixels=2_000,
        tile_size=256,
        tile_workers=1,
        png_compression=6,
        **kwargs,  # type: ignore[arg-type]
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _parent_method(parent: str) -> Callable[..., object]:
    source = _git_text("show", f"{parent}:src/inference/product_desktop.py")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ProductDesktopWorkflow":
            for child in node.body:
                if (
                    isinstance(child, ast.FunctionDef)
                    and child.name == "render_batch_previews"
                ):
                    namespace: dict[str, object] = {}
                    extracted = "from __future__ import annotations\n" + ast.unparse(
                        child
                    )
                    exec(  # noqa: S102 - exact frozen Git method in a stub-only namespace
                        compile(extracted, "<u7_15a_parent_method>", "exec"), namespace
                    )
                    return namespace["render_batch_previews"]  # type: ignore[return-value]
    raise RuntimeError("parent render_batch_previews method not found")


def _parent_limitation(parent: str, selected: tuple[Path, ...]) -> dict[str, object]:
    canonical = tuple(sorted(selected, key=lambda path: str(path.resolve()).casefold()))
    rows = tuple(
        SimpleNamespace(path=path.resolve(), sha256=_sha256(path)) for path in canonical
    )
    observed: list[Path] = []

    class Stub:
        def bind_batch_inputs(self, paths: object) -> tuple[SimpleNamespace, ...]:
            del paths
            return rows

        def render_previews(self, path: Path, amount: float) -> SimpleNamespace:
            del amount
            observed.append(path)
            return SimpleNamespace(input_path=path, input_sha256=_sha256(path))

        def close(self) -> None:
            raise AssertionError("parent limitation unexpectedly closed")

    method = _parent_method(parent)
    state, returned = method(Stub(), selected, 0.625)  # type: ignore[misc]
    return {
        "selected_order": [path.name for path in selected],
        "canonical_order": [row.path.name for row in rows],
        "observed_preview": observed[0].name,
        "returned_order": [row.path.name for row in returned],
        "canonical_first_only": (
            state.input_path == rows[0].path
            and observed == [rows[0].path]
            and tuple(returned) == rows
        ),
    }


def _rejection_controls(work: Path, order: str) -> dict[str, bool]:
    first = work / "reject-a.png"
    second = work / "reject-b.png"
    external = work / "reject-external.png"
    alias = work / "reject-alias.png"
    _source(first, 31)
    _source(second, 32)
    _source(external, 33)
    os.link(second, alias)
    calls = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("preview renderer crossed rejection boundary")

    controls = ["external", "missing", "alias", "mutation"]
    if order == "reverse":
        controls.reverse()
    results: dict[str, bool] = {}
    workflow = _workflow(work / "reject-scratch", preview_renderer=forbidden)
    try:
        for control in controls:
            if control == "external":
                representative = external
            elif control == "missing":
                representative = work / "missing.png"
            elif control == "alias":
                representative = alias
            else:
                stale = workflow.bind_batch_inputs((first, second))
                _source(second, 99)

                def stale_binding(
                    _paths: object,
                    frozen: object = stale,
                ) -> object:
                    return frozen

                original = workflow.bind_batch_inputs
                workflow.bind_batch_inputs = stale_binding  # type: ignore[method-assign]
                try:
                    workflow.render_batch_previews(
                        (first, second), 0.5, representative_path=second
                    )
                except ProductDesktopError:
                    results[control] = True
                else:
                    results[control] = False
                finally:
                    workflow.bind_batch_inputs = original  # type: ignore[method-assign]
                continue
            try:
                workflow.render_batch_previews(
                    (first, second), 0.5, representative_path=representative
                )
            except ProductDesktopError:
                results[control] = True
            else:
                results[control] = False
    finally:
        workflow.close()
    results["preview_renderer_calls_zero"] = calls == 0
    return dict(sorted(results.items()))


def build_report(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    config = json.loads(CONFIG.read_text("utf-8"))
    head = _git_text("rev-parse", "HEAD").strip()
    if _git_text("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("tracked worktree must be clean before formal execution")

    source_paths = (
        CONFIG,
        CONTRACT,
        CORE,
        UI,
        IMPLEMENTATION_TEST,
        PARENT_TEST,
        Path(__file__).resolve(),
    )
    source_before = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    work = ROOT / "tmp/u7_15a_formal"
    destination = (
        ROOT / "outputs/eval/u7_15a_desktop_batch_representative_selection/published"
    )
    if work.exists() or destination.exists():
        raise RuntimeError("owned formal path must be absent before execution")
    work.mkdir()
    later = work / "z-later.png"
    canonical_first = work / "a-first.png"
    _source(later, 1)
    _source(canonical_first, 2)
    selected = (later, canonical_first)
    if order == "reverse":
        selected = tuple(reversed(selected))
    input_before = {path.name: _sha256(path) for path in selected}

    parent = _parent_limitation(str(config["parent_head"]), selected)
    default = _workflow(work / "default-scratch")
    default_state, default_rows = default.render_batch_previews(selected, 0.625)
    default_receipt = default.export_batch(default_rows, "portra_400", destination)
    default_tree = _tree_hashes(destination)
    default.close()
    shutil.rmtree(destination)

    explicit = _workflow(work / "explicit-scratch")
    explicit_state, explicit_rows = explicit.render_batch_previews(
        selected, 0.625, representative_path=later
    )
    explicit_receipt = explicit.export_batch(explicit_rows, "portra_400", destination)
    explicit_tree = _tree_hashes(destination)
    explicit.close()

    rejection = _rejection_controls(work, order)
    tests = {
        "focused": _run(
            (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_u7_15a_desktop_batch_representative_selection.py",
            )
        ).returncode,
        "parent_ui": _run(
            (
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_u7_11a_desktop_single_look_batch.py",
            )
        ).returncode,
    }
    input_after = {path.name: _sha256(path) for path in selected}
    source_after = {
        path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths
    }
    science = {
        "parent_limitation": parent,
        "current": {
            "default_preview": default_state.input_path.name,
            "explicit_preview": explicit_state.input_path.name,
            "canonical_order": [row.path.name for row in explicit_rows],
            "default_and_explicit_rows_exact": default_rows == explicit_rows,
            "receipt_sha256": explicit_receipt.receipt_sha256,
            "receipt_exact": explicit_receipt.receipt == default_receipt.receipt,
            "tree_hashes": explicit_tree,
            "tree_exact": explicit_tree == default_tree,
        },
        "rejection_controls": rejection,
        "tests": tests,
        "input_immutable": input_before == input_after,
        "source_immutable": source_before == source_after,
    }
    gates = {
        "parent_limitation_reproduced": parent["canonical_first_only"] is True,
        "default_canonical_first_exact": (
            default_state.input_path == default_rows[0].path
        ),
        "explicit_nonfirst_preview": explicit_state.input_path == later.resolve(),
        "canonical_batch_order_unchanged": default_rows == explicit_rows,
        "batch_receipt_exact": explicit_receipt.receipt == default_receipt.receipt,
        "batch_tree_exact": explicit_tree == default_tree,
        "invalid_controls_reject_before_preview": all(rejection.values()),
        "focused_tests_pass": tests["focused"] == 0,
        "parent_ui_tests_pass": tests["parent_ui"] == 0,
        "input_immutable": input_before == input_after,
        "source_immutable": source_before == source_after,
        "tracked_clean_before": True,
    }
    scientific_identity = hashlib.sha256(_canonical_bytes(science)).hexdigest()

    shutil.rmtree(destination)
    for child in (work / "default-scratch", work / "explicit-scratch"):
        if child.exists():
            child.rmdir()
    shutil.rmtree(work)
    gates["owned_residue_zero"] = not work.exists() and not destination.exists()
    return {
        "schema": "kmcfm.u7-15a-desktop-batch-representative-selection-report.v1",
        "experiment_id": "U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION",
        "status": (
            "PASS_PRIVATE_U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION"
            if all(gates.values())
            else "FAIL_CLOSED_U7_15A_DESKTOP_BATCH_REPRESENTATIVE_SELECTION"
        ),
        "source_commit": head,
        "scientific_identity": scientific_identity,
        "science": science,
        "gates": gates,
        "bindings": source_before,
        "claim": {
            "mode": "film-inspired / Look Approximation",
            "explicit_user_authority_only": True,
            "automatic_aesthetic_routing": False,
            "calibrated_camera_rendering": False,
            "calibrated_stock_response": False,
            "physical_film_reproduction": False,
            "stock_distinguishability": False,
            "product_value_evidence": False,
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
