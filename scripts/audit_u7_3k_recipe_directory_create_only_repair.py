"""Committed-head formal audit for U7.3K directory ownership repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import create_only_directory as directory_module
from src.inference.recipe_export_request import (
    build_recipe_export_request_set,
    materialize_recipe_export_request_set,
)
from src.inference.recipe_workspace import (
    build_offline_recipe_workspace,
    materialize_offline_recipe_workspace,
)

CONFIG = ROOT / "configs/u7_3k_recipe_directory_create_only_repair_v1.json"

Materializer = Callable[[Path, Path], object]
Builder = Callable[[Path], dict[str, bytes]]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _materializers() -> dict[str, tuple[Materializer, Builder]]:
    return {
        "request_set": (
            materialize_recipe_export_request_set,
            lambda root: build_recipe_export_request_set(root)["files"],
        ),
        "workspace": (
            materialize_offline_recipe_workspace,
            build_offline_recipe_workspace,
        ),
    }


def _snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path.read_bytes())}
        for path in sorted(root.iterdir(), key=lambda item: item.name)
    }


def _expected_snapshot(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(files.items())
    }


def _run_controls(
    name: str,
    materialize: Materializer,
    build: Builder,
    work: Path,
) -> dict[str, Any]:
    history = work / f"{name}-history"
    history.mkdir()

    success = work / f"{name}-success"
    expected = build(history)
    materialize(history, success)
    success_exact = _snapshot(success) == _expected_snapshot(expected)

    initial = work / f"{name}-initial-foreign"
    original_mkdir = Path.mkdir

    def foreign_wins(path: Path, *args: object, **kwargs: object) -> None:
        if path == initial:
            original_mkdir(path, *args, **kwargs)
            (path / "foreign.bin").write_bytes(b"initial-foreign")
            raise FileExistsError("injected initial foreign directory")
        original_mkdir(path, *args, **kwargs)

    initial_rejected = False
    try:
        with patch.object(Path, "mkdir", foreign_wins):
            materialize(history, initial)
    except FileExistsError:
        initial_rejected = True
    initial_preserved = (
        initial_rejected
        and (initial / "foreign.bin").read_bytes() == b"initial-foreign"
        and [path.name for path in initial.iterdir()] == ["foreign.bin"]
    )

    added = work / f"{name}-added-foreign"
    original_write = directory_module._write_owned_file
    write_calls = 0

    def add_then_fail(path: Path, payload: bytes) -> object:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            (added / "foreign.bin").write_bytes(b"added-foreign")
            raise OSError("injected post-claim failure")
        return original_write(path, payload)

    added_rejected = False
    try:
        with patch.object(directory_module, "_write_owned_file", add_then_fail):
            materialize(history, added)
    except OSError:
        added_rejected = True
    added_preserved = (
        added_rejected
        and (added / "foreign.bin").read_bytes() == b"added-foreign"
        and [path.name for path in added.iterdir()] == ["foreign.bin"]
    )

    replaced = work / f"{name}-replaced-foreign"
    replacement = work / f"{name}-replacement.bin"
    replacement.write_bytes(b"replacement-foreign")
    original_require = directory_module._require_owned_file
    require_calls = 0

    def replace_then_fail(owned: object) -> None:
        nonlocal require_calls
        require_calls += 1
        if require_calls == 1:
            os.replace(replacement, owned.identity.path)  # type: ignore[attr-defined]
            raise directory_module.CreateOnlyDirectoryError("injected replacement")
        original_require(owned)  # type: ignore[arg-type]

    replacement_rejected = False
    try:
        with patch.object(directory_module, "_require_owned_file", replace_then_fail):
            materialize(history, replaced)
    except directory_module.CreateOnlyDirectoryError:
        replacement_rejected = True
    replacement_files = sorted(path for path in replaced.iterdir() if path.is_file())
    replacement_preserved = (
        replacement_rejected
        and len(replacement_files) == 1
        and replacement_files[0].read_bytes() == b"replacement-foreign"
    )

    clean = work / f"{name}-clean-failure"

    def clean_fail(_path: Path, _payload: bytes) -> object:
        raise OSError("injected clean failure")

    clean_rejected = False
    try:
        with patch.object(directory_module, "_write_owned_file", clean_fail):
            materialize(history, clean)
    except OSError:
        clean_rejected = True

    existing = work / f"{name}-existing"
    existing.mkdir()
    (existing / "foreign.bin").write_bytes(b"existing-foreign")
    existing_rejected = False
    try:
        materialize(history, existing)
    except ValueError:
        existing_rejected = True

    return {
        "name": name,
        "success_snapshot": _snapshot(success),
        "success_exact": success_exact,
        "initial_foreign_directory_preserved": initial_preserved,
        "foreign_added_member_preserved": added_preserved,
        "foreign_replacement_preserved": replacement_preserved,
        "clean_failure_rejected": clean_rejected,
        "clean_failure_residue_zero": clean_rejected and not clean.exists(),
        "existing_foreign_preserved": existing_rejected
        and (existing / "foreign.bin").read_bytes() == b"existing-foreign",
    }


def run_audit(order: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    source_bindings = bindings["implementation_git_lf_sha256"]
    binding_exact = all(
        _sha256(_git_bytes(bindings["implementation_commit"], path)) == expected
        for path, expected in source_bindings.items()
    )
    contract_resolves = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{bindings['contract_commit']}^{{commit}}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )

    names = ["request_set", "workspace"]
    if order == "reverse":
        names.reverse()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="u7-3k-") as raw:
        work = Path(raw)
        materializers = _materializers()
        for name in names:
            materialize, build = materializers[name]
            rows.append(_run_controls(name, materialize, build, work))
    rows.sort(key=lambda row: row["name"])

    helper_source = _git_bytes(
        bindings["implementation_commit"], "src/inference/create_only_directory.py"
    ).decode("utf-8")
    caller_sources = b"\n".join(
        _git_bytes(bindings["implementation_commit"], path)
        for path in (
            "src/inference/recipe_export_request.py",
            "src/inference/recipe_workspace.py",
        )
    ).decode("utf-8")
    no_recursive_cleanup = "rmtree" not in helper_source + caller_sources
    all_rows = lambda key: all(bool(row[key]) for row in rows)
    gates = {
        "source_bindings_exact": binding_exact and contract_resolves,
        "successful_workspace_bytes_exact": all_rows("success_exact"),
        "successful_request_set_bytes_exact": all_rows("success_exact"),
        "initial_foreign_directory_preserved": all_rows(
            "initial_foreign_directory_preserved"
        ),
        "foreign_added_member_preserved": all_rows("foreign_added_member_preserved"),
        "foreign_replacement_preserved": all_rows("foreign_replacement_preserved"),
        "owned_residue_zero_when_no_foreign_entry_remains": all_rows(
            "clean_failure_residue_zero"
        ),
        "existing_foreign_preserved": all_rows("existing_foreign_preserved"),
        "no_recursive_destination_cleanup": no_recursive_cleanup,
    }
    scientific = {
        "node_id": "U7.3K",
        "bindings": {
            "contract_commit": bindings["contract_commit"],
            "implementation_commit": bindings["implementation_commit"],
            "implementation_git_lf_sha256": source_bindings,
        },
        "rows": rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
        "product_facts": {
            "renderer_changed": False,
            "recipe_schema_changed": False,
            "successful_file_bytes_changed": False,
            "stock_response_or_calibration_claim": False,
        },
    }
    stable_identity = _sha256(_canonical_json(scientific))
    return {
        "schema": "kmcfm.u7-3k-recipe-directory-create-only-repair-result.v1",
        "status": "PASS_PRIVATE_U7_3K_RECIPE_DIRECTORY_CREATE_ONLY_REPAIR"
        if all(gates.values())
        else "FAIL_CLOSED_U7_3K_RECIPE_DIRECTORY_CREATE_ONLY_REPAIR",
        "formal_execution_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "stable_identity": stable_identity,
        **scientific,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.order)
    payload = _canonical_json(report)
    if args.output is None:
        print(payload.decode("utf-8"), end="")
    else:
        args.output.write_bytes(payload)
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
