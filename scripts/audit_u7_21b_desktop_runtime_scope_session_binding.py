from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.product_desktop import (
    ProductDesktopError,
    _load_product_runtime_scope,
    _validate_runtime_source_scope,
)

CONFIG = ROOT / "configs/u7_21b_desktop_runtime_scope_session_binding_v1.json"
SCOPE_CONFIG = ROOT / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"
CORE = ROOT / "src/inference/product_desktop.py"
TEST = ROOT / "tests/test_u7_21b_desktop_runtime_scope_session_binding.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _head() -> str:
    return _git(ROOT, "rev-parse", "HEAD")


def _repository(parent: Path, name: str) -> tuple[Path, tuple[str, ...], str]:
    root = parent / name
    (root / "src/preprocess").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "scripts").mkdir()
    files = {
        "src/preprocess/raster_decode.py": "VALUE = 1\n",
        "configs/profile.json": "{}\n",
        "scripts/render_film.py": "raise SystemExit(0)\n",
        "scripts/open_product_desktop.py": "raise SystemExit(0)\n",
        "requirements-product-v2.txt": "Pillow==12.1.1\n",
        "docs/outside.md": "outside\n",
    }
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    _git(root, "config", "user.email", "u7.21b@example.invalid")
    _git(root, "config", "user.name", "U7.21B")
    _git(root, "add", "--all")
    _git(root, "commit", "-q", "-m", "fixture")
    scope = (
        "src",
        "configs",
        "scripts/render_film.py",
        "scripts/open_product_desktop.py",
        "requirements-product-v2.txt",
    )
    return root, scope, _git(root, "rev-parse", "HEAD")


def _error(action: Callable[[], None]) -> str | None:
    try:
        action()
    except ProductDesktopError as exc:
        return str(exc)
    return None


def _remove_readonly(
    function: Callable[[Path], object], path: Path, _error: OSError
) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _case(parent: Path, name: str) -> dict[str, object]:
    root, scope, commit = _repository(parent, name)
    target = root / "src/preprocess/raster_decode.py"
    if name == "clean":
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = None
    elif name == "outside_scope":
        (root / "docs/outside.md").write_text("changed\n", encoding="utf-8")
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = None
    elif name == "modified":
        target.write_text("VALUE = 2\n", encoding="utf-8")
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = "runtime source scope changed"
    elif name == "staged":
        target.write_text("VALUE = 3\n", encoding="utf-8")
        _git(root, "add", "src/preprocess/raster_decode.py")
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = "runtime source scope changed"
    elif name == "deleted":
        target.unlink()
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = "runtime source scope changed"
    elif name == "untracked":
        (root / "src/preprocess/injected.py").write_text(
            "VALUE = 4\n", encoding="utf-8"
        )
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = "runtime source scope changed"
    elif name == "descendant_head":
        (root / "docs/new.md").write_text("new\n", encoding="utf-8")
        _git(root, "add", "docs/new.md")
        _git(root, "commit", "-q", "-m", "descendant")
        error = _error(lambda: _validate_runtime_source_scope(root, commit, scope))
        expected = "preview renderer session changed"
    elif name == "invalid_scope":
        error = _error(
            lambda: _validate_runtime_source_scope(root, commit, ("../outside",))
        )
        expected = "runtime source scope invalid"
    else:
        raise ValueError(f"unknown case: {name}")
    return {
        "case": name,
        "error": error,
        "expected": expected,
        "pass": error == expected,
    }


def build_report(*, scratch: Path, order: str) -> dict[str, object]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if scratch.exists():
        raise FileExistsError("scratch path must be absent")
    scratch.mkdir(parents=True)
    names = [
        "clean",
        "outside_scope",
        "modified",
        "staged",
        "deleted",
        "untracked",
        "descendant_head",
        "invalid_scope",
    ]
    execution = names if order == "forward" else list(reversed(names))
    try:
        records = {name: _case(scratch, name) for name in execution}
        scope_payload = json.loads(SCOPE_CONFIG.read_text(encoding="utf-8"))
        expected_scope = tuple(scope_payload["runtime_scope"])
        loaded_scope = _load_product_runtime_scope(ROOT)
        rows = [records[name] for name in names]
        gates = {
            "authoritative_scope_exact": loaded_scope == expected_scope,
            "clean_scope_accepted": records["clean"]["pass"],
            "outside_scope_change_accepted": records["outside_scope"]["pass"],
            "modified_rejected": records["modified"]["pass"],
            "staged_rejected": records["staged"]["pass"],
            "deleted_rejected": records["deleted"]["pass"],
            "untracked_rejected": records["untracked"]["pass"],
            "descendant_head_rejected": records["descendant_head"]["pass"],
            "invalid_scope_rejected": records["invalid_scope"]["pass"],
            "claim_ceiling_exact": True,
        }
        return {
            "schema_version": "neuro-film.u7-21b-desktop-runtime-scope-session-binding-result.v1",
            "source_commit": _head(),
            "status": (
                "PASS_PRIVATE_U7_21B_DESKTOP_RUNTIME_SCOPE_SESSION_BINDING"
                if all(gates.values())
                else "FAIL_CLOSED_U7_21B_DESKTOP_RUNTIME_SCOPE_SESSION_BINDING"
            ),
            "runtime_scope": list(loaded_scope),
            "cases": rows,
            "gates": gates,
            "source_bindings": {
                path.relative_to(ROOT).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (CONFIG, SCOPE_CONFIG, CORE, TEST)
            },
            "claim_ceiling": {
                "evidence_grade": "look-approximation",
                "calibrated_stock_response": False,
                "physical_film_reproduction": False,
                "scope": "private desktop session-to-runtime-source consistency",
            },
        }
    finally:
        shutil.rmtree(scratch, onexc=_remove_readonly)


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
