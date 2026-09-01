#!/usr/bin/env python3
"""Audit isolated startup of the private U7.9A product runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.install_product_runtime import _launcher_source, install_product_runtime

CONFIG = ROOT / "configs" / "u7_9c_product_runtime_startup_isolation_v1.json"
CONTRACT = (
    ROOT
    / "docs"
    / "planning"
    / "U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION_CONTRACT.md"
)
INSTALLER = ROOT / "scripts" / "install_product_runtime.py"
RENDERER = ROOT / "scripts" / "render_film.py"
REQUIREMENTS = ROOT / "requirements-product-v2.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    return int(details.st_dev), int(details.st_ino)


def _remove_tree(path: Path, expected_identity: tuple[int, int]) -> bool:
    if not os.path.lexists(path):
        return True
    details = path.stat(follow_symlinks=False)
    attributes = int(getattr(details, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if (
        not stat.S_ISDIR(details.st_mode)
        or attributes & reparse
        or _entry_identity(path) != expected_identity
    ):
        return False

    def clear_readonly(function, member, _error):  # type: ignore[no-untyped-def]
        os.chmod(member, stat.S_IWRITE)
        function(member)

    try:
        shutil.rmtree(path, onexc=clear_readonly)
    except OSError:
        return False
    return not path.exists()


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
        timeout=180,
    )


def _success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[-4000:])
    return result.stdout


def _git(*arguments: str) -> str:
    return _success(
        _run(["git", "-C", str(ROOT), *arguments], cwd=ROOT, env=dict(os.environ))
    ).strip()


def _tiny_input(path: Path) -> None:
    yy, xx = np.indices((17, 23), dtype=np.uint16)
    rgb = np.stack(
        (
            (xx * 11 + yy * 3) % 256,
            (xx * 5 + yy * 13 + 17) % 256,
            (xx * 7 + yy * 9 + 31) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)


def _launch(
    launcher: Path,
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return _run(["cmd.exe", "/d", "/c", str(launcher), *arguments], cwd=cwd, env=env)


def _source_facts() -> dict[str, str]:
    return {
        "config_sha256": _sha256(CONFIG),
        "contract_sha256": _sha256(CONTRACT),
        "installer_sha256": _sha256(INSTALLER),
        "renderer_sha256": _sha256(RENDERER),
        "requirements_sha256": _sha256(REQUIREMENTS),
    }


def _mixed_case_environment_control(
    root: Path, runtime_python: Path, hostile_environment: dict[str, str]
) -> bool:
    repository = root / "sentinel-repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    requirements = repository / "requirements.txt"
    requirements.write_text("sentinel\n", encoding="utf-8", newline="\n")
    (scripts / "render_film.py").write_text(
        "import json, os\n"
        "print(json.dumps(sorted(k for k in os.environ if k.upper().startswith('PYTHON'))))\n",
        encoding="utf-8",
        newline="\n",
    )
    git_environment = dict(os.environ)
    _success(
        _run(["git", "init", str(repository)], cwd=root, env=git_environment)
    )
    prefix = [
        "git",
        "-c",
        f"safe.directory={repository}",
        "-C",
        str(repository),
    ]
    for arguments in (
        ["config", "user.email", "u7.9c@example.invalid"],
        ["config", "user.name", "U7.9C"],
        ["add", "scripts/render_film.py", "requirements.txt"],
        ["commit", "-m", "sentinel"],
    ):
        _success(_run([*prefix, *arguments], cwd=root, env=git_environment))
    commit = _success(
        _run([*prefix, "rev-parse", "HEAD"], cwd=root, env=git_environment)
    ).strip()
    launcher_py = root / "sentinel-launch.py"
    launcher_py.write_text(
        _launcher_source(
            project_root=repository,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=_sha256(requirements),
        ),
        encoding="utf-8",
        newline="\n",
    )
    launcher_cmd = root / "sentinel-launch.cmd"
    launcher_cmd.write_text(
        f'@echo off\r\n"{runtime_python}" -I "{launcher_py}" %*\r\n',
        encoding="utf-8",
        newline="",
    )
    mixed = {
        key: value
        for key, value in hostile_environment.items()
        if not key.upper().startswith("PYTHON")
    }
    mixed["PyThOnPaTh"] = hostile_environment["PYTHONPATH"]
    mixed["pYtHoNuSeRbAsE"] = hostile_environment["PYTHONUSERBASE"]
    mixed["PyThOnInSpEcT"] = "1"
    mixed["GIT_CONFIG_COUNT"] = "1"
    mixed["GIT_CONFIG_KEY_0"] = "safe.directory"
    mixed["GIT_CONFIG_VALUE_0"] = str(repository)
    result = _launch(launcher_cmd, [], cwd=root, env=mixed)
    return result.returncode == 0 and json.loads(result.stdout) == []


def audit(root: Path, wheelhouse: Path, order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError(order)
    root = root.resolve(strict=False)
    wheelhouse = wheelhouse.resolve(strict=True)
    tmp = (ROOT / "tmp").resolve(strict=True)
    if tmp not in root.parents or tmp not in wheelhouse.parents:
        raise ValueError("formal roots must be repo-relative tmp paths")
    if os.path.lexists(root):
        raise FileExistsError("formal root must be absent")
    root.mkdir(parents=True)
    root_identity = _entry_identity(root)
    source_before = _source_facts()
    config = json.loads(CONFIG.read_text("utf-8"))
    commit = _git("rev-parse", "HEAD")
    tracked_clean = not _git("status", "--porcelain", "--untracked-files=no")
    if _git(
        "merge-base", "--is-ancestor", config["implementation"]["commit"], commit
    ):
        raise RuntimeError("frozen implementation commit is not an ancestor")
    if _git("rev-parse", "HEAD:scripts/install_product_runtime.py") != config[
        "implementation"
    ]["installer_git_blob"]:
        raise RuntimeError("installer Git blob differs from frozen implementation")
    if not tracked_clean:
        raise RuntimeError("tracked tree must be clean")

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    installation = root / "installation"
    receipt = install_product_runtime(
        installation, wheelhouse=wheelhouse, environment=environment
    )
    runtime_python = installation / "runtime" / "Scripts" / "python.exe"
    launcher = installation / "kmcfm-look.cmd"
    launcher_py = installation / "product-launch.py"
    foreign_cwd = root / "foreign-cwd"
    foreign_cwd.mkdir()

    hostile = root / "hostile-pythonpath"
    hostile.mkdir()
    marker = root / "sitecustomize-executed.txt"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
        newline="\n",
    )
    hostile_environment = dict(environment)
    hostile_environment.update(
        {
            "PYTHONPATH": str(hostile),
            "PYTHONUSERBASE": str(hostile / "user-base"),
            "PYTHONINSPECT": "1",
        }
    )

    catalog = json.loads(
        _success(
            _launch(
                launcher,
                ["--list-product-looks"],
                cwd=foreign_cwd,
                env=hostile_environment,
            )
        )
    )
    marker_absent_after_catalog = not marker.exists()

    input_path = root / "input.png"
    _tiny_input(input_path)
    looks = list(config["unchanged"]["available_looks"])
    execution_order = list(reversed(looks)) if order == "reverse" else looks
    render_rows: list[dict[str, Any]] = []
    for look in execution_order:
        direct = root / f"direct-{look}.png"
        launched = root / f"launched-{look}.png"
        _success(
            _run(
                [
                    str(runtime_python),
                    "-I",
                    str(RENDERER),
                    str(input_path),
                    "--product-look",
                    look,
                    "--output",
                    str(direct),
                ],
                cwd=ROOT,
                env={
                    key: value
                    for key, value in environment.items()
                    if not key.upper().startswith("PYTHON")
                },
            )
        )
        _success(
            _launch(
                launcher,
                [
                    str(input_path),
                    "--product-look",
                    look,
                    "--output",
                    str(launched),
                ],
                cwd=foreign_cwd,
                env=hostile_environment,
            )
        )
        render_rows.append(
            {
                "look": look,
                "image_sha256": _sha256(direct),
                "image_exact": direct.read_bytes() == launched.read_bytes(),
            }
        )
    marker_absent_after_renders = not marker.exists()
    mixed_case_environment_removed = _mixed_case_environment_control(
        root, runtime_python, hostile_environment
    )

    invalid = root / "invalid.png"
    invalid_result = _launch(
        launcher,
        [str(input_path), "--product-look", "unknown", "--output", str(invalid)],
        cwd=foreign_cwd,
        env=hostile_environment,
    )
    invalid_rejected = invalid_result.returncode != 0 and not invalid.exists()

    launcher_cmd = launcher.read_text("utf-8")
    launcher_source = launcher_py.read_text("utf-8")
    source_after = _source_facts()
    rows = sorted(render_rows, key=lambda item: item["look"])
    gates = {
        "source_identity_exact": source_before == source_after,
        "implementation_commit_and_blob_exact": True,
        "parent_receipt_schema_exact": receipt["schema"]
        == config["unchanged"]["receipt_schema"],
        "launcher_process_isolated": ' -I "' in launcher_cmd,
        "renderer_process_isolated": '[sys.executable, "-I", str(entry)' in launcher_source,
        "python_environment_removed": 'key.upper().startswith("PYTHON")'
        in launcher_source
        and mixed_case_environment_removed,
        "hostile_sitecustomize_not_executed": marker_absent_after_catalog
        and marker_absent_after_renders,
        "catalog_exact": catalog["schema_id"] == "kmcfm.product-look-catalog.v1"
        and [
            row["look_id"]
            for row in catalog["looks"]
            if row["availability"] == "available"
        ]
        == looks,
        "three_render_outputs_exact": len(rows) == 3
        and all(row["image_exact"] for row in rows),
        "invalid_look_atomic": invalid_rejected,
    }
    cleanup_complete = _remove_tree(root, root_identity)
    gates["runtime_residue_zero"] = cleanup_complete
    scientific = {
        "schema": config["schema"],
        "implementation_commit": commit,
        "source": source_before,
        "catalog_sha256": hashlib.sha256(
            json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "renders": rows,
        "gates": gates,
        "claim": config["claim"],
        "status": (
            "PASS_PRIVATE_U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION"
            if all(gates.values())
            else "FAIL_CLOSED_U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION"
        ),
    }
    stable = hashlib.sha256(
        json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        **scientific,
        "stable_identity": stable,
        "execution": {
            "python": sys.version,
            "windows": os.name == "nt",
            "runtime_python_role": "installation/runtime/Scripts/python.exe",
            "wheelhouse_file_count": len(list(wheelhouse.glob("*.whl"))),
            "scratch_cleaned_before_return": cleanup_complete,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(arguments.root, arguments.wheelhouse, arguments.order)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
