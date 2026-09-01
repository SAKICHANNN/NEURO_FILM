#!/usr/bin/env python3
"""Create a private repository-bound K-MCFM product runtime on Windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "u7_9a_private_windows_product_runtime_install_v1.json"

_IMPORTS = {
    "imageio": "imageio",
    "lazy-loader": "lazy_loader",
    "networkx": "networkx",
    "numpy": "numpy",
    "opencv-python-headless": "cv2",
    "packaging": "packaging",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "rawpy": "rawpy",
    "scikit-image": "skimage",
    "scipy": "scipy",
    "tifffile": "tifffile",
}

CommandRunner = Callable[
    [Sequence[str], Path | None, dict[str, str]], subprocess.CompletedProcess[str]
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str], cwd: Path | None, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _checked(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    cwd: Path | None,
    env: dict[str, str],
) -> str:
    result = runner(command, cwd, env)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed")[-4000:]
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result.stdout


def _git_state(project_root: Path, runner: CommandRunner, env: dict[str, str]) -> str:
    head = _checked(
        runner,
        ("git", "-C", str(project_root), "rev-parse", "HEAD"),
        cwd=project_root,
        env=env,
    ).strip()
    dirty = _checked(
        runner,
        (
            "git",
            "-C",
            str(project_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ),
        cwd=project_root,
        env=env,
    )
    if dirty.strip():
        raise RuntimeError("tracked project files must be clean before installation")
    return head


def _entry_identity(path: Path) -> tuple[int, int]:
    stat = path.stat(follow_symlinks=False)
    return int(stat.st_dev), int(stat.st_ino)


def _is_normal_directory(path: Path) -> bool:
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    attributes = int(getattr(details, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISDIR(details.st_mode) and not attributes & reparse


def _require_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    if not _is_normal_directory(path) or _entry_identity(path) != identity:
        raise RuntimeError("installation destination ownership changed")


def _cleanup_owned_directory(path: Path, identity: tuple[int, int]) -> bool:
    if not os.path.lexists(path):
        return True
    try:
        _require_owned_directory(path, identity)
    except RuntimeError:
        return False
    shutil.rmtree(path)
    return True


def _write_new_text(path: Path, text: str, *, newline: str) -> None:
    with path.open("x", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def _pinned_versions(requirements: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in requirements.read_text("utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        name, separator, version = item.partition("==")
        if not separator or not name or not version:
            raise RuntimeError(f"unsupported product requirement: {item}")
        versions[name.casefold()] = version
    return versions


def _python_path(destination: Path) -> Path:
    return destination / "runtime" / "Scripts" / "python.exe"


def _launcher_source(
    *,
    project_root: Path,
    source_commit: str,
    requirements_path: Path,
    requirements_sha256: str,
) -> str:
    values = {
        "project_root": str(project_root),
        "source_commit": source_commit,
        "requirements_path": str(requirements_path),
        "requirements_sha256": requirements_sha256,
    }
    bound = json.dumps(values, sort_keys=True)
    return f"""#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BOUND = json.loads({bound!r})
root = Path(BOUND["project_root"])
requirements = Path(BOUND["requirements_path"])

def fail(message):
    print(f"K-MCFM runtime rejected: {{message}}", file=sys.stderr)
    raise SystemExit(2)

if not root.is_dir() or not requirements.is_file():
    fail("bound repository or requirements file is missing")
digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
if digest != BOUND["requirements_sha256"]:
    fail("product requirements identity drift")
head = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    check=False, capture_output=True, text=True, encoding="utf-8"
)
if head.returncode or head.stdout.strip() != BOUND["source_commit"]:
    fail("repository commit drift")
dirty = subprocess.run(
    ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
    check=False, capture_output=True, text=True, encoding="utf-8"
)
if dirty.returncode or dirty.stdout.strip():
    fail("tracked repository drift")
entry = root / "scripts" / "render_film.py"
raise SystemExit(subprocess.call([sys.executable, str(entry), *sys.argv[1:]], cwd=root))
"""


def _metadata_probe_code() -> str:
    imports = json.dumps(_IMPORTS, sort_keys=True)
    return f"""import importlib
import importlib.metadata
import json
mapping = json.loads({imports!r})
versions = {{}}
for distribution, module in mapping.items():
    importlib.import_module(module)
    versions[distribution] = importlib.metadata.version(distribution)
for forbidden in ("omegaconf", "antlr4-python3-runtime"):
    try:
        importlib.metadata.version(forbidden)
    except importlib.metadata.PackageNotFoundError:
        continue
    raise SystemExit("forbidden distribution present: " + forbidden)
print(json.dumps(versions, sort_keys=True, separators=(",", ":")))
"""


def install_product_runtime(
    destination: Path,
    *,
    project_root: Path = ROOT,
    wheelhouse: Path | None = None,
    runner: CommandRunner = _run,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Install and return the deterministic private runtime receipt."""
    if os.name != "nt" or platform.python_implementation() != "CPython":
        raise RuntimeError("U7.9A requires Windows CPython")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("U7.9A requires CPython 3.12")

    destination = destination.resolve(strict=False)
    project_root = project_root.resolve(strict=True)
    config = json.loads(
        (project_root / CONFIG_PATH.relative_to(ROOT)).read_text("utf-8")
    )
    requirements = project_root / config["requirements"]["path"]
    if _sha256(requirements) != config["requirements"]["sha256"]:
        raise RuntimeError("product requirements identity mismatch")
    git_directory = project_root / ".git"
    if (
        destination == project_root
        or destination == git_directory
        or git_directory in destination.parents
    ):
        raise ValueError("unsafe installation destination")
    if os.path.lexists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"destination parent is missing: {destination.parent}")
    if wheelhouse is not None:
        wheelhouse = wheelhouse.resolve(strict=True)
        if not wheelhouse.is_dir():
            raise NotADirectoryError(wheelhouse)

    env = dict(os.environ if environment is None else environment)
    source_commit = _git_state(project_root, runner, env)
    destination.mkdir()
    owned_identity = _entry_identity(destination)
    try:
        runtime = destination / "runtime"
        _checked(
            runner,
            (sys.executable, "-m", "venv", str(runtime)),
            cwd=project_root,
            env=env,
        )
        python = _python_path(destination)
        if not python.is_file():
            raise RuntimeError("virtual environment did not publish python.exe")
        install_command: list[str] = [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
        ]
        if wheelhouse is not None:
            install_command.extend(("--no-index", "--find-links", str(wheelhouse)))
        install_command.extend(("-r", str(requirements)))
        _checked(runner, install_command, cwd=project_root, env=env)
        _checked(
            runner,
            (str(python), "-m", "pip", "check"),
            cwd=project_root,
            env=env,
        )
        versions = json.loads(
            _checked(
                runner,
                (str(python), "-c", _metadata_probe_code()),
                cwd=project_root,
                env=env,
            )
        )
        if {
            key.casefold(): value for key, value in versions.items()
        } != _pinned_versions(requirements):
            raise RuntimeError("installed product distribution set does not match pins")
        _require_owned_directory(destination, owned_identity)
        launcher_py = destination / "product-launch.py"
        _write_new_text(
            launcher_py,
            _launcher_source(
                project_root=project_root,
                source_commit=source_commit,
                requirements_path=requirements,
                requirements_sha256=config["requirements"]["sha256"],
            ),
            newline="\n",
        )
        launcher_cmd = destination / config["launcher_name"]
        _write_new_text(
            launcher_cmd,
            f'@echo off\r\n"{python}" "{launcher_py}" %*\r\n',
            newline="",
        )
        receipt = {
            "schema": "kmcfm.private-product-runtime-receipt.v1",
            "source_commit": source_commit,
            "project_root": str(project_root),
            "requirements": {
                "path": str(requirements),
                "sha256": config["requirements"]["sha256"],
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": ".".join(map(str, sys.version_info[:3])),
                "executable": str(python),
            },
            "distributions": versions,
            "launcher": str(launcher_cmd),
            "claim": config["claim"],
            "repository_bound": True,
            "public_distribution": False,
        }
        receipt_path = destination / config["receipt_name"]
        _write_new_text(
            receipt_path,
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            newline="\n",
        )
        return receipt
    except BaseException:
        _cleanup_owned_directory(destination, owned_identity)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the private repository-bound K-MCFM product runtime."
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument("--wheelhouse", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = install_product_runtime(args.destination, wheelhouse=args.wheelhouse)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
