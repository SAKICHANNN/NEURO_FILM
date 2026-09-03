#!/usr/bin/env python3
"""Build one private repository-independent source capsule in a verified runtime."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.install_product_runtime import (
    _entry_identity,
    _is_normal_directory,
    _native_launcher_source,
    _write_native_console_script,
)

CONFIG = ROOT / "configs/u7_22a_private_repository_independent_runtime_capsule_v1.json"
MANIFEST_SCHEMA = "kmcfm.private-runtime-source-capsule.v1"
RECEIPT_SCHEMA = "kmcfm.private-runtime-capsule-receipt.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    details = path.stat(follow_symlinks=False)
    return {"bytes": details.st_size, "sha256": _sha256(path)}


def _checked(command: Sequence[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed")[-4000:]
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result.stdout


def _committed_blobs(commit: str, paths: Sequence[str]) -> dict[str, bytes]:
    requested = {str(path).replace("\\", "/") for path in paths}
    roots = sorted({"src" if name.startswith("src/") else name for name in requested})
    result = subprocess.run(
        ["git", "archive", "--format=tar", commit, "--", *roots],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("committed source archive is unavailable")
    blobs: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            name = member.name.replace("\\", "/")
            if not member.isfile() or name not in requested:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"committed source blob unavailable: {name}")
            blobs[name] = handle.read()
    missing = requested - set(blobs)
    if missing:
        raise RuntimeError(f"committed source blobs unavailable: {sorted(missing)}")
    return blobs


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _normal_directory(path: Path, identity: tuple[int, int]) -> bool:
    return (
        os.path.lexists(path)
        and _is_normal_directory(path)
        and _entry_identity(path) == identity
    )


def _cleanup_owned(path: Path, identity: tuple[int, int]) -> bool:
    if not os.path.lexists(path):
        return True
    if not _normal_directory(path, identity):
        return False
    shutil.rmtree(path)
    return True


def _zip_info(name: str, epoch: int, *, directory: bool = False) -> zipfile.ZipInfo:
    timestamp = time.gmtime(epoch)[:6]
    info = zipfile.ZipInfo(name + ("/" if directory else ""), timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = ((0o755 if directory else 0o644) & 0xFFFF) << 16
    return info


def _source_names(commit: str, config: Mapping[str, Any]) -> list[str]:
    source_names = [
        line.strip().replace("\\", "/")
        for line in _checked(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", "src"]
        ).splitlines()
        if line.strip().endswith(".py")
    ]
    source_names.extend(str(item) for item in config["capsule"]["archive_extra_paths"])
    source_names = sorted(set(source_names))
    if not source_names or "src/__init__.py" not in source_names:
        raise RuntimeError("capsule source inventory is incomplete")
    return source_names


def _build_archive(
    path: Path,
    source_names: Sequence[str],
    blobs: Mapping[str, bytes],
    config: Mapping[str, Any],
) -> list[str]:
    epoch = int(config["capsule"]["source_date_epoch"])
    with (
        path.open("xb") as raw,
        zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive,
    ):
        archive.writestr(_zip_info("scripts/__init__.py", epoch), b"")
        for name in source_names:
            archive.writestr(_zip_info(name, epoch), blobs[name])
    return ["scripts/__init__.py", *source_names]


def _launcher_source(*, manifest_sha256: str, entrypoint: str) -> str:
    bound = json.dumps(
        {
            "entrypoint": entrypoint,
            "manifest_sha256": manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f'''#!/usr/bin/env python3
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

BOUND = json.loads({bound!r})
root = Path(sys.argv[0]).resolve().parent
manifest = root / "runtime-source-capsule.json"

def fail(message):
    print(f"K-MCFM capsule rejected: {{message}}", file=sys.stderr)
    raise SystemExit(2)

def digest(path):
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
    except OSError:
        fail("bound file is unavailable")
    return value.hexdigest()

if digest(manifest) != BOUND["manifest_sha256"]:
    fail("manifest identity changed")
try:
    payload = json.loads(manifest.read_bytes())
except Exception:
    fail("manifest is invalid")
if payload.get("schema") != "{MANIFEST_SCHEMA}":
    fail("manifest schema changed")
archive = payload.get("archive", {{}})
archive_path = root / archive.get("path", "")
if (not archive_path.is_file() or archive_path.stat().st_size != archive.get("bytes")
        or digest(archive_path) != archive.get("sha256")):
    fail("source archive identity changed")
for name, row in payload.get("external_files", {{}}).items():
    path = root.joinpath(*name.split("/"))
    if (not path.is_file() or path.stat().st_size != row.get("bytes")
            or digest(path) != row.get("sha256")):
        fail("external source identity changed")
runtime_root = root.parent.resolve()
for role, row in payload.get("parent_runtime", {{}}).items():
    path = runtime_root.joinpath(*row.get("path", "").split("/")).resolve()
    try:
        path.relative_to(runtime_root)
    except ValueError:
        fail("parent runtime containment changed")
    if (not path.is_file() or path.stat().st_size != row.get("bytes")
            or digest(path) != row.get("sha256")):
        fail("parent runtime identity changed")
entry = root.joinpath(*BOUND["entrypoint"].split("/"))
if not entry.is_file():
    fail("entrypoint is unavailable")
for key in tuple(os.environ):
    if key.upper().startswith("PYTHON") or key.startswith("KMCFM_PRIVATE_CAPSULE_"):
        os.environ.pop(key, None)
os.environ["KMCFM_PRIVATE_CAPSULE_MANIFEST"] = str(manifest)
os.environ["KMCFM_PRIVATE_CAPSULE_MANIFEST_SHA256"] = BOUND["manifest_sha256"]
sys.path.insert(0, str(archive_path))
runpy.run_path(str(entry), run_name="__main__")
'''


def _verify_parent(config: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    parent = config["parent_runtime"]
    runtime_root = (ROOT / parent["path"]).resolve(strict=True)
    try:
        runtime_root.relative_to((ROOT / "outputs").resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError("parent runtime escaped P-backed outputs") from exc
    if runtime_root.drive.casefold() != "p:":
        raise RuntimeError("parent runtime is not P-backed")
    receipt_path = runtime_root / parent["receipt_name"]
    python = runtime_root / parent["python_path"]
    for path, prefix in ((receipt_path, "receipt"), (python, "python")):
        if (
            path.stat().st_size != parent[f"{prefix}_bytes"]
            or _sha256(path) != parent[f"{prefix}_sha256"]
        ):
            raise RuntimeError(f"parent runtime {prefix} identity mismatch")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    if (
        receipt.get("schema") != parent["receipt_schema"]
        or receipt.get("claim")
        != {
            "calibrated_stock_response": False,
            "evidence_grade": "look-approximation",
            "mode": "film-inspired",
            "physical_film_reproduction": False,
            "public_release": False,
        }
        or Path(receipt["python"]["executable"]).resolve(strict=True) != python
        or receipt.get("distributions") is None
    ):
        raise RuntimeError("parent runtime receipt semantics mismatch")
    return runtime_root, python, receipt


def build_capsule(destination: Path | None = None) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text("utf-8"))
    if (
        config.get("schema")
        != "kmcfm.u7-22a-private-repository-independent-runtime-capsule.v1"
    ):
        raise RuntimeError("U7.22A config schema mismatch")
    runtime_root, python, parent_receipt = _verify_parent(config)
    head = _checked(["git", "rev-parse", "HEAD"]).strip().lower()
    _checked(
        ["git", "merge-base", "--is-ancestor", config["source_parent_commit"], head]
    )
    if _checked(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise RuntimeError("tracked project files must be clean")
    if destination is None:
        destination = (
            runtime_root / f"{config['capsule']['directory_prefix']}{head[:9]}"
        )
    destination = Path(destination).resolve(strict=False)
    if destination.parent != runtime_root or os.path.lexists(destination):
        raise FileExistsError("capsule destination must be absent below parent runtime")
    destination.mkdir()
    owned = _entry_identity(destination)
    try:
        source_names = _source_names(head, config)
        external_names = [str(item) for item in config["capsule"]["external_paths"]]
        blobs = _committed_blobs(head, [*source_names, *external_names])
        archive_path = destination / config["capsule"]["archive_name"]
        members = _build_archive(archive_path, source_names, blobs, config)
        external: dict[str, dict[str, object]] = {}
        for relative in external_names:
            target = destination.joinpath(*str(relative).split("/"))
            _write_new(target, blobs[relative])
            external[str(relative)] = _identity(target)
        (destination / "tmp").mkdir()
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "source_commit": head,
            "runtime_scope": json.loads(
                (
                    destination
                    / "configs/u7_9d_private_runtime_source_scope_binding_v1.json"
                ).read_text("utf-8")
            )["runtime_scope"],
            "archive": {
                "path": archive_path.name,
                **_identity(archive_path),
                "member_count": len(members),
            },
            "external_files": external,
            "parent_runtime": {
                "receipt": {
                    "path": config["parent_runtime"]["receipt_name"],
                    "bytes": config["parent_runtime"]["receipt_bytes"],
                    "sha256": config["parent_runtime"]["receipt_sha256"],
                },
                "python": {
                    "path": config["parent_runtime"]["python_path"],
                    "bytes": config["parent_runtime"]["python_bytes"],
                    "sha256": config["parent_runtime"]["python_sha256"],
                },
            },
            "claim": config["claim"],
        }
        manifest_path = destination / config["capsule"]["manifest_name"]
        _write_new(manifest_path, _canonical(manifest))
        manifest_sha = _sha256(manifest_path)
        launchers: dict[str, dict[str, object]] = {}
        for role, entrypoint, stem in (
            ("cli", "scripts/render_film.py", "kmcfm-capsule-look"),
            ("desktop", "scripts/open_product_desktop.py", "kmcfm-capsule-desktop"),
        ):
            source = destination / f"{stem}-launch.py"
            _write_new(
                source,
                _launcher_source(
                    manifest_sha256=manifest_sha, entrypoint=entrypoint
                ).encode("utf-8"),
            )
            command = destination / f"{stem}.cmd"
            _write_new(
                command, f'@echo off\r\n"{python}" -I "{source}" %*\r\n'.encode()
            )
            native_source = destination / "native-launcher-sources" / f"{stem}.py"
            _write_new(
                native_source,
                _native_launcher_source(source.read_text("utf-8")).encode("utf-8"),
            )
            native = destination / f"{stem}.exe"
            native_output: list[str] = []

            def native_runner(  # type: ignore[no-untyped-def]
                command, cwd, environment, _output=native_output
            ):
                result = subprocess.run(
                    list(command),
                    cwd=cwd,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                _output.append(result.stdout)
                return result

            try:
                _write_native_console_script(
                    python=python,
                    source=native_source,
                    destination=native,
                    source_date_epoch=int(config["capsule"]["source_date_epoch"]),
                    runner=native_runner,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"native launcher publication failed: {native_output[-1:]!r}"
                ) from exc
            launchers[role] = {
                "entrypoint": entrypoint,
                "python": {"path": source.name, **_identity(source)},
                "command": {"path": command.name, **_identity(command)},
                "native_source": {
                    "path": native_source.relative_to(destination).as_posix(),
                    **_identity(native_source),
                },
                "native": {"path": native.name, **_identity(native)},
            }
        file_count = sum(1 for path in destination.rglob("*") if path.is_file()) + 1
        logical_bytes = sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "source_commit": head,
            "manifest": {"path": manifest_path.name, **_identity(manifest_path)},
            "parent_runtime_receipt_sha256": config["parent_runtime"]["receipt_sha256"],
            "parent_runtime_source_commit": parent_receipt["source_commit"],
            "python": {
                "path": "../" + config["parent_runtime"]["python_path"],
                **_identity(python),
            },
            "launchers": launchers,
            "materialized_file_count": file_count,
            "logical_bytes_before_receipt": logical_bytes,
            "claim": config["claim"],
        }
        receipt_path = destination / "capsule-receipt.json"
        _write_new(receipt_path, _canonical(receipt))
        actual_files = sum(1 for path in destination.rglob("*") if path.is_file())
        actual_bytes = sum(
            path.stat().st_size for path in destination.rglob("*") if path.is_file()
        )
        if (
            actual_files > config["capsule"]["maximum_materialized_files"]
            or actual_bytes > config["capsule"]["maximum_logical_bytes"]
            or (destination / "runtime").exists()
        ):
            raise RuntimeError("capsule size or environment-duplication gate failed")
        return {
            **receipt,
            "receipt": {"path": receipt_path.name, **_identity(receipt_path)},
            "logical_bytes": actual_bytes,
        }
    except BaseException:
        _cleanup_owned(destination, owned)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            build_capsule(args.destination), sort_keys=True, separators=(",", ":")
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
