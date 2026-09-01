#!/usr/bin/env python3
"""Run the formal U7.9A private Windows runtime-install audit."""

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

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.install_product_runtime import (
    _launcher_source,
    _sha256,
    install_product_runtime,
)

CONFIG = ROOT / "configs" / "u7_9a_private_windows_product_runtime_install_v1.json"
CONTRACT = (
    ROOT
    / "docs"
    / "planning"
    / "U7_9A_PRIVATE_WINDOWS_PRODUCT_RUNTIME_INSTALL_CONTRACT.md"
)
INSTALLER = ROOT / "scripts" / "install_product_runtime.py"
RENDERER = ROOT / "scripts" / "render_film.py"
REQUIREMENTS = ROOT / "requirements-product-v2.txt"


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
        raise RuntimeError((result.stderr or result.stdout or "command failed")[-4000:])
    return result.stdout


def _source_commit() -> str:
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


def _prepare_wheelhouse(wheelhouse: Path, env: dict[str, str]) -> list[dict[str, Any]]:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    if not list(wheelhouse.glob("*.whl")):
        _require_success(
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--only-binary=:all:",
                    "--dest",
                    str(wheelhouse),
                    "-r",
                    str(REQUIREMENTS),
                ],
                cwd=ROOT,
                env=env,
            )
        )
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(
            wheelhouse.glob("*.whl"), key=lambda item: item.name.casefold()
        )
    ]


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


def _cmd_launch(
    launcher: Path, arguments: list[str], *, cwd: Path, env: dict[str, str]
):
    return _run(["cmd.exe", "/d", "/c", str(launcher), *arguments], cwd=cwd, env=env)


def _generated_launcher_control(
    *,
    runtime_python: Path,
    path: Path,
    source_commit: str,
    requirements: Path,
    requirements_sha256: str,
    env: dict[str, str],
) -> dict[str, Any]:
    path.write_text(
        _launcher_source(
            project_root=ROOT,
            source_commit=source_commit,
            requirements_path=requirements,
            requirements_sha256=requirements_sha256,
        ),
        encoding="utf-8",
        newline="\n",
    )
    result = _run(
        [str(runtime_python), str(path), "--list-product-looks"],
        cwd=path.parent,
        env=env,
    )
    return {
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "stdout_bytes": len(result.stdout.encode("utf-8")),
    }


def _existing_destination_control(path: Path, env: dict[str, str]) -> bool:
    path.mkdir()
    marker = path / "foreign.bin"
    marker.write_bytes(b"foreign")
    try:
        install_product_runtime(
            path, wheelhouse=path.parent / "wheelhouse", environment=env
        )
    except FileExistsError:
        return marker.read_bytes() == b"foreign"
    finally:
        shutil.rmtree(path)
    return False


def _failed_install_control(path: Path, env: dict[str, str]) -> bool:
    empty = path.parent / "empty-wheelhouse"
    empty.mkdir()
    try:
        try:
            install_product_runtime(path, wheelhouse=empty, environment=env)
        except RuntimeError:
            return not os.path.lexists(path)
        return False
    finally:
        shutil.rmtree(empty)


def audit(root: Path, wheelhouse: Path, order: str) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text("utf-8"))
    if order not in {"forward", "reverse"}:
        raise ValueError(order)
    root = root.resolve(strict=False)
    wheelhouse = wheelhouse.resolve(strict=False)
    tmp_root = (ROOT / "tmp").resolve(strict=True)
    if tmp_root not in root.parents or tmp_root not in wheelhouse.parents:
        raise ValueError("formal roots must remain under repo-relative tmp")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    cache = root / "pip-cache"
    temp = root / "process-temp"
    cache.mkdir()
    temp.mkdir()
    env = dict(os.environ)
    env.update(
        {
            "PIP_CACHE_DIR": str(cache),
            "TMP": str(temp),
            "TEMP": str(temp),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    source_commit = _source_commit()
    tracked_clean_before = _tracked_clean()
    wheel_rows = _prepare_wheelhouse(wheelhouse, env)
    installation = root / "installation"
    receipt = install_product_runtime(
        installation, wheelhouse=wheelhouse, environment=env
    )
    receipt_path = installation / config["receipt_name"]
    launcher = installation / config["launcher_name"]
    runtime_python = installation / "runtime" / "Scripts" / "python.exe"
    foreign_cwd = root / "foreign-cwd"
    foreign_cwd.mkdir()

    catalog_result = _cmd_launch(
        launcher, ["--list-product-looks"], cwd=foreign_cwd, env=env
    )
    catalog_text = _require_success(catalog_result).strip()
    catalog = json.loads(catalog_text)
    catalog_rows = catalog["looks"]
    available = [
        row["look_id"] for row in catalog_rows if row["availability"] == "available"
    ]
    claim_rows_pass = all(
        "look-approximation" in row["evidence_tier"]
        and "Look Approximation" in row["claim_ceiling"]
        and (
            "not a calibrated stock response" in row["claim_ceiling"]
            or "data gap" in row["claim_ceiling"]
        )
        for row in catalog_rows
        if row["availability"] == "available"
    )

    input_path = root / "input.png"
    _tiny_input(input_path)
    looks = list(config["available_looks"])
    if order == "reverse":
        looks.reverse()
    render_rows: list[dict[str, Any]] = []
    for look in looks:
        direct = root / f"direct-{look}.png"
        launched = root / f"launched-{look}.png"
        direct_result = _run(
            [
                str(runtime_python),
                str(RENDERER),
                str(input_path),
                "--product-look",
                look,
                "--output",
                str(direct),
            ],
            cwd=foreign_cwd,
            env=env,
        )
        launch_result = _cmd_launch(
            launcher,
            [
                str(input_path),
                "--product-look",
                look,
                "--output",
                str(launched),
            ],
            cwd=foreign_cwd,
            env=env,
        )
        _require_success(direct_result)
        _require_success(launch_result)
        render_rows.append(
            {
                "look_id": look,
                "direct_sha256": _sha256(direct),
                "launcher_sha256": _sha256(launched),
                "byte_exact": direct.read_bytes() == launched.read_bytes(),
            }
        )
    render_rows.sort(key=lambda row: row["look_id"])

    shadow_requirements = root / "shadow-requirements.txt"
    shadow_requirements.write_bytes(REQUIREMENTS.read_bytes() + b"# drift\n")
    requirements_control = _generated_launcher_control(
        runtime_python=runtime_python,
        path=root / "requirements-drift-launch.py",
        source_commit=source_commit,
        requirements=shadow_requirements,
        requirements_sha256=config["requirements"]["sha256"],
        env=env,
    )
    head_control = _generated_launcher_control(
        runtime_python=runtime_python,
        path=root / "head-drift-launch.py",
        source_commit="0" * 40,
        requirements=REQUIREMENTS,
        requirements_sha256=config["requirements"]["sha256"],
        env=env,
    )
    existing_pass = _existing_destination_control(root / "existing", env)
    failed_install_pass = _failed_install_control(root / "failed-install", env)

    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in (CONFIG, CONTRACT, INSTALLER, RENDERER, REQUIREMENTS)
    }
    gates = {
        "tracked_clean_before": tracked_clean_before,
        "source_commit_exact": receipt["source_commit"] == source_commit,
        "requirements_exact": receipt["requirements"]["sha256"]
        == config["requirements"]["sha256"],
        "wheelhouse_nonempty": bool(wheel_rows),
        "binary_install_and_pip_check": set(receipt["distributions"])
        == {
            line.split("==", 1)[0]
            for line in REQUIREMENTS.read_text("utf-8").splitlines()
            if line and not line.startswith("#")
        },
        "catalog_exact": available == config["available_looks"],
        "claim_ceiling_exact": claim_rows_pass,
        "foreign_cwd_launch": catalog_result.returncode == 0,
        "three_direct_launcher_exact": len(render_rows) == 3
        and all(row["byte_exact"] for row in render_rows),
        "existing_destination_preserved": existing_pass,
        "failed_install_cleanup": failed_install_pass,
        "requirements_drift_reject": requirements_control["returncode"] == 2
        and "requirements identity drift" in requirements_control["stderr"],
        "head_drift_reject": head_control["returncode"] == 2
        and "repository commit drift" in head_control["stderr"],
        "receipt_claim_exact": receipt["claim"] == config["claim"],
    }
    report = {
        "schema": "kmcfm.u7-9a-private-windows-product-runtime-install-result.v1",
        "status": "PASS_PRIVATE_WINDOWS_REPOSITORY_BOUND_PRODUCT_RUNTIME_INSTALL"
        if all(gates.values())
        else "FAIL_CLOSED_U7_9A_PRIVATE_WINDOWS_PRODUCT_RUNTIME_INSTALL",
        "automatic_pass": all(gates.values()),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "wheelhouse": wheel_rows,
        "receipt_sha256": _sha256(receipt_path),
        "launcher_sha256": _sha256(launcher),
        "catalog_sha256": hashlib.sha256(catalog_text.encode("utf-8")).hexdigest(),
        "catalog": catalog_rows,
        "renders": render_rows,
        "controls": {
            "requirements_drift": requirements_control,
            "head_drift": head_control,
        },
        "gates": gates,
        "claim": config["claim"],
    }
    shutil.rmtree(installation)
    shutil.rmtree(cache)
    shutil.rmtree(temp)
    report["formal_installation_residue_count"] = int(installation.exists())
    report["owned_cache_temp_residue_count"] = int(cache.exists()) + int(temp.exists())
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
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
