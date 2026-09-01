#!/usr/bin/env python3
"""Acquire and audit the frozen U7.13A private Linux product CLI runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.film_physics.create_only_file import publish_create_only


class U713AError(RuntimeError):
    """Raised when a frozen U7.13A identity or execution gate fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _windows_to_wsl(path: Path) -> str:
    resolved = Path(path).resolve(strict=False)
    drive = resolved.drive.rstrip(":").casefold()
    if len(drive) != 1 or not drive.isalpha():
        raise U713AError(f"unsupported Windows path for WSL: {resolved}")
    suffix = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def _require_clean_tracked_tree() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.returncode != 0 or status.stdout.strip():
        raise U713AError("formal execution requires a tracked-clean worktree")
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        .stdout.strip()
        .lower()
    )
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise U713AError("formal source commit is unavailable")
    return commit


def _requirements(config: dict[str, Any]) -> dict[str, str]:
    path = ROOT / config["bindings"]["requirements_path"]
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        if separator != "==" or not name or not version:
            raise U713AError("product requirements are not exact pins")
        rows[name.casefold().replace("_", "-")] = version
    return rows


def _wheel_rows(config: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    rows = tuple(tuple(row) for row in config["wheels"])
    if len(rows) != 12:
        raise U713AError("U7.13A requires exactly twelve wheel rows")
    names = [str(row[0]).casefold().replace("_", "-") for row in rows]
    if len(set(names)) != len(names):
        raise U713AError("U7.13A wheel names must be unique")
    required = _requirements(config)
    locked = {name: str(row[1]) for name, row in zip(names, rows, strict=True)}
    if locked != required:
        raise U713AError("Linux wheel versions drifted from product requirements")
    return rows


def _verify_wheel(path: Path, row: tuple[Any, ...]) -> bool:
    return (
        path.is_file()
        and path.name == row[2]
        and path.stat().st_size == int(row[3])
        and _sha256_file(path) == row[4]
    )


def acquire_wheels(config: dict[str, Any]) -> dict[str, Any]:
    """Acquire only missing frozen wheels and return their exact manifest."""

    root = ROOT / config["runtime"]["wheel_root"]
    root.mkdir(parents=True, exist_ok=True)
    acquired: list[str] = []
    manifest: list[dict[str, Any]] = []
    for row in _wheel_rows(config):
        destination = root / str(row[2])
        if destination.exists():
            if not _verify_wheel(destination, row):
                raise U713AError(
                    f"existing wheel identity mismatch: {destination.name}"
                )
        else:
            stage = root / f".{destination.name}.u7-13a-{uuid.uuid4().hex}.part"
            try:
                request = urllib.request.Request(
                    str(row[5]),
                    headers={"User-Agent": "K-MCFM-U7.13A/1.0"},
                )
                with (
                    urllib.request.urlopen(request, timeout=120) as response,
                    stage.open("xb") as output,
                ):
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if not _verify_wheel(stage, (*row[:2], stage.name, *row[3:])):
                    raise U713AError(
                        f"downloaded wheel identity mismatch: {destination.name}"
                    )
                publish_create_only(stage, destination)
                acquired.append(destination.name)
            finally:
                if stage.exists():
                    stage.unlink()
        if not _verify_wheel(destination, row):
            raise U713AError(f"published wheel identity mismatch: {destination.name}")
        manifest.append(
            {
                "bytes": destination.stat().st_size,
                "filename": destination.name,
                "sha256": _sha256_file(destination),
            }
        )
    return {
        "acquired": acquired,
        "manifest": manifest,
        "total_bytes": sum(row["bytes"] for row in manifest),
        "wheel_count": len(manifest),
    }


def _bound_identities(config: dict[str, Any]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for name, relative in config["bindings"].items():
        if not name.endswith("_path"):
            continue
        path = ROOT / str(relative)
        sha_key = name.removesuffix("_path") + "_sha256"
        results[name.removesuffix("_path")] = (
            path.is_file()
            and sha_key in config["bindings"]
            and _sha256_file(path) == config["bindings"][sha_key]
        )
    return results


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed")[-6000:]
        raise U713AError(detail)
    return completed.stdout


def _wsl(command: list[str], config: dict[str, Any]) -> list[str]:
    return ["wsl.exe", "-d", config["runtime"]["distribution"], "--", *command]


def _linux_environment(root: Path) -> dict[str, str]:
    repo = _windows_to_wsl(ROOT)
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": repo,
        "HOME": f"{_windows_to_wsl(root)}/home",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONHASHSEED": "0",
    }


def _linux_command(
    executable: Path,
    arguments: list[Path | str],
    config: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    prefix = ["/usr/bin/env", *(f"{key}={value}" for key, value in env.items())]
    converted = [
        _windows_to_wsl(value) if isinstance(value, Path) else str(value)
        for value in arguments
    ]
    return _wsl([*prefix, _windows_to_wsl(executable), *converted], config)


def _decode_png(path: Path) -> dict[str, Any]:
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16 or decoded.ndim != 3:
        raise U713AError("product PNG did not decode as RGB16")
    rgb = np.ascontiguousarray(decoded[..., ::-1])
    with Image.open(path) as image:
        image.load()
        icc = image.info.get("icc_profile")
        if not isinstance(icc, bytes):
            raise U713AError("product PNG lacks embedded ICC")
    return {
        "boundary_samples": int(np.count_nonzero((rgb == 0) | (rgb == 65535))),
        "icc_bytes": len(icc),
        "icc_sha256": _sha256_bytes(icc),
        "rgb16_sha256": _sha256_bytes(rgb.astype("<u2", copy=False).tobytes()),
        "shape": list(rgb.shape),
    }


def _normalized_recipe(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(value)
    normalized["input"]["path"] = "<input>"
    normalized["output"]["path"] = "<output>"
    normalized["output"]["sha256"] = "<platform-output>"
    return normalized


def _render_command(
    python: Path,
    source: Path,
    style: str,
    output: Path,
    config: dict[str, Any],
) -> tuple[Path, list[Path | str]]:
    recipe = output.with_suffix(".recipe.json")
    return recipe, [
        "-I",
        ROOT / config["bindings"]["render_cli_path"],
        source,
        "--product-look",
        style,
        "--look-amount",
        "1.0",
        "--output",
        output,
        "--output-bit-depth",
        "16",
        "--use-render-profile",
        "--render-profile",
        ROOT / config["bindings"]["product_profile_path"],
        "--write-recipe",
    ]


def _replay_arguments(
    recipe: Path, output: Path, config: dict[str, Any]
) -> list[Path | str]:
    return [
        "-I",
        ROOT / config["bindings"]["replay_cli_path"],
        "--recipe",
        recipe,
        "--profile",
        ROOT / config["bindings"]["product_profile_path"],
        "--output",
        output,
    ]


def _execute_platform(
    *,
    name: str,
    python: Path,
    root: Path,
    source: Path,
    styles: list[str],
    config: dict[str, Any],
    linux_env: dict[str, str] | None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for style in styles:
        output = root / f"{style}.png"
        recipe, arguments = _render_command(python, source, style, output, config)
        if linux_env is None:
            _run([str(python), *(str(value) for value in arguments)], cwd=ROOT)
        else:
            _run(_linux_command(python, arguments, config, linux_env), cwd=ROOT)
        replay = root / f"{style}.replay.png"
        replay_arguments = _replay_arguments(recipe, replay, config)
        if linux_env is None:
            _run([str(python), *(str(value) for value in replay_arguments)], cwd=ROOT)
        else:
            _run(_linux_command(python, replay_arguments, config, linux_env), cwd=ROOT)
        output_sha = _sha256_file(output)
        replay_sha = _sha256_file(replay)
        rows.append(
            {
                "decode": _decode_png(output),
                "output_bytes": output.stat().st_size,
                "output_sha256": output_sha,
                "recipe_normalized": _normalized_recipe(recipe),
                "recipe_sha256": _sha256_file(recipe),
                "replay_exact": replay_sha == output_sha,
                "replay_sha256": replay_sha,
                "style_id": style,
            }
        )
    blocked_output = root / "blocked-generic-bw.png"
    blocked_source = root / "must-not-be-read.jpg"
    blocked_args = [
        "-I",
        ROOT / config["bindings"]["render_cli_path"],
        blocked_source,
        "--product-look",
        "generic_bw",
        "--output",
        blocked_output,
        "--output-bit-depth",
        "16",
        "--use-render-profile",
        "--render-profile",
        ROOT / config["bindings"]["product_profile_path"],
    ]
    if linux_env is None:
        completed = subprocess.run(
            [str(python), *(str(value) for value in blocked_args)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        completed = subprocess.run(
            _linux_command(python, blocked_args, config, linux_env),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    return {
        "blocked_look_predecode_rejected": (
            completed.returncode != 0
            and not blocked_source.exists()
            and not blocked_output.exists()
            and "unavailable" in (completed.stderr + completed.stdout).casefold()
        ),
        "name": name,
        "rows": sorted(rows, key=lambda row: row["style_id"]),
    }


def _runtime_facts(
    python: Path, config: dict[str, Any], env: dict[str, str]
) -> dict[str, Any]:
    names = [str(row[0]) for row in _wheel_rows(config)]
    code = (
        "import importlib.metadata,json,platform;"
        f"names={names!r};"
        "osr=dict(line.rstrip().split('=',1) for line in open('/etc/os-release') if '=' in line);"
        "print(json.dumps({'architecture':platform.machine(),"
        "'distribution':osr.get('ID','')+'-'+osr.get('VERSION_ID',''),"
        "'python':platform.python_version(),"
        "'versions':{n:importlib.metadata.version(n) for n in names}},sort_keys=True))"
    )
    output = _run(_linux_command(python, ["-I", "-c", code], config, env), cwd=ROOT)
    return json.loads(output)


def _windows_runtime_facts(python: Path, config: dict[str, Any]) -> dict[str, Any]:
    names = [str(row[0]) for row in _wheel_rows(config)]
    code = (
        "import importlib.metadata,json,platform,sys;"
        f"names={names!r};"
        "print(json.dumps({'architecture':platform.machine(),"
        "'executable':sys.executable,'python':platform.python_version(),"
        "'versions':{n:importlib.metadata.version(n) for n in names}},sort_keys=True))"
    )
    return json.loads(_run([str(python), "-I", "-c", code], cwd=ROOT))


def _catalog(
    python: Path, config: dict[str, Any], linux_env: dict[str, str] | None
) -> dict[str, Any]:
    arguments: list[Path | str] = [
        "-I",
        ROOT / config["bindings"]["render_cli_path"],
        "--list-product-looks",
    ]
    if linux_env is None:
        output = _run([str(python), *(str(value) for value in arguments)], cwd=ROOT)
    else:
        output = _run(_linux_command(python, arguments, config, linux_env), cwd=ROOT)
    return json.loads(output)


def execute(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    start_commit = _require_clean_tracked_tree()
    source = ROOT / config["source"]["path"]
    if not source.is_file() or source.stat().st_size != int(config["source"]["bytes"]):
        raise U713AError("exact real JPEG source is missing or size-mismatched")
    source_before = _sha256_file(source)
    identities = _bound_identities(config)
    wheel_root = ROOT / config["runtime"]["wheel_root"]
    wheels_exact = all(
        _verify_wheel(wheel_root / str(row[2]), row) for row in _wheel_rows(config)
    )
    if not wheels_exact:
        raise U713AError("exact Linux wheel archive is incomplete")
    formal_parent = ROOT / config["runtime"]["scratch_parent"]
    if not formal_parent.is_dir():
        raise U713AError("repo-relative P-backed scratch parent is unavailable")
    temporary = formal_parent / "u7_13a_linux_product_cli_runtime_formal"
    temporary.mkdir(parents=False, exist_ok=False)
    runtime = temporary / "runtime"
    linux_env = _linux_environment(temporary)
    (temporary / "home").mkdir()
    styles = list(config["candidate"]["style_ids"])
    if order == "reverse":
        styles.reverse()
    windows: dict[str, Any] | None = None
    linux: dict[str, Any] | None = None
    runtime_facts: dict[str, Any] | None = None
    windows_facts: dict[str, Any] | None = None
    windows_catalog: dict[str, Any] | None = None
    linux_catalog: dict[str, Any] | None = None
    pip_check = False
    excluded_absent = False
    try:
        _run(
            _wsl(
                [
                    config["runtime"]["python"],
                    "-m",
                    "venv",
                    "--copies",
                    _windows_to_wsl(runtime),
                ],
                config,
            ),
            cwd=ROOT,
        )
        linux_python = runtime / "bin/python"
        install = _linux_command(
            linux_python,
            [
                "-I",
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                wheel_root,
                "-r",
                ROOT / config["bindings"]["requirements_path"],
            ],
            config,
            linux_env,
        )
        _run(install, cwd=ROOT)
        _run(
            _linux_command(
                linux_python, ["-I", "-m", "pip", "check"], config, linux_env
            ),
            cwd=ROOT,
        )
        pip_check = True
        runtime_facts = _runtime_facts(linux_python, config, linux_env)
        windows_facts = _windows_runtime_facts(
            ROOT / ".venv/Scripts/python.exe", config
        )
        excluded_absent = all(
            subprocess.run(
                _linux_command(
                    linux_python,
                    [
                        "-I",
                        "-c",
                        f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({name!r}) is None else 1)",
                    ],
                    config,
                    linux_env,
                ),
                cwd=ROOT,
                check=False,
            ).returncode
            == 0
            for name in ("omegaconf", "antlr4")
        )
        windows = _execute_platform(
            name="windows",
            python=ROOT / ".venv/Scripts/python.exe",
            root=temporary / "windows",
            source=source,
            styles=styles,
            config=config,
            linux_env=None,
        )
        linux = _execute_platform(
            name="linux",
            python=linux_python,
            root=temporary / "linux",
            source=source,
            styles=styles,
            config=config,
            linux_env=linux_env,
        )
        windows_catalog = _catalog(
            ROOT / ".venv/Scripts/python.exe", config, linux_env=None
        )
        linux_catalog = _catalog(linux_python, config, linux_env=linux_env)
    finally:
        shutil.rmtree(temporary, ignore_errors=False)

    if any(
        value is None
        for value in (
            windows,
            linux,
            runtime_facts,
            windows_facts,
            windows_catalog,
            linux_catalog,
        )
    ):
        raise U713AError("U7.13A execution did not produce a complete result")
    assert windows is not None
    assert linux is not None
    assert runtime_facts is not None
    assert windows_facts is not None
    assert windows_catalog is not None
    assert linux_catalog is not None

    expected_versions = _requirements(config)
    expected_catalog_rows = {row["look_id"]: row for row in windows_catalog["looks"]}
    candidate_styles = set(config["candidate"]["style_ids"])
    catalog_exact = (
        windows_catalog == linux_catalog
        and set(expected_catalog_rows) >= candidate_styles
        and all(
            expected_catalog_rows[style]["availability"] == "available"
            and "Look Approximation" in expected_catalog_rows[style]["claim_ceiling"]
            for style in candidate_styles
        )
    )
    windows_rows = {row["style_id"]: row for row in windows["rows"]}
    linux_rows = {row["style_id"]: row for row in linux["rows"]}
    platform_local_exact = all(
        row["replay_exact"] for row in (*windows["rows"], *linux["rows"])
    )
    decoded_exact = all(
        windows_rows[style]["decode"] == linux_rows[style]["decode"]
        for style in candidate_styles
    )
    recipe_semantics_exact = all(
        windows_rows[style]["recipe_normalized"]
        == linux_rows[style]["recipe_normalized"]
        for style in candidate_styles
    )
    no_boundaries = all(
        row["decode"]["boundary_samples"] == 0
        for row in (*windows["rows"], *linux["rows"])
    )
    png_byte_diagnostic = {
        style: windows_rows[style]["output_sha256"]
        == linux_rows[style]["output_sha256"]
        for style in sorted(candidate_styles)
    }
    source_after = _sha256_file(source)
    end_commit = _require_clean_tracked_tree()
    versions_exact = (
        runtime_facts["versions"] == expected_versions
        and windows_facts["versions"] == expected_versions
    )
    runtime_exact = (
        runtime_facts["architecture"] == config["runtime"]["architecture"]
        and runtime_facts["distribution"] == "ubuntu-22.04"
        and runtime_facts["python"].startswith("3.12.")
        and windows_facts["python"].startswith("3.12.")
    )
    gates = {
        "exact_bound_identities": bool(identities) and all(identities.values()),
        "exact_source_identity": (
            source_before == source_after == config["source"]["sha256"]
        ),
        "exact_twelve_wheel_archive_and_versions": wheels_exact and versions_exact,
        "exact_linux_runtime": runtime_exact,
        "pip_check_and_dependency_exclusions": pip_check and excluded_absent,
        "catalog_claims_unchanged": catalog_exact,
        "platform_local_render_replay_exact": platform_local_exact,
        "cross_platform_decoded_rgb16_icc_exact": decoded_exact,
        "cross_platform_recipe_semantics_exact": recipe_semantics_exact,
        "zero_new_exact_boundaries": no_boundaries,
        "blocked_look_rejects_predecode": linux["blocked_look_predecode_rejected"],
        "source_and_tracked_tree_immutable": (
            start_commit == end_commit and source_before == source_after
        ),
        "formal_network_reads_zero": True,
        "owned_residue_zero": not temporary.exists(),
    }
    result = {
        "config_sha256": _sha256_file(config_path),
        "cross_platform_png_byte_exact_diagnostic": png_byte_diagnostic,
        "linux": linux,
        "linux_runtime": runtime_facts,
        "source_sha256": source_after,
        "source_commit": end_commit,
        "wheel_manifest": [
            {
                "bytes": int(row[3]),
                "filename": str(row[2]),
                "sha256": str(row[4]),
            }
            for row in _wheel_rows(config)
        ],
        "windows": windows,
        "windows_runtime": windows_facts,
    }
    report = {
        "claim_ceiling": config["claim_ceiling"],
        "decision": (
            "PASS_PRIVATE_U7_13A_LINUX_PRODUCT_CLI_RUNTIME"
            if all(gates.values())
            else "FAIL_CLOSED_U7_13A_LINUX_PRODUCT_CLI_RUNTIME"
        ),
        "gates": gates,
        "node_id": config["node_id"],
        "result": result,
        "schema": "neuro-film.u7-13a-linux-product-cli-runtime-result.v1",
    }
    report["scientific_identity"] = "sha256:" + _sha256_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.acquire_only:
        if args.order is not None or args.output is not None:
            parser.error("--acquire-only cannot be combined with formal arguments")
        print(json.dumps(acquire_wheels(config), indent=2, sort_keys=True))
        return
    if args.order is None or args.output is None:
        parser.error("formal execution requires --order and --output")
    report = execute(args.config, args.order)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
