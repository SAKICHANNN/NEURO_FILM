#!/usr/bin/env python3
"""Audit the repository-bound installed native desktop launch path."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import install_product_runtime as installer
from src.film_physics.create_only_file import publish_create_only

CONFIG = ROOT / "configs" / "u7_10b_installed_runtime_desktop_launch_v1.json"
CONTRACT = (
    ROOT / "docs" / "planning" / "U7_10B_INSTALLED_RUNTIME_DESKTOP_LAUNCH_CONTRACT.md"
)
REQUIREMENTS = ROOT / "requirements-product-v2.txt"
INSTALLER = ROOT / "scripts" / "install_product_runtime.py"
AUDIT = ROOT / "scripts" / "audit_u7_10b_installed_runtime_desktop_launch.py"
AUDIT_TEST = ROOT / "tests" / "test_audit_u7_10b_installed_runtime_desktop_launch.py"
CORE_TEST = ROOT / "tests" / "test_u7_10b_installed_runtime_desktop_launch.py"
DESKTOP_ENTRY = ROOT / "scripts" / "open_product_desktop.py"
DESKTOP_CORE = ROOT / "src" / "inference" / "product_desktop.py"
DESKTOP_UI = ROOT / "src" / "inference" / "product_desktop_ui.py"
RENDERER = ROOT / "scripts" / "render_film.py"
CREATE_ONLY = ROOT / "src" / "film_physics" / "create_only_file.py"
WINDOW_TITLE = "K-MCFM Look Approximation"
EXPECTED_WHEEL_COUNT = 12
EXPECTED_WHEEL_BYTES = 111_666_287


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_report_output(path: Path) -> Path:
    output = path.resolve(strict=False)
    output_root = (ROOT / "outputs").resolve(strict=True)
    if output_root not in output.parents:
        raise ValueError("report output must remain under repo-relative outputs")
    if os.path.lexists(output):
        raise FileExistsError(f"report output already exists: {output}")
    return output


def _publish_report(path: Path, report: dict[str, Any]) -> None:
    output = _preflight_report_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.stage")
    try:
        stage.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        publish_create_only(stage, output)
    finally:
        if stage.exists():
            stage.unlink()


def _run(
    command: Sequence[str], *, cwd: Path, env: dict[str, str]
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
        timeout=180,
    )


def _success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed")[-4000:]
        raise RuntimeError(detail)
    return result.stdout


def _source_commit() -> str:
    return _success(
        _run(["git", "rev-parse", "HEAD"], cwd=ROOT, env=dict(os.environ))
    ).strip()


def _tracked_clean() -> bool:
    result = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        env=dict(os.environ),
    )
    return result.returncode == 0 and not result.stdout.strip()


def _source_git_objects(
    source_commit: str, source_paths: Sequence[Path]
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    environment = dict(os.environ)
    for path in source_paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        committed = _run(
            ["git", "rev-parse", f"{source_commit}:{relative}"],
            cwd=ROOT,
            env=environment,
        )
        materialized = _run(
            ["git", "hash-object", "--path", relative, relative],
            cwd=ROOT,
            env=environment,
        )
        expected = committed.stdout.strip() if committed.returncode == 0 else ""
        actual = materialized.stdout.strip() if materialized.returncode == 0 else ""
        facts[relative] = {
            "committed_git_blob": expected,
            "materialized_git_blob": actual,
            "exact": len(expected) == 40 and expected == actual,
        }
    return facts


def _git_safe_environment(
    environment: dict[str, str], repository: Path
) -> dict[str, str]:
    """Authorize one owned exFAT fixture repository for child Git commands."""
    result = dict(environment)
    count = int(result.get("GIT_CONFIG_COUNT", "0"))
    result["GIT_CONFIG_COUNT"] = str(count + 1)
    result[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    result[f"GIT_CONFIG_VALUE_{count}"] = repository.as_posix()
    return result


def _make_git_fixture_writable(repository: Path) -> None:
    """Clear Git object read-only bits before removing an owned fixture tree."""
    for path in repository.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWRITE)


def _tiny_input(path: Path) -> None:
    yy, xx = np.indices((43, 61), dtype=np.uint16)
    rgb = np.stack(
        (
            (xx * 17 + yy * 7) % 256,
            (xx * 5 + yy * 19 + 31) % 256,
            (xx * 13 + yy * 11 + 67) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)


def _wheel_rows(wheelhouse: Path) -> list[dict[str, Any]]:
    return [
        {"bytes": path.stat().st_size, "name": path.name, "sha256": _sha256(path)}
        for path in sorted(
            wheelhouse.glob("*.whl"), key=lambda item: item.name.casefold()
        )
    ]


def _cmd(
    launcher: Path, arguments: Sequence[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return _run(["cmd.exe", "/d", "/c", str(launcher), *arguments], cwd=cwd, env=env)


def _window_handles() -> list[int]:
    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(hwnd: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length < 1:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value == WINDOW_TITLE:
            handles.append(int(hwnd))
        return True

    if not user32.EnumWindows(callback, 0):
        raise RuntimeError("Win32 window enumeration failed")
    return handles


def _wait_for_one_window(timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handles = _window_handles()
        if len(handles) == 1:
            return handles[0]
        if len(handles) > 1:
            raise RuntimeError("multiple K-MCFM desktop windows are open")
        time.sleep(0.05)
    raise RuntimeError("installed desktop window did not appear")


def _client_box(hwnd: int) -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32
    rectangle = _Rect()
    origin = _Point(0, 0)
    if not user32.GetClientRect(hwnd, ctypes.byref(rectangle)):
        raise RuntimeError("Win32 client rectangle unavailable")
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise RuntimeError("Win32 client origin unavailable")
    width = int(rectangle.right - rectangle.left)
    height = int(rectangle.bottom - rectangle.top)
    return int(origin.x), int(origin.y), width, height


def _window_process(hwnd: int) -> tuple[int, int]:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    process_id = ctypes.c_ulong()
    if not user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id)):
        raise RuntimeError("Win32 window process unavailable")
    handle = kernel32.OpenProcess(0x00100000 | 0x001000, False, process_id.value)
    if not handle:
        raise RuntimeError("Win32 window process handle unavailable")
    return int(process_id.value), int(handle)


def _click_client(hwnd: int, x_fraction: float, y_fraction: float) -> list[int]:
    user32 = ctypes.windll.user32
    left, top, width, height = _client_box(hwnd)
    previous = _Point()
    if not user32.GetCursorPos(ctypes.byref(previous)):
        raise RuntimeError("cursor position unavailable")
    x = left + round(width * x_fraction)
    y = top + round(height * y_fraction)
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    if not user32.SetCursorPos(x, y):
        raise RuntimeError("cursor move failed")
    time.sleep(0.08)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.08)
    user32.SetCursorPos(previous.x, previous.y)
    return [x - left, y - top]


def _wait_for_preview_files(scratch: Path, timeout: float) -> list[Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        workspaces = list(scratch.glob("u7-10a-preview-*"))
        if len(workspaces) == 1:
            previews = sorted((workspaces[0] / "previews").glob("*.preview.png"))
            manifest = workspaces[0] / "previews" / "preview.json"
            if len(previews) == 3 and manifest.is_file():
                return previews
        time.sleep(0.05)
    raise RuntimeError("installed desktop previews did not become ready")


def _visual_facts(hwnd: int, output: Path) -> dict[str, Any]:
    _left, _top, width, height = _client_box(hwnd)
    image = ImageGrab.grab(window=hwnd).convert("RGB")
    image.save(output, format="PNG", compress_level=6)
    pixels = np.asarray(image)
    export = pixels[
        round(height * 0.91) : round(height * 0.985),
        round(width * 0.82) : round(width * 0.99),
    ]
    velvia = pixels[
        round(height * 0.76) : round(height * 0.83),
        round(width * 0.015) : round(width * 0.18),
    ]
    export_green = np.mean(
        (export[..., 0] > 160) & (export[..., 1] > 220) & (export[..., 2] < 190)
    )
    velvia_green = np.mean(
        (velvia[..., 0] > 160) & (velvia[..., 1] > 210) & (velvia[..., 2] < 180)
    )
    return {
        "client_height": height,
        "client_width": width,
        "export_green_fraction": float(export_green),
        "screenshot_bytes": output.stat().st_size,
        "screenshot_sha256": _sha256(output),
        "velvia_green_fraction": float(velvia_green),
    }


def _close_window(hwnd: int) -> None:
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)


def _desktop_interaction(
    *,
    launcher: Path,
    source: Path,
    scratch: Path,
    foreign_cwd: Path,
    environment: dict[str, str],
    visual_output: Path,
) -> dict[str, Any]:
    if _window_handles():
        raise RuntimeError("an existing K-MCFM desktop window would confound formal")
    scratch.mkdir()
    hostile = scratch.parent / "hostile-python"
    hostile.mkdir()
    startup_marker = hostile / "sitecustomize-ran.txt"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(startup_marker)!r}).write_text('ran')\n",
        encoding="utf-8",
        newline="\n",
    )
    mixed = dict(environment)
    mixed["PyThOnPaTh"] = str(hostile)
    mixed["PYTHONUSERBASE"] = str(hostile / "user-base")
    process = subprocess.Popen(
        [
            "cmd.exe",
            "/d",
            "/c",
            str(launcher),
            "--input",
            str(source),
            "--scratch-root",
            str(scratch),
        ],
        cwd=foreign_cwd,
        env=mixed,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    hwnd = 0
    window_process_handle = 0
    try:
        hwnd = _wait_for_one_window(20.0)
        _window_process_id, window_process_handle = _window_process(hwnd)
        preview_click = _click_client(hwnd, 0.895, 0.25)
        preview_files = _wait_for_preview_files(scratch, 90.0)
        time.sleep(0.75)
        radio_click = _click_client(hwnd, 0.04, 0.795)
        time.sleep(0.4)
        visual = _visual_facts(hwnd, visual_output)
        _close_window(hwnd)
        stdout, stderr = process.communicate(timeout=20)
    except BaseException:
        if hwnd:
            _close_window(hwnd)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        if window_process_handle:
            ctypes.windll.kernel32.CloseHandle(window_process_handle)
        raise
    window_process_terminated = (
        ctypes.windll.kernel32.WaitForSingleObject(window_process_handle, 0) == 0
    )
    ctypes.windll.kernel32.CloseHandle(window_process_handle)
    scratch_residue = sorted(path.name for path in scratch.iterdir())
    remaining_windows = _window_handles()
    result = {
        "client_geometry": [visual["client_width"], visual["client_height"]],
        "explicit_selection_action": "physical_velvia_radio_click",
        "preview_click": preview_click,
        "preview_count": len(preview_files),
        "process_returncode": process.returncode,
        "radio_click": radio_click,
        "remaining_window_count": len(remaining_windows),
        "scratch_residue_count": len(scratch_residue),
        "startup_marker_exists": startup_marker.exists(),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "visual": visual,
        "window_process_terminated": window_process_terminated,
    }
    shutil.rmtree(scratch)
    shutil.rmtree(hostile)
    return result


def _generated_desktop_control(
    *,
    runtime_python: Path,
    path: Path,
    project_root: Path,
    source_commit: str,
    requirements: Path,
    requirements_sha256: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    path.write_text(
        installer._launcher_source(
            project_root=project_root,
            source_commit=source_commit,
            requirements_path=requirements,
            requirements_sha256=requirements_sha256,
            entrypoint=Path("scripts/open_product_desktop.py"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    before = _window_handles()
    result = _run(
        [str(runtime_python), "-I", str(path), "--smoke-exit-ms", "100"],
        cwd=path.parent,
        env=environment,
    )
    after = _window_handles()
    return {
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "window_count_before": len(before),
        "window_count_after": len(after),
    }


def _tracked_drift_control(
    *, root: Path, runtime_python: Path, environment: dict[str, str]
) -> dict[str, Any]:
    repository = root / "tracked-drift-repository"
    control_environment = _git_safe_environment(environment, repository)
    (repository / "scripts").mkdir(parents=True)
    requirements = repository / "requirements.txt"
    requirements.write_bytes(b"fixture\n")
    entry = repository / "scripts" / "open_product_desktop.py"
    entry.write_text("raise SystemExit('entrypoint must remain unread')\n", "utf-8")
    _success(_run(["git", "init", "-q"], cwd=repository, env=control_environment))
    _success(
        _run(
            [
                "git",
                "-c",
                "user.name=K-MCFM",
                "-c",
                "user.email=local@invalid",
                "add",
                ".",
            ],
            cwd=repository,
            env=control_environment,
        )
    )
    _success(
        _run(
            [
                "git",
                "-c",
                "user.name=K-MCFM",
                "-c",
                "user.email=local@invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=repository,
            env=control_environment,
        )
    )
    commit = _success(
        _run(["git", "rev-parse", "HEAD"], cwd=repository, env=control_environment)
    ).strip()
    entry.write_text(
        "raise SystemExit('dirty entrypoint must remain unread')\n", "utf-8"
    )
    return _generated_desktop_control(
        runtime_python=runtime_python,
        path=root / "tracked-drift-launch.py",
        project_root=repository,
        source_commit=commit,
        requirements=requirements,
        requirements_sha256=_sha256(requirements),
        environment=control_environment,
    )


def _environment_strip_control(
    *, root: Path, runtime_python: Path, environment: dict[str, str]
) -> dict[str, Any]:
    repository = root / "environment-repository"
    control_environment = _git_safe_environment(environment, repository)
    (repository / "scripts").mkdir(parents=True)
    requirements = repository / "requirements.txt"
    requirements.write_bytes(b"fixture\n")
    entry = repository / "scripts" / "environment_probe.py"
    entry.write_text(
        "import json,os,sys\n"
        "from pathlib import Path\n"
        "keys=sorted(k for k in os.environ if k.upper().startswith('PYTHON'))\n"
        "Path(sys.argv[1]).write_text(json.dumps(keys), encoding='utf-8')\n",
        encoding="utf-8",
        newline="\n",
    )
    _success(_run(["git", "init", "-q"], cwd=repository, env=control_environment))
    _success(
        _run(
            [
                "git",
                "-c",
                "user.name=K-MCFM",
                "-c",
                "user.email=local@invalid",
                "add",
                ".",
            ],
            cwd=repository,
            env=control_environment,
        )
    )
    _success(
        _run(
            [
                "git",
                "-c",
                "user.name=K-MCFM",
                "-c",
                "user.email=local@invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            cwd=repository,
            env=control_environment,
        )
    )
    commit = _success(
        _run(["git", "rev-parse", "HEAD"], cwd=repository, env=control_environment)
    ).strip()
    launcher = root / "environment-launch.py"
    launcher.write_text(
        installer._launcher_source(
            project_root=repository,
            source_commit=commit,
            requirements_path=requirements,
            requirements_sha256=_sha256(requirements),
            entrypoint=Path("scripts/environment_probe.py"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    output = root / "environment-keys.json"
    hostile = dict(control_environment)
    hostile["PyThOnPaTh"] = "foreign"
    hostile["PYTHONUSERBASE"] = "foreign-user-base"
    result = _run(
        [str(runtime_python), "-I", str(launcher), str(output)],
        cwd=root,
        env=hostile,
    )
    keys = json.loads(output.read_text("utf-8")) if output.is_file() else None
    return {
        "child_python_keys": keys,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
    }


def _existing_destination_control(path: Path, wheelhouse: Path) -> bool:
    path.mkdir()
    marker = path / "foreign.bin"
    marker.write_bytes(b"foreign")
    try:
        installer.install_product_runtime(path, wheelhouse=wheelhouse)
    except FileExistsError:
        return marker.read_bytes() == b"foreign"
    return False


def _late_foreign_control(
    path: Path, displaced: Path, wheelhouse: Path, environment: dict[str, str]
) -> bool:
    replaced = False

    def runner(
        command: Sequence[str], cwd: Path | None, env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        nonlocal replaced
        arguments = tuple(map(str, command))
        if not replaced and "pip" in arguments and "install" in arguments:
            path.rename(displaced)
            path.mkdir()
            (path / "foreign.bin").write_bytes(b"foreign")
            replaced = True
            return subprocess.CompletedProcess(arguments, 1, "", "injected")
        return installer._run(command, cwd, env)

    try:
        installer.install_product_runtime(
            path, wheelhouse=wheelhouse, runner=runner, environment=environment
        )
    except RuntimeError:
        passed = (
            replaced
            and (path / "foreign.bin").read_bytes() == b"foreign"
            and displaced.is_dir()
        )
        shutil.rmtree(path)
        shutil.rmtree(displaced)
        return passed
    return False


def audit(
    root: Path, wheelhouse: Path, visual_output: Path, order: str
) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError(order)
    root = root.resolve(strict=False)
    wheelhouse = wheelhouse.resolve(strict=True)
    visual_output = visual_output.resolve(strict=False)
    tmp_root = (ROOT / "tmp").resolve(strict=True)
    output_root = (ROOT / "outputs").resolve(strict=True)
    if tmp_root not in root.parents or tmp_root not in wheelhouse.parents:
        raise ValueError("formal roots must remain under repo-relative tmp")
    if output_root not in visual_output.parents:
        raise ValueError("visual output must remain under repo-relative outputs")
    if os.path.lexists(root):
        raise FileExistsError(f"formal root already exists: {root}")
    if os.path.lexists(visual_output):
        raise FileExistsError(f"visual output already exists: {visual_output}")
    visual_output.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    root_identity = installer._entry_identity(root)
    environment = dict(os.environ)
    process_temp = root / "process-temp"
    process_temp.mkdir()
    environment.update(
        {
            "PIP_CACHE_DIR": str(root / "pip-cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEMP": str(process_temp),
            "TMP": str(process_temp),
        }
    )
    (root / "pip-cache").mkdir()
    source_paths = (
        CONFIG,
        CONTRACT,
        AUDIT,
        AUDIT_TEST,
        CORE_TEST,
        REQUIREMENTS,
        INSTALLER,
        DESKTOP_ENTRY,
        DESKTOP_CORE,
        DESKTOP_UI,
        RENDERER,
        CREATE_ONLY,
    )
    source_before = {path.as_posix(): _sha256(path) for path in source_paths}
    source_commit = _source_commit()
    tracked_clean_before = _tracked_clean()
    source_git_objects = _source_git_objects(source_commit, source_paths)
    if not all(row["exact"] for row in source_git_objects.values()):
        raise RuntimeError("formal sources must be exact committed Git objects")
    wheel_rows = _wheel_rows(wheelhouse)
    installation = root / "installation"
    receipt = installer.install_product_runtime(
        installation, wheelhouse=wheelhouse, environment=environment
    )
    receipt_path = installation / "product-runtime.json"
    runtime_python = installation / "runtime" / "Scripts" / "python.exe"
    foreign_cwd = root / "foreign-cwd"
    foreign_cwd.mkdir()
    launchers = receipt["launchers"]
    cli = Path(launchers["cli"]["command"])
    desktop = Path(launchers["desktop"]["command"])
    pip_check = _run(
        [str(runtime_python), "-m", "pip", "check"],
        cwd=foreign_cwd,
        env=environment,
    )

    catalog_result = _cmd(
        cli, ["--list-product-looks"], cwd=foreign_cwd, env=environment
    )
    catalog = json.loads(_success(catalog_result))
    available = [
        row["look_id"] for row in catalog["looks"] if row["availability"] == "available"
    ]
    source = root / "input.png"
    _tiny_input(source)
    styles = ["velvia_50", "portra_400", "ektar_100"]
    if order == "reverse":
        styles.reverse()
    render_rows: list[dict[str, Any]] = []
    for style in styles:
        direct = root / f"direct-{style}.png"
        launched = root / f"launched-{style}.png"
        _success(
            _run(
                [
                    str(runtime_python),
                    str(RENDERER),
                    str(source),
                    "--product-look",
                    style,
                    "--output",
                    str(direct),
                ],
                cwd=foreign_cwd,
                env=environment,
            )
        )
        _success(
            _cmd(
                cli,
                [str(source), "--product-look", style, "--output", str(launched)],
                cwd=foreign_cwd,
                env=environment,
            )
        )
        render_rows.append(
            {
                "byte_exact": direct.read_bytes() == launched.read_bytes(),
                "output_sha256": _sha256(direct),
                "style_id": style,
            }
        )
    render_rows.sort(key=lambda row: row["style_id"])

    interaction = _desktop_interaction(
        launcher=desktop,
        source=source,
        scratch=root / "gui-scratch",
        foreign_cwd=foreign_cwd,
        environment=environment,
        visual_output=visual_output,
    )

    shadow_requirements = root / "shadow-requirements.txt"
    shadow_requirements.write_bytes(REQUIREMENTS.read_bytes() + b"# drift\n")
    requirements_control = _generated_desktop_control(
        runtime_python=runtime_python,
        path=root / "requirements-drift-launch.py",
        project_root=ROOT,
        source_commit=source_commit,
        requirements=shadow_requirements,
        requirements_sha256=installer._sha256(REQUIREMENTS),
        environment=environment,
    )
    head_control = _generated_desktop_control(
        runtime_python=runtime_python,
        path=root / "head-drift-launch.py",
        project_root=ROOT,
        source_commit="0" * 40,
        requirements=REQUIREMENTS,
        requirements_sha256=installer._sha256(REQUIREMENTS),
        environment=environment,
    )
    tracked_control = _tracked_drift_control(
        root=root, runtime_python=runtime_python, environment=environment
    )
    environment_control = _environment_strip_control(
        root=root, runtime_python=runtime_python, environment=environment
    )
    invalid_result = _cmd(
        desktop, ["--smoke-exit-ms", "0"], cwd=foreign_cwd, env=environment
    )
    invalid_control = {
        "returncode": invalid_result.returncode,
        "stderr": invalid_result.stderr.strip(),
        "window_count_after": len(_window_handles()),
    }
    existing_pass = _existing_destination_control(root / "existing", wheelhouse)
    shutil.rmtree(root / "existing")
    late_foreign_pass = _late_foreign_control(
        root / "late-foreign", root / "late-displaced", wheelhouse, environment
    )

    launcher_bindings = {
        role: {
            "command_sha256": _sha256(Path(row["command"])),
            "python_sha256": _sha256(Path(row["python"])),
            "receipt_command_exact": _sha256(Path(row["command"]))
            == row["command_sha256"],
            "receipt_python_exact": _sha256(Path(row["python"]))
            == row["python_sha256"],
        }
        for role, row in launchers.items()
    }
    source_after = {path.as_posix(): _sha256(path) for path in source_paths}
    final_source_commit = _source_commit()
    tracked_clean_after = _tracked_clean()
    config_claim = json.loads(CONFIG.read_text("utf-8"))["claim_ceiling"]
    receipt_claim = {
        key: config_claim[key]
        for key in (
            "mode",
            "evidence_grade",
            "calibrated_stock_response",
            "physical_film_reproduction",
            "public_release",
        )
    }
    gates = {
        "all_four_launcher_files_receipt_bound": len(launcher_bindings) == 2
        and all(
            row["receipt_command_exact"] and row["receipt_python_exact"]
            for row in launcher_bindings.values()
        ),
        "desktop_explicit_radio_invoke": interaction["explicit_selection_action"]
        == "physical_velvia_radio_click"
        and interaction["visual"]["export_green_fraction"] > 0.15
        and interaction["visual"]["velvia_green_fraction"] > 0.001,
        "existing_and_late_foreign_preserved": existing_pass and late_foreign_pass,
        "foreign_cwd_cli_exact": catalog_result.returncode == 0
        and available == ["velvia_50", "portra_400", "ektar_100"],
        "foreign_cwd_desktop_mainloop": interaction["process_returncode"] == 0
        and interaction["client_geometry"] == [1180, 760]
        and interaction["preview_count"] == 3,
        "fresh_install_success": receipt["schema"]
        == "kmcfm.private-product-runtime-receipt.v2",
        "legacy_cli_field_retained": receipt["launcher"]
        == receipt["launchers"]["cli"]["command"],
        "owned_scratch_and_process_residue_zero": interaction["scratch_residue_count"]
        == 0
        and interaction["remaining_window_count"] == 0,
        "owned_process_terminated": interaction["process_returncode"] == 0
        and interaction["window_process_terminated"],
        "pinned_distributions_exact": receipt["distributions"]
        == installer._pinned_versions(REQUIREMENTS),
        "pip_check_success": pip_check.returncode == 0
        and "No broken requirements found" in pip_check.stdout,
        "python_environment_removed": not interaction["startup_marker_exists"]
        and environment_control["returncode"] == 0
        and environment_control["child_python_keys"] == [],
        "receipt_claim_and_private_exact": receipt["claim"] == receipt_claim
        and receipt["repository_bound"] is True
        and receipt["public_distribution"] is False,
        "source_immutable": source_before == source_after,
        "source_git_objects_exact": all(
            row["exact"] for row in source_git_objects.values()
        ),
        "source_commit_stable": final_source_commit == source_commit
        and tracked_clean_after,
        "tracked_clean_before": tracked_clean_before,
        "drift_rejects_before_tk": all(
            row["returncode"] == 2
            and row["window_count_before"] == 0
            and row["window_count_after"] == 0
            for row in (requirements_control, head_control, tracked_control)
        )
        and requirements_control["stderr"].endswith(
            "product requirements identity drift"
        )
        and head_control["stderr"].endswith("repository commit drift")
        and tracked_control["stderr"].endswith("tracked repository drift")
        and invalid_control["returncode"] != 0
        and invalid_control["window_count_after"] == 0,
        "three_cli_outputs_exact": len(render_rows) == 3
        and all(row["byte_exact"] for row in render_rows),
        "wheelhouse_exact": len(wheel_rows) == EXPECTED_WHEEL_COUNT
        and sum(row["bytes"] for row in wheel_rows) == EXPECTED_WHEEL_BYTES,
    }
    report = {
        "schema": "kmcfm.u7-10b-installed-runtime-desktop-launch-result.v1",
        "status": "PASS_PRIVATE_U7_10B_INSTALLED_RUNTIME_DESKTOP_LAUNCH"
        if all(gates.values())
        else "FAIL_CLOSED_U7_10B_INSTALLED_RUNTIME_DESKTOP_LAUNCH",
        "automatic_pass": all(gates.values()),
        "catalog": catalog["looks"],
        "claim": config_claim,
        "controls": {
            "environment_strip": environment_control,
            "head_drift": head_control,
            "invalid_argument": invalid_control,
            "requirements_drift": requirements_control,
            "tracked_drift": tracked_control,
        },
        "gates": gates,
        "interaction": interaction,
        "launcher_bindings": launcher_bindings,
        "receipt_sha256": _sha256(receipt_path),
        "renders": render_rows,
        "source": {
            str(path.relative_to(ROOT)).replace("\\", "/"): source_before[
                path.as_posix()
            ]
            for path in source_paths
        },
        "source_commit": source_commit,
        "source_git_objects": source_git_objects,
        "wheelhouse": wheel_rows,
    }
    for repository in (
        root / "tracked-drift-repository",
        root / "environment-repository",
    ):
        _make_git_fixture_writable(repository)
    if not installer._cleanup_owned_directory(root, root_identity):
        raise RuntimeError("formal root ownership changed; preserved")
    report["formal_root_residue_count"] = int(root.exists())
    report["stable_identity"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--visual-output", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _preflight_report_output(args.output)
    report = audit(args.root, args.wheelhouse, args.visual_output, args.order)
    _publish_report(args.output, report)
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
