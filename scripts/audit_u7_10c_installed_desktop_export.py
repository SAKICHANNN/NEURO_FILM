#!/usr/bin/env python3
"""Audit the installed native desktop's real export and replay path."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_u7_10b_installed_runtime_desktop_launch as parent
from scripts import install_product_runtime as installer
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

CONFIG = ROOT / "configs" / "u7_10c_installed_desktop_export_v1.json"
CONTRACT = ROOT / "docs" / "planning" / "U7_10C_INSTALLED_DESKTOP_EXPORT_CONTRACT.md"
AUDIT = ROOT / "scripts" / "audit_u7_10c_installed_desktop_export.py"
AUDIT_TEST = ROOT / "tests" / "test_audit_u7_10c_installed_desktop_export.py"
REQUIREMENTS = ROOT / "requirements-product-v2.txt"
INSTALLER = ROOT / "scripts" / "install_product_runtime.py"
PARENT_AUDIT = ROOT / "scripts" / "audit_u7_10b_installed_runtime_desktop_launch.py"
PARENT_EXPORT_TEST = ROOT / "tests" / "test_u7_10a_product_desktop_input_workflow.py"
DESKTOP_ENTRY = ROOT / "scripts" / "open_product_desktop.py"
DESKTOP_CORE = ROOT / "src" / "inference" / "product_desktop.py"
DESKTOP_UI = ROOT / "src" / "inference" / "product_desktop_ui.py"
RENDERER = ROOT / "scripts" / "render_film.py"
STYLE_ENGINE = ROOT / "src" / "inference" / "style_safe_engine.py"
PROFILE = ROOT / "configs" / "render_profiles" / "safe_rich_product_v1.json"
PRODUCT_TITLE = "K-MCFM Look Approximation"
SAVE_TITLE = "Export Look Approximation"
COMPLETE_TITLE = "Export complete"
IDOK = 1
GW_OWNER = 4
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_A = 0x41
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_N = 0x4E
VK_RETURN = 0x0D


class _KeyboardInput(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("virtual_key", ctypes.c_ushort),
        ("scan_code", ctypes.c_ushort),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouse_data", ctypes.c_ulong),
        ("flags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("extra_info", ctypes.c_size_t),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("message", ctypes.c_ulong),
        ("parameter_low", ctypes.c_ushort),
        ("parameter_high", ctypes.c_ushort),
    ]


class _InputPayload(ctypes.Union):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("keyboard", _KeyboardInput),
        ("mouse", _MouseInput),
        ("hardware", _HardwareInput),
    ]


class _Input(ctypes.Structure):
    _anonymous_: ClassVar[tuple[str, ...]] = ("payload",)
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("kind", ctypes.c_ulong),
        ("payload", _InputPayload),
    ]


def _text(hwnd: int) -> str:
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _pid(hwnd: int) -> int:
    process_id = ctypes.c_ulong()
    if not ctypes.windll.user32.GetWindowThreadProcessId(
        hwnd, ctypes.byref(process_id)
    ):
        raise RuntimeError("Win32 window process unavailable")
    return int(process_id.value)


def _owner_chain(hwnd: int) -> list[int]:
    owners: list[int] = []
    current = int(hwnd)
    while True:
        current = int(ctypes.windll.user32.GetWindow(current, GW_OWNER))
        if not current:
            return owners
        if current in owners:
            raise RuntimeError("cyclic Win32 owner chain")
        owners.append(current)


def _top_level_windows() -> list[int]:
    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(hwnd: int, _parameter: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            handles.append(int(hwnd))
        return True

    if not user32.EnumWindows(callback, 0):
        raise RuntimeError("Win32 window enumeration failed")
    return handles


def _owned_titled_windows(*, title: str, process_id: int, owner: int) -> list[int]:
    return [
        hwnd
        for hwnd in _top_level_windows()
        if _text(hwnd) == title
        and _pid(hwnd) == process_id
        and (hwnd == owner or owner in _owner_chain(hwnd))
    ]


def _wait_owned_window(
    *, title: str, process_id: int, owner: int, timeout: float
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = _owned_titled_windows(
            title=title, process_id=process_id, owner=owner
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"multiple owned windows titled {title!r}")
        time.sleep(0.05)
    raise RuntimeError(f"owned window did not appear: {title}")


def _child_windows(hwnd: int) -> list[int]:
    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    @callback_type
    def callback(child: int, _parameter: int) -> bool:
        handles.append(int(child))
        return True

    if not user32.EnumChildWindows(hwnd, callback, 0):
        # EnumChildWindows returns zero when there are no children as well as on error.
        if not handles:
            return []
        raise RuntimeError("Win32 child enumeration failed")
    return handles


def _key_chord(modifier: int, key: int) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(modifier, 0, 0, 0)
    user32.keybd_event(key, 0, 0, 0)
    user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(modifier, 0, KEYEVENTF_KEYUP, 0)


def _send_unicode_text(value: str) -> None:
    user32 = ctypes.windll.user32
    for character in value:
        code = ord(character)
        events = (_Input * 2)(
            _Input(
                kind=1,
                payload=_InputPayload(
                    keyboard=_KeyboardInput(0, code, KEYEVENTF_UNICODE, 0, 0)
                ),
            ),
            _Input(
                kind=1,
                payload=_InputPayload(
                    keyboard=_KeyboardInput(
                        0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0
                    )
                ),
            ),
        )
        sent = int(user32.SendInput(2, events, ctypes.sizeof(_Input)))
        if sent != 2:
            raise RuntimeError(f"Unicode SendInput incomplete: {sent}/2")
        time.sleep(0.002)


def _physical_dialog_button_click(dialog: int, control_id: int) -> dict[str, Any]:
    user32 = ctypes.windll.user32
    control = int(user32.GetDlgItem(dialog, control_id))
    if (
        not control
        or _class_name(control) != "Button"
        or not user32.IsWindowVisible(control)
        or not user32.IsWindowEnabled(control)
    ):
        raise RuntimeError(f"dialog button {control_id} is unavailable")
    rectangle = parent._Rect()
    previous = parent._Point()
    if not user32.GetWindowRect(control, ctypes.byref(rectangle)):
        raise RuntimeError("dialog button rectangle unavailable")
    if not user32.GetCursorPos(ctypes.byref(previous)):
        raise RuntimeError("cursor position unavailable")
    x = round((rectangle.left + rectangle.right) / 2)
    y = round((rectangle.top + rectangle.bottom) / 2)
    if not user32.SetCursorPos(x, y):
        raise RuntimeError("cursor move failed")
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.08)
    user32.SetCursorPos(previous.x, previous.y)
    return {
        "button_class": _class_name(control),
        "button_id": int(user32.GetDlgCtrlID(control)),
        "physical_click": True,
    }


def _set_save_path_and_accept(dialog: int, destination: Path) -> dict[str, Any]:
    user32 = ctypes.windll.user32
    requested = str(destination.resolve(strict=False))
    if not user32.SetForegroundWindow(dialog):
        raise RuntimeError("save dialog could not become foreground")
    time.sleep(0.1)
    _key_chord(VK_MENU, VK_N)
    time.sleep(0.08)
    _key_chord(VK_CONTROL, VK_A)
    _send_unicode_text(requested)
    time.sleep(0.4)
    accept = _physical_dialog_button_click(dialog, IDOK)
    return {
        "accept": accept,
        "dialog_class": _class_name(dialog),
        "input_method": "foreground_alt_n_unicode_sendinput",
        "requested_destination_sha256": hashlib.sha256(
            requested.casefold().encode("utf-8")
        ).hexdigest(),
    }


def _wait_for_pair(output: Path, recipe: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if output.is_file() and recipe.is_file():
            return
        time.sleep(0.05)
    raise RuntimeError("desktop export pair did not appear")


def _wait_window_closed(hwnd: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not ctypes.windll.user32.IsWindow(hwnd):
            return
        time.sleep(0.05)
    raise RuntimeError("owned window did not close")


def _dismiss_dialog(dialog: int) -> dict[str, Any]:
    user32 = ctypes.windll.user32
    if not user32.SetForegroundWindow(dialog):
        raise RuntimeError("completion dialog could not become foreground")
    time.sleep(0.08)
    user32.keybd_event(VK_RETURN, 0, 0, 0)
    user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
    return {"input_method": "foreground_physical_enter"}


def _screenshot(hwnd: int, path: Path) -> dict[str, Any]:
    _left, _top, width, height = parent._client_box(hwnd)
    image = ImageGrab.grab(window=hwnd).convert("RGB")
    image.save(path, format="PNG", compress_level=6)
    return {
        "bytes": path.stat().st_size,
        "client_geometry": [width, height],
        "sha256": parent._sha256(path),
    }


def _direct_oracle(
    *, launcher: Path, source: Path, destination: Path, cwd: Path, env: dict[str, str]
) -> tuple[bytes, bytes]:
    result = parent._cmd(
        launcher,
        [
            str(source),
            "--product-look",
            "velvia_50",
            "--look-amount",
            "1",
            "--output-bit-depth",
            "16",
            "--png-compression",
            "6",
            "--write-recipe",
            "--tile-size",
            "256",
            "--tile-workers",
            "1",
            "--output",
            str(destination),
        ],
        cwd=cwd,
        env=env,
    )
    parent._success(result)
    recipe = destination.with_suffix(".recipe.json")
    output_bytes = destination.read_bytes()
    recipe_bytes = recipe.read_bytes()
    destination.unlink()
    recipe.unlink()
    return output_bytes, recipe_bytes


def _desktop_export(
    *,
    launcher: Path,
    source: Path,
    destination: Path,
    scratch: Path,
    foreign_cwd: Path,
    environment: dict[str, str],
    visual_output: Path,
) -> dict[str, Any]:
    if parent._window_handles():
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
    main = 0
    process_handle = 0
    try:
        main = parent._wait_for_one_window(20.0)
        process_id, process_handle = parent._window_process(main)
        preview_click = parent._click_client(main, 0.895, 0.25)
        preview_files = parent._wait_for_preview_files(scratch, 90.0)
        time.sleep(0.75)
        radio_click = parent._click_client(main, 0.04, 0.795)
        time.sleep(0.35)
        export_click = parent._click_client(main, 0.895, 0.945)
        save_dialog = _wait_owned_window(
            title=SAVE_TITLE, process_id=process_id, owner=main, timeout=20.0
        )
        save_owner_chain = _owner_chain(save_dialog)
        save = _set_save_path_and_accept(save_dialog, destination)
        _wait_window_closed(save_dialog, 20.0)
        recipe_path = destination.with_suffix(".recipe.json")
        _wait_for_pair(destination, recipe_path, 120.0)
        completion = _wait_owned_window(
            title=COMPLETE_TITLE, process_id=process_id, owner=main, timeout=20.0
        )
        completion_owner_chain = _owner_chain(completion)
        completion_accept = _dismiss_dialog(completion)
        _wait_window_closed(completion, 20.0)
        time.sleep(0.35)
        visual = _screenshot(main, visual_output)
        parent._close_window(main)
        stdout, stderr = process.communicate(timeout=20)
    except BaseException:
        if main:
            parent._close_window(main)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        if process_handle:
            ctypes.windll.kernel32.CloseHandle(process_handle)
        raise
    process_terminated = (
        ctypes.windll.kernel32.WaitForSingleObject(process_handle, 0) == 0
    )
    ctypes.windll.kernel32.CloseHandle(process_handle)
    return {
        "completion_owner_bound": main in completion_owner_chain,
        "completion_accept": completion_accept,
        "explicit_actions": [
            "physical_preview_click",
            "physical_velvia_radio_click",
            "physical_export_click",
            "native_save_dialog_accept",
            "native_completion_dialog_accept",
        ],
        "export_click": export_click,
        "preview_click": preview_click,
        "preview_count": len(preview_files),
        "process_returncode": process.returncode,
        "process_stderr": stderr.strip(),
        "process_stdout": stdout.strip(),
        "process_terminated": process_terminated,
        "radio_click": radio_click,
        "remaining_product_windows": len(parent._window_handles()),
        "save_dialog": save,
        "save_owner_bound": main in save_owner_chain,
        "startup_marker_exists": startup_marker.exists(),
        "visual": visual,
    }


def _png16_rgb(path: Path) -> bool:
    header = path.read_bytes()[:29]
    return (
        len(header) == 29
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
        and header[24] == 16
        and header[25] == 2
    )


def _recipe_semantics(
    recipe: dict[str, Any], *, source: Path, output: Path, source_commit: str
) -> bool:
    profile = json.loads(PROFILE.read_text("utf-8"))
    return (
        recipe["input"]["sha256"] == parent._sha256(source)
        and Path(recipe["input"]["path"]).resolve(strict=False)
        == source.resolve(strict=False)
        and recipe["output"]["sha256"] == parent._sha256(output)
        and Path(recipe["output"]["path"]).resolve(strict=False)
        == output.resolve(strict=False)
        and recipe["output"]["bit_depth"] == 16
        and recipe["profile"]["profile_id"] == profile["profile_id"]
        and recipe["profile"]["profile_version"] == profile["profile_version"]
        and recipe["profile"]["sha256"] == parent._sha256(PROFILE)
        and recipe["render"]["style"] == "velvia_50"
        and float(recipe["render"]["look_amount"]) == 1.0
        and recipe["claim"]["evidence_grade"] == "look-approximation"
        and recipe["claim"].get("calibrated_reference_allowed") is False
        and str(recipe["software"]["commit"]).lower() == source_commit
    )


def audit(root: Path, wheelhouse: Path, visual_output: Path, order: str) -> dict[str, Any]:
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
    config = json.loads(CONFIG.read_text("utf-8"))
    parent_paths = tuple(
        ROOT / row["path"] for row in config["parent_bindings"].values()
    )
    source_paths = (
        CONFIG,
        CONTRACT,
        AUDIT,
        AUDIT_TEST,
        REQUIREMENTS,
        INSTALLER,
        PARENT_AUDIT,
        PARENT_EXPORT_TEST,
        DESKTOP_ENTRY,
        DESKTOP_CORE,
        DESKTOP_UI,
        RENDERER,
        STYLE_ENGINE,
        PROFILE,
        *parent_paths,
    )
    source_before = {path.as_posix(): parent._sha256(path) for path in source_paths}
    source_commit = parent._source_commit()
    tracked_clean_before = parent._tracked_clean()
    source_git_objects = parent._source_git_objects(source_commit, source_paths)
    if not all(row["exact"] for row in source_git_objects.values()):
        raise RuntimeError("formal sources must be exact committed Git objects")
    wheel_rows = parent._wheel_rows(wheelhouse)
    installation = root / "installation"
    receipt = installer.install_product_runtime(
        installation, wheelhouse=wheelhouse, environment=environment
    )
    runtime_python = installation / "runtime" / "Scripts" / "python.exe"
    foreign_cwd = root / "foreign-cwd"
    foreign_cwd.mkdir()
    pip_check = parent._run(
        [str(runtime_python), "-m", "pip", "check"],
        cwd=foreign_cwd,
        env=environment,
    )
    source = root / "input.png"
    parent._tiny_input(source)
    destination = root / "installed-desktop-velvia.png"
    recipe_path = destination.with_suffix(".recipe.json")
    oracle_output, oracle_recipe = _direct_oracle(
        launcher=Path(receipt["launchers"]["cli"]["command"]),
        source=source,
        destination=destination,
        cwd=foreign_cwd,
        env=environment,
    )
    interaction = _desktop_export(
        launcher=Path(receipt["launchers"]["desktop"]["command"]),
        source=source,
        destination=destination,
        scratch=root / "gui-scratch",
        foreign_cwd=foreign_cwd,
        environment=environment,
        visual_output=visual_output,
    )
    gui_output = destination.read_bytes()
    gui_recipe = recipe_path.read_bytes()
    recipe = json.loads(gui_recipe)
    replay = root / "strict-replay.png"
    replay_digest = replay_style_safe_recipe_to_file(
        recipe, profile_path=PROFILE, output_path=replay, root=ROOT
    )
    parent_evidence: dict[str, dict[str, Any]] = {}
    parent_binding_facts: dict[str, dict[str, Any]] = {}
    for role, row in config["parent_bindings"].items():
        path = ROOT / row["path"]
        actual_sha256 = parent._sha256(path)
        exact = actual_sha256 == row["sha256"]
        if not exact:
            raise RuntimeError(f"parent evidence identity drift: {role}")
        parent_evidence[role] = json.loads(path.read_text("utf-8"))
        parent_binding_facts[role] = {
            "actual_sha256": actual_sha256,
            "expected_sha256": row["sha256"],
            "exact": exact,
            "path": row["path"],
        }
    source_after = {path.as_posix(): parent._sha256(path) for path in source_paths}
    gates = {
        "direct_gui_png_byte_exact": oracle_output == gui_output,
        "direct_gui_recipe_byte_exact": oracle_recipe == gui_recipe,
        "parent_export_and_installer_foreign_controls_exact": parent_evidence[
            "installed_desktop_evidence"
        ]["gates"]["existing_and_late_foreign_preserved"]
        and str(parent_evidence["desktop_evidence"]["status"]).startswith("PASS_")
        and source_git_objects[
            "tests/test_u7_10a_product_desktop_input_workflow.py"
        ]["exact"],
        "foreign_cwd_and_python_isolation": not interaction[
            "startup_marker_exists"
        ],
        "fresh_install_and_pip_check": receipt["schema"]
        == "kmcfm.private-product-runtime-receipt.v2"
        and pip_check.returncode == 0
        and "No broken requirements found" in pip_check.stdout
        and len(wheel_rows) == parent.EXPECTED_WHEEL_COUNT
        and sum(row["bytes"] for row in wheel_rows) == parent.EXPECTED_WHEEL_BYTES,
        "gui_replay_png_byte_exact": replay.read_bytes() == gui_output
        and replay_digest == parent._sha256(destination),
        "owned_process_window_and_root_residue_zero": interaction[
            "process_returncode"
        ]
        == 0
        and interaction["process_terminated"]
        and interaction["remaining_product_windows"] == 0,
        "physical_preview_selection_export_actions": interaction[
            "explicit_actions"
        ]
        == [
            "physical_preview_click",
            "physical_velvia_radio_click",
            "physical_export_click",
            "native_save_dialog_accept",
            "native_completion_dialog_accept",
        ],
        "real_product_window_and_save_dialog": interaction["save_owner_bound"]
        and interaction["completion_owner_bound"]
        and interaction["save_dialog"]["dialog_class"] == "#32770"
        and interaction["save_dialog"]["accept"]["button_class"] == "Button"
        and interaction["save_dialog"]["accept"]["button_id"] == IDOK
        and interaction["completion_accept"]["input_method"]
        == "foreground_physical_enter",
        "receipt_and_launcher_exact": all(
            parent._sha256(Path(row["command"])) == row["command_sha256"]
            and parent._sha256(Path(row["python"])) == row["python_sha256"]
            for row in receipt["launchers"].values()
        )
        and receipt["claim"]
        == {
            key: config["claim_ceiling"][key]
            for key in (
                "mode",
                "evidence_grade",
                "calibrated_stock_response",
                "physical_film_reproduction",
                "public_release",
            )
        }
        and receipt["repository_bound"] is True
        and receipt["public_distribution"] is False,
        "source_commit_and_git_objects_exact": parent._source_commit()
        == source_commit
        and tracked_clean_before
        and parent._tracked_clean()
        and all(row["exact"] for row in source_git_objects.values())
        and source_before == source_after,
        "strict_recipe_semantics_exact": _recipe_semantics(
            recipe,
            source=source,
            output=destination,
            source_commit=source_commit,
        )
        and _png16_rgb(destination),
        "three_previews_and_visible_final_state": interaction["preview_count"] == 3
        and interaction["visual"]["client_geometry"] == [1180, 760]
        and interaction["visual"]["bytes"]
        == config["visual_lock"]["expected_screenshot_bytes"]
        and interaction["visual"]["sha256"]
        == config["visual_lock"]["expected_screenshot_sha256"]
        and config["visual_lock"]["manual_preflight_review"]
        == {
            "final_png_name_visible": "installed-desktop-velvia.png",
            "final_recipe_name_visible": "installed-desktop-velvia.recipe.json",
            "layout_crop_or_overlap_observed": False,
            "look_approximation_label_visible": True,
            "not_calibrated_label_visible": True,
            "three_preview_cards_visible": True,
            "velvia_selected": True,
        },
        "visual_matches_frozen_lock": interaction["visual"]["bytes"]
        == config["visual_lock"]["expected_screenshot_bytes"]
        and interaction["visual"]["sha256"]
        == config["visual_lock"]["expected_screenshot_sha256"],
    }
    if set(gates) != set(config["gates"]):
        raise RuntimeError("formal gate names differ from the frozen contract")
    report = {
        "schema": "kmcfm.u7-10c-installed-desktop-export-result.v1",
        "status": "PASS_PRIVATE_U7_10C_INSTALLED_DESKTOP_EXPORT"
        if all(gates.values())
        else "FAIL_CLOSED_U7_10C_INSTALLED_DESKTOP_EXPORT",
        "automatic_pass": all(gates.values()),
        "claim": config["claim_ceiling"],
        "gates": gates,
        "interaction": interaction,
        "media": {
            "gui_output_bytes": len(gui_output),
            "gui_output_sha256": parent._sha256(destination),
            "gui_recipe_bytes": len(gui_recipe),
            "gui_recipe_sha256": parent._sha256(recipe_path),
            "replay_sha256": parent._sha256(replay),
        },
        "parent_evidence": {
            role: {
                "status": value["status"],
                "automatic_pass": value.get(
                    "automatic_pass", str(value["status"]).startswith("PASS_")
                ),
            }
            for role, value in parent_evidence.items()
        },
        "parent_evidence_bindings": parent_binding_facts,
        "parent_export_safety_binding": {
            "git_object_exact": source_git_objects[
                "tests/test_u7_10a_product_desktop_input_workflow.py"
            ]["exact"],
            "test": "test_export_requires_current_preview_and_preserves_existing_pair",
        },
        "receipt_sha256": parent._sha256(installation / "product-runtime.json"),
        "source": {
            str(path.relative_to(ROOT)).replace("\\", "/"): source_before[
                path.as_posix()
            ]
            for path in source_paths
        },
        "source_commit": source_commit,
        "source_git_objects": source_git_objects,
        "wheelhouse": wheel_rows,
        "visual_lock": config["visual_lock"],
    }
    if not installer._cleanup_owned_directory(root, root_identity):
        raise RuntimeError("formal root ownership changed; preserved")
    report["formal_root_residue_count"] = int(root.exists())
    gates["owned_process_window_and_root_residue_zero"] = (
        gates["owned_process_window_and_root_residue_zero"]
        and report["formal_root_residue_count"] == 0
    )
    report["automatic_pass"] = all(gates.values())
    report["status"] = (
        "PASS_PRIVATE_U7_10C_INSTALLED_DESKTOP_EXPORT"
        if report["automatic_pass"]
        else "FAIL_CLOSED_U7_10C_INSTALLED_DESKTOP_EXPORT"
    )
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
    parent._preflight_report_output(args.output)
    report = audit(args.root, args.wheelhouse, args.visual_output, args.order)
    parent._publish_report(args.output, report)
    return 0 if report["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
