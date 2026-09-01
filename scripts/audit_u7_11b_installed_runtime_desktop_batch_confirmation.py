#!/usr/bin/env python3
"""Confirm U7.11A through the existing private installed runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_u7_10b_installed_runtime_desktop_launch as u710b
from scripts import install_product_runtime as installer
from src.inference.product_desktop import ProductDesktopError, ProductDesktopWorkflow
from src.inference.render_contract import atomic_write_json, sha256_file
from src.inference.style_safe_engine import replay_style_safe_recipe_to_file

CONFIG = ROOT / "configs/u7_11b_installed_runtime_desktop_batch_confirmation_v1.json"
CONTRACT = ROOT / (
    "docs/planning/U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION_CONTRACT.md"
)
AUDIT = ROOT / "scripts/audit_u7_11b_installed_runtime_desktop_batch_confirmation.py"
AUDIT_TEST = ROOT / (
    "tests/test_audit_u7_11b_installed_runtime_desktop_batch_confirmation.py"
)
INSTALLER = ROOT / "scripts/install_product_runtime.py"
REQUIREMENTS = ROOT / "requirements-product-v2.txt"
DESKTOP_ENTRY = ROOT / "scripts/open_product_desktop.py"
BATCH_CORE = ROOT / "src/inference/product_desktop.py"
BATCH_UI = ROOT / "src/inference/product_desktop_ui.py"
U711A_EVIDENCE = ROOT / "docs/evidence/U7_11A_DESKTOP_SINGLE_LOOK_BATCH_RESULT.json"
U710B_EVIDENCE = ROOT / (
    "docs/evidence/U7_10B_INSTALLED_RUNTIME_DESKTOP_LAUNCH_RESULT.json"
)
U79C_EVIDENCE = ROOT / (
    "docs/evidence/U7_9C_PRODUCT_RUNTIME_STARTUP_ISOLATION_RESULT.json"
)
WHEELHOUSE = ROOT / "tmp/u7_9a_wheelhouse_v1"
SCRATCH_PARENT = ROOT / "tmp"


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _environment(*, hostile: bool = False) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    if hostile:
        environment["PyThOnPaTh"] = "u7-11b-forbidden-python-path"
        environment["PYTHONUSERBASE"] = "u7-11b-forbidden-user-base"
    return environment


def _run(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: int = 240
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
        timeout=timeout,
    )


def _success(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed")[-4000:]
        raise RuntimeError(detail)
    return result.stdout


def _git_head() -> str:
    return _success(
        _run(["git", "rev-parse", "HEAD"], cwd=ROOT, env=_environment())
    ).strip().lower()


def _tracked_clean() -> bool:
    result = _run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        env=_environment(),
    )
    return result.returncode == 0 and not result.stdout.strip()


def _tiny_input(path: Path, offset: int) -> None:
    yy, xx = np.mgrid[:43, :61]
    rgb = np.stack(
        (
            (xx * 11 + yy * 3 + offset) % 251,
            (xx * 5 + yy * 13 + 17 + offset * 2) % 251,
            (xx * 7 + yy * 19 + 41 + offset * 3) % 251,
        ),
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)


def _snapshot(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _python_environment_keys(environment: dict[str, str]) -> list[str]:
    return sorted(
        key for key in environment if key.upper().startswith("PYTHON")
    )


def _member_subset(
    members: dict[str, dict[str, Any]], suffix: str
) -> dict[str, dict[str, Any]]:
    return {name: row for name, row in members.items() if name.endswith(suffix)}


def _remove_owned_snapshot_directory(
    directory: Path, expected: dict[str, dict[str, Any]]
) -> None:
    if _snapshot(directory) != expected or any(
        not path.is_file() for path in directory.iterdir()
    ):
        raise RuntimeError("published batch ownership snapshot drifted")
    shutil.rmtree(directory)


def _workflow(scratch: Path, *, command_runner: Any = None) -> ProductDesktopWorkflow:
    options: dict[str, Any] = {
        "root": ROOT,
        "scratch_root": scratch,
        "python_executable": Path(sys.executable),
        "max_preview_pixels": 3_000,
        "tile_size": 256,
        "tile_workers": 1,
        "png_compression": 6,
    }
    if command_runner is not None:
        options["command_runner"] = command_runner
    return ProductDesktopWorkflow(**options)


def _success_case(case: Path, order: str) -> dict[str, Any]:
    first = case / "B strange SOURCE.png"
    second = case / "a-source.png"
    destination = case / "published-batch"
    scratch = case / "workflow-scratch"
    scratch.mkdir()
    selected = (first, second) if order == "forward" else (second, first)
    child_commands: list[str] = []
    child_python_environment_keys: list[list[str]] = []

    def observed_runner(
        command: Sequence[str], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        child_commands.append(str(command[0]))
        child_python_environment_keys.append(_python_environment_keys(environment))
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    workflow = _workflow(scratch, command_runner=observed_runner)
    replay_paths: list[Path] = []
    try:
        state, bound = workflow.render_batch_previews(selected, 0.625)
        progress: list[list[Any]] = []
        receipt = workflow.export_batch(
            bound,
            "ektar_100",
            destination,
            progress=lambda done, total, name: progress.append([done, total, name]),
        )
        replay: dict[str, bool] = {}
        for row in receipt.receipt["jobs"]:
            recipe = json.loads(
                (destination / row["recipe_path"]).read_text(encoding="utf-8")
            )
            replay_path = scratch / f"replay-{row['output_path']}"
            replay_paths.append(replay_path)
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=ROOT
                / "configs/render_profiles/safe_rich_product_v1.json",
                output_path=replay_path,
                root=ROOT,
            )
            replay[row["output_path"]] = (
                replay_path.read_bytes()
                == (destination / row["output_path"]).read_bytes()
            )
        result = {
            "batch_id": receipt.batch_id,
            "claim": receipt.receipt["claim"],
            "child_argv0": child_commands,
            "child_python_environment_keys": child_python_environment_keys,
            "input_order": [row.basename for row in bound],
            "job_count": receipt.job_count,
            "look_amount": state.look_amount,
            "members": _snapshot(destination),
            "progress": progress,
            "batch_json_sha256": _sha256(destination / "batch.json"),
            "canonical_receipt_identity_sha256": _canonical_sha256(
                receipt.receipt
            ),
            "replay_exact": replay,
            "style_id": receipt.style_id,
            "worker_python": str(Path(sys.executable).resolve(strict=True)),
        }
        if not workflow.close():
            raise RuntimeError("workflow cleanup failed")
        for replay_path in replay_paths:
            replay_path.unlink(missing_ok=True)
        result["scratch_residue_count"] = len(tuple(scratch.iterdir()))
        return result
    finally:
        workflow.close()
        for replay_path in replay_paths:
            replay_path.unlink(missing_ok=True)
        if scratch.is_dir() and not tuple(scratch.iterdir()):
            scratch.rmdir()


def _control_cases(case: Path) -> dict[str, Any]:
    first = case / "B strange SOURCE.png"
    second = case / "a-source.png"
    controls: dict[str, Any] = {}

    child_scratch = case / "child-scratch"
    child_scratch.mkdir()
    child_calls = 0
    child_argv0: list[str] = []
    child_python_keys: list[list[str]] = []

    def fail_second(command: Any, cwd: Path, environment: dict[str, str]) -> Any:
        nonlocal child_calls
        child_calls += 1
        child_argv0.append(str(command[0]))
        child_python_keys.append(_python_environment_keys(environment))
        if child_calls == 2:
            return subprocess.CompletedProcess(command, 19, "", "injected child failure")
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    child = _workflow(child_scratch, command_runner=fail_second)
    child_destination = case / "child-failure-batch"
    try:
        _, bound = child.render_batch_previews((first, second), 0.625)
        try:
            child.export_batch(bound, "ektar_100", child_destination)
            child_rejected = False
        except ProductDesktopError as error:
            child_rejected = "child 2 failed" in str(error)
        controls["child_nonzero"] = {
            "calls": child_calls,
            "child_argv0": child_argv0,
            "child_python_environment_keys": child_python_keys,
            "destination_absent": not os.path.lexists(child_destination),
            "rejected": child_rejected,
            "stage_count": len(tuple(case.glob(".child-failure-batch.u7-11a-*.stage"))),
        }
    finally:
        child.close()
        if child_scratch.is_dir() and not tuple(child_scratch.iterdir()):
            child_scratch.rmdir()

    cancel_scratch = case / "cancel-scratch"
    cancel_scratch.mkdir()
    cancel_argv0: list[str] = []
    cancel_python_keys: list[list[str]] = []

    def observed_cancel(
        command: Sequence[str], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        cancel_argv0.append(str(command[0]))
        cancel_python_keys.append(_python_environment_keys(environment))
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    cancel = _workflow(cancel_scratch, command_runner=observed_cancel)
    cancel_destination = case / "cancel-batch"
    cancel_event = threading.Event()

    def cancel_progress(done: int, _total: int, _name: str) -> None:
        if done == 1:
            cancel_event.set()

    try:
        _, bound = cancel.render_batch_previews((first, second), 0.625)
        try:
            cancel.export_batch(
                bound,
                "ektar_100",
                cancel_destination,
                cancel_event=cancel_event,
                progress=cancel_progress,
            )
            cancel_rejected = False
        except ProductDesktopError as error:
            cancel_rejected = "cancelled safely" in str(error)
        controls["cancel_after_current"] = {
            "destination_absent": not os.path.lexists(cancel_destination),
            "child_argv0": cancel_argv0,
            "child_python_environment_keys": cancel_python_keys,
            "event_set": cancel_event.is_set(),
            "rejected": cancel_rejected,
            "stage_count": len(tuple(case.glob(".cancel-batch.u7-11a-*.stage"))),
        }
    finally:
        cancel.close()
        if cancel_scratch.is_dir() and not tuple(cancel_scratch.iterdir()):
            cancel_scratch.rmdir()

    foreign_scratch = case / "foreign-scratch"
    foreign_scratch.mkdir()
    foreign_destination = case / "late-foreign-batch"
    foreign_calls = 0
    foreign_argv0: list[str] = []
    foreign_python_keys: list[list[str]] = []

    def inject_foreign(command: Any, cwd: Path, environment: dict[str, str]) -> Any:
        nonlocal foreign_calls
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
        foreign_calls += 1
        foreign_argv0.append(str(command[0]))
        foreign_python_keys.append(_python_environment_keys(environment))
        if foreign_calls == 2:
            foreign_destination.mkdir()
            (foreign_destination / "foreign.bin").write_bytes(b"foreign")
        return result

    foreign = _workflow(foreign_scratch, command_runner=inject_foreign)
    try:
        _, bound = foreign.render_batch_previews((first, second), 0.625)
        try:
            foreign.export_batch(bound, "ektar_100", foreign_destination)
            foreign_rejected = False
        except ProductDesktopError as error:
            foreign_rejected = "destination appeared" in str(error)
        marker = foreign_destination / "foreign.bin"
        controls["late_foreign"] = {
            "calls": foreign_calls,
            "child_argv0": foreign_argv0,
            "child_python_environment_keys": foreign_python_keys,
            "marker_exact": marker.is_file() and marker.read_bytes() == b"foreign",
            "member_names": sorted(
                path.name for path in foreign_destination.iterdir()
            )
            if foreign_destination.is_dir()
            else [],
            "rejected": foreign_rejected,
            "stage_count": len(tuple(case.glob(".late-foreign-batch.u7-11a-*.stage"))),
        }
    finally:
        foreign.close()
        if foreign_destination.is_dir():
            shutil.rmtree(foreign_destination)
        if foreign_scratch.is_dir() and not tuple(foreign_scratch.iterdir()):
            foreign_scratch.rmdir()

    controls["owned_residue_count"] = len(
        tuple(case.glob(".*.u7-11a-*.stage"))
    ) + sum(
        1
        for name in ("child-scratch", "cancel-scratch", "foreign-scratch")
        if (case / name).exists()
    )
    return controls


def _worker(case: Path, order: str, action: str, output: Path) -> int:
    if action not in {"success", "controls"}:
        raise ValueError("unknown worker action")
    payload = (
        _success_case(case, order) if action == "success" else _control_cases(case)
    )
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


def _bootstrap(path: Path) -> str:
    source = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from scripts.audit_u7_11b_installed_runtime_desktop_batch_confirmation "
        "import _worker_cli\n"
        "raise SystemExit(_worker_cli())\n"
    )
    path.write_text(source, encoding="utf-8", newline="\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _worker_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--action", choices=("success", "controls"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return _worker(args.case, args.order, args.action, args.output)


def _worker_run(
    python: Path,
    bootstrap: Path,
    case: Path,
    order: str,
    action: str,
    output: Path,
    foreign_cwd: Path,
) -> dict[str, Any]:
    result = _run(
        [
            str(python),
            "-I",
            str(bootstrap),
            "--case",
            str(case),
            "--order",
            order,
            "--action",
            action,
            "--output",
            str(output),
        ],
        cwd=foreign_cwd,
        env=_environment(hostile=True),
        timeout=300,
    )
    _success(result)
    payload = json.loads(output.read_text(encoding="utf-8"))
    output.unlink()
    return payload


def _artifact_projection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "child_argv0",
            "child_python_environment_keys",
            "worker_python",
        }
    }


def _desktop_smoke(
    launcher: Path, source: Path, scratch: Path, foreign_cwd: Path
) -> dict[str, Any]:
    before = u710b._window_handles()
    environment = _environment(hostile=True)
    result = u710b._cmd(
        launcher,
        [
            "--input",
            str(source),
            "--scratch-root",
            str(scratch),
            "--smoke-exit-ms",
            "200",
        ],
        cwd=foreign_cwd,
        env=environment,
    )
    after = u710b._window_handles()
    return {
        "returncode": result.returncode,
        "stderr_empty": not result.stderr,
        "stdout_empty": not result.stdout,
        "window_count_before": len(before),
        "window_count_after": len(after),
        "scratch_residue_count": len(tuple(scratch.iterdir())),
    }


def _source_bindings(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for role, row in config["parent_bindings"].items():
        path = ROOT / row["path"]
        bindings[role] = {
            "path": row["path"],
            "actual_sha256": _sha256(path),
            "expected_sha256": row["sha256"],
            "exact": _sha256(path) == row["sha256"],
        }
    return bindings


def build_report(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    if not _tracked_clean():
        raise RuntimeError("formal execution requires a tracked-clean worktree")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_commit = _git_head()
    source_paths = (
        CONFIG,
        CONTRACT,
        AUDIT,
        AUDIT_TEST,
        INSTALLER,
        REQUIREMENTS,
        DESKTOP_ENTRY,
        BATCH_CORE,
        BATCH_UI,
        U711A_EVIDENCE,
        U710B_EVIDENCE,
        U79C_EVIDENCE,
    )
    source_before = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    source_git = u710b._source_git_objects(source_commit, source_paths)
    parent_bindings = _source_bindings(config)
    parent_u710b = json.loads(U710B_EVIDENCE.read_text(encoding="utf-8"))
    parent_u79c = json.loads(U79C_EVIDENCE.read_text(encoding="utf-8"))
    wheel_rows = u710b._wheel_rows(WHEELHOUSE.resolve(strict=True))
    formal_root = SCRATCH_PARENT / "u7-11b-formal-fixed"
    if os.path.lexists(formal_root):
        raise FileExistsError(f"formal root must be absent: {formal_root}")
    formal_root.mkdir()
    root_identity = installer._entry_identity(formal_root)
    try:
        process_temp = formal_root / "process-temp"
        process_temp.mkdir()
        environment = _environment()
        environment.update(
            {
                "PIP_CACHE_DIR": str(formal_root / "pip-cache"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TEMP": str(process_temp),
                "TMP": str(process_temp),
            }
        )
        (formal_root / "pip-cache").mkdir()
        installation = formal_root / "installation"
        install_commands: list[list[str]] = []

        def observed_installer_runner(
            command: Sequence[str], cwd: Path | None, env: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            install_commands.append([str(part) for part in command])
            return installer._run(command, cwd, env)

        receipt = installer.install_product_runtime(
            installation,
            wheelhouse=WHEELHOUSE,
            runner=observed_installer_runner,
            environment=environment,
        )
        runtime_python = installation / "runtime/Scripts/python.exe"
        receipt_path = installation / "product-runtime.json"
        pip_check = _run(
            [str(runtime_python), "-m", "pip", "check"],
            cwd=formal_root,
            env=environment,
        )
        launcher_bindings = {
            role: {
                "command_exact": _sha256(Path(row["command"]))
                == row["command_sha256"],
                "python_exact": _sha256(Path(row["python"]))
                == row["python_sha256"],
            }
            for role, row in receipt["launchers"].items()
        }
        foreign_cwd = formal_root / "foreign-cwd"
        foreign_cwd.mkdir()
        case = formal_root / "case"
        case.mkdir()
        _tiny_input(case / "B strange SOURCE.png", 1)
        _tiny_input(case / "a-source.png", 2)
        bootstrap = formal_root / "batch-worker.py"
        bootstrap_sha256 = _bootstrap(bootstrap)
        desktop_scratch = formal_root / "desktop-scratch"
        desktop_scratch.mkdir()
        desktop = _desktop_smoke(
            Path(receipt["launchers"]["desktop"]["command"]),
            case / "a-source.png",
            desktop_scratch,
            foreign_cwd,
        )
        orders = [order, "reverse" if order == "forward" else "forward"]
        cases: list[dict[str, Any]] = []
        for selection in orders:
            direct_output = formal_root / f"direct-{selection}.json"
            direct = _worker_run(
                Path(sys.executable),
                bootstrap,
                case,
                selection,
                "success",
                direct_output,
                foreign_cwd,
            )
            destination = case / "published-batch"
            _remove_owned_snapshot_directory(destination, direct["members"])
            installed_output = formal_root / f"installed-{selection}.json"
            installed = _worker_run(
                runtime_python,
                bootstrap,
                case,
                selection,
                "success",
                installed_output,
                foreign_cwd,
            )
            cases.append(
                {
                    "canonical_order_exact": direct["input_order"]
                    == installed["input_order"],
                    "canonical_receipt_identity_exact": direct[
                        "canonical_receipt_identity_sha256"
                    ]
                    == installed["canonical_receipt_identity_sha256"],
                    "direct_child_argv0": direct["child_argv0"],
                    "direct_installed_artifacts_exact": _artifact_projection(direct)
                    == _artifact_projection(installed),
                    "png16_exact": _member_subset(direct["members"], ".png")
                    == _member_subset(installed["members"], ".png"),
                    "raw_batch_json_exact": direct["batch_json_sha256"]
                    == installed["batch_json_sha256"],
                    "raw_recipes_exact": _member_subset(
                        direct["members"], ".recipe.json"
                    )
                    == _member_subset(installed["members"], ".recipe.json"),
                    "installed": installed,
                    "selection": selection,
                }
            )
            _remove_owned_snapshot_directory(destination, installed["members"])
        cases.sort(key=lambda row: row["selection"])
        controls = _worker_run(
            runtime_python,
            bootstrap,
            case,
            "forward",
            "controls",
            formal_root / "controls.json",
            foreign_cwd,
        )
        bootstrap_after_sha256 = _sha256(bootstrap)
        bootstrap.unlink()
        (case / "B strange SOURCE.png").unlink()
        (case / "a-source.png").unlink()
        case.rmdir()
        scientific = {
            "batch_cases": cases,
            "claim_ceiling": config["claim_ceiling"],
            "controls": controls,
            "desktop_smoke": desktop,
            "runtime": {
                "distributions": receipt["distributions"],
                "launcher_bindings": launcher_bindings,
                "python": receipt["python"],
                "receipt_sha256": _sha256(receipt_path),
                "repository_bound": receipt["repository_bound"],
                "source_commit": receipt["source_commit"],
            },
            "worker": {
                "bootstrap_after_sha256": bootstrap_after_sha256,
                "bootstrap_before_sha256": bootstrap_sha256,
                "installer_commands": install_commands,
            },
        }
        pip_install_commands = [
            command
            for command in install_commands
            if "-m" in command and "pip" in command and "install" in command
        ]
        network_locator_arguments = [
            argument
            for command in install_commands
            for argument in command
            if argument.casefold().startswith(("http://", "https://", "ftp://"))
        ]
        local_install_transport_exact = (
            len(pip_install_commands) == 1
            and "--no-index" in pip_install_commands[0]
            and "--find-links" in pip_install_commands[0]
            and pip_install_commands[0][
                pip_install_commands[0].index("--find-links") + 1
            ]
            == str(WHEELHOUSE.resolve(strict=True))
            and network_locator_arguments == []
        )
        scientific["worker"]["network_locator_arguments"] = (
            network_locator_arguments
        )
        scientific["worker"]["local_install_transport_exact"] = (
            local_install_transport_exact
        )
        child = controls["child_nonzero"]
        cancel = controls["cancel_after_current"]
        foreign = controls["late_foreign"]
        installed_rows = [row["installed"] for row in cases]
        gates = {
            "claim_ceiling_exact": all(
                row["claim"]
                == {
                    "calibrated_stock_response": False,
                    "evidence_grade": "look-approximation",
                    "output_label": "film-inspired / Look Approximation",
                    "physical_film_reproduction": False,
                    "stock_distinguishability": False,
                }
                for row in installed_rows
            ),
            "direct_installed_canonical_order_exact": all(
                row["canonical_order_exact"] for row in cases
            )
            and len({tuple(row["installed"]["input_order"]) for row in cases}) == 1,
            "direct_installed_png16_exact": all(
                row["png16_exact"] for row in cases
            ),
            "direct_installed_raw_recipes_exact": all(
                row["raw_recipes_exact"] for row in cases
            ),
            "direct_installed_raw_batch_receipt_exact": all(
                row["raw_batch_json_exact"]
                and row["canonical_receipt_identity_exact"]
                for row in cases
            ),
            "exact_member_counts": all(
                row["installed"]["job_count"] == 2
                and len(row["installed"]["members"]) == 5
                for row in cases
            ),
            "fresh_install_and_wheelhouse_exact": receipt["schema"]
            == "kmcfm.private-product-runtime-receipt.v2"
            and parent_u710b["gates"]["wheelhouse_exact"] is True
            and len(wheel_rows) == config["runtime"]["wheel_count"]
            and sum(row["bytes"] for row in wheel_rows)
            == config["runtime"]["wheel_bytes"]
            and _canonical_sha256(wheel_rows)
            == config["runtime"]["wheel_inventory_sha256"],
            "installed_controls_fail_closed": child["rejected"]
            and child["destination_absent"]
            and child["stage_count"] == 0
            and cancel["rejected"]
            and cancel["destination_absent"]
            and cancel["event_set"]
            and cancel["stage_count"] == 0,
            "installed_desktop_mainloop_foreign_cwd": desktop["returncode"] == 0
            and desktop["stderr_empty"]
            and desktop["window_count_before"] == 0
            and desktop["window_count_after"] == 0
            and desktop["scratch_residue_count"] == 0,
            "installed_strict_replay_exact": all(
                all(row["replay_exact"].values()) for row in installed_rows
            ),
            "installed_child_python_exact": all(
                Path(row["worker_python"]).resolve(strict=True) == runtime_python
                and len(row["child_argv0"]) == 2
                and all(
                    Path(command).resolve(strict=True) == runtime_python
                    for command in row["child_argv0"]
                )
                for row in installed_rows
            )
            and all(
                all(
                    Path(command).resolve(strict=True) == runtime_python
                    for command in control["child_argv0"]
                )
                for control in (child, cancel, foreign)
            ),
            "late_foreign_destination_preserved": foreign["rejected"]
            and foreign["marker_exact"]
            and foreign["member_names"] == ["foreign.bin"]
            and foreign["stage_count"] == 0,
            "parent_bindings_exact": all(
                row["exact"] for row in parent_bindings.values()
            ),
            "python_environment_removed": parent_u710b["gates"]
            ["python_environment_removed"]
            and all(
                parent_u79c["formal_report"]["gates"][name]
                for name in (
                    "python_environment_removed",
                    "launcher_process_isolated",
                    "renderer_process_isolated",
                )
            )
            and all(
                keys == []
                for row in installed_rows
                for keys in row["child_python_environment_keys"]
            )
            and all(
                keys == []
                for control in (child, cancel, foreign)
                for keys in control["child_python_environment_keys"]
            ),
            "local_wheelhouse_network_locator_absent": local_install_transport_exact,
            "runtime_receipt_and_pip_check_exact": receipt["source_commit"]
            == source_commit
            and receipt["requirements"]["sha256"] == _sha256(REQUIREMENTS)
            and receipt["distributions"] == installer._pinned_versions(REQUIREMENTS)
            and receipt["repository_bound"] is True
            and receipt["public_distribution"] is False
            and all(
                row["command_exact"] and row["python_exact"]
                for row in launcher_bindings.values()
            )
            and pip_check.returncode == 0
            and "No broken requirements found" in pip_check.stdout,
            "source_git_objects_exact": all(row["exact"] for row in source_git.values()),
            "worker_and_control_residue_zero": controls["owned_residue_count"] == 0,
            "worker_bytes_exact": bootstrap_sha256 == bootstrap_after_sha256,
        }
        scientific["gates"] = gates
        scientific_identity = _canonical_sha256(scientific)
        source_after = {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        }
        source_stable = (
            source_before == source_after
            and _git_head() == source_commit
            and _tracked_clean()
        )
        report = {
            "schema": "kmcfm.u7-11b-installed-runtime-desktop-batch-confirmation-result.v1",
            "status": (
                "PASS_PRIVATE_U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION"
                if all(gates.values()) and source_stable
                else "FAIL_CLOSED_U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION"
            ),
            "automatic_pass": all(gates.values()) and source_stable,
            "bootstrap_sha256": bootstrap_sha256,
            "claim": (
                "Private Windows repository-bound installed-runtime batch-core "
                "composition for deterministic film-inspired / Look Approximation "
                "output only; no native dialog automation, calibrated stock response, "
                "physical-film reproduction, arbitrary media, public installer, or "
                "cross-platform GUI claim."
            ),
            "parent_bindings": parent_bindings,
            "scientific": scientific,
            "scientific_identity_sha256": scientific_identity,
            "source": source_before,
            "source_commit": source_commit,
            "source_git_objects": source_git,
            "source_stable": source_stable,
            "wheelhouse": wheel_rows,
        }
    finally:
        if not installer._cleanup_owned_directory(formal_root, root_identity):
            raise RuntimeError("formal root ownership changed; preserved")
    report["formal_root_residue_count"] = int(formal_root.exists())
    post_cleanup_exact = (
        not formal_root.exists()
        and _git_head() == report["source_commit"]
        and _tracked_clean()
        and {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        }
        == report["source"]
    )
    report["scientific"]["gates"][
        "source_and_owned_residue_exact"
    ] = post_cleanup_exact
    report["scientific_identity_sha256"] = _canonical_sha256(report["scientific"])
    report["automatic_pass"] = all(report["scientific"]["gates"].values())
    if not report["automatic_pass"]:
        report["status"] = (
            "FAIL_CLOSED_U7_11B_INSTALLED_RUNTIME_DESKTOP_BATCH_CONFIRMATION"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", type=Path)
    parser.add_argument("--action", choices=("success", "controls"))
    args = parser.parse_args()
    if args.worker:
        if args.case is None or args.order is None or args.action is None or args.output is None:
            raise ValueError("worker arguments are incomplete")
        return _worker(args.case, args.order, args.action, args.output)
    if args.order is None or args.output is None:
        raise ValueError("formal order and output are required")
    output = args.output.resolve(strict=False)
    output_root = (ROOT / "outputs").resolve(strict=True)
    if output_root not in output.parents:
        raise ValueError("formal report must remain under repo-relative outputs")
    if os.path.lexists(output):
        raise FileExistsError(output)
    report = build_report(args.order)
    atomic_write_json(output, report)
    return 0 if report["automatic_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
