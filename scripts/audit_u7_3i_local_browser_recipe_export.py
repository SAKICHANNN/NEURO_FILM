#!/usr/bin/env python3
"""Audit one real Edge submission into the U7.3I loopback export bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_browser_export import (
    LOOPBACK_HOST,
    RecipeBrowserExportSession,
)

CONFIG = ROOT / "configs/u7_3i_local_browser_recipe_export_v1.json"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_version() -> str:
    completed = subprocess.run(
        [str(EDGE), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("Edge version query failed")
    return completed.stdout.strip()


def _wait_devtools(profile: Path, timeout: float) -> tuple[int, str]:
    deadline = time.monotonic() + timeout
    active = profile / "DevToolsActivePort"
    while time.monotonic() < deadline:
        if active.is_file():
            lines = active.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].isdigit():
                port = int(lines[0])
                try:
                    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
                        targets = json.loads(response.read())
                except (OSError, URLError, json.JSONDecodeError):
                    time.sleep(0.05)
                    continue
                pages = [target for target in targets if target.get("type") == "page"]
                if pages:
                    return port, str(pages[0]["webSocketDebuggerUrl"])
        time.sleep(0.05)
    raise RuntimeError("Edge DevTools endpoint did not become ready")


def _submit_first_form(websocket_url: str, timeout: float) -> None:
    source = r"""
const ws = new WebSocket(process.argv[1]);
const timer = setTimeout(() => { console.error('cdp timeout'); process.exit(2); }, Number(process.argv[2]) * 1000);
ws.onopen = () => ws.send(JSON.stringify({id:1,method:'Runtime.evaluate',params:{expression:"(() => { const f=document.querySelector('form'); if(!f) throw new Error('form missing'); f.requestSubmit(); return 'submitted'; })()",returnByValue:true}}));
ws.onmessage = (event) => { const value=JSON.parse(event.data); if(value.id===1){ clearTimeout(timer); if(value.error || value.result?.exceptionDetails){ console.error(JSON.stringify(value)); process.exit(3); } console.log('submitted'); ws.close(); }};
ws.onerror = () => { clearTimeout(timer); process.exit(4); };
"""
    completed = subprocess.run(
        ["node", "-e", source, websocket_url, str(timeout)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout + 5,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "submitted":
        raise RuntimeError(f"Edge form submission failed: {completed.stderr.strip()}")


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for item in reversed(children):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([*children, parent], timeout=5)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def _run_browser_export(config: dict[str, Any], scratch: Path) -> dict[str, Any]:
    output_root = scratch / "exports"
    profile = scratch / "edge-profile"
    limits = config["execution"]
    with RecipeBrowserExportSession(
        ROOT / config["history_root"],
        output_root,
        profile_path=ROOT / config["profile_path"],
        root=ROOT,
        tile_size=limits["tile_size"],
        maximum_recipe_files=limits["maximum_recipe_files"],
        maximum_recipe_bytes=limits["maximum_recipe_bytes"],
        maximum_request_bytes=limits["maximum_request_bytes"],
        maximum_form_bytes=limits["maximum_form_bytes"],
    ) as session:
        if session.request_files[0] != "request-b92a75ed8ca8349df562153aad85df26db33ab0992d6bc07503033245ac7d0b8.json":
            raise RuntimeError("the first browser form is not the frozen Ektar row")
        process = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-component-update",
                "--no-first-run",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                "--window-size=1200,900",
                session.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _, websocket_url = _wait_devtools(
                profile, limits["browser_timeout_seconds"]
            )
            _submit_first_form(websocket_url, limits["browser_timeout_seconds"])
            result = session.wait(limits["session_timeout_seconds"])
            if result is None:
                raise RuntimeError("browser export session timed out")
            token = urlsplit(session.url).query.removeprefix("token=")
            repeated = Request(
                f"http://{LOOPBACK_HOST}:{session.port}/export",
                data=urlencode(
                    {"token": token, "request_file": result.request_file}
                ).encode("ascii"),
                method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            second_status = 0
            try:
                urlopen(repeated, timeout=10)
            except HTTPError as exc:
                second_status = exc.code
            output_count = len(list(output_root.iterdir()))
            return {
                "browser_submitted": True,
                "style": result.style,
                "request_file": result.request_file,
                "recipe_path": result.receipt["recipe_path"],
                "recipe_sha256": result.receipt["recipe_sha256"],
                "output_name": result.output_path.name,
                "output_bytes": result.output_path.stat().st_size,
                "output_sha256": result.output_sha256,
                "second_submission_http_status": second_status,
                "output_count_after_second_submission": output_count,
                "server_host": LOOPBACK_HOST,
            }
        finally:
            _stop_process_tree(process)


def run() -> dict[str, Any]:
    config_payload = CONFIG.read_bytes()
    config = json.loads(config_payload)
    if not EDGE.is_file():
        raise RuntimeError("Edge executable is unavailable")
    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="u7_3i_", dir=tmp_root) as raw:
        scratch = Path(raw)
        first = _run_browser_export(config, scratch / "first")
        second = _run_browser_export(config, scratch / "second")
    frozen = config["selected_row"]
    expected_request = f"request-{frozen['recipe_sha256']}.json"
    gates = {
        "real_edge_process_submits_the_export_form": first["browser_submitted"]
        and second["browser_submitted"],
        "server_binds_loopback_only": first["server_host"] == LOOPBACK_HOST
        and second["server_host"] == LOOPBACK_HOST,
        "one_success_ends_the_session": first["second_submission_http_status"] == 400
        and second["second_submission_http_status"] == 400
        and first["output_count_after_second_submission"] == 1
        and second["output_count_after_second_submission"] == 1,
        "browser_supplies_no_input_output_or_profile_path": first["request_file"]
        == expected_request
        and second["request_file"] == expected_request,
        "request_and_recipe_identity_match_frozen_row": all(
            row["style"] == frozen["style"]
            and row["recipe_path"] == frozen["recipe_path"]
            and row["recipe_sha256"] == frozen["recipe_sha256"]
            for row in (first, second)
        ),
        "published_output_sha256_matches_frozen_u7_3d": first["output_sha256"]
        == frozen["expected_output_sha256"]
        and second["output_sha256"] == frozen["expected_output_sha256"],
        "closed_or_invalid_session_publishes_nothing": first["output_count_after_second_submission"]
        == 1
        and second["output_count_after_second_submission"] == 1,
        "owned_browser_profile_and_server_scratch_are_removed": not Path(raw).exists(),
    }
    scientific = {
        "protocol": config["schema"],
        "status": (
            "PASS_PRIVATE_LOCAL_BROWSER_RECIPE_EXPORT"
            if all(gates.values())
            else "FAIL_CLOSED_LOCAL_BROWSER_RECIPE_EXPORT"
        ),
        "contract_sha256": hashlib.sha256(config_payload).hexdigest(),
        "edge_version": _edge_version(),
        "runs": [first, second],
        "runs_output_exact": first["output_sha256"] == second["output_sha256"],
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }
    return {
        "schema": "kmcfm.u7-3i-local-browser-recipe-export-result.v1",
        "scientific": scientific,
        "stable_identity": f"sha256:{hashlib.sha256(_canonical(scientific)).hexdigest()}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if report["scientific"]["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
