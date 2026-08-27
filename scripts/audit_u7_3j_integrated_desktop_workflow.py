#!/usr/bin/env python3
"""Audit three real Edge selections through the integrated desktop workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_u7_3i_local_browser_recipe_export import (
    EDGE,
    _edge_version,
    _stop_process_tree,
    _wait_devtools,
)
from src.inference.recipe_desktop_workflow import IntegratedRecipeDesktopSession
from src.inference.recipe_export_request import build_recipe_export_request_set

CONFIG = ROOT / "configs/u7_3j_integrated_desktop_workflow_v1.json"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _submit_form_and_collect(
    websocket_url: str,
    request_file: str,
    timeout: float,
) -> dict[str, Any]:
    source = r"""
const ws = new WebSocket(process.argv[1]);
const wanted = process.argv[2];
const timer = setTimeout(() => { console.error('cdp timeout'); process.exit(2); }, Number(process.argv[3]) * 1000);
let id = 0;
function probe(){
  id += 1;
  const expression = `(() => {
    const cards=[...document.querySelectorAll('.look-card')];
    const forms=[...document.querySelectorAll('form')];
    const target=forms.find(form => form.querySelector('input[name="request_file"]')?.value === ${JSON.stringify(wanted)});
    if(!target || cards.length !== 3 || !cards.every(card => card.querySelector('img')?.complete)) return {state:'waiting'};
    const remote=[...performance.getEntriesByType('resource')].map(entry=>entry.name).filter(name => { try { const url=new URL(name); return (url.protocol==='http:' || url.protocol==='https:') && url.hostname!=='127.0.0.1'; } catch { return false; } });
    const facts={state:'submitted',cards:cards.length,forms:forms.length,images:document.querySelectorAll('img').length,buttons:document.querySelectorAll('button[type="submit"]').length,distinctHeadings:new Set([...document.querySelectorAll('.look-card h2')].map(node=>node.textContent)).size,allImagesDecoded:cards.every(card=>card.querySelector('img').naturalWidth>0),liveRegions:document.querySelectorAll('[aria-live="polite"]').length,remoteResources:remote};
    target.requestSubmit();
    return facts;
  })()`;
  ws.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,returnByValue:true}}));
}
ws.onopen = probe;
ws.onmessage = event => {
  const value=JSON.parse(event.data);
  if(value.id!==id) return;
  if(value.error || value.result?.exceptionDetails){ clearTimeout(timer); console.error(JSON.stringify(value)); process.exit(3); }
  const result=value.result?.result?.value;
  if(result?.state==='submitted'){ clearTimeout(timer); console.log(JSON.stringify(result)); ws.close(); }
  else setTimeout(probe,100);
};
ws.onerror = () => { clearTimeout(timer); process.exit(4); };
"""
    completed = subprocess.run(
        ["node", "-e", source, websocket_url, request_file, str(timeout)],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout + 5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Edge form submission failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _post(url: str, fields: dict[str, str]) -> int:
    request = Request(
        url,
        data=urlencode(fields).encode("ascii"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
        return 200
    except HTTPError as exc:
        exc.read()
        return int(exc.code)


def _run_selection(
    config: dict[str, Any],
    row: dict[str, str],
    request_file: str,
    scratch: Path,
    viewport: list[int],
) -> dict[str, Any]:
    limits = config["execution"]
    history_root = ROOT / config["history_root"]
    output_root = scratch / "exports"
    profile = scratch / "edge-profile"
    with IntegratedRecipeDesktopSession(
        history_root,
        output_root,
        profile_path=ROOT / config["profile_path"],
        root=ROOT,
        tile_size=limits["tile_size"],
        maximum_recipe_files=limits["maximum_recipe_files"],
        maximum_recipe_bytes=limits["maximum_recipe_bytes"],
        maximum_request_bytes=limits["maximum_request_bytes"],
        maximum_form_bytes=limits["maximum_form_bytes"],
    ) as session:
        with urlopen(session.url, timeout=30) as response:
            page = response.read()
        text = page.decode("utf-8")
        path_disclosure = str(ROOT) in text or "file:" in text or "P:\\" in text
        preview_payloads = re.findall(r'src="data:image/png;base64,([^"]+)"', text)
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
                f"--window-size={viewport[0]},{viewport[1]}",
                session.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _, websocket_url = _wait_devtools(
                profile, session.url, config["browser"]["timeout_seconds"]
            )
            dom = _submit_form_and_collect(
                websocket_url,
                request_file,
                config["browser"]["timeout_seconds"],
            )
            result = session.wait(config["browser"]["timeout_seconds"])
            if result is None:
                raise RuntimeError("integrated browser export timed out")
            token = urlsplit(session.url).query.removeprefix("token=")
            second = _post(
                f"http://127.0.0.1:{session.port}/export",
                {"token": token, "request_file": request_file},
            )
        finally:
            _stop_process_tree(process)
    output = result.output_path
    return {
        "style": result.style,
        "request_file": result.request_file,
        "recipe_path": row["recipe_path"],
        "recipe_sha256": row["recipe_sha256"],
        "output_name": output.name,
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "expected_output_sha256": row["expected_output_sha256"],
        "page_bytes": len(page),
        "page_path_disclosure": path_disclosure,
        "preview_count": len(preview_payloads),
        "distinct_preview_payload_count": len(set(preview_payloads)),
        "viewport": viewport,
        "dom": dom,
        "second_submission_http_status": second,
        "output_count_after_second_submission": len(list(output_root.iterdir())),
    }


def _controller(config: dict[str, Any], reverse: bool) -> dict[str, Any]:
    history_root = ROOT / config["history_root"]
    request_set = build_recipe_export_request_set(history_root)
    request_by_style = {
        str(row["style"]): str(row["request_file"])
        for row in request_set["receipt"]["requests"]
    }
    rows = list(config["rows"])
    if reverse:
        rows.reverse()
    with tempfile.TemporaryDirectory(prefix="neuro-film-u7-3j-") as temporary:
        root = Path(temporary)
        results = []
        for row in rows:
            viewport = (
                config["browser"]["narrow_viewport"]
                if row["style"] == "portra_400"
                else config["browser"]["desktop_viewport"]
            )
            results.append(
                _run_selection(
                    config,
                    row,
                    request_by_style[row["style"]],
                    root / row["style"],
                    viewport,
                )
            )
        residue = sum(1 for _ in root.rglob("*"))
    return {
        "rows": sorted(results, key=lambda item: item["style"]),
        "owned_residue_count": 0 if not Path(temporary).exists() else residue,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = _controller(config, False)
    second = _controller(config, True)
    rows = first["rows"]
    expected_styles = sorted(config["expected_styles"])
    gates = {
        "valid_recipe_count_exact": len(rows),
        "preview_count_exact": min(row["preview_count"] for row in rows),
        "distinct_style_count_exact": len({row["style"] for row in rows}),
        "distinct_preview_sha256_count_exact": min(
            row["distinct_preview_payload_count"] for row in rows
        ),
        "all_selected_outputs_exact": all(
            row["output_sha256"] == row["expected_output_sha256"] for row in rows
        ),
        "machine_local_paths_exposed": any(row["page_path_disclosure"] for row in rows),
        "non_loopback_network_requests": sum(
            len(row["dom"]["remoteResources"]) for row in rows
        ),
        "all_controls_labelled_and_operable": all(
            row["dom"]["cards"] == 3
            and row["dom"]["forms"] == 3
            and row["dom"]["buttons"] == 3
            and row["dom"]["distinctHeadings"] == 3
            and row["dom"]["liveRegions"] == 1
            for row in rows
        ),
        "repeat_submission_rejected": all(
            row["second_submission_http_status"] == 400
            and row["output_count_after_second_submission"] == 1
            for row in rows
        ),
        "owned_residue_count": first["owned_residue_count"] + second["owned_residue_count"],
        "forward_reverse_exact": first == second,
    }
    expected = config["gates"]
    passed = (
        sorted(row["style"] for row in rows) == expected_styles
        and gates["valid_recipe_count_exact"] == expected["valid_recipe_count_exact"]
        and gates["preview_count_exact"] == expected["preview_count_exact"]
        and gates["distinct_style_count_exact"] == expected["distinct_style_count_exact"]
        and gates["distinct_preview_sha256_count_exact"]
        == expected["distinct_preview_sha256_count_exact"]
        and gates["all_selected_outputs_exact"] == expected["all_selected_outputs_exact"]
        and gates["machine_local_paths_exposed"] == expected["machine_local_paths_exposed"]
        and gates["non_loopback_network_requests"] == expected["non_loopback_network_requests"]
        and gates["owned_residue_count"] == expected["owned_residue_count"]
        and gates["all_controls_labelled_and_operable"]
        and gates["repeat_submission_rejected"]
        and gates["forward_reverse_exact"]
    )
    scientific = {
        "protocol": config["schema"],
        "contract_sha256": _sha256(CONFIG),
        "edge_version": _edge_version(),
        "status": "PASS_PRIVATE_INTEGRATED_DESKTOP_WORKFLOW" if passed else "FAIL_CLOSED",
        "gates": gates,
        "rows": rows,
        "claim_ceiling": config["claim_ceiling"],
    }
    report = {
        "schema": "kmcfm.u7-3j-integrated-desktop-workflow-result.v1",
        "scientific": scientific,
        "stable_identity": f"sha256:{hashlib.sha256(_canonical(scientific)).hexdigest()}",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(_canonical(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
