#!/usr/bin/env python3
"""Audit the deterministic offline recipe workspace in fresh Edge profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_workspace import materialize_offline_recipe_workspace

CONTRACT_PATH = ROOT / "configs/u7_3e_offline_recipe_workspace_v1.json"
EDGE_PATH = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _edge_dom(index: Path, *, width: int, profile: Path) -> tuple[int, str]:
    completed = subprocess.run(
        [
            str(EDGE_PATH),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-component-update",
            "--no-first-run",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            f"--user-data-dir={profile}",
            f"--window-size={width},900",
            "--dump-dom",
            index.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    return completed.returncode, completed.stdout


def run_audit(*, reverse: bool = False) -> dict[str, Any]:
    contract_payload = CONTRACT_PATH.read_bytes()
    contract = json.loads(contract_payload)
    source_root = ROOT / contract["history_root"]
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.glob("*.recipe.json"), reverse=reverse)
    }
    with tempfile.TemporaryDirectory(prefix="u7_3e_workspace_") as temporary:
        temporary_root = Path(temporary)
        workspace = temporary_root / "workspace"
        receipt = materialize_offline_recipe_workspace(
            source_root,
            workspace,
            maximum_recipe_files=contract["limits"]["maximum_recipe_files"],
            maximum_recipe_bytes=contract["limits"]["maximum_recipe_bytes"],
        )
        file_facts = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(workspace.iterdir(), reverse=reverse)
        }
        browser: dict[str, dict[str, Any]] = {}
        for width in ((500, 1440) if reverse else (1440, 500)):
            returncode, dom = _edge_dom(
                workspace / "index.html",
                width=width,
                profile=temporary_root / f"edge-{width}",
            )
            browser[str(width)] = {
                "returncode": returncode,
                "dom_has_main": "<main>" in dom,
                "dom_has_heading": "Film workspace" in dom,
                "dom_has_history_link": 'href="history.html"' in dom,
                "dom_has_preview_link": 'href="previews.html"' in dom,
            }
    owned_temporary_roots_after_cleanup = [
        str(path)
        for path in Path(tempfile.gettempdir()).glob("u7_3e_workspace_*")
        if path.exists()
    ]
    gates = {
        "contract_source_recipes_rehash": source_hashes
        == contract["source_recipe_sha256"],
        "exact_workspace_inventory": set(file_facts)
        == set(contract["workspace_files"]),
        "three_valid_rows": receipt["catalog_counts"]
        == {"discovered": 3, "valid": 3, "invalid": 0},
        "page_hashes_match_receipt": all(
            file_facts[name] == receipt["pages"][name]
            for name in ("index.html", "history.html", "previews.html")
        ),
        "desktop_edge_dom_load": browser["1440"]["returncode"] == 0
        and all(
            value
            for key, value in browser["1440"].items()
            if key != "returncode"
        ),
        "narrow_edge_dom_load": browser["500"]["returncode"] == 0
        and all(
            value
            for key, value in browser["500"].items()
            if key != "returncode"
        ),
        "workspace_contains_no_input_files": all(
            not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".raw", ".dng"))
            for name in file_facts
        ),
        "owned_temporary_roots_removed": owned_temporary_roots_after_cleanup == [],
    }
    status = (
        "PASS_PRIVATE_OFFLINE_RECIPE_WORKSPACE"
        if all(gates.values())
        else "FAIL_CLOSED_OFFLINE_RECIPE_WORKSPACE"
    )
    scientific = {
        "protocol": contract["schema"],
        "status": status,
        "contract_sha256": _sha256(contract_payload),
        "source_recipe_sha256": source_hashes,
        "workspace_files": file_facts,
        "browser": browser,
        "gates": gates,
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": "kmcfm.u7-3e-offline-recipe-workspace-result.v1",
        "scientific": scientific,
        "stable_identity": f"sha256:{_sha256(_canonical_bytes(scientific))}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = run_audit(reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(json.dumps(report["scientific"]["gates"], sort_keys=True))
    return 0 if report["scientific"]["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
