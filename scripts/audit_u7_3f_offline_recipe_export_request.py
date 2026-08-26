#!/usr/bin/env python3
"""Run one formal U7.3F portable export-request replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.recipe_export_request import (
    build_recipe_export_request_set,
    export_recipe_request,
    materialize_recipe_export_request_set,
)

CONFIG = ROOT / "configs/u7_3f_offline_recipe_export_request_v1.json"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_contract() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    for relative, expected in value["implementation_sha256"].items():
        if _file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"implementation hash drift: {relative}")
    return value


def run(order_name: str) -> dict[str, Any]:
    contract = _load_contract()
    order_index = 0 if order_name == "forward" else 1
    order = contract["execution"]["orders"][order_index]
    rows = {row["style"]: row for row in contract["rows"]}
    if set(order) != set(rows):
        raise RuntimeError("formal order does not cover the frozen rows")
    history_root = ROOT / contract["history_root"]
    profile_path = ROOT / contract["profile_path"]
    limits = contract["execution"]

    in_memory_a = build_recipe_export_request_set(
        history_root,
        maximum_recipe_files=limits["maximum_recipe_files"],
        maximum_recipe_bytes=limits["maximum_recipe_bytes"],
    )
    in_memory_b = build_recipe_export_request_set(
        history_root,
        maximum_recipe_files=limits["maximum_recipe_files"],
        maximum_recipe_bytes=limits["maximum_recipe_bytes"],
    )
    request_set_exact = in_memory_a == in_memory_b
    row_by_style = {
        row["style"]: row for row in in_memory_a["receipt"]["requests"]
    }

    tmp_parent = ROOT / "tmp"
    tmp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"u7_3f_{order_name}_", dir=tmp_parent) as raw:
        scratch = Path(raw)
        request_root = scratch / "requests"
        materialized = materialize_recipe_export_request_set(
            history_root,
            request_root,
            maximum_recipe_files=limits["maximum_recipe_files"],
            maximum_recipe_bytes=limits["maximum_recipe_bytes"],
        )
        executions: list[dict[str, Any]] = []
        for style in order:
            frozen = rows[style]
            request_row = row_by_style[style]
            request_path = request_root / request_row["request_file"]
            output_path = scratch / f"{style}.png"
            receipt = export_recipe_request(
                history_root,
                request_path,
                profile_path=profile_path,
                output_path=output_path,
                root=ROOT,
                maximum_recipe_files=limits["maximum_recipe_files"],
                maximum_recipe_bytes=limits["maximum_recipe_bytes"],
                maximum_request_bytes=limits["maximum_request_bytes"],
                tile_size=limits["tile_size"],
            )
            output_sha256 = _file_sha256(output_path)
            executions.append(
                {
                    "style": style,
                    "recipe_path": receipt["recipe_path"],
                    "recipe_sha256": receipt["recipe_sha256"],
                    "request_file": request_row["request_file"],
                    "request_sha256": receipt["request_sha256"],
                    "output_sha256": output_sha256,
                    "expected_output_sha256": frozen["expected_output_sha256"],
                    "recipe_identity_pass": receipt["recipe_sha256"]
                    == frozen["recipe_sha256"],
                    "output_identity_pass": output_sha256
                    == frozen["expected_output_sha256"],
                }
            )
        request_payloads = [
            payload
            for name, payload in in_memory_a["files"].items()
            if name.startswith("request-")
        ]
        decoded_requests = "\n".join(
            payload.decode("utf-8") for payload in request_payloads
        )
        path_free = not any(
            token in decoded_requests
            for token in (str(ROOT), "C:\\", "D:\\", "P:\\")
        )
        materialized_exact = (
            materialized == in_memory_a["receipt"]
            and all(
                (request_root / name).read_bytes() == payload
                for name, payload in in_memory_a["files"].items()
            )
        )
        scratch_entries_before_cleanup = sorted(
            path.relative_to(scratch).as_posix()
            for path in scratch.rglob("*")
        )

    scratch_removed = not Path(raw).exists()
    gates = {
        "request_set_bytes_exact_between_builds": request_set_exact,
        "materialized_request_set_exact": materialized_exact,
        "request_files_contain_no_absolute_or_machine_paths": path_free,
        "request_recipe_sha256_matches_frozen_row": all(
            row["recipe_identity_pass"] for row in executions
        ),
        "all_output_sha256_match_frozen_u7_3d": all(
            row["output_identity_pass"] for row in executions
        ),
        "scratch_removed": scratch_removed,
    }
    return {
        "schema": "kmcfm.u7-3f-offline-recipe-export-request-report.v1",
        "node": "U7.3F",
        "order": order_name,
        "contract_sha256": _file_sha256(CONFIG),
        "request_set_receipt_sha256": _sha256(_canonical(in_memory_a["receipt"])),
        "request_set_index_sha256": in_memory_a["receipt"]["index"]["sha256"],
        "execution_rows": executions,
        "scratch_entries_before_cleanup": scratch_entries_before_cleanup,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "claim_ceiling": contract["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(report))
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
