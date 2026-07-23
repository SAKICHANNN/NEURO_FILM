#!/usr/bin/env python3
"""Run the frozen SF2.5R PROV catalogue-only two-pass audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.prov_register_recon import (  # noqa: E402
    canonical_json,
    sha256_bytes,
    summarize_catalogues,
)


def _request(base: str, query: dict[str, object]) -> dict:
    url = base + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "K-MCFM-SF2.5R-metadata-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"PROV API returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_prov_negative_register_recon_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "prov_register_recon_v1" / "report.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    passes: list[dict[str, dict]] = []
    pass_hashes: list[str] = []
    for index in range(int(config["repeat_audits"])):
        payloads = {
            name: _request(config["source"]["api_base"], query)
            for name, query in config["queries"].items()
        }
        encoded = canonical_json(payloads)
        pass_hashes.append(sha256_bytes(encoded))
        _write_atomic(args.output.parent / f"normalized_pass_{index + 1}.json", encoded)
        passes.append(payloads)
    repeat_identical = len(set(pass_hashes)) == 1
    if not repeat_identical:
        summary = {
            "decision": "unresolved_source_instability",
            "register_contents_machine_accessible": None,
        }
    else:
        summary = summarize_catalogues(
            passes[0]["negative_register"],
            passes[0]["digitised_negative_collection"],
            config,
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "software_commit": commit,
        "config_sha256": sha256_bytes(args.config.read_bytes()),
        "normalized_pass_sha256": pass_hashes,
        "normalized_passes_byte_identical": repeat_identical,
        "image_or_pixel_request_count": 0,
        **summary,
        "claim_ceiling": config["claim_ceiling"],
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_atomic(args.output, encoded)
    print(
        json.dumps(
            {
                "report": str(args.output),
                "sha256": sha256_bytes(encoded),
                "decision": report["decision"],
                "repeat_identical": repeat_identical,
                "image_or_pixel_request_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
