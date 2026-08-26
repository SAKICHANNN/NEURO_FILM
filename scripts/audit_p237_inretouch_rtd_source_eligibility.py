#!/usr/bin/env python3
"""Run the P237 official-source metadata-only eligibility audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.inretouch_source_eligibility import (
    analyze_inretouch_source_eligibility,
    canonical_json_bytes,
)

USER_AGENT = "neuro-film-p237-metadata-audit/1"


def _get(url: str, *, maximum_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read(maximum_bytes + 1)
            if len(payload) > maximum_bytes:
                raise RuntimeError(f"metadata response exceeded bound: {url}")
            return payload
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"metadata request failed: {url}") from last_error


def _json(url: str, *, maximum_bytes: int) -> tuple[dict[str, Any], int]:
    payload = _get(url, maximum_bytes=maximum_bytes)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {url}")
    return value, len(payload)


def execute(config_path: Path, *, reverse: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["official_sources"]
    repository, repository_bytes = _json(
        sources["repository_api"], maximum_bytes=256_000
    )
    branch = repository["default_branch"]
    head_data, head_bytes = _json(
        f"{sources['repository_api']}/commits/{branch}", maximum_bytes=512_000
    )
    head = head_data["sha"]
    raw_root = f"https://raw.githubusercontent.com/omarAlezaby/InRetouch/{head}"

    ordered = [
        ("readme", f"{raw_root}/readme.md", 64_000),
        ("license", f"{raw_root}/LICENSE", 16_000),
        ("dataset", f"{sources['dataset_api']}?blobs=true", 32_000_000),
    ]
    if reverse:
        ordered.reverse()
    payloads: dict[str, bytes] = {}
    for name, url, bound in ordered:
        payloads[name] = _get(url, maximum_bytes=bound)
    dataset = json.loads(payloads["dataset"])
    if not isinstance(dataset, dict):
        raise TypeError("expected Hugging Face dataset JSON object")

    report = analyze_inretouch_source_eligibility(
        config=config,
        github_repository=repository,
        github_head=head,
        readme_bytes=payloads["readme"],
        license_bytes=payloads["license"],
        dataset_metadata=dataset,
    )
    report["network_accounting"] = {
        "metadata_request_count": 5,
        "response_body_bytes": (
            repository_bytes
            + head_bytes
            + sum(len(payload) for payload in payloads.values())
        ),
    }
    # Network accounting is intentionally excluded from the stable scientific ID.
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/p237_inretouch_rtd_source_eligibility_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(report["decision"])
    return 0 if report["decision"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
