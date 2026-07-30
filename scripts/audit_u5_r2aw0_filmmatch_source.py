"""Audit the exact bounded FilmMatch source and ordered pair mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.filmmatch_paired_source import audit_source  # noqa: E402


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/u5_r2aw0_filmmatch_ektachrome_paired_source_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/external_controls/filmmatch_ektachrome_v1/integrity_audit.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["acquisition"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_source(
        root=ROOT / config["acquisition"]["root"],
        config=config,
        manifest=manifest,
    )
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": report["file_count"],
                "bytes": report["bytes"],
                "ordered_mapping_passed": report["ordered_mapping_passed"],
                "stable_evidence_id": report["stable_evidence_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
