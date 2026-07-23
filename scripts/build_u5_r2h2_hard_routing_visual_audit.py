#!/usr/bin/env python3
"""Build the frozen U5.R2H2 replay manifest and blinded presentations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.global_frontier import sha256_file  # noqa: E402
from src.eval.hard_routing_visual_audit import (  # noqa: E402
    build_replay_manifest,
    write_presentations,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "u5_r2h2_hard_routing_visual_audit_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / str(config["output_dir"])
    manifest = build_replay_manifest(ROOT, config)
    manifest.update(
        {
            "software_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "config_sha256": sha256_file(config_path),
        }
    )
    replay_path = output_dir / "replay_manifest.json"
    _write_json(replay_path, manifest)
    presentation = write_presentations(manifest, config, output_dir)
    key_path = output_dir / "blind_key.json"
    scoring_path = output_dir / "blind_scoring.json"
    index_path = output_dir / "presentation_index.json"
    _write_json(key_path, presentation.pop("blind_key"))
    _write_json(scoring_path, presentation.pop("scoring_template"))
    presentation.update(
        {
            "replay_manifest": str(replay_path.resolve()),
            "replay_manifest_sha256": sha256_file(replay_path),
            "blind_key": str(key_path.resolve()),
            "blind_key_sha256": sha256_file(key_path),
            "blind_scoring": str(scoring_path.resolve()),
            "blind_scoring_sha256": sha256_file(scoring_path),
        }
    )
    _write_json(index_path, presentation)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_sha256": sha256_file(index_path),
                "replay_manifest_sha256": sha256_file(replay_path),
                "blind_panel_count": len(presentation["blind_panels"]),
                "severe_panel_count": len(presentation["severe_panels"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
