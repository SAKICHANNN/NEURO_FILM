"""Acquire the frozen FILM-R v2 real-film scan lane and emit exact evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.real_film.filmr import (  # noqa: E402
    build_pair_manifest,
    canonical_json_bytes,
    download_file,
    fetch_article,
    hash_file,
    validate_article,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "real_film_filmr_acquisition.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "real_film" / "filmr_v2" / "files",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "outputs" / "real_film" / "filmr_v2" / "evidence",
    )
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    article = fetch_article(int(config["article_id"]))
    files = validate_article(article, config)
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "id": article["id"],
        "version": article["version"],
        "doi": article["doi"],
        "title": article["title"],
        "description": article["description"],
        "published_date": article["published_date"],
        "modified_date": article["modified_date"],
        "license": article["license"],
        "authors": article["authors"],
        "files": files,
    }
    snapshot_path = evidence_dir / "article_snapshot.json"
    snapshot_path.write_bytes(canonical_json_bytes(snapshot))
    print(f"FILM-R metadata passed: {len(files)} files / {sum(int(x['size']) for x in files)} bytes")
    data_dir = args.data_dir.resolve()
    for index, remote in enumerate(files, 1):
        download_file(remote, data_dir / str(remote["name"]))
        if index % 5 == 0 or index == len(files):
            print(f"FILM-R download {index}/{len(files)}", flush=True)
    manifest = build_pair_manifest(files, output_dir=data_dir, config=config)
    manifest_path = evidence_dir / "pairs.jsonl"
    manifest_path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in manifest
        )
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": hash_file(args.config, "sha256"),
        "article_snapshot_sha256": hash_file(snapshot_path, "sha256"),
        "pairs_manifest_sha256": hash_file(manifest_path, "sha256"),
        "article_id": article["id"],
        "article_version": article["version"],
        "doi": article["doi"],
        "license": config["license"],
        "files": len(files),
        "bytes": sum(int(item["size"]) for item in files),
        "pairs": len(manifest),
        "families_from_filenames": sorted({row["filename_family_claim"] for row in manifest}),
        "physical_roll_ids_known": 0,
        "process_ids_known": 0,
        "scanner_ids_known": 0,
        "all_remote_md5_verified": True,
        "all_local_sha256_recorded": True,
        "software_commit": _commit(),
        "claim_ceiling": config["claim_ceiling"],
    }
    report_path = evidence_dir / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"report": str(report_path), "pairs": len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
