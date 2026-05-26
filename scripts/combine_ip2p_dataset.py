#!/usr/bin/env python3
"""Combine per-style IP2P pseudo-pair datasets into one imagefolder dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine data/ip2p_train/<style>/train datasets.")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data" / "ip2p_train")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ip2p_train" / "all" / "train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    source_root = to_abs(args.source_root).resolve()
    output_dir = to_abs(args.output_dir).resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = [
        path.name
        for path in sorted(source_root.iterdir())
        if path.is_dir() and path.name != "all" and (path / args.split / "metadata.jsonl").exists()
    ]
    if not styles:
        raise RuntimeError(f"No per-style {args.split} datasets found under {source_root}")

    rows = []
    for style in styles:
        split_dir = source_root / style / args.split
        with (split_dir / "metadata.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                input_src = split_dir / row["input_image_file_name"]
                edited_src = split_dir / row["edited_image_file_name"]
                input_rel = f"{style}/input/{Path(row['input_image_file_name']).name}"
                edited_rel = f"{style}/edited/{Path(row['edited_image_file_name']).name}"
                link_or_copy(input_src, output_dir / input_rel, args.link_mode)
                link_or_copy(edited_src, output_dir / edited_rel, args.link_mode)
                rows.append(
                    {
                        **row,
                        "input_image_file_name": input_rel,
                        "edited_image_file_name": edited_rel,
                    }
                )

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "split": args.split,
        "styles": styles,
        "pairs": len(rows),
        "link_mode": args.link_mode,
    }
    (output_dir.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
