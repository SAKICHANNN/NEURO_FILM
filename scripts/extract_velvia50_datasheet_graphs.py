"""Extract exact page-8 graph images used by U5.R2H0A.

This small reproducer needs pypdf and Pillow. It never edits the source PDF.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data/physics/fujifilm_velvia_50/product_information_bulletin.pdf"
DEFAULT_OUTPUT = ROOT / "outputs/u5_r2h0_source_audit/extracted"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    page = PdfReader(str(args.pdf)).pages[7]
    for index, image in enumerate(page.images):
        stem = Path(image.name).stem
        tiff = args.output / f"velvia50_p8_{index}_{image.name}"
        png = args.output / f"velvia50_p8_{index}_{stem}.png"
        tiff.write_bytes(image.data)
        image.image.save(png)
        print(f"{tiff.name}\t{tiff.stat().st_size}\t{sha256(tiff)}")
        print(f"{png.name}\t{png.stat().st_size}\t{sha256(png)}")


if __name__ == "__main__":
    main()
