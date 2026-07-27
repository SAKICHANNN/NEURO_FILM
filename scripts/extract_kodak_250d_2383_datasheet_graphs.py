"""Extract exact Kodak graph rasters used by U5.R2AA0/AA1.

The utility validates both source PDFs before selecting embedded page images.
It never edits the PDFs and writes only to an ignored output directory.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/u5_r2aa0_source_audit/extracted"


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    sha256: str


@dataclass(frozen=True)
class GraphSpec:
    source: str
    page_zero_based: int
    image_index: int
    output_name: str
    width: int
    height: int
    png_sha256: str


SOURCES = {
    "negative_250d": SourceSpec(
        ROOT / "data/physics/kodak_vision3_250d/technical_data.pdf",
        "70adb298a7aabb285d986b720e07c87c27eb2361f0925aea2b15903c08282e16",
    ),
    "print_2383": SourceSpec(
        ROOT / "data/physics/kodak_vision_print_2383/technical_data.pdf",
        "210d5e8ec1ea3a23a003b0c95b66c3688a29088b351124470638de35062a7565",
    ),
}

GRAPHS = (
    GraphSpec(
        "negative_250d",
        2,
        0,
        "250d_characteristic.png",
        587,
        557,
        "8d4ba7acec0be4a200ba4ab2f29f928bde63f430c8fd343d9a9e36bfc973ca0d",
    ),
    GraphSpec(
        "negative_250d",
        3,
        1,
        "250d_dye_density.png",
        693,
        754,
        "a2627ceec92d64e991a7b6209d86fbfe57330d115f850abc6ffb1c62c46c7291",
    ),
    GraphSpec(
        "negative_250d",
        3,
        2,
        "250d_sensitivity.png",
        737,
        749,
        "e51e2e29c3abf6bd7b7e16ac7fe12e5be1c135e0304c61de949f74b6d4106e77",
    ),
    GraphSpec(
        "print_2383",
        3,
        0,
        "2383_characteristic.png",
        490,
        496,
        "d0c48f6dfd0f39b21ca5c6003e91614d0e387e0ff78b0e77fb3552d245a0fa0b",
    ),
    GraphSpec(
        "print_2383",
        4,
        0,
        "2383_sensitivity.png",
        428,
        397,
        "5c3afcaec19f550e9f35f3430e139f0fa0b422885f3233400e261c70306c829c",
    ),
    GraphSpec(
        "print_2383",
        4,
        1,
        "2383_dye_density.png",
        401,
        390,
        "e8fefcbaaded7d696118f0b36ca2e9e8cceafe986403fdffa19f25ef13aca56e",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(output: Path) -> list[Path]:
    readers: dict[str, PdfReader] = {}
    for name, source in SOURCES.items():
        if sha256(source.path) != source.sha256:
            raise ValueError(f"Kodak source hash mismatch: {name}")
        readers[name] = PdfReader(str(source.path))

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for graph in GRAPHS:
        images = readers[graph.source].pages[graph.page_zero_based].images
        if graph.image_index >= len(images):
            raise ValueError(f"missing embedded graph: {graph.output_name}")
        image = images[graph.image_index].image
        if image.size != (graph.width, graph.height):
            raise ValueError(f"embedded graph dimensions drifted: {graph.output_name}")
        path = output / graph.output_name
        image.save(path)
        if sha256(path) != graph.png_sha256:
            raise ValueError(f"embedded graph hash drifted: {graph.output_name}")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in extract(args.output):
        print(f"{path.name}\t{path.stat().st_size}\t{sha256(path)}")


if __name__ == "__main__":
    main()
