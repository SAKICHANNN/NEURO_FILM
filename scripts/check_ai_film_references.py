"""Integrity and visual review sheet; never read historical assessment pixels."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/ai_film_references/italy_2024_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    historical = json.loads(
        (
            ROOT
            / "data/real_film/sf3_a3u_portra_three_source_heldout_v1/nicknick/manifest.json"
        ).read_text()
    )
    old = [int(r["dhash64"], 16) for r in historical["rows"]]
    sheet = Image.new("RGB", (1000, 1200), "#222222")
    draw = ImageDraw.Draw(sheet)
    signatures = []
    for i, row in enumerate(manifest["rows"]):
        path = source / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            bits = (a[:, 1:] > a[:, :-1]).ravel()
            h = sum(int(b) << (63 - j) for j, b in enumerate(bits))
            signatures.append(h)
            im.thumbnail((240, 210))
            x, y = (i % 4) * 250, (i // 4) * 240
            sheet.paste(im, (x, y + 20))
            draw.text((x + 5, y + 3), row["path"], fill="white")
    distances = [
        (a ^ b).bit_count()
        for i, a in enumerate(signatures)
        for b in signatures[i + 1 :]
    ]
    old_distances = [(a ^ b).bit_count() for a in signatures for b in old]
    out = ROOT / "outputs/ai_reference_italy_qa_v1"
    out.mkdir(exist_ok=False)
    sheet.save(out / "contact.jpg", quality=92)
    result = {
        "count": len(signatures),
        "minimum_within_dhash": min(distances),
        "minimum_historical_dhash": min(old_distances),
        "historical_pixels_read": 0,
        "caveat": "dHash is a screening heuristic, not proof of semantic independence",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
