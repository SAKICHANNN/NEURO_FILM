import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/salut_strength_preview_v1"
    output.mkdir(exist_ok=False)
    reports = [
        (
            "salut_photo_development_v1",
            "af629d4fbb9033387eb0fd03aff7ce3aeac2de1a2beb9faeff4b689583634ffc",
        ),
        (
            "salut_photo_development_remaining_six_v1",
            "046c48c1cf9c6278cd94c6e888f5614b039deed3502e46dcbbb2c681e33c9d6b",
        ),
    ]
    report = {
        "strengths": [0.65, 0.8, 1.0],
        "formula": "float32 (1-s)*original_uint8+s*result_uint8; np.rint then uint8",
        "scope": "Display RGB output appearance strength, not model inference, linear-light or physical exposure mixing. No per-image tuning.",
        "rows": [],
    }
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    for folder, expected in reports:
        parent = root / "outputs" / folder / "attempt-0001"
        data = parent / "report.json"
        assert hashlib.sha256(data.read_bytes()).hexdigest() == expected
        for row in json.loads(data.read_text())["rows"]:
            index = row["source_index"]
            images = {}
            for label in ("original", "salut"):
                entry = row["arms"][label]
                path = parent / entry["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
                images[label] = np.asarray(Image.open(path).convert("RGB"))
            item = {
                "source_index": index,
                "parent_report_sha256": expected,
                "original": row["arms"]["original"],
                "native": row["arms"]["salut"],
                "previews": {},
            }
            variants = [("Original", Image.fromarray(images["original"]))]
            for strength in (0.65, 0.8, 1.0):
                if strength == 1:
                    path = parent / row["arms"]["salut"]["path"]
                else:
                    blend = np.rint(
                        np.float32(1 - strength) * images["original"].astype(np.float32)
                        + np.float32(strength) * images["salut"].astype(np.float32)
                    ).astype(np.uint8)
                    path = output / f"{index:02}_strength_{strength:.2f}.png"
                    Image.fromarray(blend).save(path)
                item["previews"][str(strength)] = {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                variants.append((f"Strength {strength:.2f}", Image.open(path)))
            canvas = Image.new("RGB", (1600, 1260), (240, 240, 240))
            draw = ImageDraw.Draw(canvas)
            for j, (label, im) in enumerate(variants):
                im = im.copy()
                im.thumbnail((800, 580))
                x = j % 2 * 800
                y = j // 2 * 630
                draw.text((x + 12, y + 8), label, font=font, fill=(20, 20, 20))
                canvas.paste(
                    im, (x + (800 - im.width) // 2, y + 42 + (580 - im.height) // 2)
                )
            canvas.save(output / f"{index:02}_comparison.jpg", quality=95)
            report["rows"].append(item)
    assert [r["source_index"] for r in report["rows"]] == list(range(9))
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("9 complete source/native/0.65/0.8 panels; zero inference")


if __name__ == "__main__":
    main()
