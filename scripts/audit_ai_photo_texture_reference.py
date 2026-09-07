"""Inspect existing cleared JPEG display texture without training or promotion."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.photo_texture_reference import describe_patch, select_patches


def main():
    cfg = json.loads((ROOT / "configs/ai_photo_texture_reference_v1.json").read_text())
    manifest_path = ROOT / cfg["manifest"]
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != cfg["manifest_sha256"]:
        raise ValueError("Manifest identity mismatch")
    rows = sorted(json.loads(payload)["rows"], key=lambda row: row["sha256"])
    out = ROOT / cfg["output"]
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise ValueError("Require new P-backed output")
    out.mkdir(parents=True)
    report = {"config": cfg, "images": [], "visual_review": "pending"}
    patches, spectra, meta = [], [], []
    sheet = Image.new("RGB", (8 * 132, len(rows) * 150), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        path = manifest_path.parent / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image identity mismatch")
        with Image.open(path) as im:
            # Keep encoded JPEG axes so8x8 block coordinates remain interpretable.
            a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255
        selected = select_patches(a, cfg)
        item = {
            "source": row,
            "role": "fit" if index < cfg["fit_image_count"] else "diagnostic",
            "patches": [],
        }
        draw.text(
            (0, index * 150),
            f"{row['path']} {item['role']} count={len(selected)}",
            fill="black",
        )
        for j, (g, y, x, extreme) in enumerate(selected):
            patch = a[y : y + cfg["patch"], x : x + cfg["patch"]]
            desc, residual, spectrum = describe_patch(patch)
            desc.update(
                {"x": x, "y": y, "gradient_p90": g, "extreme_fraction": extreme}
            )
            item["patches"].append(desc)
            patches.append(patch.astype(np.float32))
            spectra.append(spectrum.astype(np.float32))
            meta.append({"image_index": index, "x": x, "y": y, "role": item["role"]})
            sheet.paste(
                Image.fromarray(np.rint(patch * 255).astype(np.uint8)),
                (j * 132, index * 150 + 20),
            )
            # Scientific diagnostic visualization only:8x residual around gray.
            sheet.paste(
                Image.fromarray(
                    np.rint(np.clip(0.5 + 8 * residual, 0, 1) * 255).astype(np.uint8)
                ),
                (j * 132 + 66, index * 150 + 20),
            )
            draw.text(
                (j * 132, index * 150 + 88), f"sd={desc['luma_std']:.4f}", fill="black"
            )
            draw.text(
                (j * 132, index * 150 + 102),
                f"lag={desc['luma_lag1']:.2f}",
                fill="black",
            )
            draw.text(
                (j * 132, index * 150 + 116),
                f"b8={desc['jpeg8_boundary_ratio']:.2f}",
                fill="black",
            )
        report["images"].append(item)
        print(row["path"], len(selected), flush=True)
    report["supported_images"] = sum(
        len(item["patches"]) >= 4 for item in report["images"]
    )
    report["support_gate"] = (
        report["supported_images"] >= cfg["minimum_images_with_four_patches"]
    )
    np.savez_compressed(
        out / "patches.npz", patches=np.asarray(patches), spectra=np.asarray(spectra)
    )
    (out / "patch_metadata.json").write_text(json.dumps(meta, indent=2))
    for page in range(4):
        top, bottom = page * 5 * 150, min((page + 1) * 5 * 150, sheet.height)
        if top < bottom:
            sheet.crop((0, top, sheet.width, bottom)).save(out / f"patches_{page}.png")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("supported", report["supported_images"], "gate", report["support_gate"])


if __name__ == "__main__":
    main()
