import argparse
import hashlib
import io
import json
import shutil
import urllib.request
from pathlib import Path

import imagecodecs
import numpy as np
import tifffile
from PIL import Image, ImageCms, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "outputs/post_salut_tst100k_metadata_v1"
OUT = ROOT / "outputs/post_salut_tst100k_target_review_v1"
PIN = "3621d7cfdb6cbcb56d0419a2b75880a943cb3a28814db4838c3d08bf573ee7ef"
REVISION = "63bf13833b75407c862d608c3d81281d0a35b61f"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(report):
    pending = OUT / "report.pending"
    pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending.replace(OUT / "report.json")


def frozen():
    path = META / "candidate8_metadata.json"
    assert digest(path) == PIN
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {row[role] for row in config["triplets"] for role in ("content", "reference", "gt")}
    assert len(config["triplets"]) == 8 and len(required) == 24
    assert required == {row["path"] for row in config["files"]}
    assert len(config["files"]) == 24
    assert sum(row["bytes"] for row in config["files"]) == 20617008
    assert all(row["commit_hash"] == REVISION for row in config["files"])
    assert all(len(row["etag"]) == 64 for row in config["files"])
    return config


def acquire(config):
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(META / "candidate8_metadata.json", OUT / "candidate8_metadata.json")
    shutil.copytree(META / "terms", OUT / "terms")
    report = {
        "status": "ACQUIRING", "revision": REVISION,
        "candidate_sha256": PIN, "script_sha256": digest(Path(__file__)),
        "files": [], "triplets": [], "downloaded_bytes": 0,
        "authentication_used": False, "new_model_forwards": 0, "fits": 0,
        "scope": "Fixed eight-triplet noncommercial suitability review only; no product evaluation, training, extra reference-before or independent validation.",
        "upstream_source_mapping": "unknown", "per_image_rights_verified": False,
    }
    save(report)
    for entry in config["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        target = OUT / "originals" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{REVISION}/{entry['path']}"
        request = urllib.request.Request(url, headers={"User-Agent": "neuro-film-bounded-research/1", "Accept-Encoding": "identity"})
        pending = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(request, timeout=60) as response, pending.open("xb") as stream:
            count = 0
            while chunk := response.read(65536):
                count += len(chunk)
                if count > entry["bytes"]:
                    raise ValueError("Frozen file byte budget exceeded")
                stream.write(chunk)
        if count != entry["bytes"] or digest(pending) != entry["etag"]:
            raise ValueError(f"Frozen payload integrity failure: {entry['path']}")
        pending.rename(target)
        report["downloaded_bytes"] += count
        report["files"].append({**entry, "sha256": digest(target), "verified": True})
        save(report)
        print(f"verified {len(report['files'])}/24", flush=True)
    report["status"] = "ACQUIRED_24_EXACT_FILES"
    save(report)


def display_decode(path, destination):
    with Image.open(path) as header:
        icc = header.info.get("icc_profile")
        orientation = header.getexif().get(274, 1)
        metadata = {
            "format": header.format, "pillow_mode": header.mode,
            "encoded_size": list(header.size), "orientation": orientation,
            "icc_sha256": hashlib.sha256(icc).hexdigest() if icc else None,
            "png_gamma": header.info.get("gamma"), "png_srgb": header.info.get("srgb"),
        }
        if path.suffix.lower() not in {".tif", ".tiff", ".png"}:
            data = np.asarray(header).copy()
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tiff:
            assert len(tiff.pages) == 1
            metadata["bits_per_sample"] = tiff.pages[0].bitspersample
            data = tiff.pages[0].asarray()
    elif path.suffix.lower() == ".png":
        data = imagecodecs.png_decode(path.read_bytes())
    assert data.ndim == 3 and data.shape[2] == 3
    assert data.dtype in (np.dtype("uint8"), np.dtype("uint16"))
    metadata["decoded_dtype"] = str(data.dtype)
    metadata["decoded_shape"] = list(data.shape)
    assert orientation in (1, None), "Nonidentity orientation requires explicit precision-preserving handling"
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    if icc:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        metadata["icc_description"] = ImageCms.getProfileDescription(profile).strip()
        converted = imagecodecs.cms_transform(
            data, icc, srgb, colorspace="RGB", outcolorspace="RGB",
            outdtype=np.uint16, intent=1,
        )
        metadata["display_policy"] = "Embedded ICC to sRGB, LittleCMS relative colorimetric, uint16 output, no black-point compensation"
    else:
        converted = data.astype(np.uint16) * 257 if data.dtype == np.uint8 else data.copy()
        metadata["icc_description"] = None
        metadata["display_policy"] = "ASSUMPTION: untagged encoded RGB is sRGB; not verified colorimetry"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(destination, converted, photometric="rgb", extratags=[(34675, "B", len(srgb), srgb, False)])
    assert np.array_equal(tifffile.imread(destination), converted)
    preview = Image.fromarray(np.rint(converted.astype(np.float32) / 257).astype(np.uint8))
    png = destination.with_suffix(".png")
    preview.save(png, icc_profile=srgb)
    metadata["srgb16_path"] = str(destination.relative_to(OUT))
    metadata["srgb16_sha256"] = digest(destination)
    metadata["preview_path"] = str(png.relative_to(OUT))
    metadata["preview_sha256"] = digest(png)
    return preview, metadata


def render(config):
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ACQUIRED_24_EXACT_FILES"
    assert report["candidate_sha256"] == PIN
    report["render_script_sha256"] = digest(Path(__file__))
    report["cms_version"] = imagecodecs.cms_version()
    report["status"] = "RENDERING"
    save(report)
    for i, row in enumerate(config["triplets"], 1):
        board = Image.new("RGB", (1800, 650), (238, 238, 238))
        draw = ImageDraw.Draw(board)
        record = {"index": i, "source": row, "arms": {}}
        for column, role in enumerate(("content", "reference", "gt")):
            source = OUT / "originals" / row[role]
            entry = next(x for x in config["files"] if x["path"] == row[role])
            assert digest(source) == entry["etag"]
            preview, metadata = display_decode(source, OUT / "display_srgb16" / f"{i:02d}_{role}.tif")
            record["arms"][role] = metadata
            preview.thumbnail((590, 590), Image.Resampling.LANCZOS)
            board.paste(preview, (column * 600 + (600 - preview.width) // 2, 35 + (590 - preview.height) // 2))
            draw.text((column * 600 + 10, 8), {"content": "X / Content", "reference": "R / Reference", "gt": "Y / Supplied target"}[role], fill="black")
        a, b = record["arms"]["content"], record["arms"]["gt"]
        record["content_target_same_size"] = a["encoded_size"] == b["encoded_size"]
        record["alignment"] = "UNKNOWN: dimensions alone do not establish pixel alignment; no registration/warping performed"
        draw.text((10, 631), f"Fixed triplet {i}/8 | ICC-managed sRGB previews; see report for untagged assumptions | noncommercial research", fill="black")
        path = OUT / "review" / f"{i:02d}.png"
        path.parent.mkdir(exist_ok=True)
        board.save(path, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        record["board"] = {"path": str(path.relative_to(OUT)), "sha256": digest(path)}
        report["triplets"].append(record)
        save(report)
        print(f"rendered {i}/8", flush=True)
    report["status"] = "READY_FOR_FIXED_EIGHT_VISUAL_REVIEW_NOT_TRAINING"
    report["distinct_image_decodes"] = 24
    report["source_precision_policy"] = "All 24 downloaded bytes preserved; uint8/uint16 source decoded without Pillow RGB16 truncation; color-managed uint16 display derivatives are not source originals"
    save(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    config = frozen()
    if args.acquire:
        acquire(config)
    if args.render:
        render(config)
    if not args.acquire and not args.render:
        print(json.dumps({"files": 24, "bytes": 20617008, "revision": REVISION, "candidate_sha256": PIN}))
