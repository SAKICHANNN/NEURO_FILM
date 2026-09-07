"""Prepare pinned source or run the frozen three-pair NLUT development comparison."""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageCms, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/ai_nlut_reference_development_v1.json"
from src.models.color_lut.nlut_reference import (
    basis_chunk,
    checked_json,
    interpolate,
    load_model,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_pair(path, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            raise ValueError("RGB image required")
        icc = image.info.get("icc_profile")
        if icc:
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"),
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
            )
        else:
            image = image.copy()  # Existing display JPEG: explicit assumed sRGB.
        full = (
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float()
            / 255
        )
        small = (
            torch.from_numpy(
                np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).copy()
            )
            .permute(2, 0, 1)[None]
            .float()
            / 255
        )
    return full, small


def learning_step(model, regularizer, optimizer, content, reference, cfg):
    optimizer.zero_grad(set_to_none=True)
    output, _, _ = model(content, content, reference)
    content_loss, style_loss = model.encoder(content, reference, output)
    (content_loss + style_loss).backward()
    reg_value = 0.0
    # Same per-basis regularizer, accumulated independently before global clipping.
    for start in range(0, model.CLUTs.num, cfg["basis_chunk"]):
        end = min(start + cfg["basis_chunk"], model.CLUTs.num)
        values = regularizer(basis_chunk(model.CLUTs, start, end))
        loss = (
            (values[0] + values[1] + 10 * values[2])
            * cfg["regularization_coefficient"]
            * ((end - start) / model.CLUTs.num)
        )
        reg_value += loss.detach().item()
        loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "content": content_loss.detach().item(),
        "style": style_loss.detach().item(),
        "regularizer": reg_value,
    }


def write_image(path, values):
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite image")
    a = values.detach().cpu()[0].permute(1, 2, 0).numpy()
    with path.open("xb") as stream:
        Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(
            stream, format="PNG"
        )
    return {
        "sha256": sha(path),
        "out_of_range_fraction": float(np.mean((a < 0) | (a > 1))),
        "minimum": float(a.min()),
        "maximum": float(a.max()),
    }


def run(cfg):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded run requires the local GPU")
    owned = [
        "configs/ai_nlut_reference_development_v1.json",
        "scripts/run_ai_nlut_reference.py",
        "src/models/color_lut/nlut_reference.py",
        "tests/test_ai_nlut_reference.py",
    ]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("Commit owned implementation before execution")
    source_path = ROOT / cfg["source_manifest"]
    sources = checked_json(source_path, cfg["source_manifest_sha256"])
    reference_path = ROOT / cfg["reference_manifest"]
    references = checked_json(reference_path, cfg["reference_sha256"])
    pairs = []
    for ci, ri in zip(cfg["content_indices"], cfg["reference_indices"], strict=True):
        row, ref = sources[ci], references["rows"][ri]
        content_path = Path(row["before"])
        content_path.relative_to(ROOT)
        ref_path = reference_path.parent / ref["path"]
        if row["license"] != "CC0/Public Domain" or sha(ref_path) != ref["sha256"]:
            raise ValueError("Input identity/rights drift")
        pairs.append(
            {
                "content_index": ci,
                "reference_index": ri,
                "content_path": str(content_path),
                "content_sha256": sha(content_path),
                "reference_path": str(ref_path),
                "reference_sha256": ref["sha256"],
            }
        )
    out = ROOT / cfg["output"]
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise RuntimeError("New P-backed run required")
    out.mkdir()
    report = {
        "config": cfg,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_hashes": {p: sha(ROOT / p) for p in owned},
        "pairs": pairs,
        "rows": [],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "visual_review": "pending",
        "exact_cuda_reproduction": False,
    }
    with (out / "execution_lock.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    start_time = time.monotonic()
    # Complete network/backward/resource smoke uses only synthetic pixels.
    model, module = load_model(ROOT, cfg)
    model = model.cuda().train()
    regularizer = module.TVMN(33).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    c = (
        torch.linspace(0, 1, 3 * 256 * 256, device="cuda")
        .reshape(1, 3, 256, 256)
        .repeat(2, 1, 1, 1)
    )
    s = c.flip(-1)
    torch.cuda.reset_peak_memory_stats()
    report["synthetic_smoke"] = learning_step(model, regularizer, optimizer, c, s, cfg)
    report["smoke_peak_bytes"] = torch.cuda.max_memory_allocated()
    print(
        "SYNTHETIC", report["synthetic_smoke"], report["smoke_peak_bytes"], flush=True
    )
    del model, regularizer, optimizer, c, s
    torch.cuda.empty_cache()
    for i, pair in enumerate(pairs):
        if time.monotonic() - start_time > cfg["max_runtime_seconds"]:
            raise TimeoutError("Frozen run wall-time cap")
        torch.manual_seed(cfg["seed"])
        model, module = load_model(ROOT, cfg)
        model = model.cuda().eval()
        full, small = image_pair(Path(pair["content_path"]), cfg["size"])
        ref_full, ref_small = image_pair(Path(pair["reference_path"]), cfg["size"])
        content = small.cuda().repeat(2, 1, 1, 1)
        reference = ref_small.cuda().repeat(2, 1, 1, 1)
        with torch.no_grad():
            _, _, aux = model(content, content, reference)
            initial_lut = aux["LUT"][:1].cpu()
        model.train()
        regularizer = module.TVMN(33).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        losses = []
        pair_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for step in range(cfg["steps"]):
            if time.monotonic() - start_time > cfg["max_runtime_seconds"]:
                raise TimeoutError("Frozen run wall-time cap")
            loss = learning_step(model, regularizer, optimizer, content, reference, cfg)
            losses.append(loss)
            if step % 10 == 0 or step == cfg["steps"] - 1:
                print("PAIR", i, "STEP", step, loss, flush=True)
        with torch.no_grad():
            _, _, aux = model(content, content, reference)
            final_lut = aux["LUT"][:1].cpu()
        peak = torch.cuda.max_memory_allocated()
        del model, regularizer, optimizer, content, reference, aux
        torch.cuda.empty_cache()
        with torch.no_grad():
            rendered = {
                "identity": full,
                "simple": ((full - 0.5) * 1.15 + 0.5).clamp(0, 1),
                "pretrained": full + interpolate(initial_lut, full),
                "adapted": full + interpolate(final_lut, full),
            }
        record = {
            "pair": i,
            "shape": list(full.shape),
            "seconds": time.monotonic() - pair_start,
            "peak_bytes": peak,
            "losses": losses,
            "images": {},
        }
        for name, values in rendered.items():
            path = out / f"{i:02d}_{name}.png"
            record["images"][name] = write_image(path, values)
            record["images"][name]["mean_absolute_change"] = float(
                (values - full).abs().mean()
            )
        write_image(out / f"{i:02d}_reference.png", ref_full)
        with (out / f"{i:02d}_luts.pt").open("xb") as stream:
            torch.save({"pretrained": initial_lut, "adapted": final_lut}, stream)
        sheet = Image.new("RGB", (4 * 384, 310), "white")
        draw = ImageDraw.Draw(sheet)
        for col, name in enumerate(rendered):
            with Image.open(out / f"{i:02d}_{name}.png") as im:
                im.thumbnail((384, 280))
                sheet.paste(im, (col * 384, 25))
            draw.text((col * 384 + 5, 5), name, fill="black")
        with (out / f"{i:02d}_comparison.png").open("xb") as stream:
            sheet.save(stream, format="PNG")
        report["rows"].append(record)
    report["seconds"] = time.monotonic() - start_time
    with (out / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("COMPLETE", sha(out / "report.json"), flush=True)


def prepare(cfg):
    out = ROOT / "data/ai_models/nlut_pretrained_v1/source"
    if (out.exists() and any(out.iterdir())) or out.resolve().drive.upper() != "P:":
        raise RuntimeError("New P-backed source directory required")
    out.mkdir(exist_ok=True)
    for name, size, blob in cfg["source_files"]:
        url = f"https://raw.githubusercontent.com/semchan/NLUT/{cfg['revision']}/{name}"
        with requests.get(url, timeout=30, stream=True) as response:
            response.raise_for_status()
            raw = response.raw.read(size + 1, decode_content=True)
        identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if len(raw) != size or identity != blob:
            raise ValueError("Source identity mismatch")
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        print("VERIFIED", name, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-source", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.prepare_source:
        prepare(cfg)
    else:
        run(cfg)
