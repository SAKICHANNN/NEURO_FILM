"""Private fixed-pair VCG inference; published LUT pipeline, not generated RGB."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/ai_models/video_color_grading_v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError(f"identity mismatch: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize(cfg):
    # Verification happens before importing any downloaded Python source.
    for name, expected in cfg["source_hashes"].items():
        if sha(MODEL / "source" / name) != expected:
            raise ValueError(f"official source drift: {name}")
    manifest = checked_json(MODEL / "manifest.json", cfg["model_manifest_sha256"])
    for row in manifest["models"]:
        if sha(MODEL / row["file"]) != row["sha256"]:
            raise ValueError("converted model identity mismatch")
    clip_root = MODEL / "clip_vit_b32"
    if sha(clip_root / "pytorch_model.bin") != cfg["clip_sha256"]:
        raise ValueError("CLIP identity mismatch")
    base = (
        Path.home()
        / ".cache/huggingface/hub"
        / cfg["sd15_cache_repo"]
        / "snapshots"
        / cfg["sd15_revision"]
    )
    for name, expected in cfg["sd15_hashes"].items():
        if sha(base / name) != expected:
            raise ValueError(f"local base component drift: {name}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path[:0] = [str(MODEL / "runtime"), str(MODEL / "source"), str(ROOT)]
    import diffusers
    import numpy as np
    import torch
    import transformers
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from models.ImageEncoder import ImageEncoder
    from models.ReferenceNet import ReferenceNet
    from PIL import Image
    from pillow_lut import identity_table
    from pipeline import InferencePipeline
    from safetensors.torch import load_file
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    if diffusers.__version__ != "0.21.4" or transformers.__version__ != "4.44.2":
        raise ValueError("isolated runtime mismatch")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    unet_config = json.loads((base / "unet/config.json").read_text())
    vae_config = json.loads((base / "vae/config.json").read_text())
    clip_config = CLIPVisionConfig.from_pretrained(clip_root, local_files_only=True)
    with torch.device("meta"):
        reference = ReferenceNet.from_config(unet_config)
        unet = UNet2DConditionModel.from_config(
            unet_config, in_channels=6, out_channels=3
        )
        vae = AutoencoderKL.from_config(vae_config)
        vision = CLIPVisionModel(clip_config)
    for model, path in [
        (reference, MODEL / "referencenet_state_dict.safetensors"),
        (unet, MODEL / "unet_state_dict.safetensors"),
        (vae, base / "vae/diffusion_pytorch_model.safetensors"),
    ]:
        model.load_state_dict(load_file(path), strict=True, assign=True)
        model.eval()
    clip_state = torch.load(
        clip_root / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    # New Transformers makes this old buffer non-persistent. Verify the exact
    # published integer sequence; do not discard arbitrary state mismatches.
    positions = clip_state.pop("vision_model.embeddings.position_ids")
    if positions.dtype != torch.int64 or not torch.equal(
        positions, torch.arange(50)[None]
    ):
        raise ValueError("unexpected CLIP positional buffer")
    vision.vision_model.embeddings.position_ids = positions
    vision.load_state_dict(
        {
            key: value
            for key, value in clip_state.items()
            if key.startswith("vision_model.")
        },
        strict=True,
        assign=True,
    )
    del clip_state
    for model in (reference, unet, vae, vision):
        if any(
            tensor.is_meta
            for tensor in list(model.parameters()) + list(model.buffers())
        ):
            raise ValueError("unmaterialized model state")
    vision.eval()
    # Same published ImageEncoder.forward, but no unsafe from_pretrained pickle
    # load; the exact complete vision state is loaded explicitly above.
    encoder = ImageEncoder.__new__(ImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model = vision
    encoder.freeze()
    processor = CLIPImageProcessor.from_pretrained(clip_root, local_files_only=True)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
    )
    pipeline = InferencePipeline(vae=vae, unet=unet, scheduler=scheduler).to("cuda")
    identity = np.asarray(identity_table(16).table, dtype=np.float32).reshape(64, 64, 3)
    identity_chw = torch.from_numpy(identity.transpose(2, 0, 1).copy())[None]

    # Only the three reviewed published preprocessing functions. No download,
    # video utilities or unrelated module-level execution from utils/util.py.
    source = (MODEL / "source/utils/util.py").read_text()
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"vars", "transfer", "preprocess"}
    ]
    if len(nodes) != 3:
        raise ValueError("preprocess source mismatch")
    namespace = {"np": np, "Image": Image}
    exec(  # noqa: S102 - Exact hash-checked official functions reviewed above.
        compile(
            ast.Module(body=nodes, type_ignores=[]), "official_vcg_preprocess", "exec"
        ),
        namespace,
    )

    def predict(content, style):
        generator = torch.Generator(device="cuda").manual_seed(cfg["seed"])
        result = pipeline(
            width=cfg["size"],
            height=cfg["size"],
            num_inference_steps=cfg["steps"],
            num_actual_inference_steps=cfg["steps"],
            generator=generator,
            source_image=style,
            input_video=content,
            referencenet=reference,
            clip_image_encoder=encoder,
            clip_image_processor=processor,
            id_lut=identity_chw,
            return_dict=False,
        )
        if result.shape != (1, 3, 64, 64) or not torch.isfinite(result).all():
            raise ValueError("invalid LUT model output")
        return result[0].detach().cpu().numpy().transpose(1, 2, 0) + identity

    return predict, namespace["preprocess"]


def read_rgb(path):
    from PIL import Image, ImageCms, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            raise ValueError("expected RGB")
        if profile := image.info.get("icc_profile"):
            return ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                renderingIntent=1,
                outputMode="RGB",
            )
        return image.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ai_vcg_reference_development_v1.json",
    )
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    owned = ["scripts/run_ai_vcg_reference.py", str(args.config.relative_to(ROOT))]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *owned], cwd=ROOT):
        raise ValueError("commit owned driver/config before execution")
    out = ROOT / cfg["output"]
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise ValueError("fresh P-backed output required")
    out.mkdir()
    report = {
        "config": cfg,
        "stage": "loading",
        "rows": [],
        "visual_review": "pending",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "product_promotion": False,
        "exact_paper_reproduction": False,
    }
    started = time.monotonic()
    try:
        predict, preprocess = initialize(cfg)
        import numpy as np
        import torch
        from PIL import Image, ImageDraw, ImageFilter

        report["stage"] = "synthetic"
        torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(48)
        synthetic = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        a = predict(synthetic[None], synthetic[:, ::-1].copy())
        b = predict(synthetic[None], synthetic[:, ::-1].copy())
        report["synthetic"] = {
            "repeat_exact": bool(np.array_equal(a, b)),
            "minimum": float(a.min()),
            "maximum": float(a.max()),
        }
        if not np.array_equal(a, b):
            raise ValueError("synthetic repeat mismatch")
        print("synthetic repeat PASS", flush=True)
        report["stage"] = "development"
        prior = checked_json(ROOT / cfg["nlut_report"], cfg["nlut_report_sha256"])
        sources = checked_json(
            ROOT / cfg["source_manifest"], cfg["source_manifest_sha256"]
        )
        for i, pair in enumerate(prior["pairs"]):
            if time.monotonic() - started > cfg["max_seconds"]:
                raise TimeoutError("development wall-time cap")
            content_path, reference_path = (
                Path(pair["content_path"]),
                Path(pair["reference_path"]),
            )
            if (
                sha(content_path) != pair["content_sha256"]
                or sha(reference_path) != pair["reference_sha256"]
            ):
                raise ValueError("pair source drift")
            content, reference = read_rgb(content_path), read_rgb(reference_path)
            style = np.asarray(reference.resize((512, 512), Image.Resampling.BICUBIC))
            before = np.asarray(content)
            corrected, small = preprocess(before[None], style, 512, False)
            lut = predict(small, style)
            correction = Image.fromarray(corrected[0])
            output = correction.filter(
                ImageFilter.Color3DLUT(16, np.clip(lut, 0, 1).flatten())
            )
            simple_path = (ROOT / cfg["nlut_report"]).parent / f"{i:02d}_simple.png"
            if sha(simple_path) != prior["rows"][i]["images"]["simple"]["sha256"]:
                raise ValueError("prior simple control drift")
            simple = read_rgb(simple_path)
            legacy = read_rgb(Path(sources[i]["after"]))
            prior_path = (ROOT / cfg["nlut_report"]).parent / f"{i:02d}_adapted.png"
            if sha(prior_path) != prior["rows"][i]["images"]["adapted"]["sha256"]:
                raise ValueError("prior NLUT output drift")
            arms = {
                "identity": content,
                "simple": simple,
                "legacy_safe_lab": legacy,
                "precorrection": correction,
                "learned": output,
                "prior_nlut_negative": read_rgb(prior_path),
            }
            row = {
                "pair": pair,
                "lut_minimum": float(lut.min()),
                "lut_maximum": float(lut.max()),
                "lut_out_of_range_fraction": float(np.mean((lut < 0) | (lut > 1))),
                "images": {},
            }
            for name, image in arms.items():
                target = out / f"{i:02d}_{name}.png"
                with target.open("xb") as handle:
                    image.save(handle, format="PNG")
                row["images"][name] = sha(target)
            with (out / f"{i:02d}_lut.npy").open("xb") as handle:
                np.save(handle, lut, allow_pickle=False)
            tiles = []
            for name, image in arms.items():
                tile = Image.new("RGB", (480, 390), "white")
                small_image = image.copy()
                small_image.thumbnail((480, 360))
                tile.paste(small_image, (0, 30))
                ImageDraw.Draw(tile).text((8, 8), name, fill="black")
                tiles.append(tile)
            sheet = Image.new("RGB", (480 * len(tiles), 390), "white")
            for n, tile in enumerate(tiles):
                sheet.paste(tile, (480 * n, 0))
            sheet.save(out / f"{i:02d}_comparison.png")
            report["rows"].append(row)
            print("pair", i, "complete", flush=True)
        report["stage"] = "rendered_pending_visual_review"
        report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        with (out / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
