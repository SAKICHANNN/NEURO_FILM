import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/salut_g0_native_content_v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent():
    path = ROOT / "scripts/run_salut_photo_development_v1.py"
    assert (
        digest(path)
        == "9446a923cb149d299a650d900c987a3ab788419c8f89938ef66be4ced5240dff"
    )
    spec = importlib.util.spec_from_file_location("frozen_salut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight():
    config = json.loads(CONFIG.read_text())
    for path, expected in (
        (
            "data/external/salut_v1/checkpoint/epoch=100-step=4127466.ckpt.state.pt",
            "5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b",
        ),
        (
            "data/external/salut_v1/repo/SA-LUT/ckpts/vgg_normalised.pth",
            "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
        ),
    ):
        assert digest(ROOT / path) == expected
    manifest = ROOT / config["inputs_manifest"]
    assert digest(manifest) == config["inputs_sha256"]
    assert digest(ROOT / config["reference"]) == config["reference_sha256"]
    rows = json.loads(manifest.read_text())["rows"]
    assert [r["id"] for r in rows] == ["father_baby", "birthday", "text_card"]
    for row in rows:
        assert digest(Path(row["input_path"])) == row["input_sha256"]
        assert digest(ROOT / row["raw_path"]) == row["raw_sha256"]
    return config, {
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__)),
        "rows": rows,
    }


def worker(attempt):
    import numpy as np
    import tifffile
    import torch
    from PIL import Image
    from torch.nn import functional as F
    from torchvision.transforms.functional import to_tensor

    config, checks = preflight()
    assert checks == json.loads((attempt / "lock.json").read_text())
    base = parent()
    model, quad, loading, calls = base.build_model()
    report = {
        "status": "running",
        "new_forwards": 0,
        "rows": [],
        "checks": checks,
        "model_loading": loading,
        "scope": config["scope"],
    }
    reference = F.interpolate(
        to_tensor(Image.open(ROOT / config["reference"]).convert("RGB")).unsqueeze(0),
        size=(512, 512),
        mode="bilinear",
        align_corners=True,
    )
    with torch.inference_mode():
        for item in checks["rows"]:
            row = {"id": item["id"], "status": "forward_started"}
            report["rows"].append(row)
            base.save_report(attempt / "report.json", report)
            data = tifffile.imread(item["input_path"])
            assert (
                data.dtype == np.uint16
                and [data.shape[1], data.shape[0]] == item["size"]
            )
            full = (
                torch.from_numpy(data.astype(np.float32) / 65535)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            small = F.interpolate(
                full, size=(512, 512), mode="bilinear", align_corners=True
            )
            del full
            assert report["new_forwards"] < config["budget"]["maximum_forwards"]
            report["new_forwards"] += 1
            report["forward_count_semantics"] = (
                "started attempts, including failed attempts"
            )
            base.save_report(attempt / "report.json", report)
            _, lut, context = model(reference, small)
            height, width = data.shape[:2]
            context_full = F.interpolate(
                context, size=(height, width), mode="bilinear", align_corners=True
            )
            mean = float(context_full.mean())
            del context_full
            row["context_mean"] = mean
            np.save(attempt / (item["id"] + "_lut.npy"), lut.numpy())
            np.save(attempt / (item["id"] + "_context512.npy"), context.numpy())
            native, blended, simple = (
                np.empty_like(data),
                np.empty_like(data),
                np.empty_like(data),
            )
            for y in range(0, height, config["render_rows"]):
                pixels = (
                    torch.from_numpy(
                        data[y : y + config["render_rows"]].astype(np.float32) / 65535
                    )
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                _, rendered = quad(
                    lut[0],
                    torch.cat([torch.full_like(pixels[:, :1], mean), pixels], dim=1),
                )
                rendered = rendered.clamp(0, 1)
                mixed = (1 - config["strength"]) * pixels + config[
                    "strength"
                ] * rendered
                control = (
                    (pixels - config["simple_pivot"]) * config["simple_slope"]
                    + config["simple_pivot"]
                ).clamp(0, 1)
                for output, tensor in (
                    (native, rendered),
                    (blended, mixed),
                    (simple, control),
                ):
                    output[y : y + config["render_rows"]] = np.rint(
                        tensor.squeeze(0).permute(1, 2, 0).numpy() * 65535
                    ).astype(np.uint16)
            for arm, array in (
                ("original", data),
                ("simple", simple),
                ("g0", native),
                ("g0_strength065", blended),
            ):
                path = attempt / (item["id"] + "_" + arm)
                tifffile.imwrite(path.with_suffix(".tif"), array, photometric="rgb")
                Image.fromarray(
                    np.rint(array.astype(np.float32) / 257).astype(np.uint8)
                ).save(path.with_suffix(".png"))
            row["artifacts"] = [
                {"path": p.name, "sha256": digest(p)}
                for p in sorted(attempt.glob(item["id"] + "_*"))
            ]
            row["size"] = item["size"]
            row["status"] = "complete"
            base.save_report(attempt / "report.json", report)
            del data, native, blended, simple, small, lut, context
    report["status"] = "COMPLETE_NOT_PROMOTED"
    report["standard_lut_calls"] = len(calls)
    base.save_report(attempt / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    config, checks = preflight()
    if not args.run:
        print(json.dumps(checks, indent=2))
        return
    base = parent()
    base.__file__, base.CONFIG = __file__, CONFIG
    raise SystemExit(base.launch(config, checks))


if __name__ == "__main__":
    main()
