import argparse
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps
import torch
from torch.nn import functional as F

from scripts.run_film_restoration_probe_v1 import ROOT, decode_image, gpu_snapshot, sha, state_sha, write_json
from src.models.source_preparation_v1 import preview_frame
from src.models.semantic_film_adversarial_v1 import PhotometricCritic, SemanticField, balanced_schedule, dino_input, field_penalties, patch_candidates, photometric_patches, render_frames, render_native, train_step


CONFIG_PATH = ROOT / "configs/semantic_film_adversarial_v1.json"
BOUND_CODE = ["src/models/semantic_film_adversarial_v1.py", "scripts/run_semantic_film_adversarial_v1.py", "configs/semantic_film_adversarial_v1.json", "tests/test_semantic_film_adversarial_v1.py", "src/color_match/research/local_reference_field_v1.py", "src/models/source_preparation_v1.py", "scripts/run_film_restoration_probe_v1.py"]
SOURCE_MANIFESTS = ["film_acquisition", "italy_manifest", "parker_acquisition", "fivek_manifest", "challenge_manifest"]


def canonical_film_pixels(image: Image.Image) -> tuple[np.ndarray, dict]:
    if image.mode != "RGB":
        raise ValueError("Expected admitted RGB film reference")
    orientation = image.getexif().get(274, 1)
    metadata = {"target_precision": "encoded_sRGB_uint8_then_float32_div255", "icc_sha256": None, "original_size_wh": list(image.size), "exif_orientation": orientation, "geometry_action": "explicit_EXIF_display_orientation_before_preview"}
    icc = image.info.get("icc_profile")
    image = ImageOps.exif_transpose(image)
    if icc:
        metadata["icc_sha256"] = hashlib.sha256(icc).hexdigest()
        image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), renderingIntent=1, outputMode="RGB", flags=0)
        metadata["color_action"] = "embedded_ICC_to_sRGB8_relative_colorimetric_no_BPC"
    else:
        metadata["color_action"] = "assume_untagged_sRGB"
    array = np.asarray(image).copy()
    metadata.update({"decoded_shape_hwc": list(array.shape), "decoded_uint8_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    return array, metadata


def decode_asset(row: dict) -> tuple[torch.Tensor, dict]:
    if row["decode"] != "film":
        return decode_image(row)
    if sha(ROOT / row["path"]) != row["sha256"]:
        raise ValueError("Film asset changed")
    with Image.open(ROOT / row["path"]) as image:
        array, metadata = canonical_film_pixels(image)
    return torch.from_numpy(array.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None], metadata


def dataset_rows(config: dict) -> dict:
    load = lambda name: json.loads((ROOT / config[name]).read_text(encoding="utf-8-sig"))
    fivek = load("fivek_manifest")
    if sha(ROOT / config["fivek_manifest"]) != config["fivek_manifest_sha256"]:
        raise ValueError("FiveK manifest changed")
    sources = [{"id": row["pair_id"], "path": row["aligned_expert_path"], "sha256": row["aligned_expert_sha256"], "shape": row["shape"], "decode": "digital", "role": "development"} for row in fivek["rows"] if row["split"] == "development"]
    sources.sort(key=lambda row: row["id"])
    inland, italy, parker = load("film_acquisition"), load("italy_manifest"), load("parker_acquisition")
    references = []
    for row in inland["rows"]:
        if row["status"] != "ACQUIRED" or row["rights"]["license"] != "https://creativecommons.org/licenses/by/4.0/":
            raise ValueError("Inland rights/identity not admitted")
        references.append({"id": row["id"], "path": row["image_path"], "sha256": row["image_sha256"], "decode": "film", "collection": 0})
    if italy["license"] != "CC-BY-4.0" or italy["role"] != "development_style_reference_only":
        raise ValueError("Italy role/rights mismatch")
    for row in italy["rows"]:
        references.append({"id": "italy_" + row["path"], "path": (Path(config["italy_manifest"]).parent / row["path"]).as_posix(), "sha256": row["sha256"], "decode": "film", "collection": 1})
    if parker["license"] != "CC0 declared on parent; child-parent membership verified":
        raise ValueError("Parker parent rights/membership mismatch")
    for row in parker["rows"]:
        references.append({"id": row["id"], "path": row["path"], "sha256": row["sha256"], "decode": "film", "collection": 2, "model_group": row["metadata"]["MODEL"]})
    challenges = [{**row, "decode": "challenge"} for row in load("challenge_manifest")["sources"]]
    if len(sources) != 381 or [sum(r["collection"] == i for r in references) for i in range(3)] != [7, 19, 12] or len(challenges) != 4:
        raise ValueError("Exact admitted corpus counts required")
    if len({r["model_group"] for r in references if r["collection"] == 2}) != 12 or any(r["role"] != "already_consumed_development_only" for r in challenges):
        raise ValueError("Distinct MODEL / consumed challenge roles changed")
    if len({r["sha256"] for r in sources + references}) != len(sources) + len(references):
        raise ValueError("Duplicate source/reference bytes")
    return {"source": sources, "reference": references, "challenge": challenges}


def load_dino(config: dict, device):
    from transformers import Dinov2Model
    path = ROOT / config["dino_root"]
    for name, digest in config["dino_hashes"].items():
        if sha(path / name) != digest:
            raise ValueError("Pinned DINO artifact changed: " + name)
    return Dinov2Model.from_pretrained(path, local_files_only=True, attn_implementation="eager").eval().requires_grad_(False).to(device)


def weighted_clusters(features: torch.Tensor, weights: torch.Tensor, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    features = F.normalize(features.float(), dim=1)
    rng = np.random.default_rng(config["cluster_seed"])
    chosen = int(rng.choice(len(features), p=(weights / weights.sum()).cpu().numpy()))
    centers = [features[chosen]]
    distance = torch.full((len(features),), float("inf"), device=features.device)
    for _ in range(1, config["clusters"]):
        distance = torch.minimum(distance, (features - centers[-1]).square().sum(1))
        centers.append(features[int((distance * weights).argmax())])
    centers = torch.stack(centers)
    for _ in range(config["cluster_iterations"]):
        labels = (features @ centers.T).argmax(1)
        for cluster in range(config["clusters"]):
            selected = labels == cluster
            if selected.any():
                centers[cluster] = F.normalize((features[selected] * weights[selected, None]).sum(0), dim=0)
    return centers, (features @ centers.T).argmax(1)


def guard_gpu(config: dict) -> dict:
    snapshot = gpu_snapshot()
    if len(snapshot["gpus"]) != 1 or snapshot["gpus"][0][3] < config["minimum_free_gib"] * 1024:
        raise RuntimeError("Insufficient GPU headroom; no foreign process touched")
    torch.cuda.set_per_process_memory_fraction(config["max_process_gib"] * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    return snapshot


def configure_torch():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def freeze(config: dict) -> None:
    output = ROOT / config["output"]
    if output.exists():
        if (output / "manifest.json").exists() or (output / "run").exists():
            raise FileExistsError("Frozen or trained attempt already exists")
        previous = json.loads((output / "preparation.json").read_text())
        if previous["status"] != "PREPARATION_FAILED_NO_TRAINING" or previous["optimizer_updates"] != 0:
            raise FileExistsError("Preparation state is not a terminal zero-update failure")
        archive = output / "preparation_failures" / str(previous["pid"])
        archive.mkdir(parents=True)
        for path in list(output.iterdir()):
            if path.is_file():
                path.rename(archive / path.name)
    rows = dataset_rows(config)
    for group in rows.values():
        for row in group:
            if sha(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("Input changed: " + row["path"])
    snapshot = guard_gpu(config)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {"status": "PREPARING", "pid": os.getpid(), "gpu_snapshot": snapshot, "optimizer_updates": 0, "confirmation_assets_opened": 0}
    write_json(output / "preparation.json", receipt)
    try:
        configure_torch()
        tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "tests.test_semantic_film_adversarial_v1", "-v"], cwd=ROOT, capture_output=True, text=True)
        (output / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Meaningful contract tests failed before descriptor preparation")
        dino = load_dino(config, "cuda")
        cache, metadata = {}, {}
        with torch.no_grad():
            for name, group in rows.items():
                images, masks, boxes, decoded = [], [], [], []
                for row in group:
                    image, information = decode_asset(row)
                    small, mask, box = preview_frame(image, config["preview_side"])
                    images.append(small); masks.append(mask); boxes.append(box); decoded.append(information)
                images, masks = torch.cat(images), torch.cat(masks)
                features = []
                for start in range(0, len(images), config["descriptor_batch"]):
                    if time.monotonic() - started > config["freeze_max_seconds"]:
                        raise TimeoutError("Descriptor preparation budget exceeded")
                    part = slice(start, start + config["descriptor_batch"])
                    tokens = dino(pixel_values=dino_input(images[part].cuda(), masks[part].cuda())).last_hidden_state[:, 1:]
                    features.append(tokens.transpose(1, 2).reshape(-1, config["feature_dim"], 16, 16).cpu())
                cache[name] = {"images": images, "masks": masks, "boxes": torch.tensor(boxes), "features": torch.cat(features)}
                metadata[name] = {"decodes": decoded, "preview_boxes": boxes}
                print("cached", name, len(images), flush=True)
        del dino
        torch.cuda.empty_cache()
        candidate, descriptors, weights = {}, [], []
        for name in ("source", "reference"):
            data = cache[name]
            candidate[name] = patch_candidates(data["boxes"].tolist(), config["preview_side"], config["patch_stride"], config["critic_patch"])
            counts = np.bincount(candidate[name][:, 0], minlength=len(rows[name]))
            if np.any(counts == 0):
                raise ValueError("At least one image has no valid 64-pixel patch; no support rescue")
            c = candidate[name]
            descriptors.append(data["features"][c[:, 0], :, c[:, 1], c[:, 2]])
            image_weights = np.full(len(rows[name]), 0.5 / len(rows[name]))
            if name == "reference":
                for i, row in enumerate(rows[name]):
                    image_weights[i] = 0.5 / (3 * [7, 19, 12][row["collection"]])
            weights.extend((image_weights[c[:, 0]] / counts[c[:, 0]]).tolist())
        centers, labels = weighted_clusters(torch.cat(descriptors).cuda(), torch.tensor(weights, device="cuda"), config)
        split = len(candidate["source"])
        candidate["source"] = np.column_stack((candidate["source"], labels[:split].cpu().numpy()))
        candidate["reference"] = np.column_stack((candidate["reference"], labels[split:].cpu().numpy()))
        challenge_candidates = patch_candidates(cache["challenge"]["boxes"].tolist(), config["preview_side"], config["patch_stride"], config["critic_patch"])
        c = challenge_candidates
        challenge_features = F.normalize(cache["challenge"]["features"][c[:, 0], :, c[:, 1], c[:, 2]].cuda(), dim=1)
        challenge_labels = (challenge_features @ centers.T).argmax(1).cpu().numpy()
        candidate["challenge"] = np.column_stack((challenge_candidates, challenge_labels))
        cache["cluster_centers"] = centers.cpu()
        cache_path = output / "descriptor_cache.pt"
        torch.save(cache, cache_path)
        schedules, exposure = {}, {}
        collections = np.asarray([row["collection"] for row in rows["reference"]])
        for seed in config["seeds"]:
            schedule, support = balanced_schedule(candidate["source"], candidate["reference"], collections, config, seed)
            path = output / f"schedule_{seed}.npz"
            np.savez(path, **schedule)
            schedules[str(seed)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
            exposure[str(seed)] = support
        np.savez(output / "cluster_candidates.npz", **candidate)
        support_by_cluster = {str(k): sorted({int(collections[r[0]]) for r in candidate["reference"] if r[-1] == k}) for k in range(config["clusters"])}
        challenge_support = []
        for i in range(4):
            labels_i = challenge_labels[challenge_candidates[:, 0] == i]
            challenge_support.append({"challenge_index": i, "cluster_patch_counts": np.bincount(labels_i, minlength=config["clusters"]).tolist(), "reference_collection_support": {str(k): support_by_cluster[str(k)] for k in sorted(set(labels_i.tolist()))}})
        canary_started = time.monotonic()
        torch.manual_seed(config["seeds"][0])
        generator, critic = SemanticField(config).cuda(), PhotometricCritic(config, True).cuda()
        source, reference = cache["source"], cache["reference"]
        sr, rr = schedule["source"][0], schedule["reference"][0]
        si, ri = sr[:, 0], rr[:, 0]
        images, masks, boxes = source["images"][si].cuda(), source["masks"][si].cuda(), source["boxes"][si].tolist()
        nodes = generator(images, masks, boxes, source["features"][si].cuda())
        rendered = render_frames(images, nodes, boxes)
        fake = photometric_patches(rendered, torch.as_tensor(sr[:, 3:5]), config)
        real = photometric_patches(reference["images"][ri].cuda(), torch.as_tensor(rr[:, 3:5]), config).requires_grad_()
        cluster_ids = torch.as_tensor(schedule["clusters"][0], device="cuda")
        real_score, fake_score = critic(real, cluster_ids), critic(fake.detach(), cluster_ids)
        r1 = torch.autograd.grad(real_score.sum(), real, create_graph=True)[0].square().flatten(1).sum(1).mean()
        (F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean() + .5 * config["r1_weight"] * config["r1_interval"] * r1).backward()
        critic.eval().requires_grad_(False)
        terms = field_penalties(images, rendered, nodes, boxes, config)
        terms["adversarial"] = -critic(fake, cluster_ids).mean()
        sum(config["loss_weights"][key] * value for key, value in terms.items()).backward()
        torch.cuda.synchronize()
        receipt.update({"status": "PREPARED_PENDING_ROOT_TRAINING_ADMISSION", "elapsed_seconds": time.monotonic() - started, "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "descriptor_cache_sha256": sha(cache_path), "canary_forward_backward_seconds": time.monotonic() - canary_started, "canary_optimizer_updates": 0, "canary_note": "One real-data forward/backward including R1 and generator losses, without any optimizer or parameter update; synthetic differing-corpus tests perform actual updates. Cold timing is descriptive, not a runtime guarantee.", "generator_parameters": sum(p.numel() for p in generator.parameters()), "critic_parameters": sum(p.numel() for p in critic.parameters())})
        write_json(output / "decode_and_sampling.json", {"metadata": metadata, "sampling": exposure, "reference_collections_by_cluster": support_by_cluster, "challenge_cluster_support": challenge_support})
        variants = [{"model": f"{arm}_{seed}", "render": render} for seed in config["seeds"] for arm in config["arms"] for render in ("local", "meanfield")]
        np.random.default_rng(config["blind_review_seed"]).shuffle(variants)
        write_json(output / "blinded_review_map.json", {chr(65 + i): variant for i, variant in enumerate(variants)})
        write_json(output / "preparation.json", receipt)
        manifest = {"schema": "semantic-film-adversarial-freeze-v1", "config": config, "rows": rows, "config_sha256": sha(CONFIG_PATH), "code_sha256": {path: sha(ROOT / path) for path in BOUND_CODE}, "source_manifest_sha256": {name: sha(ROOT / config[name]) for name in SOURCE_MANIFESTS}, "descriptor_cache": {"path": cache_path.relative_to(ROOT).as_posix(), "sha256": sha(cache_path)}, "schedules": schedules, "evidence_sha256": {name: sha(output / name) for name in ("decode_and_sampling.json", "preparation.json", "cluster_candidates.npz", "tests.txt", "blinded_review_map.json")}, "confirmation_assets_opened": 0}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": receipt["status"], "manifest_sha256": sha(output / "manifest.json"), "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        receipt.update({"status": "PREPARATION_FAILED_NO_TRAINING", "failure": repr(exc), "elapsed_seconds": time.monotonic() - started})
        write_json(output / "preparation.json", receipt)
        raise


def save_output(folder: Path, name: str, value: torch.Tensor) -> dict:
    array = value.detach().cpu().numpy()[0].transpose(1, 2, 0)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Nonfinite or out-of-range render")
    png, floats = folder / (name + ".png"), folder / (name + ".npy")
    Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(png)
    np.save(floats, array)
    return {"image_path": png.relative_to(ROOT).as_posix(), "image_sha256": sha(png), "float_path": floats.relative_to(ROOT).as_posix(), "float_sha256": sha(floats), "size_wh": [array.shape[1], array.shape[0]]}


def blind_panel(folder: Path) -> None:
    panel = Image.new("RGB", (1440, 1140), "#181818")
    draw = ImageDraw.Draw(panel)
    for index, label in enumerate(["original"] + list("ABCDEFGH")):
        with Image.open(folder / (label + ".png")) as image:
            image.thumbnail((476, 350), Image.Resampling.LANCZOS)
            left, top = (index % 3) * 480, (index // 3) * 380
            panel.paste(image, (left + (480 - image.width) // 2, top + 25 + (350 - image.height) // 2))
            draw.text((left + 8, top + 5), label, fill="white")
    panel.save(folder / "blind_contact.jpg", quality=94)


def run(config: dict, expected_sha: str) -> None:
    output = ROOT / config["output"]
    if sha(output / "manifest.json") != expected_sha:
        raise ValueError("Reviewed manifest SHA required")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["config"] != config or manifest["config_sha256"] != sha(CONFIG_PATH):
        raise ValueError("Configuration drift")
    for path, digest in manifest["code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Code drift: " + path)
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, stdout=subprocess.DEVNULL)
    if subprocess.check_output(["git", "diff", "HEAD", "--", *BOUND_CODE], cwd=ROOT):
        raise ValueError("Bound code must be committed")
    for name, digest in manifest["source_manifest_sha256"].items():
        if sha(ROOT / config[name]) != digest:
            raise ValueError("Source manifest drift")
    for name, digest in manifest["evidence_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError("Preparation evidence drift")
    for record in [manifest["descriptor_cache"], *manifest["schedules"].values()]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Cache/schedule drift")
    for group in manifest["rows"].values():
        for row in group:
            if sha(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("Input bytes drift")
    destination = output / "run"
    if destination.exists():
        raise FileExistsError("Single attempt exists; no restart")
    snapshot = guard_gpu(config)
    destination.mkdir()
    configure_torch()
    started = time.monotonic()
    report = {"status": "RUNNING", "pid": os.getpid(), "manifest_sha256": expected_sha, "gpu_snapshot": snapshot, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "models": {}, "rows": [], "confirmation_assets_opened": 0, "claim_ceiling": config["claim_ceiling"]}
    write_json(destination / "report.json", report)

    def guard(stage):
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("45 minute cap: " + stage)
        if torch.cuda.max_memory_reserved() > config["max_process_gib"] * 1024 ** 3 or torch.cuda.mem_get_info()[0] < 512 * 1024 ** 2:
            raise RuntimeError("GPU memory cap: " + stage)

    try:
        cache = torch.load(ROOT / manifest["descriptor_cache"]["path"], weights_only=True, map_location="cpu")
        review_mapping = json.loads((output / "blinded_review_map.json").read_text())
        aliases = {(value["model"], value["render"]): key for key, value in review_mapping.items()}
        for name in ("source", "reference", "challenge"):
            for key in ("images", "masks", "features"):
                cache[name][key] = cache[name][key].cuda()
        models = {}
        for seed in config["seeds"]:
            with np.load(ROOT / manifest["schedules"][str(seed)]["path"], allow_pickle=False) as arrays:
                schedule = {key: arrays[key] for key in arrays.files}
            for arm in config["arms"]:
                name = f"{arm}_{seed}"
                torch.manual_seed(seed)
                generator, critic = SemanticField(config).cuda(), PhotometricCritic(config, arm == "conditional").cuda()
                initial = {"generator": state_sha(generator), "critic": state_sha(critic)}
                go = torch.optim.Adam(generator.parameters(), lr=config["generator_lr"], betas=tuple(config["adam_betas"]))
                co = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], betas=tuple(config["adam_betas"]))
                trace_path = destination / f"{name}_trace.jsonl"
                with trace_path.open("x", encoding="utf-8") as stream:
                    for step in range(config["steps"]):
                        guard(name)
                        sr, rr = schedule["source"][step], schedule["reference"][step]
                        si, ri = torch.as_tensor(sr[:, 0], device="cuda"), torch.as_tensor(rr[:, 0], device="cuda")
                        source, reference = cache["source"], cache["reference"]
                        terms = train_step(generator, critic, go, co, source["images"][si], source["masks"][si], source["boxes"][sr[:, 0]].tolist(), source["features"][si], torch.as_tensor(sr[:, 3:5]), reference["images"][ri], torch.as_tensor(rr[:, 3:5]), torch.as_tensor(schedule["clusters"][step], device="cuda"), step, config)
                        stream.write(json.dumps({"step": step + 1, **terms}) + "\n")
                        if (step + 1) % 100 == 0:
                            stream.flush()
                            write_json(destination / "progress.json", {"model": name, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                            print(name, step + 1, terms, flush=True)
                checkpoint = destination / (name + ".pt")
                torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(), "manifest_sha256": expected_sha}, checkpoint)
                report["models"][name] = {"initial_state_sha256": initial, "final_state_sha256": {"generator": state_sha(generator), "critic": state_sha(critic)}, "checkpoint_sha256": sha(checkpoint), "trace_sha256": sha(trace_path), "steps": config["steps"], "seed": seed, "arm": arm}
                models[name] = generator.eval().requires_grad_(False)
                del critic, go, co
                write_json(destination / "report.json", report)
        with torch.no_grad():
            for i, row in enumerate(manifest["rows"]["challenge"]):
                guard("native render")
                native, decode = decode_asset(row)
                native = native.cuda()
                folder = destination / f"challenge_{i:02d}"
                folder.mkdir()
                record = {"source": row, "decode": decode, "original": save_output(folder, "original", native), "models": {}}
                challenge = cache["challenge"]
                for name, model in models.items():
                    nodes = model(challenge["images"][i:i + 1], challenge["masks"][i:i + 1], challenge["boxes"][i:i + 1].tolist(), challenge["features"][i:i + 1])
                    local = render_native(native, nodes, config["native_tile_size"])
                    mean = render_native(native, nodes.mean((-2, -1), keepdim=True).expand_as(nodes), config["native_tile_size"])
                    path = folder / (name + "_coefficients.npy")
                    np.save(path, nodes.cpu().numpy())
                    record["models"][name] = {"local": save_output(folder, aliases[name, "local"], local), "meanfield": save_output(folder, aliases[name, "meanfield"], mean), "coefficients_sha256": sha(path), "mean_absolute_change": float((local - native).abs().mean()), "local_vs_meanfield_mae": float((local - mean).abs().mean()), "spatial_coefficient_range": float((nodes.amax((-2, -1)) - nodes.amin((-2, -1))).max())}
                    guard("native render " + name)
                blind_panel(folder)
                report["rows"].append(record)
                write_json(destination / "report.json", report)
        report["status"] = "COMPLETE_DEVELOPMENT_VISUAL_REVIEW_PENDING"
    except Exception as exc:
        report.update({"status": "FAILED_NO_RESTART", "failure": repr(exc)})
        raise
    finally:
        report.update({"elapsed_seconds": time.monotonic() - started, "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        write_json(destination / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "peak_reserved_bytes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    if args.freeze:
        freeze(config)
    elif args.manifest_sha256:
        run(config, args.manifest_sha256)
    else:
        parser.error("--run requires reviewed --manifest-sha256")
