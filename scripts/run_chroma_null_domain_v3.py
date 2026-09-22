import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.run_film_restoration_probe_v1 import ROOT, sha, state_sha, write_json
from scripts.run_semantic_film_adversarial_v1 import configure_torch, decode_asset, guard_gpu, save_output
from scripts.run_chroma_constrained_film_v2 import load_schedule, verify_base
from src.models.semantic_film_adversarial_v1 import PhotometricCritic, balanced_schedule
from src.models.chroma_constrained_film_v2 import ChromaField, apply_chroma_field, render_native, train_step


CONFIG_PATH = ROOT / "configs/chroma_null_domain_v3.json"
BOUND_CODE = ["scripts/run_chroma_null_domain_v3.py", "configs/chroma_null_domain_v3.json", "tests/test_chroma_null_domain_v3.py", "src/models/chroma_constrained_film_v2.py", "scripts/run_chroma_constrained_film_v2.py", "configs/chroma_constrained_film_v2.json", "src/models/semantic_film_adversarial_v1.py", "scripts/run_semantic_film_adversarial_v1.py", "src/color_match/research/local_reference_field_v1.py", "src/models/source_preparation_v1.py", "scripts/run_film_restoration_probe_v1.py"]


def load_config() -> tuple[dict, dict, dict, dict]:
    v3 = json.loads(CONFIG_PATH.read_text())
    v2_path = ROOT / v3["base_v2_manifest"]
    if sha(v2_path) != v3["base_v2_manifest_sha256"]:
        raise ValueError("Base v2 manifest changed")
    v2 = json.loads(v2_path.read_text())
    base = json.loads((ROOT / v2["config"]["base_manifest"]).read_text())
    if sha(ROOT / v2["config"]["base_manifest"]) != v2["config"]["base_manifest_sha256"]:
        raise ValueError("Base semantic manifest changed")
    report = json.loads((ROOT / v3["base_v2_report"]).read_text())
    if report["status"] != "COMPLETE_DEVELOPMENT_VISUAL_REVIEW_PENDING" or report["manifest_sha256"] != v3["base_v2_manifest_sha256"]:
        raise ValueError("Base v2 run incomplete or mismatched")
    if v3["seeds"] != v2["config"]["seeds"] or v3["arm"] not in v2["config"]["arms"]:
        raise ValueError("Seeds/arm must match v2")
    return {**v2["config"], **v3}, v2, base, report


def null_split(source_rows: list, count: int) -> tuple[list, list, np.ndarray]:
    ranked = sorted(range(len(source_rows)), key=lambda i: hashlib.sha256(source_rows[i]["id"].encode("utf-8")).hexdigest())
    references = sorted(ranked[:count])
    sources = sorted(ranked[count:])
    collections = np.full(len(source_rows), -1, dtype=np.int64)
    for rank, index in enumerate(ranked[:count]):
        collections[index] = rank % 3
    return sources, references, collections


def null_candidates(source_candidates: np.ndarray, sources: list, references: list) -> tuple[np.ndarray, np.ndarray]:
    return source_candidates[np.isin(source_candidates[:, 0], sources)], source_candidates[np.isin(source_candidates[:, 0], references)]


def panels(folder: Path, labels: list, crops: list) -> None:
    labels = ["original"] + labels
    panel = Image.new("RGB", (1440, 1140), "#181818")
    draw = ImageDraw.Draw(panel)
    for index, label in enumerate(labels):
        with Image.open(folder / (label + ".png")) as image:
            image.thumbnail((476, 350), Image.Resampling.LANCZOS)
            left, top = (index % 3) * 480, (index // 3) * 380
            panel.paste(image, (left + (480 - image.width) // 2, top + 25 + (350 - image.height) // 2))
            draw.text((left + 8, top + 5), label, fill="white")
    panel.save(folder / "blind_contact.jpg", quality=94)
    for crop_index, crop in enumerate(crops):
        panel = Image.new("RGB", (1200, 1272), "#181818")
        draw = ImageDraw.Draw(panel)
        for cell, label in enumerate(labels):
            with Image.open(folder / (label + ".png")) as image:
                tile = image.crop(tuple(crop))
                tile.thumbnail((396, 396), Image.Resampling.LANCZOS)
                left, top = cell % 3 * 400, cell // 3 * 424
                panel.paste(tile, (left + (400 - tile.width) // 2, top + 24 + (400 - tile.height) // 2))
                draw.text((left + 6, top + 5), label, fill="white")
        panel.save(folder / f"blind_crop{crop_index}.png")


def freeze(config: dict, v2: dict, base: dict, report: dict) -> None:
    output = ROOT / config["output"]
    if (output / "manifest.json").exists() or (output / "run").exists():
        raise FileExistsError("Frozen or trained attempt already exists")
    verify_base(base)
    snapshot = guard_gpu(config)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {"status": "PREPARING", "pid": os.getpid(), "gpu_snapshot": snapshot, "optimizer_updates": 0, "confirmation_assets_opened": 0}
    write_json(output / "preparation.json", receipt)
    try:
        configure_torch()
        tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "tests.test_chroma_null_domain_v3", "tests.test_chroma_constrained_film_v2", "-v"], cwd=ROOT, capture_output=True, text=True)
        (output / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Contract tests failed before schedule preparation")
        sources, references, collections = null_split(base["rows"]["source"], config["null_reference_count"])
        base_output = ROOT / base["config"]["output"]
        with np.load(base_output / "cluster_candidates.npz", allow_pickle=False) as arrays:
            source_candidates = arrays["source"]
        source_pool, reference_pool = null_candidates(source_candidates, sources, references)
        schedules, exposure = {}, {}
        for seed in config["seeds"]:
            schedule, support = balanced_schedule(source_pool, reference_pool, collections, config, seed)
            if np.isin(schedule["source"][..., 0], references).any() or not np.isin(schedule["reference"][..., 0], references).all():
                raise ValueError("Null split leaked")
            path = output / f"schedule_{seed}.npz"
            np.savez(path, **schedule)
            schedules[str(seed)] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
            exposure[str(seed)] = support
        cache = torch.load(ROOT / base["descriptor_cache"]["path"], weights_only=True, map_location="cpu")
        initial = {}
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            generator, critic = ChromaField(config, config["arm"]), PhotometricCritic(config, True)
            expected = report["models"][f"{config['arm']}_{seed}"]["initial_state_sha256"]
            initial[str(seed)] = {"generator": state_sha(generator), "critic": state_sha(critic)}
            if initial[str(seed)] != expected:
                raise ValueError("Initial state does not match recorded v2 arm-A initial state")
        schedule = load_schedule({"schedules": schedules}, config["seeds"][0])
        torch.manual_seed(config["seeds"][0])
        generator, critic = ChromaField(config, config["arm"]).cuda(), PhotometricCritic(config, True).cuda()
        go, co = torch.optim.Adam(generator.parameters(), lr=0.), torch.optim.Adam(critic.parameters(), lr=0.)
        before = state_sha(generator)
        source = cache["source"]
        for step in range(2):
            sr, rr = schedule["source"][step], schedule["reference"][step]
            si, ri = torch.as_tensor(sr[:, 0]), torch.as_tensor(rr[:, 0])
            terms = train_step(generator, critic, go, co, source["images"][si].cuda(), source["masks"][si].cuda(), source["boxes"][si].tolist(), source["features"][si].cuda(), torch.as_tensor(sr[:, 3:5]), source["images"][ri].cuda(), torch.as_tensor(rr[:, 3:5]), torch.as_tensor(schedule["clusters"][step], device="cuda"), step, config)
        torch.cuda.synchronize()
        if state_sha(generator) != before:
            raise RuntimeError("Zero-learning-rate canary changed parameters")
        split = {"pseudo_reference_indices": references, "pseudo_reference_ids": [base["rows"]["source"][i]["id"] for i in references], "pseudo_collections": {str(i): int(collections[i]) for i in references}, "collection_counts": np.bincount(collections[references], minlength=3).tolist(), "source_count": len(sources), "reference_candidate_rows": int(len(reference_pool)), "source_candidate_rows": int(len(source_pool))}
        write_json(output / "null_split.json", {"split": split, "sampling": exposure})
        receipt.update({"status": "PREPARED_PENDING_ROOT_TRAINING_ADMISSION", "elapsed_seconds": time.monotonic() - started, "initial_state_sha256": initial, "canary_terms_step2": terms, "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        variants = [{"model": f"null_{seed}", "render": render} for seed in config["seeds"] for render in ("local", "meanfield")] + [{"model": f"{config['arm']}_{seed}", "render": render} for seed in config["seeds"] for render in ("local", "meanfield")]
        np.random.default_rng(config["blind_review_seed"]).shuffle(variants)
        write_json(output / "blinded_review_map.json", {label: variant for label, variant in zip(config["blind_labels"], variants)})
        write_json(output / "preparation.json", receipt)
        manifest = {"schema": "chroma-null-domain-freeze-v3", "config": config, "config_sha256": sha(CONFIG_PATH), "code_sha256": {path: sha(ROOT / path) for path in BOUND_CODE}, "rows": base["rows"], "descriptor_cache": base["descriptor_cache"], "schedules": schedules, "evidence_sha256": {name: sha(output / name) for name in ("preparation.json", "tests.txt", "blinded_review_map.json", "null_split.json")}, "confirmation_assets_opened": 0}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": receipt["status"], "manifest_sha256": sha(output / "manifest.json"), "elapsed_seconds": receipt["elapsed_seconds"], "collection_counts": split["collection_counts"]}), flush=True)
    except Exception as exc:
        receipt.update({"status": "PREPARATION_FAILED_NO_TRAINING", "failure": repr(exc), "elapsed_seconds": time.monotonic() - started})
        write_json(output / "preparation.json", receipt)
        raise


def run(config: dict, v2: dict, base: dict, report_v2: dict, expected_sha: str) -> None:
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
    for name, digest in manifest["evidence_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError("Preparation evidence drift")
    for record in manifest["schedules"].values():
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Schedule drift")
    verify_base(base)
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
        cache = torch.load(ROOT / base["descriptor_cache"]["path"], weights_only=True, map_location="cpu")
        aliases = {(value["model"], value["render"]): key for key, value in json.loads((output / "blinded_review_map.json").read_text()).items()}
        for name in ("source", "challenge"):
            for key in ("images", "masks", "features"):
                cache[name][key] = cache[name][key].cuda()
        models = {}
        for seed in config["seeds"]:
            schedule = load_schedule(manifest, seed)
            name = f"null_{seed}"
            torch.manual_seed(seed)
            generator, critic = ChromaField(config, config["arm"]).cuda(), PhotometricCritic(config, True).cuda()
            initial = {"generator": state_sha(generator), "critic": state_sha(critic)}
            if initial != report_v2["models"][f"{config['arm']}_{seed}"]["initial_state_sha256"]:
                raise ValueError("Initial state mismatch with v2 arm A")
            go = torch.optim.Adam(generator.parameters(), lr=config["generator_lr"], betas=tuple(config["adam_betas"]))
            co = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], betas=tuple(config["adam_betas"]))
            trace_path = destination / f"{name}_trace.jsonl"
            tail = []
            with trace_path.open("x", encoding="utf-8") as stream:
                for step in range(config["steps"]):
                    guard(name)
                    sr, rr = schedule["source"][step], schedule["reference"][step]
                    si, ri = torch.as_tensor(sr[:, 0], device="cuda"), torch.as_tensor(rr[:, 0], device="cuda")
                    source = cache["source"]
                    terms = train_step(generator, critic, go, co, source["images"][si], source["masks"][si], source["boxes"][sr[:, 0]].tolist(), source["features"][si], torch.as_tensor(sr[:, 3:5]), source["images"][ri], torch.as_tensor(rr[:, 3:5]), torch.as_tensor(schedule["clusters"][step], device="cuda"), step, config)
                    stream.write(json.dumps({"step": step + 1, **terms}) + "\n")
                    if step >= config["steps"] - 100:
                        tail.append(terms)
                    if (step + 1) % 100 == 0:
                        stream.flush()
                        write_json(destination / "progress.json", {"model": name, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                        print(name, step + 1, {key: round(value, 5) for key, value in terms.items()}, flush=True)
            checkpoint = destination / (name + ".pt")
            torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(), "manifest_sha256": expected_sha}, checkpoint)
            report["models"][name] = {"initial_state_sha256": initial, "final_state_sha256": {"generator": state_sha(generator), "critic": state_sha(critic)}, "checkpoint_sha256": sha(checkpoint), "trace_sha256": sha(trace_path), "steps": config["steps"], "seed": seed, "last100_mean": {key: float(np.mean([t[key] for t in tail])) for key in ("sigma_95_fraction", "theta_95_fraction", "tint_95_fraction", "gamut_scaled_fraction", "adversarial", "real_score", "fake_score")}}
            models[name] = generator.eval().requires_grad_(False)
            del critic, go, co
            write_json(destination / "report.json", report)
        with torch.no_grad():
            gray = torch.linspace(.05, .95, 19, device="cuda")[None, None, :, None].expand(1, 3, -1, 1).contiguous()
            for i, row in enumerate(base["rows"]["challenge"]):
                guard("native render")
                native, decode = decode_asset(row)
                native = native.cuda()
                folder = destination / f"challenge_{i:02d}"
                folder.mkdir()
                record = {"source": {key: row[key] for key in ("id", "path", "sha256") if key in row}, "decode": decode, "original": save_output(folder, "original", native), "models": {}, "copied_v2": {}}
                v2_row = report_v2["rows"][i]
                if v2_row["original"]["float_sha256"] != record["original"]["float_sha256"]:
                    raise ValueError("Challenge decode differs from v2")
                challenge = cache["challenge"]
                for name, model in models.items():
                    nodes = model(challenge["images"][i:i + 1], challenge["masks"][i:i + 1], challenge["boxes"][i:i + 1].tolist(), challenge["features"][i:i + 1])
                    mean_nodes = nodes.mean((-2, -1), keepdim=True)
                    local, local_stats = render_native(native, nodes, config)
                    mean, mean_stats = render_native(native, mean_nodes.expand_as(nodes), config)
                    neutral, _ = apply_chroma_field(gray, mean_nodes.expand(-1, -1, 19, 1), config)
                    path = folder / (name + "_coefficients.npy")
                    np.save(path, nodes.cpu().numpy())
                    record["models"][name] = {"local": save_output(folder, aliases[name, "local"], local), "meanfield": save_output(folder, aliases[name, "meanfield"], mean), "coefficients_sha256": sha(path), "local_gamut": local_stats, "meanfield_gamut": mean_stats, "mean_absolute_change": float((local - native).abs().mean()), "meanfield_mean_absolute_change": float((mean - native).abs().mean()), "local_vs_meanfield_mae": float((local - mean).abs().mean()), "meanfield_nodes": mean_nodes.flatten().tolist(), "meanfield_neutral_ramp": {"input": gray[0, 0, :, 0].tolist(), "output_rgb": neutral[0, :, :, 0].T.tolist()}}
                    guard("native render " + name)
                for seed in config["seeds"]:
                    name = f"{config['arm']}_{seed}"
                    for render in ("local", "meanfield"):
                        source_record = v2_row["models"][name][render]
                        label = aliases[name, render]
                        for key, suffix in (("image", ".png"), ("float", ".npy")):
                            if sha(ROOT / source_record[key + "_path"]) != source_record[key + "_sha256"]:
                                raise ValueError("v2 render drift")
                            shutil.copyfile(ROOT / source_record[key + "_path"], folder / (label + suffix))
                        record["copied_v2"][f"{name}_{render}"] = {"label": label, "from": source_record}
                panels(folder, config["blind_labels"], row.get("existing_crop_boxes", []))
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
    config, v2, base, report = load_config()
    if args.freeze:
        freeze(config, v2, base, report)
    elif args.manifest_sha256:
        run(config, v2, base, report, args.manifest_sha256)
    else:
        parser.error("--run requires reviewed --manifest-sha256")
