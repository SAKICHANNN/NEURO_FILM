import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from scripts.run_film_restoration_probe_v1 import ROOT, sha, state_sha, write_json
from scripts.run_semantic_film_adversarial_v1 import configure_torch, decode_asset, guard_gpu, save_output
from src.models.semantic_film_adversarial_v1 import PhotometricCritic
from src.color_match.research.local_reference_field_v1 import interpolate_nodes
from src.models.chroma_constrained_film_v2 import ChromaField, apply_chroma_field, render_frames, render_native, train_step


CONFIG_PATH = ROOT / "configs/chroma_constrained_film_v2.json"
BOUND_CODE = ["src/models/chroma_constrained_film_v2.py", "scripts/run_chroma_constrained_film_v2.py", "configs/chroma_constrained_film_v2.json", "tests/test_chroma_constrained_film_v2.py", "src/models/semantic_film_adversarial_v1.py", "scripts/run_semantic_film_adversarial_v1.py", "configs/semantic_film_adversarial_v1.json", "src/color_match/research/local_reference_field_v1.py", "src/models/source_preparation_v1.py", "scripts/run_film_restoration_probe_v1.py"]
BLIND_LABELS = list("PQRSTUVW")
SATURATION_KEYS = ["sigma_95_fraction", "theta_95_fraction", "tint_95_fraction", "gamut_scaled_fraction"]


def load_config() -> tuple[dict, dict]:
    v2 = json.loads(CONFIG_PATH.read_text())
    base_path = ROOT / v2["base_manifest"]
    if sha(base_path) != v2["base_manifest_sha256"]:
        raise ValueError("Base semantic manifest changed")
    base = json.loads(base_path.read_text())
    overlap = sorted(set(base["config"]) & set(v2) - {"schema", "output", "seeds", "arms", "blind_review_seed", "steps", "batch_size", "max_seconds", "max_process_gib", "claim_ceiling", "decision"})
    if overlap:
        raise ValueError("Unreviewed override of base hyperparameters: " + ", ".join(overlap))
    for key in ("seeds", "steps", "batch_size", "max_seconds", "max_process_gib"):
        if v2[key] != base["config"][key]:
            raise ValueError("Schedule/budget must match the base experiment: " + key)
    return {**base["config"], **v2}, base


def verify_base(base: dict) -> None:
    for path, digest in base["code_sha256"].items():
        if path.startswith("configs/semantic") and sha(ROOT / path) != digest:
            raise ValueError("Base config drift: " + path)
    for record in [base["descriptor_cache"], *base["schedules"].values()]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Base cache/schedule drift")
    for group in base["rows"].values():
        for row in group:
            if sha(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("Input bytes drift: " + row["path"])


def load_schedule(base: dict, seed: int) -> dict:
    with np.load(ROOT / base["schedules"][str(seed)]["path"], allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def numeric_sweep(cache: dict, config: dict) -> dict:
    """Every cached source/reference/challenge frame in float32 at identity and at small and full random fields; any roundoff-contract failure aborts the freeze."""
    generator = torch.Generator().manual_seed(config["blind_review_seed"])
    frames, worst_low, worst_high = 0, 0., 0.
    for name in ("source", "reference", "challenge"):
        for index, (t, l, h, w) in enumerate(cache[name]["boxes"].tolist()):
            image = cache[name]["images"][index:index + 1, :, t:t + h, l:l + w].cuda()
            for arm in config["arms"]:
                bounds = torch.tensor(config["field_bounds"][arm])[None, :, None, None]
                for strength in (0., 1e-3, 2e-3, 1e-2, 1.):
                    nodes = (strength * bounds * (2 * torch.rand(1, 6, config["grid_size"], config["grid_size"], generator=generator) - 1)).cuda()
                    with torch.no_grad():
                        output, _ = apply_chroma_field(image, interpolate_nodes(nodes, (h, w), (h, w)), config)
                    worst_low, worst_high = min(worst_low, float(output.min())), max(worst_high, float(output.max()))
                    frames += 1
    if worst_low < 0 or worst_high > 1:
        raise FloatingPointError("Sweep output outside [0,1]")
    return {"frames": frames, "strengths_of_bound": [0., 1e-3, 2e-3, 1e-2, 1.], "dtype": "float32 input, float64 colour path", "output_min": worst_low, "output_max": worst_high, "failures": 0}


def freeze(config: dict, base: dict) -> None:
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
        tests = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-m", "unittest", "tests.test_chroma_constrained_film_v2", "-v"], cwd=ROOT, capture_output=True, text=True)
        (output / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Contract tests failed before canary")
        cache = torch.load(ROOT / base["descriptor_cache"]["path"], weights_only=True, map_location="cpu")
        receipt["numeric_sweep"] = numeric_sweep(cache, config)
        schedule = load_schedule(base, config["seeds"][0])
        source, reference = cache["source"], cache["reference"]
        canary = {}
        for arm in config["arms"]:
            torch.manual_seed(config["seeds"][0])
            generator, critic = ChromaField(config, arm).cuda(), PhotometricCritic(config, True).cuda()
            go = torch.optim.Adam(generator.parameters(), lr=0.)
            co = torch.optim.Adam(critic.parameters(), lr=0.)
            before = state_sha(generator)
            arm_started = time.monotonic()
            for step in range(2):
                sr, rr = schedule["source"][step], schedule["reference"][step]
                si, ri = torch.as_tensor(sr[:, 0]), torch.as_tensor(rr[:, 0])
                terms = train_step(generator, critic, go, co, source["images"][si].cuda(), source["masks"][si].cuda(), source["boxes"][si].tolist(), source["features"][si].cuda(), torch.as_tensor(sr[:, 3:5]), reference["images"][ri].cuda(), torch.as_tensor(rr[:, 3:5]), torch.as_tensor(schedule["clusters"][step], device="cuda"), step, config)
            torch.cuda.synchronize()
            if state_sha(generator) != before:
                raise RuntimeError("Zero-learning-rate canary changed parameters")
            canary[arm] = {"terms_step2": terms, "seconds_two_steps": time.monotonic() - arm_started, "generator_parameters": sum(p.numel() for p in generator.parameters()), "critic_parameters": sum(p.numel() for p in critic.parameters())}
            del generator, critic, go, co
        torch.cuda.empty_cache()
        receipt.update({"status": "PREPARED_PENDING_ROOT_TRAINING_ADMISSION", "elapsed_seconds": time.monotonic() - started, "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "canary": canary, "canary_note": "Two real-data train_step calls per arm with zero learning rate: full critic/R1/generator forward-backward including the custom gamut backward, parameters verified unchanged. Timing is cold and descriptive."})
        variants = [{"model": f"{arm}_{seed}", "render": render} for seed in config["seeds"] for arm in config["arms"] for render in ("local", "meanfield")]
        np.random.default_rng(config["blind_review_seed"]).shuffle(variants)
        write_json(output / "blinded_review_map.json", {label: variant for label, variant in zip(BLIND_LABELS, variants)})
        write_json(output / "preparation.json", receipt)
        manifest = {"schema": "chroma-constrained-film-freeze-v2", "config": config, "base_manifest_sha256": config["base_manifest_sha256"], "config_sha256": sha(CONFIG_PATH), "code_sha256": {path: sha(ROOT / path) for path in BOUND_CODE}, "rows": base["rows"], "descriptor_cache": base["descriptor_cache"], "schedules": base["schedules"], "evidence_sha256": {name: sha(output / name) for name in ("preparation.json", "tests.txt", "blinded_review_map.json", "root_gamut_geometry_preflight.json")}, "confirmation_assets_opened": 0}
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"status": receipt["status"], "manifest_sha256": sha(output / "manifest.json"), "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    except Exception as exc:
        receipt.update({"status": "PREPARATION_FAILED_NO_TRAINING", "failure": repr(exc), "elapsed_seconds": time.monotonic() - started})
        write_json(output / "preparation.json", receipt)
        raise


def blind_panels(folder: Path, crops: list) -> None:
    labels = ["original"] + BLIND_LABELS
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


def run(config: dict, base: dict, expected_sha: str) -> None:
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
        for name in ("source", "reference", "challenge"):
            for key in ("images", "masks", "features"):
                cache[name][key] = cache[name][key].cuda()
        models = {}
        for seed in config["seeds"]:
            schedule = load_schedule(base, seed)
            for arm in config["arms"]:
                name = f"{arm}_{seed}"
                torch.manual_seed(seed)
                generator, critic = ChromaField(config, arm).cuda(), PhotometricCritic(config, True).cuda()
                initial = {"generator": state_sha(generator), "critic": state_sha(critic)}
                go = torch.optim.Adam(generator.parameters(), lr=config["generator_lr"], betas=tuple(config["adam_betas"]))
                co = torch.optim.Adam(critic.parameters(), lr=config["critic_lr"], betas=tuple(config["adam_betas"]))
                trace_path = destination / f"{name}_trace.jsonl"
                tail = []
                with trace_path.open("x", encoding="utf-8") as stream:
                    for step in range(config["steps"]):
                        guard(name)
                        sr, rr = schedule["source"][step], schedule["reference"][step]
                        si, ri = torch.as_tensor(sr[:, 0], device="cuda"), torch.as_tensor(rr[:, 0], device="cuda")
                        source, reference = cache["source"], cache["reference"]
                        terms = train_step(generator, critic, go, co, source["images"][si], source["masks"][si], source["boxes"][sr[:, 0]].tolist(), source["features"][si], torch.as_tensor(sr[:, 3:5]), reference["images"][ri], torch.as_tensor(rr[:, 3:5]), torch.as_tensor(schedule["clusters"][step], device="cuda"), step, config)
                        stream.write(json.dumps({"step": step + 1, **terms}) + "\n")
                        if step >= config["steps"] - 100:
                            tail.append(terms)
                        if (step + 1) % 100 == 0:
                            stream.flush()
                            write_json(destination / "progress.json", {"model": name, "step": step + 1, "elapsed_seconds": time.monotonic() - started})
                            print(name, step + 1, {key: round(value, 5) for key, value in terms.items()}, flush=True)
                checkpoint = destination / (name + ".pt")
                torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(), "manifest_sha256": expected_sha}, checkpoint)
                report["models"][name] = {"initial_state_sha256": initial, "final_state_sha256": {"generator": state_sha(generator), "critic": state_sha(critic)}, "checkpoint_sha256": sha(checkpoint), "trace_sha256": sha(trace_path), "steps": config["steps"], "seed": seed, "arm": arm, "last100_mean": {key: float(np.mean([t[key] for t in tail])) for key in SATURATION_KEYS}}
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
                record = {"source": {key: row[key] for key in ("id", "path", "sha256") if key in row}, "decode": decode, "original": save_output(folder, "original", native), "models": {}}
                challenge = cache["challenge"]
                for name, model in models.items():
                    nodes = model(challenge["images"][i:i + 1], challenge["masks"][i:i + 1], challenge["boxes"][i:i + 1].tolist(), challenge["features"][i:i + 1])
                    mean_nodes = nodes.mean((-2, -1), keepdim=True)
                    local, local_stats = render_native(native, nodes, config)
                    mean, mean_stats = render_native(native, mean_nodes.expand_as(nodes), config)
                    neutral, _ = apply_chroma_field(gray, mean_nodes.expand(-1, -1, 19, 1), config)
                    path = folder / (name + "_coefficients.npy")
                    np.save(path, nodes.cpu().numpy())
                    bounds = model.bounds
                    record["models"][name] = {"local": save_output(folder, aliases[name, "local"], local), "meanfield": save_output(folder, aliases[name, "meanfield"], mean), "coefficients_sha256": sha(path), "local_gamut": local_stats, "meanfield_gamut": mean_stats, "mean_absolute_change": float((local - native).abs().mean()), "local_vs_meanfield_mae": float((local - mean).abs().mean()), "node_95_fraction_by_field": ((nodes.abs() / bounds) > .95).float().mean((0, 2, 3)).tolist(), "meanfield_nodes": mean_nodes.flatten().tolist(), "meanfield_neutral_ramp": {"input": gray[0, 0, :, 0].tolist(), "output_rgb": neutral[0, :, :, 0].T.tolist()}}
                    guard("native render " + name)
                blind_panels(folder, row.get("existing_crop_boxes", []))
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
    config, base = load_config()
    if args.freeze:
        freeze(config, base)
    elif args.manifest_sha256:
        run(config, base, args.manifest_sha256)
    else:
        parser.error("--run requires reviewed --manifest-sha256")
