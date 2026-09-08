import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/salut_v1/repo/SA-LUT"
CONFIG = ROOT / "configs/salut_photo_development_v1.json"

PINS = {
    "core/module/model.py": "fae31ab74cf858e595593d85b8699cd9796e2da07bf00c3485f64f660e5e6e20",
    "core/module/net.py": "e560939d06fa2cab32279ef671635f5340f13a245ff2377814cb56adcb3867b1",
    "core/module/clut4d.py": "6d3368e9e49242223d94f51c449f22f20db209c9a057bb1a4c2cdc4333bc3576",
    "core/dataset/utils.py": "f97da5c4fbb8de1c37c442ebdb0c7d6d5b2188bdeffed0d640a91709767a67e1",
    "core/assets/Standard.cube": "f99223675b29933952da2153bdb3137dd749d12964d0753db85e47576ca4578d",
}


def build_model():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(48)
    report = {
        "identity": "salut_pytorch_cpu_photo_development_v1",
        "seed": 48,
        "input_shape": [1, 3, 512, 512],
        "context_target": [512, 512],
        "query_chunk": 512,
        "threads": 2,
        "timeout_seconds": 600,
        "torch": torch.__version__,
        "source_sha256": PINS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_ceiling": "CPU photographic development with adapted interpolation and full-key query chunking; not official CUDA parity or independent validation.",
    }
    for relative, expected in PINS.items():
        assert hashlib.sha256((BASE / relative).read_bytes()).hexdigest() == expected

    def execute(nodes, namespace, path):
        module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - selected reviewed hash-pinned AST only.

    def tree(relative):
        return ast.parse((BASE / relative).read_text(encoding="utf-8")).body

    net = {"torch": torch, "nn": nn}
    execute(
        [
            node
            for node in tree("core/module/net.py")
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Assign))
        ],
        net,
        BASE / "core/module/net.py",
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": functional,
        "np": np,
        "math": math,
        "net": types.SimpleNamespace(**net),
        "__file__": str(BASE / "core/module/model.py"),
    }
    execute(
        [
            node
            for node in tree("core/module/clut4d.py")
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ],
        namespace,
        BASE / "core/module/clut4d.py",
    )
    execute(
        [
            node
            for node in tree("core/dataset/utils.py")
            if isinstance(node, ast.FunctionDef) and node.name == "read_3dlut_from_file"
        ],
        namespace,
        BASE / "core/dataset/utils.py",
    )
    vgg_path = BASE / "ckpts/vgg_normalised.pth"
    report["vgg_sha256"] = hashlib.sha256(vgg_path.read_bytes()).hexdigest()
    assert (
        report["vgg_sha256"]
        == "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a"
    )
    vgg_state = torch.load(vgg_path, map_location="cpu", weights_only=True)
    namespace["safe_vgg"] = lambda: vgg_state

    def tri(lut, image):
        assert image.shape[0] == 1 and torch.isfinite(image).all()
        assert image.min() >= 0 and image.max() <= 1
        dim = lut.shape[-1]
        step = torch.tensor(1.000001 / (dim - 1), dtype=image.dtype)
        values = image[0].reshape(3, -1)
        lower = torch.floor(values / step).long()
        fraction = torch.fmod(values, step) / step
        output = torch.zeros_like(values)
        for b in (0, 1):
            for g in (0, 1):
                for r in (0, 1):
                    weight = torch.ones(values.shape[1])
                    for axis, upper in enumerate((r, g, b)):
                        weight *= fraction[axis] if upper else 1 - fraction[axis]
                    output += lut[:, lower[2] + b, lower[1] + g, lower[0] + r] * weight
        return output.reshape_as(image)

    standard_calls = []

    class Tri(nn.Module):
        def forward(self, lut, image):
            result = tri(lut[0], image)
            standard_calls.append(hashlib.sha256(result.numpy().tobytes()).hexdigest())
            return result

    class Quad(nn.Module):
        def forward(self, lut, image):
            context = image[:, :1]
            assert (
                torch.isfinite(context).all()
                and context.min() >= 0
                and (context.max() <= 1)
            )
            output = (1 - context) * tri(lut[:, 0], image[:, 1:]) + context * tri(
                lut[:, 1], image[:, 1:]
            )
            return (lut, output)

    namespace.update(TrilinearInterpolation=Tri, QuadrilinearInterpolation_4D=Quad)

    class SafeLoad(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and (node.func.value.id == "torch")
                and (node.func.attr == "load")
            ):
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="safe_vgg", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                    node,
                )
            return node

    names = {
        "adaptive_instance_normalization",
        "ConvLayer",
        "SplattingBlock2",
        "CrossAttentionContextGenerator",
        "ResidualBlock",
        "ChannelAttention",
        "VLog2StyleNet4D",
    }
    nodes = [
        SafeLoad().visit(node)
        for node in tree("core/module/model.py")
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    execute(nodes, namespace, BASE / "core/module/model.py")
    checkpoint = (
        ROOT / "data/external/salut_v1/checkpoint/epoch=100-step=4127466.ckpt.state.pt"
    )
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (
        report["checkpoint_sha256"]
        == "5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {
        key.removeprefix("vlog2stylenet."): value
        for key, value in state.items()
        if key.startswith("vlog2stylenet.")
    }
    assert len(state) == 88
    model = namespace["VLog2StyleNet4D"](
        dim=17, num_basis=state["CLUTs.LUTs"].shape[0]
    ).eval()
    missing = set(model.state_dict()) - set(state)
    assert missing == {"tvmn.weight_c"}
    assert not set(state) - set(model.state_dict())
    assert "tvmn.weight_c" not in dict(model.named_parameters())
    buffer = dict(model.named_buffers())["tvmn.weight_c"].clone()
    assert list(buffer.shape) == [3, 1, 17, 17, 17] and (buffer == 2).all()
    report["checkpoint_model_key_count"] = len(state)
    report["deterministic_buffer_adaptation"] = {
        "key": "tvmn.weight_c",
        "shape": list(buffer.shape),
        "constant_value": 2,
        "sha256": hashlib.sha256(buffer.numpy().tobytes()).hexdigest(),
        "source": "hash-pinned TV_4D constructor clut4d.py:122-127",
        "effect": "Weights context TV regularizer only; evaluation output/LUT/context do not consume tvmn.",
        "prior_attempt": "outputs/ai_nonlut_science_bridge_v1/full_network_cpu/report.json: strict load failed before any forward; retained unchanged",
    }
    state["tvmn.weight_c"] = buffer
    model.load_state_dict(state, strict=True)
    report["strict_model_keys"] = {
        key: list(value.shape) for key, value in state.items()
    }
    model.context_extractor.target_resolution = (512, 512)

    def chunk_attention(q, k, v, channels, temperature):
        return torch.cat(
            [
                functional.softmax(
                    torch.bmm(part, k) / channels**0.5 * temperature, dim=-1
                ).bmm(v)
                for part in q.split(512, dim=1)
            ],
            dim=1,
        )

    namespace["chunk_attention"] = chunk_attention
    context_class = next(
        node for node in nodes if node.name == "CrossAttentionContextGenerator"
    )
    forward = next(
        node
        for node in context_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    replacement = ast.parse(
        "aggregated = chunk_attention(Q_flat, K_flat, V_flat, C_attn, self.attn_temperature)"
    ).body[0]
    body = []
    replaced = 0
    for statement in forward.body:
        if isinstance(statement, ast.Assign) and isinstance(
            statement.targets[0], ast.Name
        ):
            name = statement.targets[0].id
            if name in {"attn_scores", "attn_weights"}:
                replaced += 1
                continue
            if (
                name == "aggregated"
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and (statement.value.func.attr == "bmm")
            ):
                body.append(replacement)
                replaced += 1
                continue
        body.append(statement)
    assert replaced == 4
    forward.body = body
    forward.name = "chunk_forward"
    execute([forward], namespace, BASE / "core/module/model.py")
    model.context_extractor.forward = types.MethodType(
        namespace["chunk_forward"], model.context_extractor
    )
    return (model, Quad(), report, standard_calls)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight():
    import numpy as np
    import psutil
    from PIL import Image

    config = read_json(CONFIG)
    assert config["source_indices"] == [0, 1, 2]
    assert (
        config["seed"],
        config["size"],
        config["query_chunk"],
        config["threads"],
    ) == (48, 512, 512, 2)
    assert config["budget"] == {
        "maximum_forwards": 3,
        "maximum_worker_seconds": 600,
        "maximum_rss_bytes": 4294967296,
        "poll_seconds": 0.1,
    }
    for key in ("baseline_config", "ncc_report", "adapter_probe", "official_cli"):
        assert digest(ROOT / config[key]) == config[key + "_sha256"], key
    for relative, expected in PINS.items():
        assert digest(BASE / relative) == expected, relative
    for relative, expected in (
        (
            "ckpts/vgg_normalised.pth",
            "804ca2835ecf7539f0cd2a7ac3c18ce81e6f8468969ae7117ac0c148d286bb4a",
        ),
        (
            "../checkpoint/epoch=100-step=4127466.ckpt.state.pt",
            "5ad3685cf6bbf14230d71580b9b93f03e76bf6d7dfaceecd0a3527341cc92f2b",
        ),
    ):
        path = (
            BASE / relative
            if not relative.startswith("../")
            else ROOT
            / "data/external/salut_v1/checkpoint/epoch=100-step=4127466.ckpt.state.pt"
        )
        assert digest(path) == expected
    baseline = read_json(ROOT / config["baseline_config"])
    assert (
        digest(ROOT / baseline["source_manifest"]) == baseline["source_manifest_sha256"]
    )
    manifest = read_json(ROOT / baseline["source_manifest"])
    reference = baseline["reference"]
    assert digest(ROOT / reference["path"]) == reference["sha256"]
    assert digest(ROOT / reference["manifest"]) == reference["manifest_sha256"]
    assert Path(reference["path"]).name == "18.jpg"
    ncc = read_json(ROOT / config["ncc_report"])
    rows = []
    for index in config["source_indices"]:
        source = Path(manifest[index]["before"])
        assert source.resolve().is_relative_to((ROOT / "outputs").resolve())
        expected = baseline["sources"][index]
        assert (
            source.stat().st_size == expected["bytes"]
            and digest(source) == expected["sha256"]
        )
        old = next(row for row in ncc["rows"] if row["source_index"] == index)
        assert (
            old["source_sha256"] == expected["sha256"]
            and old["reference_sha256"] == reference["sha256"]
        )
        arms = {}
        for name, old_name in (
            ("original", "identity"),
            ("simple", "simple"),
            ("old_ncc", "learned_no_precorrection"),
        ):
            entry = old["arms"][old_name]
            path = (ROOT / config["ncc_report"]).parent / entry["path"]
            assert digest(path) == entry["sha256"]
            with Image.open(path) as image:
                assert list(image.size) == entry["size"]
            arms[name] = {
                "path": str(path),
                "sha256": entry["sha256"],
                "size": entry["size"],
            }
        with (
            Image.open(source) as image,
            Image.open(arms["original"]["path"]) as original,
        ):
            assert np.array_equal(
                np.asarray(image.convert("RGB")), np.asarray(original.convert("RGB"))
            )
        rows.append(
            {
                "source_index": index,
                "source_path": str(source),
                "source_sha256": expected["sha256"],
                "baseline_arms": arms,
            }
        )
    return config, {
        "status": "CPU_PHOTO_DRYRUN_READY",
        "config_sha256": digest(CONFIG),
        "runner_sha256": digest(Path(__file__)),
        "reference": reference,
        "queue": rows,
        "psutil_version": psutil.__version__,
        "new_forwards": 0,
        "available_memory_bytes": psutil.virtual_memory().available,
    }


def photo_worker(attempt):
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional
    from torchvision.transforms.functional import to_pil_image, to_tensor

    config, checks = preflight()
    frozen = read_json(attempt / "lock.json")
    assert (
        frozen["config_sha256"] == checks["config_sha256"]
        and frozen["runner_sha256"] == checks["runner_sha256"]
    )
    report = {
        "identity": config["identity"],
        "status": "loading",
        "new_forwards": 0,
        "rows": [],
        "config_sha256": checks["config_sha256"],
        "runner_sha256": checks["runner_sha256"],
        "reference_sha256": checks["reference"]["sha256"],
        "scope": config["scope"],
    }
    report_path = attempt / "report.json"
    for item in checks["queue"]:
        row = {
            "source_index": item["source_index"],
            "source_sha256": item["source_sha256"],
            "status": "queued",
            "arms": {},
        }
        for name, entry in item["baseline_arms"].items():
            destination = attempt / (
                str(item["source_index"]).zfill(2) + "_" + name + ".png"
            )
            shutil.copyfile(entry["path"], destination)
            assert digest(destination) == entry["sha256"]
            row["arms"][name] = {
                "path": destination.name,
                "sha256": digest(destination),
                "size": entry["size"],
            }
        report["rows"].append(row)
    save_report(report_path, report)
    model, quad, loading, standard_calls = build_model()
    report["model_loading"] = loading
    report["model_loading"].update(
        identity=config["identity"],
        input_shape=[1, 3, 512, 512],
        context_target=[512, 512],
        timeout_seconds=600,
        claim_ceiling=config["scope"],
    )
    started = time.monotonic()
    with Image.open(ROOT / checks["reference"]["path"]) as image:
        reference = to_tensor(image.convert("RGB")).unsqueeze(0)
    reference = functional.interpolate(
        reference, size=(512, 512), mode="bilinear", align_corners=True
    )

    def save_tensor(path, tensor):
        assert torch.isfinite(tensor).all()
        to_pil_image(tensor.squeeze(0).clamp(0, 1)).save(path)
        with Image.open(path) as image:
            return {"path": path.name, "sha256": digest(path), "size": list(image.size)}

    def save_array(path, tensor):
        np.save(path, tensor.numpy(), allow_pickle=False)
        return {
            "path": path.name,
            "sha256": digest(path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.numpy().dtype),
        }

    with torch.inference_mode():
        for item, row in zip(checks["queue"], report["rows"], strict=True):
            assert report["new_forwards"] < config["budget"]["maximum_forwards"]
            with Image.open(item["source_path"]) as image:
                full = to_tensor(image.convert("RGB")).unsqueeze(0)
            small = functional.interpolate(
                full, size=(512, 512), mode="bilinear", align_corners=True
            )
            report["new_forwards"] += 1
            row["status"] = "forward_started"
            report["status"] = "running"
            save_report(report_path, report)
            forward_start = time.monotonic()
            _, lut, context = model(reference, small)
            row["forward_seconds"] = time.monotonic() - forward_start
            context_full = functional.interpolate(
                context, size=full.shape[-2:], mode="bilinear", align_corners=True
            )
            mean = context_full.mean()
            row["context_mean"] = float(mean)
            row["context_minmax"] = [
                float(context_full.min()),
                float(context_full.max()),
            ]
            prefix = str(item["source_index"]).zfill(2)
            row["fused_lut"] = save_array(attempt / (prefix + "_fused_lut.npy"), lut)
            row["context512"] = save_array(
                attempt / (prefix + "_context512.npy"), context
            )
            row["context_full"] = save_array(
                attempt / (prefix + "_context_full.npy"), context_full
            )
            row["context_preview"] = save_tensor(
                attempt / (prefix + "_context.png"), context_full
            )
            _, styled = quad(lut[0], torch.cat([context_full, full], dim=1))
            row["arms"]["salut"] = save_tensor(
                attempt / (prefix + "_salut.png"), styled
            )
            _, g0 = quad(
                lut[0],
                torch.cat([torch.full_like(context_full, mean.item()), full], dim=1),
            )
            row["arms"]["g0"] = save_tensor(attempt / (prefix + "_g0.png"), g0)
            row["spatial_minus_g0_max_abs"] = float(abs(styled - g0).max())
            row["spatial_minus_g0_mean_abs"] = float(abs(styled - g0).mean())
            row["status"] = "complete"
            report["elapsed_after_loading_seconds"] = time.monotonic() - started
            report["standard_lut_calls"] = len(standard_calls)
            report["standard_lut_output_sha256"] = list(standard_calls)
            save_report(report_path, report)
            del full, small, lut, context, context_full, styled, g0
    report["status"] = "COMPLETE_NOT_PROMOTED"
    save_report(report_path, report)


def launch(config, checks):
    import psutil

    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / "attempt-0001"
    attempt.mkdir(exist_ok=False)
    save_report(attempt / "lock.json", checks)
    shutil.copyfile(CONFIG, attempt / "config.json")
    started = time.monotonic()
    peak = 0
    reason = None
    with (attempt / "worker.log").open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                str(attempt),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        observed = psutil.Process(process.pid)
        owned = {observed.pid: observed.create_time()}
        process_peaks = {}
        sampled_tree_peak = 0
        while process.poll() is None:
            try:
                for child in observed.children(recursive=True):
                    owned[child.pid] = child.create_time()
                tree_rss = 0
                for pid, created in list(owned.items()):
                    try:
                        member = psutil.Process(pid)
                        if member.create_time() != created:
                            continue
                        memory = member.memory_info()
                        tree_rss += memory.rss
                        process_peaks[pid] = max(
                            process_peaks.get(pid, 0),
                            memory.rss,
                            getattr(memory, "peak_wset", 0),
                        )
                    except psutil.NoSuchProcess:
                        continue
                sampled_tree_peak = max(sampled_tree_peak, tree_rss)
                peak = max(peak, sum(process_peaks.values()))
            except psutil.NoSuchProcess:
                break
            except psutil.AccessDenied:
                reason = "STOPPED_RESOURCE_STATE_UNKNOWN"
            if peak > config["budget"]["maximum_rss_bytes"]:
                reason = "STOPPED_RSS_LIMIT"
            elif (
                time.monotonic() - started > config["budget"]["maximum_worker_seconds"]
            ):
                reason = "STOPPED_TIME_LIMIT"
            if reason:
                for pid, created in reversed(list(owned.items())):
                    try:
                        member = psutil.Process(pid)
                        if member.create_time() == created:
                            member.kill()
                    except psutil.NoSuchProcess:
                        continue
                break
            time.sleep(config["budget"]["poll_seconds"])
        code = process.wait()
    report_path = attempt / "report.json"
    report = (
        read_json(report_path)
        if report_path.exists()
        else {"rows": [], "new_forwards": 0}
    )
    report["supervisor"] = {
        "exit_code": code,
        "peak_observed_rss_bytes": peak,
        "sampled_tree_rss_peak_bytes": sampled_tree_peak,
        "owned_pid_create_times": owned,
        "individual_peak_rss_bytes": process_peaks,
        "worker_wall_seconds": time.monotonic() - started,
        "rss_limitation": "0.1-second worker-tree sampling including venv redirector descendants, PID/create-time bound. Budget uses conservative sum of per-process historical peaks (Windows peak working set where available), not simultaneous allocator reservation. Only observed owned tree is terminated.",
    }
    if reason or code:
        report["status"] = reason or "WORKER_FAILED"
    save_report(report_path, report)
    print(
        json.dumps(
            {
                "attempt": str(attempt),
                "status": report["status"],
                "new_forwards": report["new_forwards"],
                "supervisor": report["supervisor"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "COMPLETE_NOT_PROMOTED" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        photo_worker(args.worker)
        return 0
    config, checks = preflight()
    if args.run:
        return launch(config, checks)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
