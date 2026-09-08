import ast
import hashlib
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/ai_nonlut_science_bridge_v1/full_network_cpu/attempt-0002"
BASE = ROOT / "data/external/salut_v1/repo/SA-LUT"
PINS = {
    "core/module/model.py": "fae31ab74cf858e595593d85b8699cd9796e2da07bf00c3485f64f660e5e6e20",
    "core/module/net.py": "e560939d06fa2cab32279ef671635f5340f13a245ff2377814cb56adcb3867b1",
    "core/module/clut4d.py": "6d3368e9e49242223d94f51c449f22f20db209c9a057bb1a4c2cdc4333bc3576",
    "core/dataset/utils.py": "f97da5c4fbb8de1c37c442ebdb0c7d6d5b2188bdeffed0d640a91709767a67e1",
    "core/assets/Standard.cube": "f99223675b29933952da2153bdb3137dd749d12964d0753db85e47576ca4578d",
}


def save(report):
    (DEST / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def worker():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(48)
    started = time.monotonic()
    report = {
        "identity": "salut_pytorch_cpu_adaptation_small_synthetic_v1",
        "status": "preparing",
        "forward_attempts": 0,
        "completed_forwards": 0,
        "seed": 48,
        "input_shape": [1, 3, 64, 64],
        "context_target": [64, 64],
        "threads": 2,
        "timeout_seconds": 120,
        "torch": torch.__version__,
        "source_sha256": PINS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_ceiling": "Adapted CPU dense/chunk consistency only; reduced resolution, synthetic inputs, no official CUDA parity or photographic evidence.",
    }
    save(report)
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
                and context.max() <= 1
            )
            output = (1 - context) * tri(lut[:, 0], image[:, 1:]) + context * tri(
                lut[:, 1], image[:, 1:]
            )
            return lut, output

    namespace.update(TrilinearInterpolation=Tri, QuadrilinearInterpolation_4D=Quad)

    class SafeLoad(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and node.func.attr == "load"
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
    assert not (set(state) - set(model.state_dict()))
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
        "prior_attempt": "../report.json: strict load failed before any forward; retained unchanged",
    }
    state["tvmn.weight_c"] = buffer
    save(report)
    model.load_state_dict(state, strict=True)
    report["strict_model_keys"] = {
        key: list(value.shape) for key, value in state.items()
    }
    model.context_extractor.target_resolution = (64, 64)
    style, content = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    report["input_sha256"] = [
        hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in (style, content)
    ]

    def chunk_attention(q, k, v, channels, temperature):
        return torch.cat(
            [
                functional.softmax(
                    torch.bmm(part, k) / (channels**0.5) * temperature, dim=-1
                ).bmm(v)
                for part in q.split(64, dim=1)
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
                and statement.value.func.attr == "bmm"
            ):
                body.append(replacement)
                replaced += 1
                continue
        body.append(statement)
    assert replaced == 4
    forward.body = body
    forward.name = "chunk_forward"
    execute([forward], namespace, BASE / "core/module/model.py")
    outputs = []
    with torch.inference_mode():
        for mode in ("dense", "chunk64"):
            if mode == "chunk64":
                model.context_extractor.forward = types.MethodType(
                    namespace["chunk_forward"], model.context_extractor
                )
            report.update(status="running_" + mode, forward_attempts=len(outputs) + 1)
            save(report)
            outputs.append(model(style, content))
            report["completed_forwards"] = len(outputs)
            report[mode + "_elapsed_seconds"] = time.monotonic() - started
            save(report)
    report["comparisons"] = {}
    for name, dense, chunk in zip(("output", "lut", "context"), *outputs, strict=True):
        difference = abs(dense - chunk)
        report["comparisons"][name] = {
            "shape": list(dense.shape),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "finite": bool(torch.isfinite(dense).all() and torch.isfinite(chunk).all()),
            "dense_sha256": hashlib.sha256(dense.numpy().tobytes()).hexdigest(),
            "chunk_sha256": hashlib.sha256(chunk.numpy().tobytes()).hexdigest(),
        }
    report["standard_lut_output_sha256"] = standard_calls
    report["standard_lut_calls"] = len(standard_calls)
    report["elapsed_seconds"] = time.monotonic() - started
    report["passed"] = len(standard_calls) == 2 and all(
        item["finite"] and item["max_abs_error"] < 1e-5
        for item in report["comparisons"].values()
    )
    report["status"] = "passed" if report["passed"] else "failed_comparison"
    save(report)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST / "report.json").exists(), (
        "Existing attempt must not be rerun automatically"
    )
    with (DEST / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait(timeout=120)
            failure = "worker_failed" if code else None
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            failure = "timeout_120_seconds"
    report_path = DEST / "report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    if failure:
        report.update(status=failure, passed=False)
        save(report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "strict_model_keys"},
            indent=2,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
