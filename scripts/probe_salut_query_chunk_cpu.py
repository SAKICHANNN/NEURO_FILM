import ast
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def main():
    torch.set_num_threads(2)
    torch.manual_seed(48)
    root = Path(__file__).resolve().parents[1]
    source = root / "data/external/salut_v1/repo/SA-LUT/core/module/model.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "fae31ab74cf858e595593d85b8699cd9796e2da07bf00c3485f64f660e5e6e20"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ChannelAttention"
    )
    namespace = {"nn": nn}
    exec(  # noqa: S102 - hash-pinned local class avoids CUDA imports.
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    checkpoint = (
        root / "data/external/salut_v1/checkpoint/epoch=100-step=4127466.ckpt.state.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    tensors = {
        k: list(v.shape) for k, v in state.items() if isinstance(v, torch.Tensor)
    }
    net_source = source.with_name("net.py")
    assert (
        hashlib.sha256(net_source.read_bytes()).hexdigest()
        == "e560939d06fa2cab32279ef671635f5340f13a245ff2377814cb56adcb3867b1"
    )
    net_tree = ast.parse(net_source.read_text(encoding="utf-8"))
    vgg_node = next(
        n
        for n in net_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "vgg" for t in n.targets)
    )
    exec(  # noqa: S102 - hash-pinned VGG declaration only.
        compile(ast.Module(body=[vgg_node], type_ignores=[]), str(net_source), "exec"),
        namespace,
    )
    vgg_path = source.parents[2] / "ckpts/vgg_normalised.pth"
    vgg_state = torch.load(vgg_path, map_location="cpu", weights_only=True)
    namespace["vgg"].load_state_dict(vgg_state, strict=True)
    cases = []
    with torch.inference_mode():
        for dtype in (torch.float32, torch.float64):
            for chunk in (1, 7, 32, 129):
                q = torch.randn(2, 128, 16, dtype=dtype)
                k = torch.randn(2, 16, 128, dtype=dtype)
                v = torch.randn(2, 128, 16, dtype=dtype)
                temperature = 1.37
                dense = ((q @ k) / 4 * temperature).softmax(-1) @ v
                split = torch.cat(
                    [
                        ((x @ k) / 4 * temperature).softmax(-1) @ v
                        for x in q.split(chunk, dim=1)
                    ],
                    dim=1,
                )
                ca = namespace["ChannelAttention"](16).to(dtype=dtype).eval()
                dense_ca = ca(dense.transpose(1, 2).reshape(2, 16, 8, 16))
                split_ca = ca(split.transpose(1, 2).reshape(2, 16, 8, 16))
                atol = 2e-6 if dtype == torch.float32 else 1e-12
                torch.testing.assert_close(split, dense, atol=atol, rtol=atol)
                torch.testing.assert_close(split_ca, dense_ca, atol=atol, rtol=atol)
                cases.append(
                    {
                        "dtype": str(dtype),
                        "chunk": chunk,
                        "attention_max_abs": (dense - split).abs().max().item(),
                        "channel_attention_max_abs": (dense_ca - split_ca)
                        .abs()
                        .max()
                        .item(),
                    }
                )
    report = {
        "status": "microprobe_passed_not_full_network_parity",
        "device": "cpu",
        "seed": 48,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_tensor_shapes": tensors,
        "cases": cases,
        "claim_ceiling": "Small synthetic query partition test with complete K/V and channel attention after concatenation; no image, CUDA or full-network execution.",
    }
    report["vgg"] = {
        "strict_load": True,
        "key_count": len(vgg_state),
        "sha256": hashlib.sha256(vgg_path.read_bytes()).hexdigest(),
    }
    output = root / "outputs/ai_nonlut_science_bridge_v1/query_chunk_cpu.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "tensor_count": len(tensors), "cases": cases}
        )
    )


if __name__ == "__main__":
    main()
