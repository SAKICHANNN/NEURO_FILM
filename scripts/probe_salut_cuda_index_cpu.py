import hashlib
import json
import re
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "data/external/salut_v1/repo/SA-LUT/quadrilinear_cpp/src/quadrilinear4d_kernel.cu"
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        source_hash
        == "a18d0b62f0abb3c5572d38c7adafbfe19ab1a64ca47b8fbe7fa07c54b74055ce"
    )
    model = root / "data/external/salut_v1/repo/SA-LUT/core/module/model.py"
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    assert (
        model_hash == "fae31ab74cf858e595593d85b8699cd9796e2da07bf00c3485f64f660e5e6e20"
    )
    code = source.read_text(encoding="utf-8").split("int QuadriLinearForwardLaucher")[0]
    dim = 17
    binsize = np.float32(1.000001 / (dim - 1))
    rng = np.random.default_rng(48)
    nodes = np.arange(dim, dtype=np.float32) * binsize
    edges = np.unique(
        np.clip(
            np.r_[
                0.0,
                1.0,
                nodes,
                np.nextafter(nodes, np.float32(0)),
                np.nextafter(nodes, np.float32(2)),
            ],
            0,
            1,
        ).astype(np.float32)
    )
    points = np.vstack(
        [
            rng.random((1024, 4), dtype=np.float32),
            np.array(
                [
                    [c, r, g, b]
                    for c in [0, 0.5, 1]
                    for r in [0, 1]
                    for g in [0, 1]
                    for b in [0, 1]
                ],
                np.float32,
            ),
            np.array([[0.37, x, x, x] for x in edges], np.float32),
        ]
    )
    lut = rng.random((3, 2, dim, dim, dim), dtype=np.float32)
    variables = {"dim": dim, "context_id": 0, "context_d": points[:, 0]}
    for name, values in zip("rgb", points[:, 1:].T, strict=True):
        variables[name + "_id"] = np.floor(values / binsize).astype(int)
        variables[name + "_d"] = np.fmod(values, binsize) / binsize
    expressions = {}
    for kind, pattern in (
        ("indices", r"int (id[01]{4})\s*=\s*(.*?);"),
        ("weights", r"float (w[01]{4})\s*=\s*(.*?);"),
    ):
        expressions[kind] = {
            name: eval(" ".join(expr.split()), {"__builtins__": {}}, variables)
            for name, expr in re.findall(pattern, code, re.DOTALL)
        }
    indices, weights = expressions["indices"], expressions["weights"]
    order = re.findall(
        r"(w[01]{4}) \* lut\[(id[01]{4})\]",
        code.split("output[index] =")[1].split(";")[0],
    )
    assert len(indices) == len(weights) == len(order) == 16
    output = np.zeros((3, len(points)), np.float32)
    identity_output = np.zeros_like(output)
    axes = np.indices((dim, dim, dim))
    identity = np.stack([axes[2], axes[1], axes[0]]) / (dim - 1)
    identity = np.repeat(identity[:, None], 2, axis=1).reshape(3, -1).astype(np.float32)
    for weight, index in order:
        output += weights[weight][None, :] * lut.reshape(3, -1)[:, indices[index]]
        identity_output += weights[weight][None, :] * identity[:, indices[index]]
    reference = []
    for context, red, green, blue in points.astype(np.float64):
        coordinates = np.array([red, green, blue]) / float(binsize)
        ri, gi, bi = np.floor(coordinates).astype(int)
        tr, tg, tb = coordinates - np.floor(coordinates)
        cube = lut[:, :, bi : bi + 2, gi : gi + 2, ri : ri + 2].astype(np.float64)
        value = cube[..., 0] * (1 - tr) + cube[..., 1] * tr
        value = value[..., 0] * (1 - tg) + value[..., 1] * tg
        value = value[..., 0] * (1 - tb) + value[..., 1] * tb
        reference.append(value[:, 0] * (1 - context) + value[:, 1] * context)
    difference = abs(output - np.array(reference).T)
    identity_difference = abs(identity_output - points[:, 1:].T / (16 * float(binsize)))
    report = {
        "source": str(source.relative_to(root)),
        "source_sha256": source_hash,
        "model_sha256": model_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": 48,
        "samples": len(points),
        "dtype": "float32 expressions / float64 independent reference",
        "storage_axes": ["output_channel", "context", "B", "G", "R"],
        "shape": [3, 2, 17, 17, 17],
        "channel_shift": 9826,
        "binsize_numerator": 1.000001,
        "binsize_float32": float(binsize),
        "index_min": int(min(i.min() for i in indices.values())),
        "index_max": int(max(i.max() for i in indices.values())),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "identity_max_abs_error": float(identity_difference.max()),
        "max_weight_sum_error": float(np.max(abs(sum(weights.values()) - 1))),
        "minimum_weight": float(min(w.min() for w in weights.values())),
        "rgb_one_lower_index": int(np.floor(np.float32(1) / binsize)),
        "rgb_negative_lower_index": int(np.floor(np.float32(-1e-7) / binsize)),
        "rgb_1point00001_lower_index": int(np.floor(np.float32(1.00001) / binsize)),
        "passed": bool(difference.max() < 1e-6 and identity_difference.max() < 1e-6),
        "prior_attempt_correction": "Initial ephemeral reference reduced the wrong axis; corrected to R then G then B then context. Initial 0.69359 difference was a reference bug, not a CUDA finding.",
        "claim_ceiling": "CPU arithmetic comparison only; no CUDA execution, compilation, full network, image inference or backward validation. Published CPU extension uses incompatible context/grid indexing and is not this reference.",
    }
    assert report["passed"] and report["index_min"] == 0 and report["index_max"] == 9825
    destination = root / "outputs/ai_nonlut_science_bridge_v1/cuda_index_cpu.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
