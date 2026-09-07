"""Private NLUT reference adapter, not a production renderer.

Official classes are loaded from locally pinned semchan/NLUT MIT source.
Only redundant unsafe encoder loading is removed. Tensor contractions are
reassociated to avoid materializing all2048 full LUTs at once. No RGB generator.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


def checked_json(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Manifest hash mismatch")
    return json.loads(raw)


def cube_to_lut(cube):
    return torch.stack(
        (cube[:, 0].permute(0, 2, 3, 1), cube[:, 1].permute(0, 2, 1, 3), cube[:, 2]),
        dim=1,
    )


def basis_chunk(base, start, end):
    d, n, s, w = base.dim, end - start, base.s, base.w
    core = base.LUTs.reshape(s, base.num, 3, w)[:, start:end].reshape(s * n * 3, w)
    cube = base.s_Layers.mm(core.mm(base.w_Layers).reshape(s, n * 3 * d * d))
    cube = cube.reshape(d, n * 3, d * d).permute(1, 0, 2).reshape(n, 3, d, d, d)
    return cube_to_lut(cube)


def fused_lut(base, weights):
    d = base.dim
    core = base.LUTs.reshape(base.s, base.num, 3, base.w)
    combined = torch.einsum("bn,sncw->bscw", weights, core)
    combined = combined @ base.w_Layers
    cube = torch.einsum("ds,bscq->bdcq", base.s_Layers, combined)
    return cube_to_lut(cube.permute(0, 2, 1, 3).reshape(-1, 3, d, d, d))


def interpolate(luts, images):
    """Official RGB-fastest LUT layout and padded-bin convention; no clamp."""
    if images.ndim != 4 or images.shape[1] != 3 or luts.ndim != 5:
        raise ValueError("Expected NCHW image and NCBBB LUT")
    if luts.shape[0] not in (1, images.shape[0]):
        raise ValueError("LUT batch mismatch")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Input outside official SDR domain")
    d = luts.shape[-1]
    binsize = images.new_tensor(1.000001 / (d - 1))
    coords = images / binsize
    lower = coords.floor().long()
    frac = torch.fmod(images, binsize) / binsize
    r, g, b = lower.unbind(1)
    wr, wg, wb = frac.unbind(1)
    index = r + g * d + b * d * d
    flat = luts.reshape(luts.shape[0], 3, -1).expand(images.shape[0], -1, -1)
    out = torch.zeros_like(images)
    for dr, dg, db in (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ):
        weight = (
            (wr if dr else 1 - wr) * (wg if dg else 1 - wg) * (wb if db else 1 - wb)
        )
        ids = (index + dr + dg * d + db * d * d).flatten(1)
        values = flat.gather(2, ids[:, None].expand(-1, 3, -1)).reshape_as(images)
        out = out + values * weight[:, None]
    return out


class Interpolation(nn.Module):
    def forward(self, lut, image):
        return interpolate(lut, image)


def load_official(root: Path, cfg):
    source_root = root / "data/ai_models/nlut_pretrained_v1/source"
    sources = {}
    for name, size, blob in cfg["source_files"]:
        raw = (source_root / name).read_bytes()
        identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if len(raw) != size or identity != blob:
            raise ValueError("Official source identity mismatch")
        sources[name] = raw
    feature_path = root / cfg["feature_manifest"]
    feature = checked_json(feature_path, cfg["feature_sha256"])
    for row in feature["files"]:
        if (
            hashlib.sha256((feature_path.parent / row["path"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError("Feature source identity mismatch")
    spec = importlib.util.spec_from_file_location(
        "pinned_nlut_net", feature_path.parent / "net.py"
    )
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    tree = ast.parse(sources["nlut_models.py"])
    names = {
        "calc_mean_std",
        "AdaIN",
        "ConvLayer",
        "SplattingBlock2",
        "NLUTNet",
        "CLUT",
        "TVMN",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    removed = 0
    for node in nodes:
        if node.name == "NLUTNet":
            init = next(
                x
                for x in node.body
                if isinstance(x, ast.FunctionDef) and x.name == "__init__"
            )
            kept = []
            for statement in init.body:
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and ast.unparse(statement.value.func) == "vgg.load_state_dict"
                ):
                    removed += 1
                else:
                    kept.append(statement)
            init.body = kept
    if removed != 1 or len(nodes) != len(names):
        raise ValueError("Pinned AST adaptation contract changed")
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "transforms": transforms,
        "net": net,
        "cube_to_lut": cube_to_lut,
        "TrilinearInterpolation": Interpolation,
    }
    exec(  # noqa: S102 - exact Git-blob checked, reviewed classes only
        compile(
            ast.Module(body=nodes, type_ignores=[]), "pinned_nlut_models.py", "exec"
        ),
        namespace,
    )
    return SimpleNamespace(**namespace, OriginalCLUT=namespace["CLUT"])


def stream_combine(self, weight, TVMN):
    if TVMN is not None:
        raise ValueError("Use separately accumulated exact chunk regularizer")
    return fused_lut(self, weight), 0


def load_model(root, cfg):
    module = load_official(root, cfg)
    path = root / cfg["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    model = module.NLUTNet("2048+32+32", dim=33)
    state = torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.CLUTs.combine = MethodType(stream_combine, model.CLUTs)
    return model, module
