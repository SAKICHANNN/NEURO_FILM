"""Acquire pinned feature-only artifacts and safely smoke their state dictionary."""

import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/ai_nlut_feature_source_v1.json").read_text())
    out = ROOT / cfg["destination"]
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise RuntimeError("Require new P-backed artifact directory")
    out.mkdir(parents=True)
    rows = []
    total = 0
    for f in cfg["files"]:
        url = f"https://raw.githubusercontent.com/semchan/NLUT/{cfg['revision']}/{f['path']}"
        with urlopen(url, timeout=60) as r:
            data = r.read(f["bytes"] + 1)
        if len(data) != f["bytes"]:
            raise ValueError("size mismatch")
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != f["git_blob"]:
            raise ValueError("Git object mismatch")
        total += len(data)
        if total > cfg["max_total_bytes"]:
            raise ValueError("budget exceeded")
        path = out / f["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        rows.append(dict(f, sha256=hashlib.sha256(data).hexdigest(), url=url))
        print(f["path"], len(data), flush=True)
    # All code/data blobs were checked before executing the reviewed net module.
    spec = importlib.util.spec_from_file_location(
        "pinned_nlut_features", out / "net.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(
        out / "models/vgg_normalised.pth", map_location="cpu", weights_only=True
    )
    module.vgg.load_state_dict(state, strict=True)
    net = module.Net(module.vgg).eval()
    torch.set_num_threads(4)
    x = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64).requires_grad_()
    features = net.encode_with_intermediate(x)
    assert all(torch.isfinite(y).all() for y in features)
    sum(y.square().mean() for y in features).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in net.parameters())
    report = {
        "config": cfg,
        "files": rows,
        "feature_shapes": [list(y.shape) for y in features],
        "finite_gradient": True,
        "frozen_encoder": True,
        "pixels_read": 0,
        "full_nlut_reproduction": False,
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("PASS", report["feature_shapes"], flush=True)


if __name__ == "__main__":
    main()
