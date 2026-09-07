"""Bounded official NLUT checkpoint acquisition; no code or pickle execution."""

import hashlib
import json
import subprocess
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 236254785
URL = "https://drive.usercontent.google.com/download?id=1tVO0GFa5GeuuXmvPHq9iYpfwSObrUMiv&export=download&confirm=t"


def inspect_state(checkpoint):
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Expected state_dict checkpoint")
    rows = []
    for name, value in checkpoint["state_dict"].items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Only named tensors admitted")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite model state")
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "elements": value.numel(),
            }
        )
    if not rows:
        raise ValueError("Empty model")
    return rows


def main():
    out = ROOT / "data/ai_models/nlut_pretrained_v1"
    if out.exists() or out.resolve().drive.upper() != "P:":
        raise RuntimeError("New P-backed destination required")
    out.mkdir(parents=True)
    path = out / "336999_style_lut.pth.partial"
    digest = hashlib.sha256()
    count = 0
    with requests.get(URL, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("HTML response is not checkpoint")
        with path.open("xb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                count += len(chunk)
                if count > EXPECTED_BYTES:
                    raise ValueError("Checkpoint byte budget exceeded")
                stream.write(chunk)
                digest.update(chunk)
    if count != EXPECTED_BYTES:
        raise ValueError(f"Wrong size: {count}")
    target = path.with_suffix("")
    path.rename(target)
    identity = {
        "url": URL,
        "bytes": count,
        "sha256": digest.hexdigest(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source": "https://github.com/semchan/NLUT",
        "scope": "internal pretrained comparison only; no redistribution clearance",
    }
    with (out / "source_lock.json").open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2)
    print("DOWNLOADED", count, digest.hexdigest(), flush=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    rows = inspect_state(checkpoint)
    report = {
        **identity,
        "tensor_rows": rows,
        "tensor_elements": sum(r["elements"] for r in rows),
        "top_level_keys": list(checkpoint),
        "weights_only": True,
        "inference": False,
        "pixels_read": 0,
    }
    with (out / "inspection.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("INSPECTED", len(rows), report["tensor_elements"], flush=True)


if __name__ == "__main__":
    main()
