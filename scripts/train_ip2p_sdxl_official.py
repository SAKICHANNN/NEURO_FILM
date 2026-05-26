#!/usr/bin/env python3
"""Launch official diffusers SDXL InstructPix2Pix fine-tuning.

The upstream script expands a regular SDXL UNet from 4 input channels to the
8-channel InstructPix2Pix layout, so the default base is SDXL Base rather than
an already-converted IP2P checkpoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    "v0.38.0/examples/instruct_pix2pix/train_instruct_pix2pix_sdxl.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch official SDXL-IP2P training on local film pseudo-pairs.")
    parser.add_argument("--style", default="all", help="Style name or all. Used for default paths only.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--pretrained-model", default=DEFAULT_MODEL)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--conditioning-dropout-prob", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--checkpointing-steps", type=int, default=250)
    parser.add_argument("--checkpoints-total-limit", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--official-script",
        type=Path,
        default=ROOT / ".cache" / "diffusers_examples" / "train_instruct_pix2pix_sdxl.py",
    )
    parser.add_argument("--official-script-url", default=DEFAULT_SCRIPT_URL)
    parser.add_argument("--refresh-official-script", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def ensure_script(path: Path, url: str, refresh: bool) -> Path:
    path = to_abs(path)
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official diffusers script: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def default_dataset_dir(style: str) -> Path:
    if style == "all":
        return ROOT / "data" / "ip2p_train" / "all" / "train"
    return ROOT / "data" / "ip2p_train" / style / "train"


def build_command(args: argparse.Namespace, script_path: Path, dataset_dir: Path, output_dir: Path) -> list[str]:
    launcher = [sys.executable] if args.direct else [sys.executable, "-m", "accelerate.commands.launch"]
    command = launcher + [
        str(script_path),
        "--pretrained_model_name_or_path",
        args.pretrained_model,
        "--train_data_dir",
        str(dataset_dir),
        "--original_image_column",
        "input_image",
        "--edited_image_column",
        "edited_image",
        "--edit_prompt_column",
        "edit_prompt",
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_train_steps",
        str(args.steps),
        "--learning_rate",
        str(args.lr),
        "--conditioning_dropout_prob",
        str(args.conditioning_dropout_prob),
        "--seed",
        str(args.seed),
        "--mixed_precision",
        args.mixed_precision,
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--checkpoints_total_limit",
        str(args.checkpoints_total_limit),
        "--output_dir",
        str(output_dir),
        "--gradient_checkpointing",
    ]
    if args.resume_from_checkpoint:
        command += ["--resume_from_checkpoint", args.resume_from_checkpoint]
    if args.enable_xformers:
        command.append("--enable_xformers_memory_efficient_attention")
    if args.use_8bit_adam:
        command.append("--use_8bit_adam")
    if args.allow_tf32:
        command.append("--allow_tf32")
    if args.center_crop:
        command.append("--center_crop")
    if args.random_flip:
        command.append("--random_flip")
    return command


def main() -> int:
    args = parse_args()
    dataset_dir = to_abs(args.dataset_dir or default_dataset_dir(args.style)).resolve()
    if not (dataset_dir / "metadata.jsonl").exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {dataset_dir}")
    output_dir = to_abs(args.output_dir or (ROOT / "outputs" / "ip2p_sdxl_finetune" / args.style)).resolve()
    script_path = ensure_script(args.official_script, args.official_script_url, args.refresh_official_script)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, script_path, dataset_dir, output_dir)

    print("Command:")
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
