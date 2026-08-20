"""Blind EditReward score lock for U5.R2EDITREWARD0.

The scorer deliberately has no argument for either private mapping or direct
preference results.  Those inputs belong to a later aggregation process after
two complete score locks satisfy the frozen mechanics gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

CONTRACT_SCHEMA = (
    "neuro-film.u5-r2editreward0-consumed-source-aware-preference-contract.v1"
)
SCORE_SCHEMA = "neuro-film.u5-r2editreward0-blind-score-lock.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _validate_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.seek(0)
        if int(getattr(image, "n_frames", 1)) != 1:
            raise ValueError(f"multi-frame image is forbidden: {path}")
        width, height = image.size
        mode = image.mode
        image.verify()
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image geometry: {path}")
    return width, height, mode


@dataclass(frozen=True)
class ScoreRow:
    dataset: str
    source_id: str
    presentation_id: str
    role: str
    label: str
    source_path: Path
    candidate_path: Path
    source_sha256: str
    candidate_sha256: str
    source_geometry: tuple[int, int]
    candidate_geometry: tuple[int, int]

    @property
    def row_id(self) -> str:
        return f"{self.dataset}:{self.presentation_id}:{self.label}"


def _presentation_index(presentation_id: str) -> int:
    prefix, number = presentation_id.split("-", maxsplit=1)
    if prefix != "R1":
        raise ValueError(f"only frozen round 1 is allowed: {presentation_id}")
    index = int(number) - 1
    if not 0 <= index < 12:
        raise ValueError(f"presentation index out of range: {presentation_id}")
    return index


def _round_one_rows(public: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in public.get("rows", []) if row.get("round") == 1]
    rows.sort(key=lambda row: row["presentation_id"])
    if len(rows) != 12:
        raise ValueError(f"expected 12 round-one rows, found {len(rows)}")
    if [_presentation_index(row["presentation_id"]) for row in rows] != list(range(12)):
        raise ValueError("round-one presentation order is not exact")
    return rows


def _source_facts(dataset: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if dataset == "p401":
        facts = manifest.get("input_facts", [])
        identifiers = [fact.get("id") for fact in facts]
        expected = [f"p399_{index:02d}" for index in range(12)]
    else:
        facts = manifest.get("rows", [])
        identifiers = [fact.get("source_id") for fact in facts]
        expected = [f"p402_{index:02d}" for index in range(12)]
    if identifiers != expected:
        raise ValueError(f"{dataset} source order mismatch")
    return facts


def build_inventory(
    contract_path: Path,
    p401_public_root: Path,
    p401_source_root: Path,
    p402_public_root: Path,
    p402_source_root: Path,
) -> tuple[dict[str, Any], list[ScoreRow]]:
    contract = _load_json(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("unexpected U5.R2EDITREWARD0 contract schema")

    rows: list[ScoreRow] = []
    dataset_inputs = {
        "p401": (p401_public_root, p401_source_root),
        "p402": (p402_public_root, p402_source_root),
    }
    for dataset, (public_root, source_root) in dataset_inputs.items():
        frozen = contract["consumed_inputs"][dataset]
        public_path = public_root / "public_manifest.json"
        source_manifest_path = source_root / "manifest.json"
        if dataset == "p401":
            source_manifest_path = source_root.parent / "manifest.json"
        _validate_hash(
            public_path, frozen["public_manifest_sha256"], f"{dataset} public manifest"
        )
        _validate_hash(
            source_manifest_path,
            frozen["source_manifest_sha256"],
            f"{dataset} source manifest",
        )
        public = _load_json(public_path)
        source_manifest = _load_json(source_manifest_path)
        if source_manifest.get("manifest_id") != frozen["source_manifest_id"]:
            raise ValueError(f"{dataset} source manifest identity mismatch")
        public_rows = _round_one_rows(public)
        facts = _source_facts(dataset, source_manifest)

        for public_row in public_rows:
            index = _presentation_index(public_row["presentation_id"])
            fact = facts[index]
            source_id = fact.get("id") if dataset == "p401" else fact.get("source_id")
            source_path = source_root / fact["file_name"]
            expected_source_sha = fact.get("file_sha256") or fact.get("byte_sha256")
            _validate_hash(
                source_path, expected_source_sha, f"{dataset} {source_id} source"
            )
            source_width, source_height, _ = _validate_image(source_path)
            labels = public_row.get("labels", [])
            assets = public_row.get("assets", {})
            if labels != list(assets) or len(labels) not in {2, 4}:
                raise ValueError(f"{dataset} label/asset order mismatch")
            for label in labels:
                candidate_path = public_root / assets[label]
                candidate_sha = sha256_file(candidate_path)
                candidate_width, candidate_height, _ = _validate_image(candidate_path)
                rows.append(
                    ScoreRow(
                        dataset=dataset,
                        source_id=source_id,
                        presentation_id=public_row["presentation_id"],
                        role=public_row["role"],
                        label=label,
                        source_path=source_path,
                        candidate_path=candidate_path,
                        source_sha256=expected_source_sha,
                        candidate_sha256=candidate_sha,
                        source_geometry=(source_width, source_height),
                        candidate_geometry=(candidate_width, candidate_height),
                    )
                )

    if len(rows) != 72 or len({row.row_id for row in rows}) != 72:
        raise ValueError(
            "blind inventory must contain exactly 72 unique candidate rows"
        )
    return contract, rows


def wrong_source_controls(rows: Iterable[ScoreRow]) -> list[tuple[ScoreRow, ScoreRow]]:
    grouped: dict[str, list[ScoreRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset, []).append(row)
    controls: list[tuple[ScoreRow, ScoreRow]] = []
    for dataset in ("p401", "p402"):
        dataset_rows = grouped[dataset]
        by_source: dict[str, list[ScoreRow]] = {}
        for row in dataset_rows:
            by_source.setdefault(row.source_id, []).append(row)
        source_ids = sorted(by_source)
        if len(source_ids) != 12:
            raise ValueError(f"{dataset} must contain 12 sources")
        for index, source_id in enumerate(source_ids):
            candidates = sorted(by_source[source_id], key=lambda row: row.label)
            selector = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[0], 16)
            selected = candidates[selector % len(candidates)]
            wrong_source_id = source_ids[(index + 1) % len(source_ids)]
            wrong_source_row = min(
                by_source[wrong_source_id], key=lambda row: row.label
            )
            controls.append((selected, wrong_source_row))
    if len(controls) != 24:
        raise ValueError("wrong-source control must contain exactly 24 rows")
    return controls


def _prepend_runtime_paths(
    python_deps: Path, transformers_source: Path, official_source: Path
) -> None:
    for path in (official_source, python_deps, transformers_source):
        resolved = str(path.resolve())
        if resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)


class EditRewardRuntime:
    def __init__(
        self,
        *,
        contract: dict[str, Any],
        python_deps: Path,
        transformers_source: Path,
        official_source: Path,
        model_dir: Path,
        processor_dir: Path,
        offload_dir: Path,
    ) -> None:
        _prepend_runtime_paths(python_deps, transformers_source, official_source)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor

        if transformers.__version__ != "4.57.0":
            raise RuntimeError(
                f"unexpected transformers version: {transformers.__version__}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen local mechanics path")

        external = contract["external_asset"]
        transformers_commit = subprocess.check_output(
            ["git", "-C", str(transformers_source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if transformers_commit != contract["resource_execution"]["transformers_commit"]:
            raise ValueError("transformers source commit mismatch")
        editreward_commit = subprocess.check_output(
            ["git", "-C", str(official_source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if editreward_commit != external["repository_commit"]:
            raise ValueError("EditReward source commit mismatch")
        for relative, expected in contract["official_source_sha256"].items():
            _validate_hash(
                official_source / relative,
                expected,
                f"official source {relative}",
            )

        seed = int(contract["resource_execution"]["random_seed"])
        import numpy as np

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

        _validate_hash(
            model_dir / "model.safetensors",
            external["model_sha256"],
            "EditReward model",
        )
        _validate_hash(
            model_dir / "config.json",
            contract["base_processor"]["files"]["config.json"]["sha256"],
            "EditReward model config",
        )
        for name, facts in contract["base_processor"]["files"].items():
            path = processor_dir / name
            if path.stat().st_size != facts["bytes"]:
                raise ValueError(f"processor size mismatch: {name}")
            _validate_hash(path, facts["sha256"], f"processor {name}")

        official_package = official_source / "EditReward"
        namespace = types.ModuleType("EditReward")
        namespace.__path__ = [str(official_package)]
        namespace.__package__ = "EditReward"
        sys.modules["EditReward"] = namespace

        from EditReward.model.qwen2_5_vl_trainer import (
            Qwen2_5_VLRewardModelBT_MultiHead,
        )

        processor = AutoProcessor.from_pretrained(
            processor_dir,
            padding_side="right",
            local_files_only=True,
        )
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)
        model_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        model_config.vocab_size = len(processor.tokenizer)
        offload_dir.mkdir(parents=True, exist_ok=True)
        model, loading_info = Qwen2_5_VLRewardModelBT_MultiHead.from_pretrained(
            model_dir,
            config=model_config,
            output_dim=2,
            reward_token="special",
            special_token_ids=special_token_ids,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            rm_head_type="ranknet_multi_head",
            pooling_strategy="mean",
            device_map="auto",
            max_memory={0: "9GiB", "cpu": "24GiB"},
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            local_files_only=True,
            output_loading_info=True,
        )
        load_failures = {
            "missing_keys": loading_info.get("missing_keys", []),
            "unexpected_keys": loading_info.get("unexpected_keys", []),
            "mismatched_keys": loading_info.get("mismatched_keys", []),
            "error_msgs": loading_info.get("error_msgs", []),
        }
        if any(load_failures.values()):
            raise RuntimeError(f"strict load failed: {load_failures}")
        model.eval()
        self.torch = torch
        self.model = model
        self.processor = processor
        self.load_facts = {
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "processor_class": type(processor).__name__,
            "state_tensor_count": external["tensor_count"],
            "state_parameter_count": external["parameter_count"],
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "device_map": {
                key: str(value) for key, value in model.hf_device_map.items()
            },
        }

    def _prepare_batch(
        self, source: Path, candidate: Path, instruction: str, dimension: str
    ):
        from EditReward.dataset.data_collator_qwen_edit import prompt_with_special_token
        from EditReward.dataset.prompts import (
            INSTRUCTION_EDIT_FOLLOWING,
            INSTRUCTION_EDIT_QUALITY,
        )
        from EditReward.dataset.utils import process_vision_info

        if dimension == "dim1":
            base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=instruction)
        elif dimension == "dim2":
            base_prompt = INSTRUCTION_EDIT_QUALITY.format(text_prompt=instruction)
        else:
            raise ValueError(dimension)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(source),
                        "min_pixels": 200704,
                        "max_pixels": 200704,
                    },
                    {
                        "type": "image",
                        "image": str(candidate),
                        "min_pixels": 200704,
                        "max_pixels": 200704,
                    },
                    {"type": "text", "text": base_prompt + prompt_with_special_token},
                ],
            }
        ]
        image_inputs, _ = process_vision_info([message])
        batch = self.processor(
            text=self.processor.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        entry_device = self.model.device
        return {key: value.to(entry_device) for key, value in batch.items()}

    def score(
        self, source: Path, candidate: Path, instruction: str
    ) -> tuple[float, float]:
        batch_dim1 = self._prepare_batch(source, candidate, instruction, "dim1")
        batch_dim2 = self._prepare_batch(source, candidate, instruction, "dim2")
        with self.torch.inference_mode():
            logits = self.model(
                return_dict=True,
                batch_dim1=batch_dim1,
                batch_dim2=batch_dim2,
            )["logits"]
        mean = float(logits[0, 0].detach().float().cpu().item())
        log_sigma = float(logits[0, 1].detach().float().cpu().item())
        if not math.isfinite(mean) or not math.isfinite(log_sigma):
            raise ValueError("EditReward produced a non-finite score")
        return mean, log_sigma


def _row_payload(
    row: ScoreRow, mean: float, log_sigma: float, source_kind: str
) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "dataset": row.dataset,
        "source_id": row.source_id,
        "presentation_id": row.presentation_id,
        "role": row.role,
        "label": row.label,
        "source_kind": source_kind,
        "source_sha256": row.source_sha256,
        "candidate_sha256": row.candidate_sha256,
        "source_geometry": list(row.source_geometry),
        "candidate_geometry": list(row.candidate_geometry),
        "mean": mean,
        "log_sigma": log_sigma,
    }


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--p401-public-root", type=Path, required=True)
    parser.add_argument("--p401-source-root", type=Path, required=True)
    parser.add_argument("--p402-public-root", type=Path, required=True)
    parser.add_argument("--p402-source-root", type=Path, required=True)
    parser.add_argument("--python-deps", type=Path, required=True)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--order", choices=("canonical", "reverse"), default="canonical"
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = _common_parser().parse_args()
    contract, rows = build_inventory(
        args.contract.resolve(),
        args.p401_public_root.resolve(),
        args.p401_source_root.resolve(),
        args.p402_public_root.resolve(),
        args.p402_source_root.resolve(),
    )
    runtime = EditRewardRuntime(
        contract=contract,
        python_deps=args.python_deps.resolve(),
        transformers_source=args.transformers_source.resolve(),
        official_source=args.official_source.resolve(),
        model_dir=args.model_dir.resolve(),
        processor_dir=args.processor_dir.resolve(),
        offload_dir=args.offload_dir.resolve(),
    )
    instruction = contract["fixed_instruction"]
    primary_rows = sorted(rows, key=lambda row: row.row_id)
    controls = wrong_source_controls(rows)
    if args.order == "reverse":
        primary_rows.reverse()
        controls.reverse()
    if args.smoke:
        primary_rows = primary_rows[:1]
        controls = []

    primary = []
    for row in primary_rows:
        mean, log_sigma = runtime.score(
            row.source_path, row.candidate_path, instruction
        )
        primary.append(_row_payload(row, mean, log_sigma, "correct"))

    wrong_source = []
    for candidate_row, wrong_source_row in controls:
        mean, log_sigma = runtime.score(
            wrong_source_row.source_path,
            candidate_row.candidate_path,
            instruction,
        )
        payload = _row_payload(candidate_row, mean, log_sigma, "cyclic-wrong")
        payload["wrong_source_id"] = wrong_source_row.source_id
        payload["wrong_source_sha256"] = wrong_source_row.source_sha256
        wrong_source.append(payload)

    payload = {
        "schema": SCORE_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "status": "MECHANICS_SMOKE_ONLY" if args.smoke else "BLIND_SCORE_LOCK_COMPLETE",
        "contract_sha256": sha256_file(args.contract.resolve()),
        "order": args.order,
        "smoke": args.smoke,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "model_sha256": contract["external_asset"]["model_sha256"],
        "load": runtime.load_facts,
        "inventory": {
            "primary_count": len(primary),
            "wrong_source_count": len(wrong_source),
            "private_mapping_reads": 0,
            "direct_result_reads": 0,
        },
        "primary": sorted(primary, key=lambda row: row["row_id"]),
        "wrong_source": sorted(wrong_source, key=lambda row: row["row_id"]),
        "claim_ceiling": contract["claim_ceiling"],
    }
    _atomic_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "rows": payload["inventory"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
