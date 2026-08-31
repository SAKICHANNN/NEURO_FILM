"""Committed-head audit for U7.3L recipe replay ownership repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import style_safe_engine
from src.inference.style_safe_engine import (
    StyleSafeEngineError,
    replay_style_safe_recipe_to_file,
)
from src.preprocess import output_encode
from src.preprocess.output_encode import (
    save_srgb8,
    save_srgb16_png,
    save_srgb16_tiff,
)

CONFIG = ROOT / "configs/u7_3l_recipe_replay_create_only_repair_v1.json"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def _pixels() -> np.ndarray:
    values = np.arange(7 * 11 * 3, dtype=np.float32).reshape(7, 11, 3)
    return values / float(values.max())


def _recipe_for(
    work: Path,
    *,
    name: str,
    suffix: str,
    bit_depth: int,
    format_name: str,
) -> tuple[dict[str, object], bytes]:
    expected = work / f"expected-{name}{suffix}"
    if bit_depth == 8:
        actual_format = save_srgb8(_pixels(), expected)
    elif format_name == "PNG":
        actual_format = save_srgb16_png(_pixels(), expected, compression_level=6)
    else:
        actual_format = save_srgb16_tiff(_pixels(), expected)
    if actual_format != format_name:
        raise RuntimeError("audit encoder format drift")
    payload = expected.read_bytes()
    recipe: dict[str, object] = {
        "render": {},
        "output": {
            "bit_depth": bit_depth,
            "format": format_name,
            "icc_profile_fingerprint_sha256": (
                style_safe_engine.srgb_icc_profile_fingerprint_sha256()
            ),
            "png_compression": 6,
            "sha256": _sha256(payload),
        },
    }
    return recipe, payload


def _success_rows(work: Path, order: str) -> list[dict[str, Any]]:
    definitions = [
        ("jpeg8", ".jpg", 8, "JPEG"),
        ("png8", ".png", 8, "PNG"),
        ("png16", ".png", 16, "PNG"),
        ("tiff8", ".tiff", 8, "TIFF"),
        ("tiff16", ".tiff", 16, "TIFF"),
    ]
    if order == "reverse":
        definitions.reverse()
    rows: list[dict[str, Any]] = []
    for name, suffix, bit_depth, format_name in definitions:
        recipe, expected = _recipe_for(
            work,
            name=name,
            suffix=suffix,
            bit_depth=bit_depth,
            format_name=format_name,
        )
        recipe_before = copy.deepcopy(recipe)
        output = work / f"replay-{name}{suffix}"
        with patch.object(
            style_safe_engine,
            "replay_style_safe_recipe",
            return_value=_pixels(),
        ):
            digest = replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=output,
                root=work,
            )
        rows.append(
            {
                "name": name,
                "bytes": len(expected),
                "sha256": _sha256(expected),
                "successful_output_bytes_exact": output.read_bytes() == expected,
                "returned_digest_exact": digest == _sha256(expected),
                "recipe_immutable": recipe == recipe_before,
                "temporary_residue": len(list(work.glob(f".{output.name}.*.tmp"))),
            }
        )
    rows.sort(key=lambda row: row["name"])
    return rows


def _failure_controls(work: Path) -> dict[str, bool]:
    recipe, _expected = _recipe_for(
        work,
        name="controls",
        suffix=".png",
        bit_depth=8,
        format_name="PNG",
    )
    rendered = patch.object(
        style_safe_engine,
        "replay_style_safe_recipe",
        return_value=_pixels(),
    )

    late = work / "late.png"
    late_foreign = b"late-foreign-output"
    real_publish = output_encode.publish_create_only

    def late_inject(stage: Path, final: Path):
        final.write_bytes(late_foreign)
        return real_publish(stage, final)

    late_rejected = False
    try:
        with rendered, patch.object(output_encode, "publish_create_only", late_inject):
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=late,
                root=work,
            )
    except FileExistsError:
        late_rejected = True

    replaced = work / "replaced.png"
    replacement = work / "replacement.bin"
    replacement_foreign = b"postpublication-foreign"
    replacement.write_bytes(replacement_foreign)

    def replace_then_hash(path: Path) -> str:
        os.replace(replacement, path)
        return "0" * 64

    replacement_rejected = False
    try:
        with (
            patch.object(
                style_safe_engine,
                "replay_style_safe_recipe",
                return_value=_pixels(),
            ),
            patch.object(style_safe_engine, "sha256_file", replace_then_hash),
        ):
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=replaced,
                root=work,
            )
    except StyleSafeEngineError:
        replacement_rejected = True

    owned = work / "owned-mismatch.png"
    owned_rejected = False
    try:
        with (
            patch.object(
                style_safe_engine,
                "replay_style_safe_recipe",
                return_value=_pixels(),
            ),
            patch.object(style_safe_engine, "sha256_file", return_value="0" * 64),
        ):
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=owned,
                root=work,
            )
    except StyleSafeEngineError:
        owned_rejected = True

    existing = work / "existing.png"
    existing_foreign = b"existing-foreign"
    existing.write_bytes(existing_foreign)
    existing_rejected = False
    try:
        with patch.object(
            style_safe_engine,
            "replay_style_safe_recipe",
            side_effect=AssertionError("render must not run"),
        ):
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=existing,
                root=work,
            )
    except StyleSafeEngineError:
        existing_rejected = True

    symlink = work / "dangling.png"
    with (
        patch.object(Path, "exists", return_value=False),
        patch.object(Path, "is_symlink", return_value=True),
        patch.object(
            style_safe_engine,
            "replay_style_safe_recipe",
            side_effect=AssertionError("render must not run"),
        ),
    ):
        symlink_rejected = False
        try:
            replay_style_safe_recipe_to_file(
                recipe,
                profile_path=work / "unused-profile.json",
                output_path=symlink,
                root=work,
            )
        except StyleSafeEngineError:
            symlink_rejected = True

    return {
        "late_foreign_destination_preserved": (
            late_rejected and late.read_bytes() == late_foreign
        ),
        "postpublication_replacement_preserved": (
            replacement_rejected and replaced.read_bytes() == replacement_foreign
        ),
        "owned_mismatch_residue_zero": owned_rejected and not owned.exists(),
        "existing_destination_preserved": (
            existing_rejected and existing.read_bytes() == existing_foreign
        ),
        "dangling_symlink_rejected_before_render": symlink_rejected,
        "temporary_residue_zero": not any(work.glob(".*.tmp")),
    }


def run_audit(order: str) -> dict[str, Any]:
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be forward or reverse")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    source_bindings = bindings["implementation_git_lf_sha256"]
    source_bindings_exact = all(
        _sha256(_git_bytes(bindings["implementation_commit"], path)) == expected
        for path, expected in source_bindings.items()
    )
    commits_resolve = all(
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        == 0
        for commit in (bindings["contract_commit"], bindings["implementation_commit"])
    )

    with tempfile.TemporaryDirectory(prefix="u7-3l-") as raw:
        work = Path(raw)
        success_rows = _success_rows(work, order)
        controls = _failure_controls(work)

    all_success = all(
        row["successful_output_bytes_exact"]
        and row["returned_digest_exact"]
        and row["recipe_immutable"]
        and row["temporary_residue"] == 0
        for row in success_rows
    )
    gates = {
        "source_bindings_exact": source_bindings_exact and commits_resolve,
        "successful_output_bytes_exact": all_success,
        "late_foreign_destination_preserved": controls[
            "late_foreign_destination_preserved"
        ],
        "postpublication_replacement_preserved": controls[
            "postpublication_replacement_preserved"
        ],
        "owned_mismatch_residue_zero": controls["owned_mismatch_residue_zero"],
        "existing_and_symlink_destinations_reject": (
            controls["existing_destination_preserved"]
            and controls["dangling_symlink_rejected_before_render"]
        ),
        "temporary_residue_zero": controls["temporary_residue_zero"],
        "source_and_recipe_inputs_immutable": all(
            row["recipe_immutable"] for row in success_rows
        ),
    }
    scientific = {
        "schema": config["schema"],
        "node_id": config["node_id"],
        "bindings": bindings,
        "success_rows": success_rows,
        "controls": controls,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
        "stop_rule": config["stop_rule"],
    }
    stable_identity = _sha256(_canonical_json(scientific))
    status = (
        "PASS_PRIVATE_U7_3L_RECIPE_REPLAY_CREATE_ONLY_REPAIR"
        if all(gates.values())
        else "FAIL_CLOSED_U7_3L_RECIPE_REPLAY_CREATE_ONLY_REPAIR"
    )
    report: dict[str, Any] = {
        "schema": config["schema"] + ".report",
        "status": status,
        "stable_identity": stable_identity,
        "scientific": scientific,
    }
    report["report_identity"] = _sha256(_canonical_json(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_json(report))
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
