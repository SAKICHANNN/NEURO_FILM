"""Audit exact official BRACE sources without requesting payloads or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class P285Error(RuntimeError):
    """Raised when a frozen P285 source or execution boundary differs."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _verify_binding(bindings: dict[str, object], prefix: str) -> bool:
    path = ROOT / str(bindings[f"{prefix}_path"])
    body = path.read_bytes()
    return len(body) == int(bindings[f"{prefix}_bytes"]) and _sha256(body) == str(
        bindings[f"{prefix}_sha256"]
    )


def _fetch(source: dict[str, object], maximum_bytes: int) -> dict[str, object]:
    request = urllib.request.Request(
        str(source["url"]),
        headers={
            "Accept": "application/vnd.github+json,text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "neuro-film-p285-source-audit/1.0",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read(maximum_bytes + 1)
                if len(body) > maximum_bytes:
                    raise P285Error("official response exceeds frozen byte ceiling")
                if int(response.status) != 200:
                    raise P285Error("official response did not return HTTP 200")
                final_url = response.geturl()
                http_status = int(response.status)
            break
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt == 2:
                raise P285Error("official source transport failed after bounded retries") from error
            time.sleep(0.25 * (attempt + 1))
    else:  # pragma: no cover - loop always breaks or raises
        raise P285Error("official source transport failed") from last_error
    if len(body) != int(source["bytes"]) or _sha256(body) != source["sha256"]:
        raise P285Error("official response identity differs from frozen source")
    return {
        "body": body,
        "body_bytes": len(body),
        "body_sha256": _sha256(body),
        "final_url": final_url,
        "http_status": http_status,
    }


def _extract_facts(
    config: dict[str, Any], responses: dict[str, dict[str, object]]
) -> dict[str, object]:
    commit = json.loads(bytes(responses["commit"]["body"]))
    tree = json.loads(bytes(responses["tree"]["body"]))
    readme = bytes(responses["readme"]["body"]).decode("utf-8", errors="strict")
    lower = re.sub(r"\s+", " ", readme.casefold()).strip()
    paths = sorted(str(item["path"]) for item in tree["tree"])
    blobs = sorted(
        str(item["path"]) for item in tree["tree"] if item["type"] == "blob"
    )
    repository = config["official_repository"]
    expected = config["expected"]

    official_identity = all(
        (
            commit["sha"] == repository["commit"],
            commit["tree"]["sha"] == repository["tree"],
            [item["sha"] for item in commit["parents"]] == [repository["parent"]],
            tree["sha"] == repository["tree"],
            not bool(tree["truncated"]),
            len(paths) == int(expected["tree_entry_count"]),
            paths == expected["tree_paths"],
            expected["paper_title"].casefold() in lower,
            expected["arxiv_id"] in readme,
        )
    )
    observation_phrases = (
        "bracketed raw dataset",
        "automated multi-exposure capture tool",
        "multi-frame bracketed raw restoration paradigm",
    )
    new_physical_observation = all(phrase in lower for phrase in observation_phrases)

    license_names = {
        "license",
        "license.md",
        "license.txt",
        "copying",
        "copying.md",
        "notice",
        "notice.txt",
    }
    license_paths = [path for path in blobs if Path(path).name.casefold() in license_names]
    code_paths = [
        path
        for path in blobs
        if Path(path).suffix.casefold()
        in {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".ipynb", ".py", ".sh"}
    ]
    manifest_paths = [
        path
        for path in blobs
        if any(
            token in Path(path).name.casefold()
            for token in ("manifest", "checksum", "sha256", "md5")
        )
    ]
    model_paths = [
        path
        for path in blobs
        if Path(path).suffix.casefold() in {".ckpt", ".onnx", ".pth", ".pt", ".safetensors"}
    ]
    todo_count = len(re.findall(r"(?m)^TODO\s*$", readme))
    executable_release = bool(code_paths) and todo_count == 0
    dataset_locator_count = len(
        set(
            re.findall(
                r"https?://[^)\s]+",
                readme[readme.find("Download Pretrained Models and Datasets") :],
            )
        )
    )
    # The sole dataset badge points back to the repository and is not a payload locator.
    separate_dataset_locator = any(
        "github.com/ZZH-qwq/BRACE" not in url
        for url in re.findall(r"https?://[^)\s]+", readme)
        if "dataset" in url.casefold() or "drive" in url.casefold()
    )
    public_inventory = bool(manifest_paths) and separate_dataset_locator
    exact_checkpoint = bool(model_paths) and any(
        token in lower for token in ("sha256", "checksum", "md5")
    )
    group_isolation = bool(
        re.search(r"(?:scene|capture|session|subject)[-_ ](?:disjoint|split|id)", lower)
    )
    commercial_code_rights = bool(license_paths) and any(
        token in lower for token in ("apache-2.0", "mit license", "bsd license")
    )
    commercial_dataset_rights = bool(license_paths) and any(
        token in lower
        for token in (
            "creative commons attribution 4.0",
            "cc by 4.0",
            "public domain",
        )
    )
    return {
        "code_commercial_compatible": commercial_code_rights,
        "code_paths": code_paths,
        "dataset_commercial_compatible": commercial_dataset_rights,
        "dataset_locator_count": dataset_locator_count,
        "exact_public_archive_inventory": public_inventory,
        "exact_reference_checkpoint": exact_checkpoint,
        "executable_release": executable_release,
        "group_isolation": group_isolation,
        "license_paths": license_paths,
        "manifest_paths": manifest_paths,
        "model_paths": model_paths,
        "new_physical_observation": new_physical_observation,
        "official_identity": official_identity,
        "readme_todo_count": todo_count,
        "repository_blob_count": len(blobs),
        "repository_path_count": len(paths),
        "tree_paths": paths,
    }


def execute(config_path: Path, *, reverse: bool = False) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bindings = config["bindings"]
    binding_gates = {
        "contract": _verify_binding(bindings, "contract"),
        "runner": _verify_binding(bindings, "runner"),
        "test": _verify_binding(bindings, "test"),
    }
    if not all(binding_gates.values()):
        raise P285Error("frozen local binding differs")
    maximum = int(config["sources"]["maximum_response_bytes_each"])
    source_keys = ("commit", "tree", "readme")
    traversal = tuple(reversed(source_keys)) if reverse else source_keys
    responses = {key: _fetch(config["sources"][key], maximum) for key in traversal}
    total_bytes = sum(int(item["body_bytes"]) for item in responses.values())
    if total_bytes > int(config["sources"]["maximum_total_network_bytes"]):
        raise P285Error("formal network bytes exceed frozen total ceiling")
    facts = _extract_facts(config, responses)
    gates = {
        "commercial_compatible_code_rights": facts["code_commercial_compatible"],
        "commercial_compatible_dataset_rights": facts[
            "dataset_commercial_compatible"
        ],
        "exact_public_archive_inventory": facts["exact_public_archive_inventory"],
        "exact_reference_checkpoint": facts["exact_reference_checkpoint"],
        "executable_release": facts["executable_release"],
        "group_isolation": facts["group_isolation"],
        "new_physical_observation": facts["new_physical_observation"],
        "official_identity": facts["official_identity"],
    }
    decision = (
        "PASS_PRIVATE_BRACE_BRACKET_RAW_SOURCE_READINESS"
        if all(gates.values())
        else "NOT_READY_BRACE_RELEASE_AND_RIGHTS_GAP_NOT_SCIENTIFIC_RESULT"
    )
    transport = {
        key: {name: value for name, value in item.items() if name != "body"}
        for key, item in responses.items()
    }
    report = {
        "archive_or_member_requests": 0,
        "bindings": binding_gates,
        "candidate_count": "2/3",
        "checkpoint_or_model_requests": 0,
        "claim_ceiling": config["claim_ceiling"],
        "dataset_requests": 0,
        "decision": decision,
        "experiment_id": "P285",
        "facts": facts,
        "figure_blob_requests": 0,
        "gates": gates,
        "network_bytes": total_bytes,
        "pixel_reads": 0,
        "repository_clone_or_checkout": 0,
        "schema": "neuro-film.p285-brace-bracket-raw-source-readiness-result.v1",
        "training_or_inference_runs": 0,
        "transport": transport,
    }
    report["scientific_identity"] = "sha256:" + _sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    report = execute(args.config.resolve(), reverse=args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))


if __name__ == "__main__":
    main()
