from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2s_product_yaml_runtime_decoupling_v1.json"
REQUIREMENTS = ROOT / "requirements-product-v2.txt"
PARENT_AUDIT = ROOT / "scripts/audit_u7_2r_product_effect_argument_preflight.py"
PARENT_CONFIG = ROOT / "configs/u7_2r_product_effect_argument_preflight_v1.json"
PROFILE_YAML = ROOT / "configs/color_rendering_profiles.yaml"
SCRATCH_PARENT = ROOT / "tmp"
BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools", "wheel"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    payload = subprocess.check_output(["git", *arguments], cwd=ROOT)
    return payload if binary else payload.decode("utf-8").strip()


def _canonical_distribution(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        canonical = _canonical_distribution(name)
        if canonical in result or not canonical or not version:
            raise ValueError("requirements manifest is not uniquely and fully pinned")
        result[canonical] = version
    return result


def _run(
    arguments: list[str], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _python_in(venv: Path) -> Path:
    return venv / "Scripts/python.exe"


def _inventory(python: Path, env: dict[str, str]) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json;"
        "print(json.dumps(sorted((d.metadata['Name'],d.version) "
        "for d in m.distributions()),separators=(',',':')))"
    )
    result = _run([str(python), "-c", code], env=env)
    if result.returncode != 0:
        return {}
    rows = json.loads(result.stdout)
    return {
        _canonical_distribution(str(name)): str(version)
        for name, version in rows
        if _canonical_distribution(str(name)) not in BOOTSTRAP_DISTRIBUTIONS
    }


def _import_probe(python: Path, env: dict[str, str]) -> dict[str, Any]:
    code = """
import hashlib
import json
import sys
from pathlib import Path
import scripts.pipeline_color_baseline
import src.inference
import src.inference.render_contract
from src.inference.yaml_config import load_yaml_mapping
document = load_yaml_mapping(Path('configs/color_rendering_profiles.yaml'))
canonical = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
print(json.dumps({
    'antlr4_loaded': any(name == 'antlr4' or name.startswith('antlr4.') for name in sys.modules),
    'omegaconf_loaded': 'omegaconf' in sys.modules,
    'profile_canonical_bytes': len(canonical),
    'profile_canonical_sha256': hashlib.sha256(canonical).hexdigest(),
}, sort_keys=True, separators=(',', ':')))
"""
    result = _run([str(python), "-c", code], env=env)
    if result.returncode != 0:
        return {"returncode": result.returncode}
    return {"returncode": 0, **json.loads(result.stdout)}


def _normalize_parent_report(payload: bytes) -> bytes:
    report = json.loads(payload)
    report.pop("execution_commit", None)
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode()


def _source_locks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for role, binding in sorted(config["source_locks"].items()):
        path = ROOT / binding["path"]
        blob = _git("cat-file", "blob", binding["git_blob"], binary=True)
        assert isinstance(blob, bytes)
        rows[role] = {
            "blob_sha256": _sha256_bytes(blob),
            "blob_sha256_exact": _sha256_bytes(blob) == binding["sha256"],
            "current_sha256": _sha256(path) if path.is_file() else None,
            "current_sha256_exact": path.is_file()
            and _sha256(path) == binding["sha256"],
        }
    return rows


def _build_environment(
    *, root: Path, environment_id: str, nested_order: str, config: dict[str, Any]
) -> dict[str, Any]:
    venv = root / environment_id
    cache = root / f"{environment_id}-pip-cache"
    nested_report = root / f"{environment_id}-u7_2r.json"
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = str(cache)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    create = _run(
        ["py", config["python"]["windows_launcher_selector"], "-m", "venv", str(venv)],
        env=env,
    )
    python = _python_in(venv)
    install = (
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-input",
                "--only-binary=:all:",
                "--requirement",
                str(REQUIREMENTS),
            ],
            env=env,
        )
        if create.returncode == 0 and python.is_file()
        else None
    )
    check = (
        _run([str(python), "-m", "pip", "check"], env=env)
        if install is not None and install.returncode == 0
        else None
    )
    inventory = (
        _inventory(python, env) if check is not None and check.returncode == 0 else {}
    )
    import_probe = (
        _import_probe(python, env)
        if inventory == config["required_distributions"]
        else {"returncode": None}
    )
    nested = (
        _run(
            [
                str(python),
                str(PARENT_AUDIT),
                "--config",
                str(PARENT_CONFIG),
                "--order",
                nested_order,
                "--output",
                str(nested_report),
            ],
            env=env,
        )
        if import_probe.get("returncode") == 0
        else None
    )
    nested_payload = nested_report.read_bytes() if nested_report.is_file() else b""
    nested_json = json.loads(nested_payload) if nested_payload else {}
    normalized = _normalize_parent_report(nested_payload) if nested_payload else b""
    return {
        "environment_id": environment_id,
        "python_version": (
            _run([str(python), "--version"], env=env).stdout.strip()
            if python.is_file()
            else None
        ),
        "create_returncode": create.returncode,
        "install_returncode": install.returncode if install is not None else None,
        "pip_check_returncode": check.returncode if check is not None else None,
        "inventory": inventory,
        "import_probe": import_probe,
        "nested_order": nested_order,
        "nested_returncode": nested.returncode if nested is not None else None,
        "nested_report_bytes": len(nested_payload),
        "nested_report_sha256": _sha256_bytes(nested_payload)
        if nested_payload
        else None,
        "nested_normalized_bytes": len(normalized),
        "nested_normalized_sha256": _sha256_bytes(normalized) if normalized else None,
        "nested_status": nested_json.get("status"),
        "nested_failed_gates": sorted(
            name for name, passed in nested_json.get("gates", {}).items() if not passed
        ),
    }


def build_report(config_path: Path, order: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = config["required_distributions"]
    source_locks = _source_locks(config)
    labels = ["environment-a", "environment-b"]
    execution_labels = labels if order == "forward" else list(reversed(labels))
    nested_orders = {"environment-a": "forward", "environment-b": "reverse"}
    work_root: Path | None = None
    rows: list[dict[str, Any]] = []
    try:
        SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
        work_root = Path(tempfile.mkdtemp(prefix="u7_2s-", dir=SCRATCH_PARENT))
        for label in execution_labels:
            rows.append(
                _build_environment(
                    root=work_root,
                    environment_id=label,
                    nested_order=nested_orders[label],
                    config=config,
                )
            )
    finally:
        if work_root is not None:
            resolved = work_root.resolve()
            if (
                resolved.parent != SCRATCH_PARENT.resolve()
                or not resolved.name.startswith("u7_2s-")
            ):
                raise RuntimeError("refusing to remove unexpected U7.2S work root")
            shutil.rmtree(resolved, ignore_errors=True)

    rows.sort(key=lambda row: row["environment_id"])
    terminal = config["u7_2r_terminal_report"]
    forbidden = {
        _canonical_distribution(name) for name in config["forbidden_distribution_roots"]
    }
    nested_exact = (
        len(rows) == config["environment_count"]
        and all(
            row["nested_returncode"] == terminal["expected_returncode"] for row in rows
        )
        and all(row["nested_status"] == terminal["expected_status"] for row in rows)
        and all(
            row["nested_failed_gates"] == terminal["expected_failed_gates"]
            for row in rows
        )
        and all(
            row["nested_normalized_bytes"] == terminal["normalized_scientific_bytes"]
            and row["nested_normalized_sha256"]
            == terminal["normalized_scientific_sha256"]
            for row in rows
        )
    )
    gates = {
        "binary_only_fresh_installs_pass": len(rows) == config["environment_count"]
        and all(
            row["create_returncode"] == row["install_returncode"] == 0 for row in rows
        ),
        "forbidden_research_and_removed_distributions_absent": all(
            forbidden.isdisjoint(row["inventory"]) for row in rows
        ),
        "installed_inventories_exact_and_fully_pinned": all(
            row["inventory"] == required for row in rows
        ),
        "nested_u7_2r_terminal_science_exact": nested_exact,
        "nested_u7_2r_reports_byte_exact": len(rows) == config["environment_count"]
        and len({row["nested_report_sha256"] for row in rows}) == 1,
        "omegaconf_and_antlr_absent_from_product_imports": all(
            row["import_probe"].get("returncode") == 0
            and row["import_probe"].get("omegaconf_loaded") is False
            and row["import_probe"].get("antlr4_loaded") is False
            for row in rows
        ),
        "owned_runtime_residue_zero": work_root is not None and not work_root.exists(),
        "pip_check_passes": all(row["pip_check_returncode"] == 0 for row in rows),
        "python_3_12_exact": all(
            row["python_version"] == f"Python {config['python']['qualified_patch']}"
            for row in rows
        ),
        "requirements_manifest_exact": _requirements(REQUIREMENTS) == required,
        "source_locks_exact": all(
            row["blob_sha256_exact"] and row["current_sha256_exact"]
            for row in source_locks.values()
        ),
        "tracked_profile_yaml_exact_in_both_environments": len(rows)
        == config["environment_count"]
        and len(
            {
                (
                    row["import_probe"].get("profile_canonical_bytes"),
                    row["import_probe"].get("profile_canonical_sha256"),
                )
                for row in rows
            }
        )
        == 1,
    }
    return {
        "schema_version": "neuro-film.u7-2s-product-yaml-runtime-decoupling-result.v1",
        "status": "PASS" if all(gates.values()) else "FAIL_CLOSED",
        "execution_commit": _git("rev-parse", "HEAD"),
        "runner_git_blob": _git(
            "rev-parse", f"HEAD:{Path(__file__).relative_to(ROOT).as_posix()}"
        ),
        "config_sha256": _sha256(config_path),
        "source_locks": source_locks,
        "requirements": required,
        "environments": rows,
        "gates": gates,
        "claim_ceiling": config["claim_ceiling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--order", choices=("forward", "reverse"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.config, args.order)
    encoded = (
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
