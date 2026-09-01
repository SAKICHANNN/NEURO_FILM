from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pipeline_color_baseline import load_profile_values
from src.inference import migrate_legacy_safe_rich
from src.inference.yaml_config import YamlConfigError, load_yaml_mapping
from tests.historical_evidence_binding import assert_historical_evidence_binding

ROOT = Path(__file__).resolve().parents[1]
PROFILE_YAML = ROOT / "configs" / "color_rendering_profiles.yaml"
CONFIG = ROOT / "configs" / "u7_2s_product_yaml_runtime_decoupling_v1.json"


def _requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        canonical = name.lower().replace("_", "-").replace(".", "-")
        assert canonical not in result
        result[canonical] = version
    return result


def _normalized_parent_report() -> bytes:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(
        (ROOT / config["u7_2r_terminal_report"]["path"]).read_text(encoding="utf-8")
    )
    report.pop("execution_commit", None)
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode()


def test_tracked_profile_yaml_has_frozen_canonical_identity() -> None:
    document = load_yaml_mapping(PROFILE_YAML)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert canonical == (
        '{"profiles":{"safe_rich":{"defaults":{"chroma_curve_strength":0.45,'
        '"dither":0.35,"gamut_mode":"source","gamut_safe":true,"grain":0.0,'
        '"highlight_ceiling_l":99.0,"luma_strength":0.02,"output_margin":4,'
        '"preserve_luma_detail":0.9,"shadow_floor_l":1.0,"strength":0.35,'
        '"tone_rolloff":0.04,"use_guardrails":true},"description":"First '
        "conservative profile: rich chroma, locked luminance, no hard output "
        'clipping.","styles":{"ektar_100":{"chroma_curve_strength":0.44,'
        '"strength":0.36},"hp5":{"chroma_curve_strength":0.0,'
        '"luma_strength":0.1,"preserve_luma_detail":0.85,"strength":0.72},'
        '"portra_400":{"chroma_curve_strength":0.45,"strength":0.35},'
        '"portra_800":{"chroma_curve_strength":0.44,"strength":0.36},'
        '"tri_x_400":{"chroma_curve_strength":0.0,"luma_strength":0.1,'
        '"preserve_luma_detail":0.85,"strength":0.74},"velvia_50":{'
        '"chroma_curve_strength":0.45,"luma_strength":0.02,"strength":0.35},'
        '"vision3_250d":{"chroma_curve_strength":0.44,"strength":0.36},'
        '"vision3_500t":{"chroma_curve_strength":0.44,"strength":0.36}}}},'
        '"schema_version":1}'
    )
    profile = load_profile_values(PROFILE_YAML, "safe-rich", "velvia_50")
    assert profile["strength"] == 0.35
    assert profile["gamut_mode"] == "source"


@pytest.mark.parametrize(
    "payload",
    [
        "a: 1\na: 2\n",
        "- one\n- two\n",
        "value: !unsafe thing\n",
        "value: [unterminated\n",
        'value: "${environment:HOME}"\n',
    ],
)
def test_strict_loader_rejects_nonliteral_or_ambiguous_documents(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(YamlConfigError):
        load_yaml_mapping(path)


def test_product_imports_do_not_load_omegaconf_or_antlr() -> None:
    for module in (
        "scripts.pipeline_color_baseline",
        "src.inference.render_contract",
        "src.inference",
    ):
        code = (
            f"import {module}; import json,sys; "
            "print(json.dumps({'omegaconf': 'omegaconf' in sys.modules, "
            "'antlr4': any(name == 'antlr4' or name.startswith('antlr4.') "
            "for name in sys.modules)}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"omegaconf": False, "antlr4": False}


def test_existing_profile_migration_remains_exact() -> None:
    migrated = migrate_legacy_safe_rich(
        PROFILE_YAML,
        ROOT / "configs" / "film_color_stats.json",
        ROOT / "configs" / "color_guardrails.json",
        root=ROOT,
    )
    tracked = json.loads(
        (ROOT / "configs" / "render_profiles" / "safe_rich_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert migrated == tracked


def test_v2_manifest_is_exactly_v1_minus_omegaconf_and_antlr() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v1 = _requirements(ROOT / "requirements-product.txt")
    for distribution, version in config["removed_distributions"].items():
        assert v1.pop(distribution) == version
    assert _requirements(ROOT / config["requirements_path"]) == v1
    assert v1 == config["required_distributions"]
    assert set(config["removed_distributions"]).isdisjoint(v1)


def test_protocol_locks_terminal_parent_without_reclassifying_it() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for binding in config["source_locks"].values():
        assert_historical_evidence_binding(ROOT, binding)
    terminal = config["u7_2r_terminal_report"]
    report_path = ROOT / terminal["path"]
    assert report_path.stat().st_size == terminal["bytes"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == terminal["sha256"]
    normalized = _normalized_parent_report()
    assert len(normalized) == terminal["normalized_scientific_bytes"]
    assert (
        hashlib.sha256(normalized).hexdigest()
        == terminal["normalized_scientific_sha256"]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == terminal["expected_status"] == "FAIL_CLOSED"
    assert [name for name, passed in report["gates"].items() if not passed] == terminal[
        "expected_failed_gates"
    ]
