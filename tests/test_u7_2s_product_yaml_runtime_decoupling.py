from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.pipeline_color_baseline import load_profile_values
from src.inference import migrate_legacy_safe_rich
from src.inference.yaml_config import YamlConfigError, load_yaml_mapping

ROOT = Path(__file__).resolve().parents[1]
PROFILE_YAML = ROOT / "configs" / "color_rendering_profiles.yaml"


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
