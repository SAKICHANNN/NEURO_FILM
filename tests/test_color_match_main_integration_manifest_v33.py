from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import pytest
from scripts.build_reference_match_integration_manifest import IntegrationManifestError
from scripts.build_reference_match_integration_manifest_v33 import SCHEMA_ID,PAYLOAD_SCOPE,V33_REQUIRED_CONTRACT_SCHEMA_PATHS,V33_REQUIRED_PUBLIC_EXPORTS,build_manifest,build_schema,encode_json,validate_manifest,validate_manifest_schema
from src.color_match.strict_json import strict_json_loads
ROOT=Path(__file__).resolve().parents[1];MANIFEST=ROOT/"configs/reference_match_main_integration_manifest_v33.json";SCHEMA=ROOT/"configs/schemas/reference_match_main_integration_manifest_v33.schema.json";PRIOR=ROOT/"configs/reference_match_main_integration_manifest_v32.json"
def frozen():return strict_json_loads(MANIFEST.read_text(encoding="utf-8"))
def blob(path):
    files={r["path"]:r["blob"] for r in frozen()["payload_files"]}
    return subprocess.check_output(["git","cat-file","blob",files[path]],cwd=ROOT).decode()
def test_v33_rebuilds_and_binds_cross_language_output_capabilities():
    v=frozen();assert build_manifest(repo=ROOT,payload_commit=v["payload_commit"],base_commit=v["base_commit"],main_commit=v["main_commit"])==v
    assert encode_json(build_schema())==SCHEMA.read_text(encoding="utf-8");validate_manifest(repo=ROOT,manifest=v);validate_manifest_schema(v)
    assert v["schema_id"]==SCHEMA_ID and v["payload_scope"]==PAYLOAD_SCOPE and not v["overlap_paths"]
    assert v["required_public_exports"]==list(V33_REQUIRED_PUBLIC_EXPORTS)
    assert v["supersedes_manifest"]["sha256"]==hashlib.sha256(PRIOR.read_bytes()).hexdigest()
    schema_paths=[row["path"] for row in v["contract_schemas"]]
    assert schema_paths==list(V33_REQUIRED_CONTRACT_SCHEMA_PATHS)
    row=v["contract_schemas"][-1];assert row["path"]=="configs/schemas/reference_file_output_capabilities_v1.schema.json"
    assert row["sha256"]==hashlib.sha256(blob(row["path"]).encode()).hexdigest()
    assert '"reference_file_output_capabilities_payload"' in blob("src/color_match/__init__.py")
@pytest.mark.parametrize("mutation",[lambda v:v.update(payload_scope="bad"),lambda v:v["contract_schemas"][-1].update(sha256="0"*64),lambda v:v.update(extra=True)])
def test_v33_tamper(mutation):
    v=deepcopy(frozen());mutation(v)
    with pytest.raises(IntegrationManifestError,match="schema violation|differs from commit-derived"):validate_manifest(repo=ROOT,manifest=v)
