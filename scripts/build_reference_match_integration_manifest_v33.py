"""Build the P148 integration manifest."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
if __package__:
    from scripts.build_reference_match_integration_manifest import IntegrationManifestError
    from scripts.build_reference_match_integration_manifest_v25 import REQUIRED_CONTRACT_SCHEMA_PATHS
    from scripts.build_reference_match_integration_manifest_v32 import V32_REQUIRED_PUBLIC_EXPORTS
    from scripts.reference_match_integration_manifest_common import build_versioned_manifest, encode_json, validate_manifest_schema as validate_schema_common, validate_versioned_manifest
else:
    from build_reference_match_integration_manifest import IntegrationManifestError
    from build_reference_match_integration_manifest_v25 import REQUIRED_CONTRACT_SCHEMA_PATHS
    from build_reference_match_integration_manifest_v32 import V32_REQUIRED_PUBLIC_EXPORTS
    from reference_match_integration_manifest_common import build_versioned_manifest, encode_json, validate_manifest_schema as validate_schema_common, validate_versioned_manifest
SCHEMA_ID="neuro-film.reference-match-main-integration-manifest.v33";CLAIM_CEILING="review-ready-not-merged";PAYLOAD_SCOPE="P1-P148"
PRIOR_MANIFEST_PATH="configs/reference_match_main_integration_manifest_v32.json";PRIOR_MANIFEST_SCHEMA_ID="neuro-film.reference-match-main-integration-manifest.v32"
ROOT=Path(__file__).resolve().parents[1];V32_SCHEMA_PATH=ROOT/"configs/schemas/reference_match_main_integration_manifest_v32.schema.json";DEFAULT_SCHEMA_PATH=ROOT/"configs/schemas/reference_match_main_integration_manifest_v33.schema.json"
VERIFICATION_COMMANDS=("git merge-tree --write-tree <payload_commit> <main_commit>","python -m pytest -q -k color_match","python -m pytest -q")
V33_REQUIRED_PUBLIC_EXPORTS=V32_REQUIRED_PUBLIC_EXPORTS+("reference_file_output_capabilities_payload",)
V33_REQUIRED_CONTRACT_SCHEMA_PATHS=REQUIRED_CONTRACT_SCHEMA_PATHS+("configs/schemas/reference_file_output_capabilities_v1.schema.json",)
def build_schema(*,v32_schema_path:Path=V32_SCHEMA_PATH)->dict[str,Any]:
    try:result=deepcopy(json.loads(v32_schema_path.read_text(encoding="utf-8")))
    except (OSError,json.JSONDecodeError) as exc:raise IntegrationManifestError("v32 integration schema unavailable") from exc
    result["$id"]="https://neuro-film.local/schemas/reference-match-main-integration-manifest-v33";result["title"]="Neuro-Film Reference Match Main Integration Manifest V33";result["properties"]["schema_id"]["const"]=SCHEMA_ID;result["properties"]["payload_scope"]["const"]=PAYLOAD_SCOPE;result["properties"]["required_public_exports"]["const"]=list(V33_REQUIRED_PUBLIC_EXPORTS)
    capability_schema_path="configs/schemas/reference_file_output_capabilities_v1.schema.json"
    contract_paths=result["$defs"]["contract_schema"]["properties"]["path"]["enum"];contract_paths.append(capability_schema_path)
    contracts=result["properties"]["contract_schemas"];contracts["minItems"]=23;contracts["maxItems"]=23;contracts["prefixItems"].append({"allOf":[{"$ref":"#/$defs/contract_schema"},{"properties":{"path":{"const":capability_schema_path}}}]})
    prior=result["$defs"]["prior_manifest"]["properties"];prior["path"]["const"]=PRIOR_MANIFEST_PATH;prior["schema_id"]["const"]=PRIOR_MANIFEST_SCHEMA_ID;Draft202012Validator.check_schema(result);return result
def build_manifest(*,repo:Path,payload_commit:str,base_commit:str,main_commit:str)->dict[str,Any]:return build_versioned_manifest(repo=repo,payload_commit=payload_commit,base_commit=base_commit,main_commit=main_commit,schema_id=SCHEMA_ID,payload_scope=PAYLOAD_SCOPE,required_public_exports=V33_REQUIRED_PUBLIC_EXPORTS,required_contract_schema_paths=V33_REQUIRED_CONTRACT_SCHEMA_PATHS,prior_manifest_path=PRIOR_MANIFEST_PATH,prior_manifest_schema_id=PRIOR_MANIFEST_SCHEMA_ID,verification_commands=VERIFICATION_COMMANDS,claim_ceiling=CLAIM_CEILING)
def validate_manifest_schema(manifest:Any,*,schema_path:Path=DEFAULT_SCHEMA_PATH)->None:validate_schema_common(manifest,schema_path=schema_path)
def validate_manifest(*,repo:Path,manifest:dict[str,Any],schema_path:Path=DEFAULT_SCHEMA_PATH)->None:validate_versioned_manifest(repo=repo,manifest=manifest,schema_path=schema_path,rebuild=build_manifest)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--payload");p.add_argument("--base");p.add_argument("--main");p.add_argument("--output",type=Path);p.add_argument("--output-schema",type=Path);p.add_argument("--verify-manifest",type=Path);a=p.parse_args()
    if a.output_schema:a.output_schema.write_text(encode_json(build_schema()),encoding="utf-8",newline="\n")
    elif a.verify_manifest:validate_manifest(repo=a.repo,manifest=json.loads(a.verify_manifest.read_text(encoding="utf-8")))
    else:
        if not all((a.payload,a.base,a.main)):p.error("--payload, --base and --main are required")
        value=encode_json(build_manifest(repo=a.repo,payload_commit=a.payload,base_commit=a.base,main_commit=a.main))
        if a.output:a.output.write_text(value,encoding="utf-8",newline="\n")
        else:print(value,end="")
