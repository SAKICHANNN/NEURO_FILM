import json
from pathlib import Path
from scripts.evaluate_u6_p4gh_cloud_residual_chart_artifact import evaluate
ROOT=Path(__file__).resolve().parents[1];CONTRACT=ROOT/'configs/u6_p4gh_cloud_residual_chart_artifact_v1.json'
def test_p4gh_contract_uses_existing_chart_ceilings():
 c=json.loads(CONTRACT.read_text());assert c['gates']['maximum_neutral_chroma_p99']==.09;assert c['gates']['maximum_edge_overshoot']==.025;assert c['gates']['maximum_new_boundary_fraction']==.0005


def test_p4gh_formal_result_closes_on_neutral_chroma():
 result=json.loads((ROOT/'docs/evidence/U6_P4GH_CLOUD_RESIDUAL_CHART_ARTIFACT_RESULT.json').read_text())
 stable=result['stable']
 assert result['automatic_pass'] is False
 assert stable['decision']=='close_cloud_residual_after_chart_artifact_gate'
 assert stable['gates']['neutral_chroma'] is False
 assert stable['gates']['patch_chroma'] is False
 assert stable['gates']['overshoot'] is True
 assert stable['gates']['undershoot'] is True
 assert stable['gates']['boundary'] is True
 assert stable['residual']['neutral_chroma_p99']<stable['physics']['neutral_chroma_p99']
 assert result['stable_evidence_id']=='43b2c6ee164a35304d91cb3ef2acc37e48a6e394cf0e447d888bcfa66d558fcb'
