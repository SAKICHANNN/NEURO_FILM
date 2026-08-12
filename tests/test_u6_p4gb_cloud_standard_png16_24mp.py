import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_p4gb_contract_freezes_complete_transaction():
 c=json.loads((ROOT/'configs/u6_p4gb_cloud_standard_png16_24mp_v1.json').read_text());assert c['scenario']['height']*c['scenario']['width']==24_000_000;assert c['gates']['require_exact_png_repeat'];assert c['gates']['require_restart_verification'];assert c['gates']['require_cleanup']

def test_p4gb_frozen_result_passes_transaction_resource_gates():
 r=json.loads((ROOT/'docs/evidence/U6_P4GB_CLOUD_STANDARD_PNG16_24MP_RESULT.json').read_text());assert r['status']=='PASS';assert r['gates']['all_pass'];assert r['identities']['png_bytes']>0;assert max(r['metrics']['peak_process_tree_rss_bytes'])<=1073741824
