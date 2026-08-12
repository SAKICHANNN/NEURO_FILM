import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_p4gg_contract_freezes_retained_24mp_transaction():
 c=json.loads((ROOT/'configs/u6_p4gg_cloud_residual_png16_24mp_v1.json').read_text());assert c['scenario']['height']*c['scenario']['width']==24_000_000;assert c['gates']['require_raw_identity_from_p4ge_mechanism'];assert c['gates']['require_restart_verification'];assert c['gates']['require_cleanup']

def test_p4gg_frozen_result_passes_all_transaction_gates():
 r=json.loads((ROOT/'docs/evidence/U6_P4GG_CLOUD_RESIDUAL_PNG16_24MP_RESULT.json').read_text());assert r['status']=='PASS';assert r['gates']['all_pass'];assert r['identities']['png_bytes']>60_000_000;assert max(r['metrics']['peak_process_tree_rss_bytes'])<1073741824
