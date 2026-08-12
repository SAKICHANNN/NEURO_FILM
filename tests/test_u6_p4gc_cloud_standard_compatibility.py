import json
from pathlib import Path

from scripts.evaluate_u6_p4gc_cloud_standard_compatibility import scene,spread
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def test_p4gc_fresh_scene_generator_is_exact_and_nonconstant():
 a=scene(1709,32,48);b=scene(1709,32,48);c=scene(2017,32,48);assert np.array_equal(a,b);assert not np.array_equal(a,c);assert spread(a)[0]>.1

def test_p4gc_formal_result_closes_exact_profile_pairing():
 r=json.loads((ROOT/'docs/evidence/U6_P4GC_CLOUD_STANDARD_COMPATIBILITY_RESULT.json').read_text());assert r['decision']=='close-exact-composition-profile-incompatible';assert r['pass_rate']==0.;assert all(x['scan_p99_p01_range']>0 and x['physics_only_p99_p01_range']>0 and x['combined_p99_p01_range']==0 for x in r['rows'])
