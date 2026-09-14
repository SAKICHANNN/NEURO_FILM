import numpy as np
import pytest
import hashlib

from src.eval.fable_cdfe_visual_panel import render_codes, render_comfort_panel
from src.eval.fable_reference_photometry import transform


def test_native_render_matches_direct_transform_and_gate_blocks(tmp_path):
    codes = np.random.default_rng(8).integers(0, 256, (13, 19, 3), dtype=np.uint8)
    u = np.array([.7, -.8, .5, -.4])
    expected = np.floor(255*transform(codes/255, u, slope_limit=1.25, offset_limit=.35)+.5).astype(np.uint8)
    np.testing.assert_array_equal(render_codes(codes, u), expected)
    with pytest.raises(ValueError, match='numeric gate'):
        render_comfort_panel(tmp_path/'panel', {}, {'gates': {'numeric_passed': False}}, {})
    assert not (tmp_path/'panel').exists()


def test_complete_fixed_panel_preserves_all_native_outputs(tmp_path):
    queries = []
    for q in range(4):
        path = tmp_path / f'q{q}.npy'
        np.save(path, np.full((3, 5, 3), 12000+q*3000, dtype=np.uint16))
        queries.append({'identity': f'q{q}', 'native_path': str(path),
                        'native_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    ids = [str(j) for j in range(32)]
    cases = [{'donor': str(j//2), 'query': f'q{j%4}', 'treatment': str(j%32)} for j in range(64)]
    plan = {'queries': queries, 'comfort': {'cases': cases},
            'treatments': [{'id': i, 'u': [0., .7, -.5, .4]} for i in ids]}
    report = {'gates': {'numeric_passed': True}, 'identities': ids, 'treatment_ids': ids}
    arrays = {key: np.zeros((32, 32, 4)) for key in ['learned_u', 'paired_oracle_u']}
    directory = tmp_path / 'panel'
    manifest = render_comfort_panel(directory, plan, report, arrays)
    assert len(manifest['cases']) == 64
    assert len(manifest['image_sha256']) == 196
    assert [row['index'] for row in manifest['cases']] == list(range(64))
    from PIL import Image
    for name, digest in manifest['image_sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        with Image.open(directory/name) as image:
            assert image.size == (5, 3)
    page = (directory/'index.html').read_text(encoding='utf-8')
    assert page.count('<section>') == 64
    assert page.count('<img ') == 256
