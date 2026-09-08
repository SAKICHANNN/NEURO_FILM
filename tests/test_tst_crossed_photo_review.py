import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from scipy.interpolate import RegularGridInterpolator
import tifffile

ROOT = Path('C:/Users/hhvrf/Documents/neuro_film')
spec = importlib.util.spec_from_file_location('photo_review', ROOT / 'scripts/prepare_tst_crossed_photo_review.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def grid():
    axis = np.linspace(0, 1, 7)
    return np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(343, 3)


def test_explicit_root_metadata_preflight():
    result = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'scripts/prepare_tst_crossed_photo_review.py')],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    checks = json.loads(result.stdout)
    assert checks['status'] in ('WAIT_S3_COMPLETE', 'READY') and checks['photo_reads'] == 0


def test_running_s3_never_reads_partial_records(tmp_path, monkeypatch):
    (tmp_path / 'running.lock').write_text('live')
    monkeypatch.setattr(runner, 'read', lambda path: pytest.fail('partial S3 read'))
    assert runner.completed_s3({'s3_output': str(tmp_path)}) is None


def test_native_render_independent_interpolation_and_rounding():
    source = np.random.default_rng(12).integers(0, 65536, (17, 29, 3), dtype=np.uint16)
    values = np.clip(grid() * [.8, 1., .9] + [.1, 0., .05], 0, 1)
    actual = runner.render_native(source, values, 47)
    axis = np.linspace(0, 1, 7)
    expected = RegularGridInterpolator((axis, axis, axis), values.reshape(7, 7, 7, 3))(source.astype(float)/65535)
    frozen = runner.s2.core.render_absolute(source.reshape(-1, 3).astype(float)/65535, values, 7).reshape(source.shape)
    np.testing.assert_allclose(frozen, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual, np.rint(np.clip(frozen, 0, 1)*65535).astype(np.uint16))
    independent_codes = np.rint(np.clip(expected, 0, 1)*65535).astype(np.uint16)
    mismatch = actual != independent_codes
    assert np.max(np.abs(actual.astype(int)-independent_codes.astype(int))) <= 1
    scaled = expected[mismatch]*65535
    np.testing.assert_allclose(scaled-np.floor(scaled), .5, atol=1e-9, rtol=0)


def test_patch_rectangles_are_exact_native_pixels_and_source_only():
    source = np.random.default_rng(7).integers(0, 65536, (87, 105, 3), dtype=np.uint16)
    boxes = runner.donor_boxes(source.shape, 32)
    board, layout = runner.panel([source, source], ['original', 'unchanged'], boxes, 'synthetic native evidence', 64)
    decoded = np.asarray(Image.open(io.BytesIO(runner.png_bytes(board))))
    for row in layout:
        x0, y0, x1, y1 = row['source_xyxy']
        bx0, by0, bx1, by1 = row['board_xyxy']
        np.testing.assert_array_equal(decoded[by0:by1, bx0:bx1], np.asarray(runner.display(source[y0:y1, x0:x1])))
    assert boxes == runner.donor_boxes(np.zeros_like(source).shape, 32)


def test_cache_limit_preserves_existing_artifacts(tmp_path):
    (tmp_path / 'existing').write_bytes(b'keep')
    with pytest.raises(runner.s2.IdentityFailure, match='CACHE_LIMIT'):
        runner.put_artifact(tmp_path, 'new.png', Image.new('RGB', (10, 10)), {'bytes': 4, 'limit': 5})
    assert (tmp_path / 'existing').read_bytes() == b'keep' and not (tmp_path / 'new.png').exists()


def test_full_synthetic_candidate_five_sheets_and_external_decisions(tmp_path):
    shape = (128, 128, 3)
    source = np.random.default_rng(8).integers(0, 65536, shape, dtype=np.uint16)
    after = np.clip(source.astype(np.int32)+1200, 0, 65535).astype(np.uint16)
    parent = tmp_path / 's2'
    parent.mkdir()
    np.save(parent / 'absolute_grid.npy', grid())
    np.savez_compressed(parent / 'coordinates.npz', fit_flat_indices=np.arange(512), score_flat_indices=np.arange(512, 1024), shape=np.array(shape))
    artifacts = [{'path': name, 'sha256': runner.digest(parent / name), 'bytes': (parent / name).stat().st_size} for name in ('absolute_grid.npy', 'coordinates.npz')]
    runner.save(parent / 'record.json', {'synthetic': True})
    runner.save(tmp_path / 's3_record.json', {'synthetic': True})
    files, candidate = [], {'before': 'P.tif', 'after': 'R.tif'}
    for name, values, role in [('P.tif', source, 'before'), ('R.tif', after, 'after')]:
        path = tmp_path / name
        tifffile.imwrite(path, values, photometric='rgb')
        files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': runner.digest(path)})
        token = runner.hashlib.sha256(name.encode()).hexdigest()
        header_path = tmp_path / 'headers' / (token+'.json')
        runner.save(header_path, {'header': {'eligible_decode': True, 'format': 'TIFF', 'width': 128, 'height': 128}})
        candidate[role+'_header_sha256'] = runner.digest(header_path)
        candidate[role+'_encoded_sha256'] = runner.digest(path)
    candidate['before_canonical_sha256'] = runner.s2.corpus.canonical_hash(source)
    runner.save(tmp_path / 'files.json', {'files': files})
    probe_rows = [{'split': 'fit', 'component': str(i), 'category': 'synthetic', 'representative': str(i), 'crops': [{'box_xyxy_exclusive': [0, 0, 40, 50]}, {'box_xyxy_exclusive': [70, 70, 120, 120]}]} for i in range(4)]
    runner.save(tmp_path / 'probes.json', {'rows': probe_rows})
    probes = [runner.s3.core.VisibilityProbe(str(i), 'fit', source) for i in range(4)]
    item = {'operator_key': 'synthetic', 'split': 'fit', 's2_record_path': str(parent / 'record.json'), 's2_record_sha256': runner.digest(parent / 'record.json'),
            's3_record_path': str(tmp_path / 's3_record.json'), 's3_record_sha256': runner.digest(tmp_path / 's3_record.json'),
            'candidate': candidate, 'canonical_after': runner.s2.corpus.canonical_hash(after), 'artifacts': artifacts}
    config = {'bound_tol': 1e-12, 'acquisition_config': str(tmp_path / 'files.json'), 'originals': str(tmp_path), 'headers': str(tmp_path / 'headers'),
              'chunk_pixels': 32768, 'donor_patch_side': 32, 'overview_side': 128, 'probe_manifest': str(tmp_path / 'probes.json')}
    directory = tmp_path / 'review'
    directory.mkdir()
    budget = {'bytes': 0, 'limit': 20_000_000}
    result = runner.prepare_candidate(config, item, probes, directory, budget)
    assert result['status'] == 'PREPARED_AWAITING_EXTERNAL_REVIEW' and not result['training_admitted']
    assert len(list(directory.glob('*.png'))) == 5 and not list(directory.glob('*.tif'))
    assert budget['bytes'] == sum(p.stat().st_size for p in directory.iterdir())
    template = runner.read(directory / 'review_template.json')
    assert template['status'] == 'PENDING_EXTERNAL_REVIEW' and template['final_decision'] is None
    assert not template['reviewed_artifact_hashes'] and len(template['required_artifact_hashes']) == 5
    assert len(result['support']) == 4 and all(r['native_probe_pixels'] == 16384 for r in result['support'])
    result.update(identity={}, input=item)
    runner.save(directory / 'record.json', result)
    assert runner.reusable(directory, {}, item) == result
    (directory / '01_probe.png').write_bytes(b'changed')
    with pytest.raises(runner.s2.IdentityFailure, match='ARTIFACT'):
        runner.reusable(directory, {}, item)


def test_process_tree_cpu_limit_has_durable_failure(tmp_path, monkeypatch):
    config = {'output': str(tmp_path), 'cpu_seconds': .8, 'wall_seconds': 20, 'study_cpu_seconds': 100, 'study_wall_seconds': 100, 'rss_bytes': 2147483648}
    checks = {'status': 'READY', 'identity': {}, 'accounting': {'stage_cpu': 0, 'stage_wall': 0, 'study_cpu': 0, 'study_wall': 0}}
    monkeypatch.setattr(runner, 'preflight', lambda *args: (config, {}, checks))
    original = subprocess.Popen
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda args, **kwargs: original([sys.executable, '-c', 'while True: pass'], **kwargs))
    assert runner.launch(tmp_path / 'config.json', 0) == 1
    record = runner.read(tmp_path / 'attempts/0001/accounting.json')
    assert record['failure'] == 'CPU_LIMIT' and record['aggregate_cpu_seconds'] >= .8
    assert record['peak_tree_rss_bytes'] > 0 and not (tmp_path / 'batches/0000.json').exists()


def test_completed_s3_only_survivors_receive_review_queue(tmp_path):
    output = tmp_path / 's3'
    runner.save(tmp_path / 's3_config.json', {})
    (tmp_path / 's3_entry.py').write_text('synthetic')
    items = [{'operator_key': str(i), 'split': 'fit', 's2_status': runner.s3.SOLVED if i < 2 else 'NUMERICAL_FAILURE_NOT_ADMITTED'} for i in range(3)]
    parent = tmp_path / 's2/record.json'
    runner.save(parent, {'candidate': {'before': 'synthetic'}, 'identity': {'s2': True}, 'canonical_after': 'canonical', 'artifacts': [], 'status': runner.s3.SOLVED})
    items[0].update(record_path=str(parent), record_sha256=runner.digest(parent))
    inputs = {'rows': items, 's2_identity': {'s2': True}}
    identity = {'input_sha256': runner.s3.object_hash(inputs), 'entry_sha256': runner.digest(tmp_path / 's3_entry.py'), 'config_sha256': runner.digest(tmp_path / 's3_config.json')}
    runner.save(output / 'input_lock.json', {'identity': identity, **inputs})
    statuses = [runner.SURVIVOR, 'PROVISIONAL_OPERATOR_NOT_VISIBLE_PAIR_UNREVIEWED', 'NOT_ELIGIBLE_S2_OUTCOME_RETAINED']
    for item, status in zip(items, statuses):
        runner.save(output / 'candidate' / item['operator_key'] / 'record.json', {'identity': identity, 'input': item, 'status': status, 'training_admitted': False,
             'strength': 1., 'threshold_mean_delta_e2000': 4., 'computed_visible_count': 3, 'uncomputed_count': 0, 'visibility_possible': True})
    accounting = {'exit_code': 0, 'failure': None, 'live': False}
    runner.save(output / 'attempts/0001/accounting.json', accounting)
    counts = dict(runner.Counter(statuses))
    runner.save(output / 'report.json', {'status': 'COMPLETE_VISIBILITY_SCREEN_NOT_ADMISSION', 'identity': identity, 'processed': 3,
         'counts': counts, 'counts_by_split': {'fit': counts}, 'last_attempt_accounting': accounting})
    config = {'s3_output': str(output), 's3_config': str(tmp_path / 's3_config.json'), 's3_entry': str(tmp_path / 's3_entry.py'), 'expected_candidates': 3}
    result = runner.completed_s3(config)
    assert len(result['outcomes']) == 3 and [r['operator_key'] for r in result['survivors']] == ['0']
    report = runner.read(output / 'report.json')
    report['processed'] = 2
    runner.save(output / 'report.json', report)
    with pytest.raises(runner.s2.IdentityFailure, match='INCOMPLETE'):
        runner.completed_s3(config)
