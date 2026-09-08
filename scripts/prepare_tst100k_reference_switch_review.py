import hashlib
import importlib.util
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/tst100k_reference_switch_fit_review_v1.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    assert len(config['groups']) == 24 and len(config['files']) == 120
    assert all(g['split'] == 'development_fit' for g in config['groups'])
    needed = {r[k] for g in config['groups'] for r in g['triplets']
              for k in ('content', 'reference', 'gt')}
    assert needed == {f['path'] for f in config['files']}
    assert sum(f['bytes'] for f in config['files']) == config['total_bytes'] == 90560009
    decoder_path = ROOT / 'scripts/prepare_tst100k_target_review.py'
    assert digest(decoder_path) == config['decoder_script_sha256']
    spec = importlib.util.spec_from_file_location('tst_decoder', decoder_path)
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    out = ROOT / config['output']
    out.mkdir(parents=True, exist_ok=False)
    decoder.OUT = out
    shutil.copyfile(CONFIG, out / 'config.json')
    shutil.copytree(ROOT / config['terms_source'], out / 'terms')
    report = {'status': 'ACQUIRING', 'config_sha256': digest(CONFIG),
              'script_sha256': digest(Path(__file__)), 'files': [], 'groups': [],
              'check_files_acquired': 0, 'model_forwards': 0, 'training': 0,
              'scope': config['scope'], 'per_image_rights_verified': False}

    def save():
        pending = out / 'report.pending'
        pending.write_text(json.dumps(report, indent=2), encoding='utf-8')
        pending.replace(out / 'report.json')

    def fetch(entry):
        relative = Path(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert entry['revision'] == config['revision']
        target = out / 'originals' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/ToneStyle/TST100K/resolve/{config['revision']}/{entry['path']}"
        request = urllib.request.Request(url, headers={'Accept-Encoding': 'identity'})
        partial = target.with_suffix(target.suffix + '.part')
        count = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as stream:
            while chunk := response.read(65536):
                count += len(chunk)
                assert count <= entry['bytes']
                stream.write(chunk)
        assert count == entry['bytes'] and digest(partial) == entry['etag']
        partial.rename(target)
        return {**entry, 'sha256': digest(target)}

    save()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, f) for f in config['files']]
        for job in as_completed(jobs):
            report['files'].append(job.result())
            save()
    assert len(report['files']) == 120
    report['status'] = 'RENDERING'
    save()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    for group in config['groups']:
        index = group['group_index']
        first, second = group['triplets']
        roles = [('X', first['content']), ('R1', first['reference']), ('Y1', first['gt']),
                 ('R2', second['reference']), ('Y2', second['gt'])]
        board = Image.new('RGB', (2500, 600), (238, 238, 238))
        draw = ImageDraw.Draw(board)
        record = {'group_index': index, 'split': group['split'], 'arms': {}}
        for column, (role, relative) in enumerate(roles):
            source = out / 'originals' / relative
            expected = next(f['etag'] for f in config['files'] if f['path'] == relative)
            assert digest(source) == expected
            preview, metadata = decoder.display_decode(source, out / 'display_srgb16' / f'{index:02d}_{role}.tif')
            metadata['original_path'] = relative
            record['arms'][role] = metadata
            preview.thumbnail((490, 540), Image.Resampling.LANCZOS)
            board.paste(preview, (column * 500 + (500-preview.width)//2, 30+(540-preview.height)//2))
            draw.text((column*500+8, 8), role, fill='black')
        draw.text((8, 580), f'Fit group {index:02d}: supplied targets; fixed strength1; no model prediction', fill='black')
        dest = out / 'review' / f'{index:02d}.png'
        dest.parent.mkdir(exist_ok=True)
        board.save(dest, icc_profile=profile)
        record['board'] = {'path': str(dest.relative_to(out)), 'sha256': digest(dest)}
        record['same_size_x_y1_y2'] = len({tuple(record['arms'][r]['encoded_size']) for r in ('X', 'Y1', 'Y2')}) == 1
        report['groups'].append(record)
        save()
    report['status'] = 'READY_FOR_24_FIT_GROUP_REVIEW_NOT_TRAINING'
    report['total_original_bytes'] = sum(f['bytes'] for f in report['files'])
    report['distinct_pixel_files_decoded'] = 120
    save()
    print(report['status'], len(report['files']), len(report['groups']))


if __name__ == '__main__':
    main()
