import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    start = time.perf_counter()
    cp = ROOT/'configs/fable_blueneg_native_v1.json'
    cfg = json.loads(cp.read_text())
    assert digest(ROOT/'data/raw/blueneg/meta.json') == cfg['metadata_sha256']
    assert digest(ROOT/'data/raw/blueneg/remote_inventory.json') == cfg['inventory_sha256']
    root = ROOT/cfg['output_root']
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in cfg['files']:
        if record['role'] != 'development':
            continue
        print(json.dumps({'event': 'fetch', 'roll_id': record['roll_id'], 'bytes': record['size']}), flush=True)
        path = Path(hf_hub_download(repo_id=cfg['repo_id'], repo_type='dataset',
            filename=record['remote_path'], revision=cfg['revision'], local_dir=root))
        assert path.stat().st_size == record['size']
        actual = digest(path)
        assert actual == record['sha256']
        rows.append({'roll_id': record['roll_id'], 'path': str(path.relative_to(ROOT)),
                     'bytes': path.stat().st_size, 'sha256': actual})
        report = {'status': 'COMPLETE' if len(rows) == 3 else 'PARTIAL', 'files': rows,
            'config_sha256': digest(cp), 'entry_sha256': digest(Path(__file__)),
            'wall_seconds': time.perf_counter()-start, 'pixels_decoded': False,
            'sealed_validation_downloaded': False, 'required_credit': cfg['required_credit']}
        (root/'download_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'verified', 'roll_id': record['roll_id']}), flush=True)


if __name__ == '__main__':
    main()
