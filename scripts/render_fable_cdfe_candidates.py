import hashlib
import json
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

from src.preprocess.fable_canonical_raw import linear16_to_q8
from src.preprocess.fable_screened_render import render_screened_raw, file_sha256
from src.preprocess.fable_render_lock import verify_render_lock


def main():
    root = Path(__file__).resolve().parents[1]
    relative = ['outputs/fable_cdfe68_exposure_v1/strict_metadata_prefilter.json',
                'outputs/fable_cdfe68_acquisition_v1/ledger.json',
                'configs/fable_cdfe68_screen_lock_v1.json']
    blobs = [(root / name).read_bytes() for name in relative]
    prefilter, acquisition, seal = map(json.loads, blobs)
    verify_render_lock(root, json.loads((root / seal['render_lock']).read_text()))
    selected = [r for r in prefilter['rows'] if r['metadata_prefilter_pass'] is True]
    if len(selected) != 114 or len({r['identity'] for r in selected}) != 114:
        raise ValueError('fixed 114-candidate frame changed')
    output = root / 'outputs/fable_cdfe68_native_candidates_v1'
    output.mkdir(exist_ok=False)
    frame = {'status': 'NATIVE_RENDER_QUALIFICATION_NOT_ROLE_ADMISSION',
             'source_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in zip(relative, blobs)},
             'runner_sha256': file_sha256(Path(__file__)),
             'identities': [r['identity'] for r in selected]}
    (output / 'frame.json').write_text(json.dumps(frame, indent=2))
    with (output / 'renders.jsonl').open('x', encoding='utf-8') as stream:
        for candidate in selected:
            identity = candidate['identity']
            source = acquisition['objects'][identity]
            if source['sha256'] != candidate['raw_sha256']:
                raise ValueError('prefilter and transport RAW binding differ')
            record = {'identity': identity, 'raw_sha256': source['sha256']}
            try:
                linear, evidence = render_screened_raw(Path(source['path']), root=root,
                    seal=seal, expected_raw_sha256=source['sha256'])
            except (ValueError, rawpy.LibRawError) as error:
                record.update(status='RENDER_CONTRACT_REJECTED_NO_REPLACEMENT', error=str(error))
            else:
                stem = hashlib.sha256(identity.encode()).hexdigest()
                native = output / (stem + '.npy')
                np.save(native, linear, allow_pickle=False)
                preview = Image.fromarray(linear16_to_q8(linear))
                preview.thumbnail((1200, 1200))
                preview_path = output / (stem + '.jpg')
                preview.save(preview_path, quality=95)
                record.update(status='LOCKED_NATIVE_RENDER_PASSED_NOT_ROLE_ADMISSION',
                    native_path=str(native), native_sha256=file_sha256(native),
                    preview_path=str(preview_path), preview_sha256=file_sha256(preview_path),
                    evidence=evidence)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(identity, record['status'], flush=True)
    (output / 'complete.json').write_text(json.dumps({'count': len(selected),
        'renders_sha256': file_sha256(output / 'renders.jsonl')}))


if __name__ == '__main__':
    main()
