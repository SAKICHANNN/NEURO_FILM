import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src.data.fable_bounded_fetch import fetch_once, save_ledger


ROOT = Path(__file__).resolve().parents[1]
FRAME = 'outputs/fable_cdfe68_exposure_v1/metadata_action_frame.json'
FRAME_SHA = 'a84e0cf89b46236da39e82dd37b56e850e8066d0ff3d0792704a488aba2831d5'


def entries(frame):
    result = [{'identity': r['identity'], 'url': r['url'], 'expected_raw_sha256': r['expected_raw_sha256']}
              for r in frame['history_rows'] if r['action'] == 'REMOTE_ONCE_PENDING']
    result.extend({'identity': r['identity'], 'url': r['source']['dng_url'], 'expected_raw_sha256': None}
                  for r in frame['candidate_rows'])
    if len(result) != 452 or len({r['identity'] for r in result}) != 452:
        raise ValueError('fixed remote frame changed')
    for row in result:
        url = urlsplit(row['url'])
        if url.scheme != 'https' or url.netloc not in {'data.csail.mit.edu', 'raw.pixls.us'} or url.query or url.fragment:
            raise ValueError('unexpected original URL')
    return result


def validate_resume(ledger, plan, cap):
    if ledger['frame_sha256'] != FRAME_SHA or ledger['entries'] != plan:
        raise ValueError('frame or execution plan changed')
    if set(ledger['objects']) - {r['identity'] for r in plan}:
        raise ValueError('unknown ledger identity')
    if ledger['charged_body_bytes'] != sum(r['charged_bytes'] for r in ledger['objects'].values()) or not 0 <= ledger['charged_body_bytes'] <= cap:
        raise ValueError('ledger accounting mismatch')
    for record in ledger['objects'].values():
        if record['status'] == 'STARTED':
            raise ValueError('uncertain attempt blocks automatic resume')
        if record['status'] in ('FETCHED', 'LOCAL_REUSED'):
            with Path(record['path']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
                    raise ValueError('retained original hash mismatch')


def local_reuse(root, plan):
    wanted = {r['identity'].split('/', 1)[1] + '.dng': r for r in plan if r['identity'].startswith('fivek/')}
    roots = [root / 'data', root / 'outputs/fivek_auto_optimize/freeze_v1/gold64_original_samples']
    command = ['rg', '--files', '--no-ignore', '-g', '*.dng', '-g', '*.DNG', *[str(p) for p in roots if p.exists()]]
    listing = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
    matches = {}
    for name in sorted(listing.stdout.splitlines()):
        path = Path(name)
        if path.name not in wanted:
            continue
        row = wanted[path.name]
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if row.get('expected_raw_sha256') and digest != row['expected_raw_sha256']:
            raise ValueError('local original conflicts with recorded digest')
        if row['identity'] in matches and matches[row['identity']]['sha256'] != digest:
            raise ValueError('same local identity has conflicting bytes')
        matches[row['identity']] = {'identity': row['identity'], 'url': row['url'],
            'status': 'LOCAL_REUSED', 'path': str(path), 'sha256': digest, 'charged_bytes': 0}
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    blob = (ROOT / FRAME).read_bytes()
    if hashlib.sha256(blob).hexdigest() != FRAME_SHA:
        raise ValueError('frozen metadata frame digest differs')
    frame = json.loads(blob)
    plan = entries(frame)
    output = ROOT / 'outputs/fable_cdfe68_acquisition_v1'
    if not args.execute and not args.prepare:
        print(json.dumps({'status': 'DRY_RUN', 'remote_frame': len(plan), 'cap': frame['combined_new_transfer_ceiling_bytes']}))
        return
    output.mkdir(exist_ok=True)
    lock = output / 'runner.lock'
    handle = lock.open('x')
    try:
        ledger_path = output / 'ledger.json'
        ledger = json.loads(ledger_path.read_bytes()) if ledger_path.exists() else {
            'frame_sha256': FRAME_SHA, 'entries': plan, 'objects': local_reuse(ROOT, plan), 'charged_body_bytes': 0}
        cap = frame['combined_new_transfer_ceiling_bytes']
        validate_resume(ledger, plan, cap)
        save_ledger(ledger_path, ledger)
        if not args.execute:
            print(json.dumps({'status': 'PREPARED', 'local_reused': len(ledger['objects']), 'entries': len(plan)}))
            return
        with requests.Session() as session:
            for entry in plan:
                fetch_once(entry, session=session, directory=output, ledger_path=ledger_path, ledger=ledger, cap=cap)
        ledger['status'] = 'BOUNDED_TRANSPORT_PASS_COMPLETE_NOT_SOURCE_ADMISSION'
        save_ledger(ledger_path, ledger)
    finally:
        handle.close()
        lock.unlink()


if __name__ == '__main__':
    main()
