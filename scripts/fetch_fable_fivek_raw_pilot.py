import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/fable_source_population_bridge_v1'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=36)
    args = parser.parse_args()
    if not 1 <= args.limit <= 36:
        raise ValueError('limit must be in 1..36')
    manifest_path = BASE / 'raw_pilot_manifest_v1.json'
    manifest = json.loads(manifest_path.read_text())
    evidence = json.loads((ROOT / 'docs/evidence/FABLE_FIVEK_COMPARISONS_20260914.json').read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != evidence['stage_a']['pins'][manifest_path.name]:
        raise ValueError('pilot pin mismatch')
    if manifest['parent_sha256'] != hashlib.sha256((BASE / 'provisional_manifest_v1.json').read_bytes()).hexdigest():
        raise ValueError('parent pin mismatch')
    rows = manifest['rows']
    if len(rows) != 36 or len({r['identity'] for r in rows}) != 36:
        raise ValueError('fixed pilot must contain 36 unique identities')
    output = BASE / 'raw_pilot'
    output.mkdir(exist_ok=True)
    ledger_path = output / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        'manifest_sha256': manifest_sha, 'status': 'PREPARED', 'body_bytes': 0, 'objects': {}}
    if ledger['manifest_sha256'] != manifest_sha:
        raise ValueError('ledger input changed')
    if ledger['status'] == 'ATTEMPT_TERMINATED' or any(
            a['status'] == 'STARTED' for r in ledger['objects'].values() for a in r['attempts']):
        raise ValueError('terminal or uncertain attempt; no automatic restart')
    for identity, record in ledger['objects'].items():
        if identity not in {r['identity'] for r in rows} or record['status'] != 'FETCHED':
            raise ValueError('unexpected or incomplete existing object')
        if hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('existing RAW hash mismatch')
    session = requests.Session()
    session.headers['Accept-Encoding'] = 'identity'
    processed = 0
    for row in rows:
        identity = row['identity']
        if identity in ledger['objects']:
            continue
        if processed == args.limit:
            break
        source = row['source']
        url = urlsplit(source['dng_url'])
        expected = '/graphics/fivek/img/dng/' + source['source_name'] + '.dng'
        if url.scheme != 'https' or url.netloc != 'data.csail.mit.edu' or url.path != expected or url.query or url.fragment:
            raise ValueError('unexpected fixed RAW URL')
        encoded_url = urlunsplit((url.scheme, url.netloc, quote(url.path, safe='/%'), '', ''))
        record = {'identity': identity, 'url': source['dng_url'], 'status': 'PENDING', 'attempts': []}
        ledger['objects'][identity] = record
        for number in range(manifest['max_attempts_per_object']):
            attempt = {'number': number + 1, 'status': 'STARTED', 'body_bytes': 0}
            record['attempts'].append(attempt)
            ledger['status'] = 'RUNNING'
            save(ledger_path, ledger)
            started = time.monotonic()
            data = bytearray()
            terminal = False
            try:
                remaining = manifest['max_total_body_bytes_including_retries'] - ledger['body_bytes']
                if remaining <= 0:
                    terminal = True
                    raise ValueError('total byte cap reached')
                with session.get(encoded_url, timeout=(15, 30), stream=True, allow_redirects=False) as response:
                    attempt['http_status'] = response.status_code
                    length = response.headers.get('Content-Length')
                    length = int(length) if length is not None else None
                    if response.status_code != 200:
                        raise ValueError(f'HTTP {response.status_code}; no redirect fallback')
                    if length is not None and (length > manifest['max_object_bytes'] or length > remaining):
                        terminal = True
                        raise ValueError('announced body exceeds frozen cap')
                    while True:
                        capacity = min(manifest['max_object_bytes'] - len(data), remaining)
                        if capacity <= 0:
                            if length is not None and len(data) == length:
                                break
                            terminal = True
                            raise ValueError('body cap reached without verified EOF')
                        allowance = min(65536, capacity)
                        attempt['body_bytes'] += allowance
                        ledger['body_bytes'] += allowance
                        try:
                            block = response.raw.read(allowance)
                        except Exception:
                            terminal = True
                            attempt['accounting'] = 'CONSERVATIVE_RESERVED_BYTES_READ_UNCERTAIN'
                            raise
                        unused = allowance - len(block)
                        attempt['body_bytes'] -= unused
                        ledger['body_bytes'] -= unused
                        if not block:
                            break
                        data.extend(block)
                        remaining -= len(block)
                    if length is not None and len(data) != length:
                        raise ValueError('incomplete body')
                if data[:4] not in (b'II*\x00', b'MM\x00*'):
                    raise ValueError('body is not a classic TIFF/DNG container')
                target = output / (source['dataset_id'] + '.dng')
                if target.exists():
                    raise ValueError('unaccounted target already exists')
                target.write_bytes(data)
                record.update(status='FETCHED', path=target.relative_to(ROOT).as_posix(),
                              sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
                attempt['status'] = 'SUCCESS'
            except Exception as error:
                attempt.update(status='FAILED', error=type(error).__name__ + ': ' + str(error)[:300])
            attempt['elapsed_seconds'] = time.monotonic() - started
            save(ledger_path, ledger)
            if record['status'] == 'FETCHED' or terminal:
                break
        if record['status'] != 'FETCHED':
            record['status'] = 'FAILED'
            ledger['status'] = 'ATTEMPT_TERMINATED'
            save(ledger_path, ledger)
            break
        processed += 1
        ledger['status'] = 'FETCH_COMPLETE' if len(ledger['objects']) == 36 else 'BATCH_PAUSED'
        save(ledger_path, ledger)
    print(json.dumps({'status': ledger['status'], 'fetched': sum(
        r['status'] == 'FETCHED' for r in ledger['objects'].values()), 'body_bytes': ledger['body_bytes']}))


def main():
    lock = BASE / 'raw_pilot_runner.lock'
    handle = lock.open('x')
    try:
        run()
    finally:
        handle.close()
        lock.unlink()


if __name__ == '__main__':
    main()
