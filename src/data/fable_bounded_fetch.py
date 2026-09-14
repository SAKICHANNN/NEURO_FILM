import hashlib
import json
from pathlib import Path


def save_ledger(path, ledger):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    temporary.replace(path)


def fetch_once(entry: dict, *, session, directory: Path, ledger_path: Path, ledger: dict, cap: int) -> dict:
    identity = entry['identity']
    if any(r['status'] == 'STARTED' for r in ledger['objects'].values()):
        raise ValueError('uncertain prior attempt; no automatic restart')
    if identity in ledger['objects']:
        return ledger['objects'][identity]
    remaining = cap - ledger['charged_body_bytes']
    record = {'identity': identity, 'url': entry['url'], 'status': 'BUDGET_EXHAUSTED', 'charged_bytes': 0}
    ledger['objects'][identity] = record
    if remaining <= 0:
        save_ledger(ledger_path, ledger)
        return record
    record['status'] = 'STARTED'
    # Reserve the remaining budget before a request so a crash cannot hide uncertain reads.
    record['charged_bytes'] = remaining
    ledger['charged_body_bytes'] += remaining
    save_ledger(ledger_path, ledger)
    consumed = 0
    pending_read = 0
    allowance = remaining
    part = directory / (hashlib.sha256(identity.encode()).hexdigest() + '.part')
    try:
        with session.get(entry['url'], stream=True, timeout=(15, 45), allow_redirects=False,
                         headers={'Accept-Encoding': 'identity'}) as response:
            length = response.headers.get('Content-Length')
            length = int(length) if length is not None else None
            if response.status_code != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('non-200 response or unsupported content encoding')
            if length is not None and (length < 0 or length > remaining):
                raise ValueError('announced body exceeds remaining budget')
            if length is not None:
                allowance = length
                ledger['charged_body_bytes'] -= remaining - length
                record['charged_bytes'] = length
                save_ledger(ledger_path, ledger)
            digest = hashlib.sha256()
            with part.open('xb') as stream:
                while consumed < allowance:
                    pending_read = min(1024 * 1024, allowance - consumed)
                    block = response.raw.read(pending_read)
                    pending_read = 0
                    if not block:
                        break
                    consumed += len(block)
                    digest.update(block)
                    stream.write(block)
            if length is not None and consumed != length:
                raise ValueError('incomplete declared body')
            if length is None and consumed == allowance:
                raise ValueError('budget limit reached without established EOF')
            actual = digest.hexdigest()
            if entry.get('expected_raw_sha256') and actual != entry['expected_raw_sha256']:
                raise ValueError('downloaded original hash mismatch')
            target = part.with_suffix('.raw')
            if target.exists():
                raise ValueError('unaccounted completed target exists')
            part.replace(target)
            record.update(status='FETCHED', path=str(target), sha256=actual, bytes=consumed)
            ledger['charged_body_bytes'] -= record['charged_bytes'] - consumed
            record['charged_bytes'] = consumed
    except Exception as error:
        charge = consumed + pending_read
        ledger['charged_body_bytes'] -= record['charged_bytes'] - charge
        record['charged_bytes'] = charge
        record.update(status='FAILED_NO_RETRY', error=f'{type(error).__name__}: {error}',
                      accounting='Application body bytes plus in-flight read allowance; not TCP/TLS wire bytes')
    record['bytes_returned_to_caller'] = consumed
    save_ledger(ledger_path, ledger)
    return record
