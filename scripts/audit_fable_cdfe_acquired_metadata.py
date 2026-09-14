import hashlib
import io
import json
import logging
from pathlib import Path

import rawpy

from scripts.audit_fable_cdfe_local_history import read_fields
from src.preprocess.fable_raw_eligibility import inspect_numeric_metadata, inspect_decoder_numeric


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'outputs/fable_cdfe68_acquisition_v1/ledger.json'
    blob = source.read_bytes()
    ledger = json.loads(blob)
    if ledger.get('status') != 'BOUNDED_TRANSPORT_PASS_COMPLETE_NOT_SOURCE_ADMISSION':
        raise ValueError('transport not complete; do not open changing acquisition frame')
    output = root / 'outputs/fable_cdfe68_acquisition_v1/metadata.jsonl'
    with output.open('x', encoding='utf-8') as stream:
        for entry in ledger['entries']:
            record = ledger['objects'][entry['identity']]
            result = {'identity': entry['identity'], 'transport_status': record['status'],
                      'ledger_sha256': hashlib.sha256(blob).hexdigest()}
            if record['status'] in ('FETCHED', 'LOCAL_REUSED'):
                path = Path(record['path'])
                with path.open('rb') as raw:
                    digest = hashlib.file_digest(raw, 'sha256').hexdigest()
                if digest != record['sha256']:
                    raise ValueError('acquired original changed')
                result['raw_sha256'] = digest
                warnings = io.StringIO()
                handler = logging.StreamHandler(warnings)
                logger = logging.getLogger('tifffile')
                logger.addHandler(handler)
                try:
                    result['ifd_metadata'] = read_fields(path)
                    if entry['identity'].startswith('fivek/'):
                        numeric = inspect_numeric_metadata(path)
                        with rawpy.imread(str(path)) as raw:
                            result['decoder_numeric'] = inspect_decoder_numeric(raw, numeric)
                            result['pattern'] = raw.raw_pattern.tolist() if raw.raw_pattern is not None else None
                            result['camera_whitebalance'] = raw.camera_whitebalance
                            result['geometry'] = {'width': raw.sizes.width, 'height': raw.sizes.height,
                                                  'flip': raw.sizes.flip, 'pixel_aspect': raw.sizes.pixel_aspect}
                        result['numeric'] = numeric
                    result['status'] = 'METADATA_READ_NOT_RELATION_ADMISSION'
                except (ValueError, OSError, TypeError, rawpy.LibRawError) as error:
                    result.update(status='METADATA_UNRESOLVED_NO_RETRY', error=f'{type(error).__name__}: {error}')
                finally:
                    logger.removeHandler(handler)
                    result['reader_warnings'] = warnings.getvalue().splitlines()
            else:
                result['status'] = 'UNAVAILABLE_NO_RETRY'
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
    print('Fixed acquisition metadata pass complete')


if __name__ == '__main__':
    main()
