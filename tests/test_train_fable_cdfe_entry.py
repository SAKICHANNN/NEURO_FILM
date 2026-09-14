import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('module,extra', [('scripts.train_fable_cdfe', []),
    ('scripts.validate_fable_cdfe', []), ('scripts.validate_fable_cdfe', ['--assessment'])])
def test_draft_cannot_allocate_or_start_training(tmp_path, module, extra):
    contract = tmp_path / 'draft.json'
    contract.write_text(json.dumps({'status': 'DRAFT_NOT_TRAINING_AUTHORIZATION'}))
    result = subprocess.run([sys.executable, '-m', module,
                             '--contract', str(contract), *extra],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'sealed training contract required' in result.stderr
