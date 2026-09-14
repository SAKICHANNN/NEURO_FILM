import json
import subprocess
import sys
from pathlib import Path


def test_draft_cannot_allocate_or_start_training(tmp_path):
    contract = tmp_path / 'draft.json'
    contract.write_text(json.dumps({'status': 'DRAFT_NOT_TRAINING_AUTHORIZATION'}))
    result = subprocess.run([sys.executable, '-m', 'scripts.train_fable_cdfe',
                             '--contract', str(contract)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'draft cannot execute' in result.stderr
