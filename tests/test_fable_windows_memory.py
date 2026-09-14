import subprocess
import sys


def test_windows_rejects_allocation_above_process_limit():
    code = '''
from src.data.fable_windows_memory import limit_current_process_committed_memory
job = limit_current_process_committed_memory(64*1024**2)
small = bytearray(1024)
try:
    oversized = bytearray(128*1024**2)
except MemoryError:
    print('LIMIT_ENFORCED')
else:
    raise AssertionError('allocation escaped limit')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'LIMIT_ENFORCED' in result.stdout
