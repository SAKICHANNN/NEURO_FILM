import importlib.metadata
import platform
import sys


def computation_runtime() -> dict:
    return {'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ['numpy', 'torch', 'scipy', 'Pillow', 'rawpy', 'pywin32']}}
