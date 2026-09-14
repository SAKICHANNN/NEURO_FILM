import hashlib

from src.training.fable_source_closure import local_import_closure


def test_transitive_relative_import_and_initializer_are_bound(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/__init__.py').write_text('from . import side\n')
    (tmp_path / 'src/side.py').write_text('VALUE = 3\n')
    (tmp_path / 'src/a.py').write_text('from src.b import fn\n')
    (tmp_path / 'src/b.py').write_text('from .side import VALUE\ndef fn(): return VALUE\n')
    closure = local_import_closure(tmp_path, ['src/a.py'])
    assert set(closure) == {'src/a.py', 'src/b.py', 'src/side.py', 'src/__init__.py'}
    assert closure['src/side.py'] == hashlib.sha256((tmp_path/'src/side.py').read_bytes()).hexdigest()
