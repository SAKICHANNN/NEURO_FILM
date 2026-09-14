import pytest

from scripts.build_fable_cdfe_history_graph import camera_day, components


def test_transitive_union_retains_history_link_across_dates():
    edges = [{'left': 'history', 'right': 'pilot'}, {'left': 'pilot', 'right': 'candidate'}]
    assert components(['history', 'pilot', 'candidate', 'other'], edges) == {
        'candidate': ['candidate', 'history', 'pilot'], 'other': ['other']}


def test_unknown_edge_endpoint_rejected():
    with pytest.raises(KeyError):
        components(['known'], [{'left': 'known', 'right': 'missing'}])


def test_camera_day_explicit_alias_and_strict_date():
    aliases = {'CANON REBEL': 'CANON 300D'}
    assert camera_day(' Canon   Rebel ', '2006:09:20 04:59:41', aliases) == ('CANON 300D', '2006-09-20')
    assert camera_day('Canon Rebel X', '2006:09:20 04:59:41', aliases) is None
    assert camera_day('Canon Rebel', '2006:02:30 04:59:41', aliases) is None
    assert camera_day('Canon Rebel', None, aliases) is None
