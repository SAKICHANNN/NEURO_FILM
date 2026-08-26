from __future__ import annotations

import pytest

from scripts.audit_p245_dng_profile_huesatmap_real_file_guard import (
    _encoded_url,
    _snapshot_row,
)


def test_encoded_url_preserves_frozen_route_and_quotes_filename() -> None:
    result = _encoded_url(
        "https://raw.pixls.us/getfile.php/1052/nice/DJI - FC220 - 16bit (4:3).DNG"
    )
    assert result.startswith("https://raw.pixls.us/getfile.php/1052/nice/")
    assert " " not in result
    assert "DJI%20-%20FC220" in result


def test_snapshot_row_requires_one_exact_id_route() -> None:
    snapshot = {
        "data": [
            ["a", "https://raw.pixls.us/getfile.php/1/nice/a.dng"],
            ["b", "https://raw.pixls.us/getfile.php/2/nice/b.dng"],
        ]
    }
    assert _snapshot_row(snapshot, 2)[0] == "b"
    with pytest.raises(RuntimeError, match="found 0"):
        _snapshot_row(snapshot, 3)

