import copy
import hashlib

import pytest

from scripts.prepare_creative_look_portraits import validate_metadata, verify_payload


def fixture():
    row = {
        "page_id": 1,
        "title": "File:Example.jpg",
        "page_revision": 2,
        "size": 3,
        "sha1": hashlib.sha1(b"abc").hexdigest(),
        "width": 10,
        "height": 20,
        "url": "https://upload.wikimedia.org/example.jpg",
    }
    info = {key: row[key] for key in ("size", "sha1", "width", "height", "url")}
    info["extmetadata"] = {"LicenseShortName": {"value": "CC0"}}
    return {"rows": [row]}, {
        "1": {"title": row["title"], "revisions": [{"revid": 2}], "imageinfo": [info]}
    }


def test_exact_metadata_and_payload():
    config, pages = fixture()
    validate_metadata(config, pages)
    verify_payload(b"abc", config["rows"][0])


@pytest.mark.parametrize("key", ["size", "sha1", "width", "height", "url"])
def test_metadata_drift(key):
    config, pages = fixture()
    pages["1"]["imageinfo"][0][key] = "wrong"
    with pytest.raises(ValueError):
        validate_metadata(config, pages)


def test_license_and_revision_drift():
    config, pages = fixture()
    other = copy.deepcopy(pages)
    other["1"]["imageinfo"][0]["extmetadata"]["LicenseShortName"]["value"] = "unknown"
    with pytest.raises(ValueError):
        validate_metadata(config, other)
    pages["1"]["revisions"][0]["revid"] = 3
    with pytest.raises(ValueError):
        validate_metadata(config, pages)


@pytest.mark.parametrize("payload", [b"ab", b"abcd", b"xyz"])
def test_payload_mismatch(payload):
    config, _ = fixture()
    with pytest.raises(ValueError):
        verify_payload(payload, config["rows"][0])
