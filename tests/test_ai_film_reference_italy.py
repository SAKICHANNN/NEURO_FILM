from scripts.acquire_ai_film_reference_italy import Images, sha


def test_image_parser_ignores_links():
    parser = Images()
    parser.feed('<img src="a.jpg"><a href="b.jpg">link</a>')
    assert parser.urls == ["a.jpg"]


def test_hash_known_vector():
    assert sha(b"test") == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
