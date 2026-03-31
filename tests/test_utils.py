from egyptian_id_ocr.utils import normalize_digits, normalize_arabic_text, validate_egyptian_id


def test_normalize_digits():
    assert normalize_digits("١٢٣٤٥") == "12345"


def test_normalize_arabic_text():
    assert normalize_arabic_text("أحـمد   علي") == "احمد علي"


def test_validate_egyptian_id():
    ok, _ = validate_egyptian_id("30201011234567")
    assert ok
    ok2, _ = validate_egyptian_id("1123")
    assert not ok2
