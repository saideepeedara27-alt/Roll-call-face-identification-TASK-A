"""Tests for intelligent filename parsing and identity-key normalization."""

from rollcall.filename_parser import parse_identity_filename


def test_standard_name():
    parsed = parse_identity_filename("Priya Sharma.jpg")
    assert parsed.display_name == "Priya Sharma"
    assert parsed.identity_key == "priya_sharma"


def test_underscores_and_mixed_case():
    parsed = parse_identity_filename("rahul_VERMA-photo2.PNG")
    assert parsed.display_name == "Rahul Verma"
    assert parsed.identity_key == "rahul_verma"


def test_multi_photo_numeric_suffix_maps_to_same_identity():
    p1 = parse_identity_filename("Ananya_Rao_1.jpg")
    p2 = parse_identity_filename("ananya rao 2.jpeg")
    p3 = parse_identity_filename("Ananya Rao (1).jpg")
    assert p1.identity_key == p2.identity_key == p3.identity_key == "ananya_rao"
    assert p1.display_name == "Ananya Rao"


def test_path_traversal_stripped():
    parsed = parse_identity_filename("../../secret/Meera_Nair.jpg")
    assert parsed.original_filename == "Meera_Nair.jpg"
    assert parsed.identity_key == "meera_nair"
