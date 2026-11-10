from utils import extract_domain


def test_extract_domain_only_strips_a_leading_www_label() -> None:
    assert extract_domain("https://www.example.com/jobs") == "example.com"
    assert extract_domain("https://notwww.example.com") == "notwww.example.com"
    assert extract_domain("https://shop.www.example.com") == "shop.www.example.com"
    assert extract_domain("https://www.mywww.example.com") == "mywww.example.com"
