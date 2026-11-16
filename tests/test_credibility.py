import builtins

import pytest

import credibility as cr


@pytest.mark.parametrize("url,tier,reason", [
    ("https://en.wikipedia.org/wiki/Eiffel_Tower", "High", "list:"),
    ("https://m.facebook.com/x", "Low", "list:"),
    ("https://www.instagram.com/reel/1", "Low", "list:"),
    ("https://www.cdc.gov/a", "High", "rule:"),
    ("https://www.gov.uk/x", "High", "rule:"),
    ("https://web.mit.edu/x", "High", "rule:"),
    ("https://www.iitm.ac.in/", "High", "rule:"),
    ("https://www.who.int/", "High", "list:"),
    ("https://pmc.ncbi.nlm.nih.gov/articles/PMC4853817", "High", "list:"),
    ("https://www.reuters.com/x", "High", "list:"),
    ("https://www.infowars.com/x", "Low", "rating:"),
    ("https://www.bbc.co.uk/news", "High", "rating:"),
    ("https://www.nytimes.com/x", "High", "rating:"),
    ("https://www.foxnews.com/x", "Medium", "rating:"),
    ("https://some-unknown-blog.example/x", "Unrated", "unrated"),
    ("not a url", "Unrated", "unrated"),
    ("", "Unrated", "unrated"),
    (None, "Unrated", "unrated"),
])
def test_trust(url, tier, reason):
    t = cr.trust(url)
    assert t["tier"] == tier and t["trust_reason"].startswith(reason), t
    assert t["points"] == cr.TIERS[tier]


def test_low_list_beats_rating(monkeypatch):
    monkeypatch.setattr(cr, "_ratings", lambda: {"youtube.com": 0.99})
    assert cr.trust("https://youtube.com/watch")["tier"] == "Low"


@pytest.mark.parametrize("host,reg", [
    ("www.bbc.co.uk", "bbc.co.uk"), ("edition.cnn.com", "cnn.com"),
    ("a.b.example.com.au", "example.com.au"), ("localhost:8501", "localhost"),
])
def test_registered_domain(host, reg):
    assert cr.registered_domain(host) == reg


def test_high_domains():
    hd = cr.high_domains()
    assert len(hd) <= 300 and len(set(hd)) == len(hd)
    assert "wikipedia.org" in hd and set(cr.HIGH_LIST) <= set(hd)
    assert hd == cr.high_domains()
    assert len(cr.high_domains(10)) == 10


def test_csv_read_once(monkeypatch):
    cr._ratings.cache_clear()
    reads = []
    real = builtins.open

    def counting(f, *a, **k):
        if str(f).endswith("domain_pc1.csv"):
            reads.append(f)
        return real(f, *a, **k)
    monkeypatch.setattr(builtins, "open", counting)
    for _ in range(100):
        cr.trust("https://www.nytimes.com/x")
    assert len(reads) == 1
