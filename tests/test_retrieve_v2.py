import asyncio

import pytest
import tavily

import evidence_cache as ec
import retrieve_evidence as rv


def claim(text="Paris is in France.", line=0, queries=("paris france", "paris france fact check")):
    return {"claim_text": text, "line_id": line, "speaker": "A", "start": 0.0,
            "queries": list(queries)}


def res(url, score=0.9, content="chunk one [...] chunk two", title="t"):
    return {"url": url, "score": score, "content": content, "title": title}


class Fake:
    """AsyncTavilyClient stand-in. responder(query) -> list of results or raises."""
    calls, in_flight, max_in_flight = [], 0, 0
    responder = staticmethod(lambda q: [res("https://a.org/x")])
    latency = 0

    def __init__(self, **kw):
        pass

    async def search(self, **kw):
        cls = type(self)
        cls.calls.append(kw)
        cls.in_flight += 1
        cls.max_in_flight = max(cls.max_in_flight, cls.in_flight)
        try:
            await asyncio.sleep(cls.latency)
            return {"results": cls.responder(kw["query"])}
        finally:
            cls.in_flight -= 1


@pytest.fixture
def fake(monkeypatch):
    class F(Fake):
        calls, in_flight, max_in_flight, latency = [], 0, 0, 0
    monkeypatch.setattr(tavily, "AsyncTavilyClient", F)
    monkeypatch.delenv("VERASCOPE_QUERIES", raising=False)
    return F


def run(claims):
    return asyncio.run(rv.retrieve_v2(claims))


def test_params(fake):
    run([claim()])
    assert len(fake.calls) == 2
    for kw in fake.calls:
        assert kw["search_depth"] == "basic" and kw["max_results"] == 8
        assert kw["chunks_per_source"] == 3 and kw["include_raw_content"] is False
        assert kw["include_domains_mode"] == "prefer" and len(kw["include_domains"]) <= 300
    assert [kw["query"] for kw in fake.calls] == ["paris france", "paris france fact check"]


def test_merge_same_domain(fake):
    fake.responder = staticmethod(lambda q: [res("https://en.wikipedia.org/A", 0.7, "a [...] b")]
                                  if "fact" not in q else
                                  [res("https://wikipedia.org/B", 0.95, "c [...] d [...] e")])
    ev, fails, _ = run([claim()])
    (src,) = ev["paris is in france"]["sources"]
    assert src["found_by"] == ["fact_check", "neutral"]
    assert src["url"] == "https://wikipedia.org/B" and src["tavily_score"] == 0.95
    assert src["chunks"] == ["c", "d", "e"] and src["source_id"] == 1
    assert ev["paris is in france"]["credits_used"] == 2 and not fails


@pytest.mark.parametrize("content,want", [
    ("a [...] b [...] c", ["a", "b", "c"]), ("no separator", ["no separator"]),
    ("[...]", []), ("x" * 3000, ["x" * 1500]),
])
def test_split_chunks(content, want):
    assert rv._split_chunks(content) == want


def test_score_min_fallback(fake):
    fake.responder = staticmethod(lambda q: [res("https://a.org", 0.3), res("https://b.org", 0.2)])
    ev, fails, _ = run([claim()])
    assert [s["domain"] for s in ev["paris is in france"]["sources"]] == ["a.org"]
    assert not fails


def test_zero_results(fake):
    fake.responder = staticmethod(lambda q: [])
    _, fails, _ = run([claim()])
    assert fails["paris is in france"] == "zero usable results"


def test_partial_and_total_failure(fake):
    def r(q):
        if "fact" not in q:
            raise RuntimeError("boom")
        return [res("https://a.org")]
    fake.responder = staticmethod(r)
    ev, fails, _ = run([claim()])
    e = ev["paris is in france"]
    assert len(e["sources"]) == 1 and e["search_failures"][0]["query_label"] == "neutral"
    assert not fails

    def boom(q):
        raise RuntimeError("down")
    fake.responder = staticmethod(boom)
    ev, fails, _ = run([claim()])
    assert "paris is in france" in fails and ev["paris is in france"]["sources"] == []


def test_one_query_env(fake, monkeypatch):
    monkeypatch.setenv("VERASCOPE_QUERIES", "1")
    run([claim()])
    assert [kw["query"] for kw in fake.calls] == ["paris france"]


def test_single_query_claim(fake):
    run([claim(queries=["only one"])])
    assert len(fake.calls) == 1


def test_dedupe(fake):
    _, _, back = run([claim(line=i) for i in (1, 5, 9)])
    assert len(fake.calls) == 2 and back["paris is in france"] == [1, 5, 9]


def test_concurrency(fake):
    fake.latency = 0.05
    run([claim(text=f"claim {i}") for i in range(10)])
    assert len(fake.calls) == 20 and fake.max_in_flight <= 4


def test_replay(fake, monkeypatch, tmp_path):
    monkeypatch.setenv("VERASCOPE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    a = run([claim()])
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")

    def dead(q):
        raise AssertionError("called in replay")
    fake.responder = staticmethod(dead)
    b = run([claim()])
    assert b[0]["paris is in france"]["credits_used"] == 0
    for e in (a[0], b[0]):
        e["paris is in france"].pop("credits_used")
    assert a == b
