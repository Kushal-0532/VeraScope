import argparse
import asyncio
import json

import pytest

import evidence_cache as ec
import flags
import retrieve_evidence as re_


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("VERASCOPE_CACHE_REFRESH", raising=False)
    ec.reset_stats()
    return tmp_path


def files(d):
    return [p for p in d.rglob("*") if p.is_file()]


class Counter:
    def __init__(self, value=None):
        self.n, self.value = 0, value

    def __call__(self):
        self.n += 1
        return self.value if self.value is not None else {"n": self.n}


def test_record_then_replay(cache, monkeypatch):
    fn = Counter()
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    assert ec.get_or_call("k", {"a": 1}, fn) == {"n": 1}
    assert ec.get_or_call("k", {"a": 1}, fn) == {"n": 1}
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    assert ec.get_or_call("k", {"a": 1}, fn) == {"n": 1}
    assert fn.n == 1
    assert ec.stats()["k"] == {"hits": 2, "misses": 1, "writes": 1}


def test_replay_miss(monkeypatch):
    fn = Counter()
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    with pytest.raises(ec.CacheMiss):
        ec.get_or_call("k", {"a": 1}, fn)
    assert fn.n == 0


def test_off_mode(cache, monkeypatch):
    monkeypatch.delenv("VERASCOPE_CACHE", raising=False)
    fn = Counter()
    ec.get_or_call("k", {"a": 1}, fn)
    ec.get_or_call("k", {"a": 1}, fn)
    assert fn.n == 2 and files(cache) == []


def test_key_stability():
    assert ec.key("k", a=1, b=2) == ec.key("k", b=2, a=1)
    assert ec.key("k", model="m1", x=1) != ec.key("k", model="m2", x=1)
    assert ec.key("k", text="a") != ec.key("k", text="b")
    assert ec.key("k1", a=1) != ec.key("k2", a=1)


def test_failure_not_cached(cache, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "record")

    def boom():
        raise ValueError("nope")
    with pytest.raises(ValueError):
        ec.get_or_call("k", {"a": 1}, boom)
    assert files(cache) == []


def test_refresh_overwrites(cache, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    fn = Counter()
    ec.get_or_call("k", {"a": 1}, fn)
    (p,) = files(cache)
    m1 = p.stat().st_mtime_ns
    monkeypatch.setenv("VERASCOPE_CACHE_REFRESH", "1")
    assert ec.get_or_call("k", {"a": 1}, fn) == {"n": 2}
    assert json.loads(p.read_text())["value"] == {"n": 2}
    assert p.stat().st_mtime_ns >= m1 and fn.n == 2


def test_non_serialisable(cache, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    with pytest.raises(TypeError):
        ec.get_or_call("k", {"a": 1}, lambda: object())
    assert files(cache) == []


def test_async_concurrent_one_key(cache, monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    calls = []

    async def afn():
        calls.append(1)
        await asyncio.sleep(0.01)
        return {"v": 1}

    async def main():
        return await asyncio.gather(*(ec.aget_or_call("k", {"a": 1}, afn) for _ in range(20)))
    assert asyncio.run(main()) == [{"v": 1}] * 20
    assert 1 <= len(calls) <= 20
    (p,) = files(cache)  # exactly one file, no leftover .tmp
    assert json.loads(p.read_text())["value"] == {"v": 1}


def test_tavily_record_replay(monkeypatch):
    resp = {"results": [{"url": "https://x.org", "content": "c", "score": 0.9}]}

    class Fake:
        n = 0

        async def search(self, **kw):
            Fake.n += 1
            return resp

    class Dead:
        async def search(self, **kw):
            raise AssertionError("network called in replay")

    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    a = asyncio.run(re_._search_one(Fake(), "q", 4))
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    b = asyncio.run(re_._search_one(Dead(), "q", 4))
    assert a == b == resp["results"] and Fake.n == 1


def test_replay_miss_propagates_through_retrieve(monkeypatch):
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    with pytest.raises(ec.CacheMiss):
        asyncio.run(re_.retrieve([{"claim_text": "x is y", "line_id": 0}]))


def test_flags_mutually_exclusive():
    ap = flags.add_args(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        ap.parse_args(["--record", "--replay"])


def test_pipeline_record_replay_all_three_kinds(cache, monkeypatch):
    """Groq + Tavily + NLI recorded once (fakes below the cache), replayed with
    every backend raising -> identical verdicts."""
    import claim_detect
    import tavily
    import verify_claims
    import verify_pipeline as vp

    lines = [(0.0, 2.0, "A", "Paris is the capital of France.", None),
             (2.0, 4.0, "B", "I like tea.", None)]

    async def groq_fake(client, model, messages):
        return {"content": {"claims": [{"line_id": 0, "claim_text": "Paris is the capital of France."}]},
                "usage": None}

    class TavFake:
        def __init__(self, **kw):
            pass

        async def search(self, **kw):
            return {"results": [{"url": "https://a.org", "content": "Paris is the capital of France. " * 10,
                                 "score": 0.9}]}

    monkeypatch.setattr(claim_detect, "_groq_call", groq_fake)
    monkeypatch.setattr(tavily, "AsyncTavilyClient", TavFake)
    monkeypatch.setattr(verify_claims, "_call_raw", lambda c, s, local: 0.97)
    monkeypatch.setenv("VERASCOPE_CACHE", "record")
    rec, _ = asyncio.run(vp.verify_lines(lines))
    assert {p.parts[-3] for p in files(cache)} == {"groq", "tavily", "nli"}

    def dead(*a, **k):
        raise AssertionError("backend called in replay")

    class TavDead(TavFake):
        async def search(self, **kw):
            dead()

    monkeypatch.setattr(claim_detect, "_groq_call", dead)
    monkeypatch.setattr(tavily, "AsyncTavilyClient", TavDead)
    monkeypatch.setattr(verify_claims, "_call_raw", dead)
    monkeypatch.setenv("VERASCOPE_CACHE", "replay")
    rep, _ = asyncio.run(vp.verify_lines(lines))
    assert rep == rec and rec
