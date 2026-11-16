import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import adjudicate as adj
from test_claim_detect_queries import walk

WIKI = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
REPLICA = "Berlin has its own Eiffel Tower replica in the amusement park."


def s(i, tier1, url="https://blog-x.example/p", chunk=None, failed=None):
    d = {"source_id": i, "url": url, "domain": url.split("/")[2], "tier1": tier1,
         "chunks": [chunk or f"source {i} says something long enough here"],
         "best_chunk": chunk or f"source {i} says something long enough here"}
    if failed:
        d = {"source_id": i, "url": url, "chunks": [], "failed": failed}
    return d


HIGH = "https://www.reuters.com/a"
MED = "https://www.foxnews.com/a"
LOW = "https://www.infowars.com/a"


@pytest.mark.parametrize("sources,want", [
    ([s(1, "agree", HIGH), s(2, "agree", HIGH)], False),
    ([s(1, "agree"), s(2, "unsure")], True),
    ([s(1, "agree", LOW), s(2, "disagree", LOW)], False),
    ([s(1, "agree", LOW), s(2, "agree", LOW), s(3, "agree", LOW), s(4, "disagree", HIGH)], True),
    ([s(1, "agree", MED), s(2, "disagree", LOW)], True),
    ([s(1, None, failed="no text"), s(2, None, failed="nli error: X")], False),
])
def test_trigger(sources, want):
    assert adj.needs_tier2(sources) is want


SRC = {"source_id": 1, "chunks": [WIKI, "Second chunk — with an “em dash” inside it."]}


@pytest.mark.parametrize("quote,ok", [
    (WIKI, True),
    ("the eiffel   tower is a WROUGHT-iron lattice tower", True),
    ('"Second chunk – with an "em dash" inside it."', True),
    ("…lattice tower on the Champ de Mars…", True),
    ("The Eiffel Tower is located in Berlin, Germany.", False),
    ("Paris", False),
    (REPLICA, False),
    ("", False),
    (None, False),
])
def test_quote(quote, ok):
    assert adj.verify_quote(quote, SRC) is ok


def stats():
    return adj._new_stats()


def test_merge():
    srcs = [s(1, "agree", chunk=REPLICA), s(2, "agree", chunk=WIKI), s(3, "disagree"),
            s(4, None, failed="no text")]
    llm = {"sources": [
        {"source_id": 1, "stance": "off_topic", "quote": REPLICA},
        {"source_id": 2, "stance": "disagree", "quote": "made-up sentence that is not there"},
        {"source_id": 3, "stance": "maybe", "quote": "source 3 says something"},
        {"source_id": 99, "stance": "agree", "quote": "x" * 20},
    ]}
    st = stats()
    out = adj.merge(srcs, llm, st)
    assert out[0]["stance"] == "off_topic" and out[0]["decided_by"] == "llm" and out[0]["quote"] == REPLICA
    assert out[1]["stance"] == "agree" and out[1]["decided_by"] == "nli" and out[1]["quote"] is None
    assert out[2]["stance"] == "disagree" and out[2]["decided_by"] == "nli"
    assert out[3] == srcs[3]
    assert st["quote_failures"] == 1 and st["invalid_ids"] == 1 and st["invalid_stances"] == 1


def test_tier1_mapping():
    out = adj.merge([s(1, "unsure"), s(2, "agree"), s(3, "disagree")], None, stats())
    assert [o["stance"] for o in out] == ["no_side", "agree", "disagree"]


class FakeGroq:
    def __init__(self, fail=0, reply=None):
        self.calls, self.fail, self.reply = [], fail, reply
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail:
            raise RateLimit()
        content = self.reply or {"sources": []}
        return NS(choices=[NS(message=NS(content=json.dumps(content)))],
                  usage=NS(model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 50}))


class RateLimit(Exception):
    status_code = 429
    response = NS(headers={"retry-after": "1"})


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, d):
        self.t += d


def runit(items, client, budget=None):
    c = Clock()
    pacer = adj.TpmPacer(tpm=10**9, clock=c, sleep=c.sleep)
    return asyncio.run(adj.run(items, budget=budget, client=client, pacer=pacer, sleep=c.sleep))


def hard(i):
    return {"claim_text": f"claim {i}", "sources": [s(1, "unsure"), s(2, "agree")]}


def test_budget():
    g = FakeGroq()
    out, st = runit([hard(i) for i in range(17)], g, budget=15)
    assert len(g.calls) == 15 and st["tier2_calls"] == 15 and st["tier2_skipped"] == 2
    assert all(src["decided_by"] == "nli" for it in out[15:] for src in it["sources"])


def test_budget_env(monkeypatch):
    monkeypatch.setenv("VERASCOPE_TIER2_MAX", "2")
    g = FakeGroq()
    _, st = runit([hard(i) for i in range(4)], g)
    assert len(g.calls) == 2 and st["tier2_skipped"] == 2


def test_easy_claim_no_call():
    g = FakeGroq()
    out, st = runit([{"claim_text": "c", "sources": [s(1, "agree", HIGH)]}], g)
    assert g.calls == [] and st["tier2_calls"] == 0


def test_pacer():
    c = Clock()
    p = adj.TpmPacer(tpm=8000, clock=c, sleep=c.sleep)
    times = []

    async def go():
        for _ in range(3):
            await p.wait(3000)
            times.append(c.t)
    asyncio.run(go())
    assert times[:2] == [0.0, 0.0] and times[2] >= 60


def test_pacer_never_over_90pct():
    c = Clock()
    p = adj.TpmPacer(tpm=8000, clock=c, sleep=c.sleep)
    log = []

    async def go():
        for n in [2500, 1000, 3000, 2000, 500, 3500, 100] * 3:
            await p.wait(n)
            log.append((c.t, n))
            c.t += 5
    asyncio.run(go())
    for t0, _ in log:
        assert sum(n for t, n in log if t0 <= t < t0 + 60) <= 0.9 * 8000


def test_429_retry_then_success():
    g = FakeGroq(fail=2, reply={"sources": [{"source_id": 2, "stance": "agree",
                                             "quote": "source 2 says something long"}]})
    out, st = runit([hard(0)], g)
    assert len(g.calls) == 3 and st["tier2_errors"] == 0
    assert out[0]["sources"][1]["decided_by"] == "llm"


def test_429_gives_up():
    g = FakeGroq(fail=4)
    out, st = runit([hard(0), hard(1)], g)
    assert st["tier2_errors"] == 1 and out[0]["sources"][0]["decided_by"] == "nli"
    assert len(out) == 2  # run continued


def test_prompt():
    long = "word " * 200
    msgs = adj.build_messages("c", [s(1, "agree", chunk=long), s(2, None, failed="no text")])
    user = msgs[1]["content"]
    assert "[2]" not in user and "[1]" in user
    body = user.split('"')[1]
    assert len(body) <= 400


def test_schema_strict():
    walk(adj._SCHEMA["json_schema"]["schema"])
