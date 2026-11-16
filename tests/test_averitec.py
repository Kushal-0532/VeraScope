import datetime
import json
from pathlib import Path

import pytest

import credibility
import passages
from experiments import averitec as av

FIX = Path(__file__).parent / "fixtures"
DEV, TOPK = FIX / "synthetic_averitec.json", FIX / "synthetic_averitec_top_k.jsonl"


@pytest.fixture
def claims(tmp_path):
    """The 4 valid fixture claims (the 5th has an unknown label), with their pool."""
    valid = [c for c in json.loads(DEV.read_text()) if c["label"] != "Mostly True"]
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(valid))
    return av.load_dev(p, TOPK)


def test_unknown_label_raises_naming_it():
    with pytest.raises(ValueError, match="Mostly True"):
        av.load_dev(DEV)


def test_label_map_and_pool(claims):
    assert [c["gold"] for c in claims] == ["supported", "disputed", "mixed", "nei"]
    assert [c["id"] for c in claims] == [0, 1, 2, 3]
    assert len(claims[0]["top_k"]) == 6  # U+2028 inside a sentence didn't split the row
    assert set(av.PRED_MAP.values()) == set(av.LABEL_MAP.values())


def _items(counts):
    out, i = [], 0
    for gold, n in counts.items():
        for _ in range(n):
            out.append({"id": i, "gold": gold})
            i += 1
    return out


def test_sample_deterministic_and_quotas():
    items = _items({"supported": 120, "disputed": 300, "mixed": 40, "nei": 35})
    a, b = av.sample(items, seed=42), av.sample(items, seed=42)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert [x["id"] for x in a] != [x["id"] for x in av.sample(items, seed=7)]
    assert {g: sum(x["gold"] == g for x in a) for g in av.QUOTAS} == av.QUOTAS
    assert [x["id"] for x in a] == sorted(x["id"] for x in a)


def test_sample_caps_short_class():
    got = av.sample(_items({"supported": 100, "disputed": 100, "mixed": 5, "nei": 100}))
    assert sum(x["gold"] == "mixed" for x in got) == 5 and len(got) == 125


def test_leak_filter(claims):
    c = claims[0]
    kept = av.leak_filter(c, c["top_k"])
    urls = [s["url"] for s in kept]
    assert "https://www.factcheck-a.com/a-is-true" not in urls  # exact fact-check URL
    assert "https://factcheck-a.com/other" not in urls  # same domain as the fact-check
    assert "https://twitter.com/x/status/1" not in urls  # original / cached claim URL
    assert "https://twitter.com/y/status/2" in urls  # same site as the claim, other page
    assert len(kept) == 3


def test_parse_claim_date():
    assert av.parse_claim_date("9-9-2020") == datetime.date(2020, 9, 9)
    assert av.parse_claim_date("9-10-2020") == datetime.date(2020, 10, 9)
    assert av.parse_claim_date("31-10-2020") == datetime.date(2020, 10, 31)
    for bad in ("2020-10-31", "32-1-2020", "None", "", None):
        assert av.parse_claim_date(bad) is None


def test_item_shape_runs_phase31(claims):
    item = av.build_item(claims[0])
    assert set(item) == {"id", "claim", "gold", "claim_date", "sources"}
    assert item["claim_date"] == "2020-09-09"
    srcs = item["sources"]
    assert [s["domain"] for s in srcs] == ["twitter.com", "news.org"]
    assert srcs[1]["url"] == "https://news.org/a"  # archive wrapper removed (trust tiers the real host)
    assert srcs[1]["chunks"] == ["News reports A.", "More news on A."]
    for s in srcs:
        assert {"source_id", "url", "domain", "chunks"} <= set(s)
        assert s["domain"] == credibility.registered_domain(s["url"])
    assert [s["source_id"] for s in srcs] == [1, 2]
    out = passages.select(item["claim"], srcs, lambda c, ch: [0.9] * len(ch))
    assert all(s["tier1"] == "agree" and s["best_chunk"] for s in out)


def test_pool_caps_and_empty(claims):
    c = dict(claims[1], top_k=[{"sentence": f"s{j}", "url": f"https://d{i}.org/{j}"}
                               for i in range(12) for j in range(5)])
    srcs = av.evidence_pool(c)
    assert len(srcs) == av.MAX_SOURCES and all(len(s["chunks"]) == av.PER_SOURCE for s in srcs)
    assert av.build_item(claims[2])["sources"] == []  # empty pool
    assert av.build_item(claims[3])["sources"] == []  # whitespace-only sentence


def test_selfcheck():
    av._selfcheck()
