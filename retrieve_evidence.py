#!/usr/bin/env python3
"""Evidence retrieval over deduped claims from Phase 10.

claims: [{"claim_text", "line_id", "speaker", "start"}, ...] ->
    (evidence, failures, backmap)
    evidence: {claim_id: [{"url", "snippet"}, ...]}
    failures: {claim_id: reason}
    backmap:  {claim_id: [line_id, ...]}

claim_id is the normalized claim text (lowercase, stripped punctuation,
collapsed whitespace) so dedupe and the back-map fall out of the same key.
See specs/docs/tavily-search.md.
"""

import asyncio
import logging
import os
import re
import sys

MAX_RESULTS = 4
SEARCH_DEPTH = "basic"
SNIPPET_MIN = 200
RAW_CONTENT_CAP = 1500
SCORE_MIN = 0.5

log = logging.getLogger("retrieve_evidence")

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_key():
    if not os.environ.get("TAVILY_API_KEY"):
        die("no TAVILY_API_KEY: get a free key at https://app.tavily.com")


def normalize(text):
    text = text.lower().strip()
    text = _PUNCT_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def _dedupe(claims):
    """-> ({claim_id: claim_text}, {claim_id: [line_id, ...]})"""
    texts = {}
    backmap = {}
    for c in claims:
        cid = normalize(c["claim_text"])
        texts.setdefault(cid, c["claim_text"])
        backmap.setdefault(cid, []).append(c["line_id"])
    return texts, backmap


def _snippet_from(result):
    content = result.get("content") or ""
    if len(content) >= SNIPPET_MIN:
        return content
    raw = result.get("raw_content")
    if raw:
        return raw[:RAW_CONTENT_CAP]
    return content


async def _search_one(client, query, max_results):
    resp = await client.search(
        query=query,
        include_raw_content=True,
        max_results=max_results,
        search_depth=SEARCH_DEPTH,
    )
    return resp.get("results", [])


async def retrieve(claims, concurrency=4, max_results=MAX_RESULTS):
    texts, backmap = _dedupe(claims)
    if not texts:
        return {}, {}, {}

    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "tvly-test-placeholder"))
    sem = asyncio.Semaphore(concurrency)

    async def one(cid, query):
        async with sem:
            try:
                results = await _search_one(client, query, max_results)
                return cid, results, None
            except Exception as e:
                return cid, None, e

    pairs = await asyncio.gather(*(one(cid, q) for cid, q in texts.items()))

    evidence, failures = {}, {}
    for cid, results, err in pairs:
        if err is not None:
            log.warning("search failed for %r: %r", cid, err)
            evidence[cid] = []
            failures[cid] = repr(err)
            continue

        filtered = [r for r in results if r.get("score", 0) > SCORE_MIN]
        if not filtered and results:
            # keep the single best result rather than nothing (weak premise > no premise)
            filtered = [max(results, key=lambda r: r.get("score", 0))]

        items = [{"url": r["url"], "snippet": _snippet_from(r)[:RAW_CONTENT_CAP]}
                 for r in filtered if r.get("url")]
        evidence[cid] = items
        if not items:
            failures[cid] = "zero usable results"
        log.info("%r -> %d evidence item(s)", cid, len(items))

    return evidence, failures, backmap


def _selfcheck():
    global _search_one
    orig = _search_one

    calls = []

    async def fake(client, query, max_results):
        calls.append(query)
        if query == "boom":
            raise RuntimeError("simulated network failure")
        if query == "empty query":
            return []
        if query == "low score":
            return [{"url": "http://low", "title": "t", "content": "c" * 300, "score": 0.1}]
        return [
            {"url": "http://a", "title": "t", "content": "x" * 300, "score": 0.9},
            {"url": "http://b", "title": "t", "content": "short", "raw_content": "y" * 3000, "score": 0.7},
            {"url": "http://c", "title": "t", "content": "z" * 300, "score": 0.4},  # filtered out
        ]

    _search_one = fake

    claims = [
        {"claim_text": "Unemployment hit 3.4 percent in January 2023.", "line_id": 2, "speaker": "A", "start": 6.0},
        {"claim_text": "unemployment hit 3.4 percent in january 2023", "line_id": 9, "speaker": "A", "start": 40.0},
        {"claim_text": "Inflation was above six percent.", "line_id": 4, "speaker": "B", "start": 14.0},
        {"claim_text": "boom", "line_id": 5, "speaker": "B", "start": 20.0},
        {"claim_text": "empty query", "line_id": 6, "speaker": "B", "start": 21.0},
        {"claim_text": "low score", "line_id": 7, "speaker": "B", "start": 22.0},
    ]
    evidence, failures, backmap = asyncio.run(retrieve(claims))

    # dedupe: 5 unique queries issued, not 6
    assert len(calls) == 5, calls
    unemployment_id = normalize("Unemployment hit 3.4 percent in January 2023.")
    assert sorted(backmap[unemployment_id]) == [2, 9], backmap

    # ranked/score filtering: 2 of 3 results kept (score > 0.5), raw_content fallback used
    inflation_id = normalize("Inflation was above six percent.")
    assert len(evidence[inflation_id]) == 2, evidence[inflation_id]
    assert evidence[inflation_id][1]["snippet"] == "y" * RAW_CONTENT_CAP

    # snippet cap honored
    assert all(len(item["snippet"]) <= RAW_CONTENT_CAP
               for items in evidence.values() for item in items)

    # exception on one claim doesn't lose the others
    assert evidence["boom"] == [] and "boom" in failures
    assert unemployment_id in evidence and evidence[unemployment_id]

    # zero results -> empty evidence + failure entry, no raise
    assert evidence["empty query"] == [] and "empty query" in failures

    # all-filtered-out falls back to best single result, not empty
    assert len(evidence["low score"]) == 1
    assert evidence["low score"][0]["url"] == "http://low"
    assert "low score" not in failures

    _search_one = orig
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
