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

import evidence_cache

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
    params = dict(query=query, include_raw_content=True,
                  max_results=max_results, search_depth=SEARCH_DEPTH)
    resp = await evidence_cache.aget_or_call(
        "tavily", {"endpoint": "tavily.search", **params},
        lambda: client.search(**params))
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
            except evidence_cache.CacheMiss:
                raise
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


# ---- v2 (Phase 30): multi-query, one source per registered domain ----------

V2_MAX_RESULTS = 8
V2_CHUNKS = 3
CHUNK_CAP = 1500
QUERY_LABELS = ["neutral", "fact_check"]
_CHUNK_SEP = "[...]"


def _split_chunks(content):
    return [c.strip()[:CHUNK_CAP] for c in (content or "").split(_CHUNK_SEP) if c.strip()]


def _keep(results):
    """score > SCORE_MIN; if none pass, the single best (v1 behaviour)."""
    kept = [r for r in results if r.get("score", 0) > SCORE_MIN and r.get("url")]
    if not kept:
        with_url = [r for r in results if r.get("url")]
        kept = [max(with_url, key=lambda r: r.get("score", 0))] if with_url else []
    return kept


def _merge(results_by_query):
    """{label: [tavily result, ...]} -> sources, one per registered domain."""
    import credibility
    groups = {}
    for label, results in results_by_query.items():
        for r in results:
            d = credibility.registered_domain(r["url"]) or r["url"]
            groups.setdefault(d, []).append((label, r))
    sources = []
    for d, hits in groups.items():
        hits.sort(key=lambda h: -h[1].get("score", 0))
        primary = hits[0][1]
        chunks = []
        for _, r in hits:
            for c in _split_chunks(r.get("content")):
                if c not in chunks and len(chunks) < V2_CHUNKS:
                    chunks.append(c)
        sources.append({"url": primary["url"], "domain": d, "title": primary.get("title", ""),
                        "chunks": chunks, "tavily_score": primary.get("score", 0),
                        "found_by": sorted({label for label, _ in hits})})
    sources.sort(key=lambda s: (-s["tavily_score"], s["domain"]))
    for i, s in enumerate(sources, 1):
        s["source_id"] = i
    return sources


async def _search_v2(client, query, credits):
    import credibility
    params = dict(query=query, search_depth=SEARCH_DEPTH, max_results=V2_MAX_RESULTS,
                  chunks_per_source=V2_CHUNKS, include_raw_content=False,
                  include_domains=credibility.high_domains(),
                  include_domains_mode="prefer")

    async def call():
        credits.append(1)  # only runs on a real call (cache miss / off)
        return await client.search(**params)
    resp = await evidence_cache.aget_or_call("tavily", {"endpoint": "tavily.search", **params}, call)
    return resp.get("results", [])


async def retrieve_v2(claims, concurrency=4):
    """-> (evidence, failures, backmap); evidence[cid] = {"sources", "search_failures",
    "credits_used"}. A claim whose every query failed is in failures, never 'disputed'."""
    texts, backmap = _dedupe(claims)
    if not texts:
        return {}, {}, {}
    n_q = int(os.environ.get("VERASCOPE_QUERIES", 2))
    queries = {}
    for c in claims:
        queries.setdefault(normalize(c["claim_text"]), c.get("queries") or [c["claim_text"]])

    from tavily import AsyncTavilyClient
    client = AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY", "tvly-test-placeholder"))
    sem = asyncio.Semaphore(concurrency)

    async def one(cid, label, q):
        credits = []
        async with sem:
            try:
                return cid, label, await _search_v2(client, q, credits), None, len(credits)
            except evidence_cache.CacheMiss:
                raise
            except Exception as e:
                return cid, label, None, e, len(credits)

    jobs = [one(cid, QUERY_LABELS[i], q)
            for cid in texts for i, q in enumerate(queries[cid][:n_q])]
    out = await asyncio.gather(*jobs)

    evidence, failures = {}, {}
    per = {cid: {"results": {}, "search_failures": [], "credits_used": 0} for cid in texts}
    for cid, label, results, err, credits in out:
        per[cid]["credits_used"] += credits
        if err is not None:
            log.warning("search %s failed for %r: %r", label, cid, err)
            per[cid]["search_failures"].append({"query_label": label, "error": repr(err)})
        else:
            per[cid]["results"][label] = _keep(results)
    for cid, b in per.items():
        sources = _merge(b["results"])
        evidence[cid] = {"sources": sources, "search_failures": b["search_failures"],
                         "credits_used": b["credits_used"]}
        if not b["results"]:
            failures[cid] = "every query failed"
        elif not sources:
            failures[cid] = "zero usable results"
        log.info("%r -> %d source(s), %d credit(s)", cid, len(sources), b["credits_used"])
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
    assert _split_chunks("a [...] b [...] c") == ["a", "b", "c"]
    assert _split_chunks("[...]") == []
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
