#!/usr/bin/env python3
"""NLI verification + verdict aggregation over (claim, evidence-snippet) pairs.

Hosted mDeBERTa via HF Inference (serverless), zero-shot-classification task.
See specs/docs/hf-nli-inference.md for the measured score table and gotchas.
"""

import asyncio
import logging
import os
import sys
import threading

import evidence_cache

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SNIPPET_CAP = 1500
SUPPORT_T = 0.85
CONTRADICT_T = 0.15

log = logging.getLogger("verify_claims")

_client = None


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _get_client():
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(token=os.environ.get("HF_TOKEN"))
    return _client


_local = None
_local_lock = threading.Lock()


def _call_local(claim_text, snippet):
    """Same model on CPU via transformers. With one candidate label the pipeline
    scores entailment vs contradiction, matching the hosted API's semantics."""
    with _local_lock:  # ponytail: one lock, pipeline isn't thread-safe
        out = _get_local()(snippet, candidate_labels=[claim_text], hypothesis_template="{}")
    return out["scores"][0]


def _get_local():
    """Load the pipeline once. Call with _local_lock held."""
    global _local
    if _local is None:
        import torch
        from transformers import pipeline
        device = os.environ.get("VERASCOPE_NLI_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        _local = pipeline("zero-shot-classification", model=NLI_MODEL, device=device)
    return _local


def _call_model(claim_text, snippet):
    """The one thing that actually talks to HF. Isolated so offline tests can
    monkeypatch below the truncation logic, not replace it. Recorded/replayed
    through evidence_cache (local and hosted keyed separately)."""
    local = os.environ.get("VERASCOPE_LOCAL_NLI") == "1"
    return evidence_cache.get_or_call(
        "nli", _nli_parts(claim_text, snippet, local), lambda: _call_raw(claim_text, snippet, local))


def _call_raw(claim_text, snippet, local):
    if local:
        return _call_local(claim_text, snippet)
    client = _get_client()
    out = client.zero_shot_classification(
        snippet,
        candidate_labels=[claim_text],
        hypothesis_template="{}",
        model=NLI_MODEL,
    )
    return out[0].score


def _nli_parts(claim_text, snippet, local):
    return {"model": NLI_MODEL, "backend": "local" if local else "hf-inference",
            "premise": snippet, "hypothesis": claim_text, "template": "{}"}


def _score_batch_raw(claim_text, snippets, local):
    """One pipeline call for all premises of one claim (GPU-friendly)."""
    if not local:  # hosted has no batch endpoint; HF serverless is 402 anyway
        return [_call_raw(claim_text, s, False) for s in snippets]
    with _local_lock:
        out = _get_local()(list(snippets), candidate_labels=[claim_text], hypothesis_template="{}")
    out = [out] if isinstance(out, dict) else out
    return [o["scores"][0] for o in out]


def score_pairs(claim_text, chunks):
    """premise=each chunk, hypothesis=claim -> [P(entailment)]. Cached per pair
    (same key as score_pair), misses scored in one batch. Local NLI is the Stage 4
    default; VERASCOPE_LOCAL_NLI=0 forces the hosted API."""
    local = os.environ.get("VERASCOPE_LOCAL_NLI", "1") != "0"
    chunks = [c[:SNIPPET_CAP] for c in chunks]
    if not chunks:
        return []
    return evidence_cache.get_many(
        "nli", [_nli_parts(claim_text, c, local) for c in chunks],
        lambda miss: _score_batch_raw(claim_text, [chunks[i] for i in miss], local))


def score_pair(claim_text, snippet):
    """premise=snippet, hypothesis=claim_text -> P(entailment)."""
    return _call_model(claim_text, snippet[:SNIPPET_CAP])


def _label(score):
    if score >= SUPPORT_T:
        return "supports"
    if score <= CONTRADICT_T:
        return "contradicts"
    return "neutral"


def _aggregate(scores):
    """scores: list of raw entailment probabilities for one claim's pairs."""
    if not scores:
        return {"verdict": "unclear", "confidence": 0.0}

    supporting = [s for s in scores if s >= SUPPORT_T]
    contradicting = [s for s in scores if s <= CONTRADICT_T]

    if not supporting and not contradicting:
        verdict, deciding = "unclear", scores
    elif contradicting and not supporting:
        verdict, deciding = "disputed", contradicting
    elif supporting and not contradicting:
        verdict, deciding = "supported", supporting
    else:
        verdict, deciding = "disputed", contradicting  # conflict -> disputed (Prior Decision 7)

    confidence = sum(abs(s - 0.5) * 2 for s in deciding) / len(deciding)
    return {"verdict": verdict, "confidence": confidence}


async def verify(evidence_map, concurrency=8):
    """evidence_map: {claim_id: [{"url", "snippet"}, ...]}.
    Returns {claim_id: {"verdict", "confidence", "evidence_used", "n_scored", "n_failed"}}.
    A failed pair is missing evidence, never a contradiction (Prior Decision 8).
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(claim_id, item):
        async with sem:
            try:
                s = await asyncio.to_thread(score_pair, claim_id, item["snippet"])
                return claim_id, item, s, None
            except evidence_cache.CacheMiss:
                raise
            except Exception as e:
                return claim_id, item, None, e

    tasks = [one(claim_id, item)
             for claim_id, items in evidence_map.items() for item in items]
    results = await asyncio.gather(*tasks) if tasks else []

    per_claim = {claim_id: {"evidence_used": [], "scores": [], "n_failed": 0}
                 for claim_id in evidence_map}
    for claim_id, item, s, err in results:
        bucket = per_claim[claim_id]
        if err is not None:
            bucket["n_failed"] += 1
            log.warning("pair failed for %r (%s): %r", claim_id, item.get("url"), err)
            continue
        bucket["evidence_used"].append({
            "url": item["url"], "snippet": item["snippet"],
            "score": s, "label": _label(s),
        })
        bucket["scores"].append(s)

    out = {}
    for claim_id, bucket in per_claim.items():
        agg = _aggregate(bucket["scores"])
        out[claim_id] = {
            "verdict": agg["verdict"],
            "confidence": agg["confidence"],
            "evidence_used": bucket["evidence_used"],
            "n_scored": len(bucket["scores"]),
            "n_failed": bucket["n_failed"],
        }
        log.info("%r -> %s (conf %.2f, %d scored, %d failed)",
                  claim_id, out[claim_id]["verdict"], out[claim_id]["confidence"],
                  out[claim_id]["n_scored"], out[claim_id]["n_failed"])
    return out


# ---- Verdict v2 (Phase 33, SPEC Part IV "Verdict rules (locked)") ---------

LABELS = {
    "strongly_supported": ("✅", "Strongly supported"),
    "mostly_supported": ("🟢", "Mostly supported"),
    "mixed": ("🟡", "Mixed"),
    "mostly_disputed": ("🟠", "Mostly disputed"),
    "strongly_disputed": ("❌", "Strongly disputed"),
    "not_enough_evidence": ("⚪", "Not enough evidence"),
    "no_sources": ("⚪", "No sources found"),
    "couldnt_check": ("⛔", "Couldn't check"),
    "not_checked": ("⏳", "Not checked yet"),  # UI only
}
_EPS = 1e-9
_CAP = {"strongly_supported": "mostly_supported", "strongly_disputed": "mostly_disputed"}
_STANCES = ("agree", "disagree", "no_side", "off_topic")


def _band(share):
    if share >= 0.80 - _EPS:
        return "strongly_supported"
    if share >= 0.60 - _EPS:
        return "mostly_supported"
    if share > 0.40 + _EPS:
        return "mixed"
    if share > 0.20 + _EPS:
        return "mostly_disputed"
    return "strongly_disputed"


def _cap(label):
    return _CAP.get(label, label)


def _tier_points(src):
    import credibility
    tier = src.get("tier") or credibility.trust(src.get("url", ""))["tier"]
    return tier, src.get("points") or credibility.TIERS[tier]


def _warnings(share, voting, n_failed, n_total):
    """voting: [(stance, tier)] of agree/disagree sources."""
    out = []
    high = {st for st, tier in voting if tier == "High"}
    lean = "agree" if share >= 0.60 - _EPS else "disagree" if share <= 0.40 + _EPS else None
    text = {"disagree": "A highly trusted source disagrees", "agree": "A highly trusted source agrees"}
    if lean:
        other = "disagree" if lean == "agree" else "agree"
        if other in high:
            out.append({"key": "trusted_minority", "text": text[other]})
    elif len(high) == 1:
        out.append({"key": "trusted_minority", "text": text[next(iter(high))]})
    if len(voting) == 1:
        out.append({"key": "single_source", "text": "Only one source took a side"})
    if all(tier in ("Low", "Unrated") for _, tier in voting):
        out.append({"key": "low_trust_only", "text": "Only low-trust sources took a side"})
    if n_failed:
        out.append({"key": "partial_failures", "text": f"{n_failed} of {n_total} couldn't be read"})
    return out


def verdict(sources, retrieval_failed=False):
    """Scored sources (stance / failed / tier / points) -> label, share, points, counts,
    warnings. Only agree/disagree vote; failures never count as disagreement."""
    counts = dict.fromkeys(_STANCES, 0)
    counts["failed"] = 0
    voting, a_pts, d_pts = [], 0, 0
    for src in sources:
        if src.get("failed"):
            counts["failed"] += 1
            continue
        st = src.get("stance")
        if st in counts:
            counts[st] += 1
        if st in ("agree", "disagree"):
            tier, pts = _tier_points(src)
            voting.append((st, tier))
            if st == "agree":
                a_pts += pts
            else:
                d_pts += pts

    def out(label, share=None, warnings=()):
        emoji, text = LABELS[label]
        has_share = share is not None
        return {"label": label, "emoji": emoji, "label_text": text,
                "share": share, "agree_points": a_pts if has_share else None,
                "disagree_points": d_pts if has_share else None,
                "warnings": list(warnings), "counts": counts, "sources": sources}

    if not sources:
        return out("couldnt_check" if retrieval_failed else "no_sources")
    if counts["failed"] == len(sources):
        return out("couldnt_check")
    partial = ([{"key": "partial_failures",
                 "text": f"{counts['failed']} of {len(sources)} couldn't be read"}]
               if counts["failed"] else [])
    if a_pts + d_pts == 0:
        return out("not_enough_evidence", warnings=partial)
    share = a_pts / (a_pts + d_pts)
    warns = _warnings(share, voting, counts["failed"], len(sources))
    label = _band(share)
    if any(w["key"] in ("trusted_minority", "single_source", "low_trust_only") for w in warns):
        label = _cap(label)
    return out(label, share, warns)


def _selfcheck():
    global _call_model
    orig = _call_model

    def fake(claim_text, snippet):
        assert len(snippet) <= SNIPPET_CAP, len(snippet)
        return {"SUP": 0.99, "CON": 0.01, "NEU": 0.5}.get(snippet, 0.5)

    def flaky(claim_text, snippet):
        if snippet == "FAIL":
            raise RuntimeError("simulated failure")
        return fake(claim_text, snippet)

    _call_model = flaky
    ev = {
        "c_supported": [{"url": "u1", "snippet": "SUP"}],
        "c_disputed": [{"url": "u1", "snippet": "CON"}],
        "c_unclear_neutral": [{"url": "u1", "snippet": "NEU"}],
        "c_disputed_conflict": [{"url": "u1", "snippet": "SUP"}, {"url": "u2", "snippet": "CON"}],
        "c_empty": [],
        "c_allfail": [{"url": "u1", "snippet": "FAIL"}],
    }
    out = asyncio.run(verify(ev))
    assert out["c_supported"]["verdict"] == "supported"
    assert out["c_disputed"]["verdict"] == "disputed"
    assert out["c_unclear_neutral"]["verdict"] == "unclear"
    assert out["c_disputed_conflict"]["verdict"] == "disputed"
    assert out["c_empty"]["verdict"] == "unclear" and out["c_empty"]["n_scored"] == 0
    assert out["c_allfail"]["verdict"] == "unclear" and out["c_allfail"]["n_failed"] == 1
    for v in out.values():
        assert 0.0 <= v["confidence"] <= 1.0

    # truncation: snippet longer than SNIPPET_CAP must be cut before reaching _call_model
    _call_model = fake
    long_snippet = "z" * (SNIPPET_CAP + 500)
    asyncio.run(verify({"c_long": [{"url": "u", "snippet": long_snippet}]}))  # fake asserts the cap

    # aggregation table, all five rows, direct
    T = [
        ([], "unclear"),
        ([0.5, 0.6], "unclear"),
        ([0.99, 0.9], "supported"),
        ([0.01, 0.05], "disputed"),
        ([0.99, 0.01], "disputed"),
    ]
    for scores, want in T:
        got = _aggregate(scores)["verdict"]
        assert got == want, (scores, got, want)

    _call_model = orig
    v = verdict([{"stance": "agree", "tier": "High"}, {"stance": "agree", "tier": "High"},
                 {"stance": "disagree", "tier": "High"}])
    assert v["label"] == "mostly_supported" and v["warnings"][0]["key"] == "trusted_minority"
    assert verdict([], retrieval_failed=True)["label"] == "couldnt_check"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
