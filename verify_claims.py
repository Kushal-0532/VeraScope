#!/usr/bin/env python3
"""NLI verification + verdict aggregation over (claim, evidence-snippet) pairs.

Hosted mDeBERTa via HF Inference (serverless), zero-shot-classification task.
See specs/docs/hf-nli-inference.md for the measured score table and gotchas.
"""

import asyncio
import logging
import os
import sys

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


def _call_model(claim_text, snippet):
    """The one thing that actually talks to HF. Isolated so offline tests can
    monkeypatch below the truncation logic, not replace it."""
    client = _get_client()
    out = client.zero_shot_classification(
        snippet,
        candidate_labels=[claim_text],
        hypothesis_template="{}",
        model=NLI_MODEL,
    )
    return out[0].score


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
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
