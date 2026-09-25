#!/usr/bin/env python3
"""Stage 2 orchestration: detect -> retrieve -> verify -> fan back to lines.

Split into two composable stages so a UI can show claims immediately (cheap,
one Groq pass) and defer the expensive part (Tavily + HF, real credits) to an
explicit user action:

  detect_only(lines, on_stage=None) -> (claims, n_lines)
      claims: [{"claim_text", "line_id", "speaker", "start"}, ...] straight
      from claim_detect.py. No retrieval, no verification, no cost beyond
      one batched Groq call.

  verify_claims_list(claims, n_lines, on_stage=None) -> (verdicts_by_line, stats)
      Takes claims already detected (e.g. by detect_only) and runs
      retrieve -> verify -> fan back. This is the part that spends
      Tavily/HF credits and takes real time.

  verify_lines(lines, on_stage=None) -> (verdicts_by_line, stats)
      Convenience wrapper: detect_only + verify_claims_list in one call, for
      callers that don't need the two-step UX (CLI, tests).

verdicts_by_line: {line_id: [record, ...]}  (a line can hold >1 claim)
record: {"claim_text", "verdict", "confidence", "evidence_used"}
stats: counts + per-stage elapsed + per-stage failures, for the UI.

Sequential across stages (a genuine dependency chain); all concurrency lives
inside each stage. See specs/phases/13-verify-pipeline.md.
"""

import asyncio
import logging
import sys
import time

from claim_detect import detect_claims
from retrieve_evidence import retrieve
from verify_claims import verify

log = logging.getLogger("verify_pipeline")


def _pad(lines):
    """claim_detect.py's contract is 5-tuples (start, end, speaker, text, conf).
    Stage 1 lines may be 4-tuples (confidence not yet attached) — pad so this
    module works either way, per the phase's own no-ordering-imposed rule."""
    return [t if len(t) == 5 else (*t, None) for t in lines]


async def detect_only(lines, on_stage=None):
    """Cheap first stage: batched Groq claim detection only. No Tavily/HF
    calls, no credits beyond the Groq batch. -> (claims, n_lines)."""
    lines = _pad(lines)
    if on_stage:
        on_stage("Detecting claims...")
    claims = await detect_claims(lines)
    log.info("detect: %d claims from %d lines", len(claims), len(lines))
    return claims, len(lines)


async def verify_claims_list(claims, n_lines, on_stage=None):
    """Expensive second stage: retrieve evidence + NLI verify + fan back to
    lines, for claims already produced by detect_only. This is the part
    that spends Tavily/HF credits."""
    stats = {
        "n_lines": n_lines,
        "n_claims": len(claims),
        "n_unique_claims": 0,
        "verdict_counts": {"supported": 0, "disputed": 0, "unclear": 0},
        "n_retrieve_failures": 0,
        "n_verify_failures": 0,
        "elapsed": {},
    }

    if not claims:
        stats["elapsed"]["retrieve"] = 0.0
        stats["elapsed"]["verify"] = 0.0
        if on_stage:
            on_stage("Retrieving evidence...")
            on_stage("Verifying claims...")
        return {}, stats

    if on_stage:
        on_stage("Retrieving evidence...")
    t0 = time.perf_counter()
    evidence, retrieve_failures, backmap = await retrieve(claims)
    stats["elapsed"]["retrieve"] = time.perf_counter() - t0
    stats["n_unique_claims"] = len(backmap)
    stats["n_retrieve_failures"] = len(retrieve_failures)
    log.info("retrieve: %d unique claims, %d failures, %.2fs",
              len(backmap), len(retrieve_failures), stats["elapsed"]["retrieve"])

    if on_stage:
        on_stage("Verifying claims...")
    t0 = time.perf_counter()
    verdicts = await verify(evidence)
    stats["elapsed"]["verify"] = time.perf_counter() - t0
    log.info("verify: %d claims verified in %.2fs", len(verdicts), stats["elapsed"]["verify"])

    # claim_text per claim_id: any original claim_text that normalizes to it works
    from retrieve_evidence import normalize
    claim_text_by_id = {}
    for c in claims:
        claim_text_by_id.setdefault(normalize(c["claim_text"]), c["claim_text"])

    verdicts_by_line = {}
    for claim_id, line_ids in backmap.items():
        record = verdicts.get(claim_id)
        if record is None:
            continue
        stats["n_verify_failures"] += record.get("n_failed", 0)
        stats["verdict_counts"][record["verdict"]] += 1
        out_record = {
            "claim_text": claim_text_by_id.get(claim_id, claim_id),
            "verdict": record["verdict"],
            "confidence": record["confidence"],
            "evidence_used": record["evidence_used"],
        }
        for line_id in line_ids:
            verdicts_by_line.setdefault(line_id, []).append(out_record)

    return verdicts_by_line, stats


async def verify_lines(lines, on_stage=None):
    """Convenience: detect_only + verify_claims_list in one call, timing
    the detect stage itself (verify_claims_list times its own retrieve/verify)."""
    t0 = time.perf_counter()
    claims, n_lines = await detect_only(lines, on_stage=on_stage)
    detect_elapsed = time.perf_counter() - t0
    log.info("detect: %d claims in %.2fs", len(claims), detect_elapsed)

    verdicts_by_line, stats = await verify_claims_list(claims, n_lines, on_stage=on_stage)
    stats["elapsed"]["detect"] = detect_elapsed
    return verdicts_by_line, stats


def _selfcheck():
    import claim_detect, retrieve_evidence, verify_claims

    async def fake_detect(lines, **kw):
        return [
            {"claim_text": "X is true.", "line_id": 1, "speaker": "A", "start": 1.0},
            {"claim_text": "x is true", "line_id": 4, "speaker": "B", "start": 4.0},
            {"claim_text": "Y is false.", "line_id": 4, "speaker": "B", "start": 4.0},
        ]

    async def fake_retrieve(claims, **kw):
        ev = {c: [{"url": "http://e", "snippet": "s"}] for c in {"x is true", "y is false"}}
        bm = {"x is true": [1, 4], "y is false": [4]}
        return ev, {}, bm

    async def fake_verify(evidence_map, **kw):
        return {
            "x is true": {"verdict": "supported", "confidence": 0.9, "evidence_used": [], "n_failed": 0},
            "y is false": {"verdict": "disputed", "confidence": 0.8, "evidence_used": [], "n_failed": 0},
        }

    global detect_claims, retrieve, verify
    orig = (detect_claims, retrieve, verify)
    detect_claims, retrieve, verify = fake_detect, fake_retrieve, fake_verify

    lines = [(float(i), i + 1.0, "A", f"l{i}", 0.9) for i in range(6)]
    seen = []
    verdicts, stats = asyncio.run(verify_lines(lines, on_stage=seen.append))

    assert len(verdicts[1]) == 1 and len(verdicts[4]) == 2, verdicts
    assert verdicts[1][0]["verdict"] == "supported"
    assert {r["verdict"] for r in verdicts[4]} == {"supported", "disputed"}
    assert len(seen) == 3 and seen == [
        "Detecting claims...", "Retrieving evidence...", "Verifying claims..."]
    assert 0 not in verdicts
    assert stats["n_lines"] == 6 and stats["n_claims"] == 3 and stats["n_unique_claims"] == 2
    assert stats["verdict_counts"]["supported"] == 1 and stats["verdict_counts"]["disputed"] == 1
    total = sum(stats["elapsed"].values())
    # trivially fast fakes: sum of stage times must equal wall clock within 1s
    assert total < 1.0

    # empty and total-failure paths
    async def none(lines, **kw):
        return []
    detect_claims = none
    v, s = asyncio.run(verify_lines([(0.0, 1.0, "A", "hi", 0.9)]))
    assert v == {} and s["n_claims"] == 0

    # 4-tuple lines (no confidence attached yet) also work
    detect_claims, retrieve, verify = fake_detect, fake_retrieve, fake_verify
    lines4 = [(float(i), i + 1.0, "A", f"l{i}") for i in range(6)]
    v4, s4 = asyncio.run(verify_lines(lines4))
    assert v4.keys() == verdicts.keys(), v4

    # every stage failing (simulates "no keys at all"): claim_detect.py's own
    # per-batch try/except already degrades an auth failure to a dropped
    # batch, not a raise (verified in Phase 10's selfcheck) — so from
    # verify_lines's perspective this looks identical to the zero-claims path
    # already exercised above. No new exception surface to cover here.

    # two-step flow (the Claims tab's actual usage): detect_only is cheap and
    # standalone, verify_claims_list runs later on its output, independently.
    detect_claims, retrieve, verify = fake_detect, fake_retrieve, fake_verify
    seen2 = []
    claims2, n_lines2 = asyncio.run(detect_only(lines, on_stage=seen2.append))
    assert len(claims2) == 3 and n_lines2 == 6
    assert seen2 == ["Detecting claims..."]  # only the detect message, not retrieve/verify

    seen3 = []
    verdicts2, stats2 = asyncio.run(verify_claims_list(claims2, n_lines2, on_stage=seen3.append))
    assert seen3 == ["Retrieving evidence...", "Verifying claims..."]
    assert verdicts2.keys() == verdicts.keys()
    assert stats2["n_lines"] == 6 and stats2["n_claims"] == 3

    # detect_only on zero claims still reports the right n_lines
    detect_claims = none
    claims3, n_lines3 = asyncio.run(detect_only([(0.0, 1.0, "A", "hi", 0.9)]))
    assert claims3 == [] and n_lines3 == 1
    v3, s3 = asyncio.run(verify_claims_list([], 1))
    assert v3 == {} and s3["n_claims"] == 0

    detect_claims, retrieve, verify = orig
    print("selfcheck ok")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
