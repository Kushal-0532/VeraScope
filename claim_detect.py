#!/usr/bin/env python3
"""Batched claim detection over Stage 1's fused transcript lines.

lines: [(start, end, speaker, text, conf), ...] -> [Claim, ...] where a claim
is {"claim_text", "line_id", "speaker", "start"}. Uses Groq openai/gpt-oss-20b
with strict json_schema output. See specs/docs/groq-structured-outputs.md.
"""

import asyncio
import json
import logging
import math
import os
import sys

MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 20
CONTEXT = 2  # lines of context on each side of a batch

log = logging.getLogger("claim_detect")

SYSTEM_PROMPT = """\
You extract check-worthy factual claims from a transcript excerpt.

Numbered lines are candidates. Lines marked [context] are shown only to help
you resolve pronouns and referents — never emit a claim whose line_id is a
context line.

A line is check-worthy only when ALL of these hold:
- it asserts something about the world that is verifiable against a public
  source: a number, date, event, quantity, attribution, or causal claim
- it is not an opinion, prediction, hypothetical, question, joke, or a
  statement purely about the speaker's own feelings or intentions
- it is specific enough to search for ("things are getting worse" is not,
  "inflation hit 9.1 percent in June 2022" is)
- it would matter if it were false

For each check-worthy line, emit claim_text: one self-contained declarative
sentence with all pronouns and referents resolved, no hedges ("I think",
"apparently") carried through, and no speaker attribution baked in. If a line
contains two independent factual claims, emit two objects with the same
line_id.

Most lines are not check-worthy. Returning few or zero claims is normal and
expected.
"""

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claim_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line_id": {"type": "integer"},
                            "claim_text": {"type": "string"},
                        },
                        "required": ["line_id", "claim_text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    },
}


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_key():
    if not os.environ.get("GROQ_API_KEY"):
        die("no GROQ_API_KEY: set it in ~/.env or the environment "
            "(https://console.groq.com/keys)")


def _render_batch(lines, lo, hi):
    """lo:hi is the eligible (non-context) slice; render [lo-CONTEXT, hi+CONTEXT)."""
    ctx_lo = max(0, lo - CONTEXT)
    ctx_hi = min(len(lines), hi + CONTEXT)
    rows = []
    for i in range(ctx_lo, ctx_hi):
        _, _, speaker, text, _ = lines[i]
        tag = "" if lo <= i < hi else " [context]"
        rows.append(f"[{i}]{tag} {speaker}: {text}")
    return "\n".join(rows)


async def _detect_batch(client, lines, lo, hi, model):
    batch_text = _render_batch(lines, lo, hi)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": batch_text},
        ],
        temperature=0.0,
        response_format=_SCHEMA,
        # gpt-oss-20b defaults reasoning_effort to "medium"; on garbled/repetitive
        # real ASR text it can burn the entire completion budget on hidden reasoning
        # and emit nothing, which Groq reports as a 400 json_validate_failed with an
        # empty failed_generation rather than a token-limit error. "low" is enough
        # for this task (find check-worthy sentences in plain English) and leaves
        # room for the actual answer. Found live on audio2.wav's batch 0 (2026-08-23).
        reasoning_effort="low",
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    claims = []
    for c in data.get("claims", []):
        line_id = c["line_id"]
        if not (lo <= line_id < hi):
            log.warning("dropping claim with out-of-range line_id %d (batch %d:%d)",
                        line_id, lo, hi)
            continue
        speaker, start = lines[line_id][2], lines[line_id][0]
        claims.append({
            "claim_text": c["claim_text"],
            "line_id": line_id,
            "speaker": speaker,
            "start": start,
        })
    return claims


async def detect_claims(lines, concurrency=4, model=MODEL):
    sem = asyncio.Semaphore(concurrency)
    from groq import AsyncGroq
    client = AsyncGroq(max_retries=6, api_key=os.environ.get("GROQ_API_KEY", "sk-test-placeholder"))

    n = len(lines)
    batches = [(lo, min(lo + BATCH_SIZE, n)) for lo in range(0, n, BATCH_SIZE)]

    async def one(lo, hi):
        async with sem:
            try:
                claims = await _detect_batch(client, lines, lo, hi, model)
                log.info("batch %d:%d -> %d claims from %d lines", lo, hi, len(claims), hi - lo)
                return claims
            except Exception as e:
                log.warning("batch %d:%d failed: %r", lo, hi, e)
                return e

    results = await asyncio.gather(*(one(lo, hi) for lo, hi in batches), return_exceptions=True)

    all_claims = []
    for r in results:
        if isinstance(r, Exception):
            continue
        all_claims.extend(r)

    log.info("detect_claims: %d claims from %d lines (%d batches)", len(all_claims), n, len(batches))
    return all_claims


def _selfcheck():
    global _detect_batch
    orig = _detect_batch

    calls = []

    async def fake(client, lines, lo, hi, model):
        calls.append((lo, hi))
        out = []
        for i in range(lo, hi):
            if "percent" in lines[i][3]:
                out.append({"claim_text": lines[i][3], "line_id": i,
                            "speaker": lines[i][2], "start": lines[i][0]})
        return out

    _detect_batch = fake
    lines = [(float(i), i + 1.0, "S", f"line {i}", 0.9) for i in range(45)]
    lines[10] = (10.0, 11.0, "S", "unemployment fell to 3.4 percent", 0.9)
    lines[25] = (25.0, 26.0, "S", "inflation hit 9.1 percent", 0.9)
    claims = asyncio.run(detect_claims(lines))
    assert len(calls) == math.ceil(45 / BATCH_SIZE) == 3, calls
    assert {c["line_id"] for c in claims} == {10, 25}
    assert claims[0]["speaker"] == "S" and claims[0]["start"] == 10.0

    async def flaky(client, lines, lo, hi, model):
        if lo == 20:  # batch containing line 25
            raise RuntimeError("simulated failure")
        return await fake(client, lines, lo, hi, model)

    _detect_batch = flaky
    claims = asyncio.run(detect_claims(lines))
    assert {c["line_id"] for c in claims} == {10}, claims  # batch 0 survives, batch 1 dropped

    _detect_batch = orig
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
