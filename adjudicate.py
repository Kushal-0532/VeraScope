"""Tier-2 adjudicator (Phase 32, Prior Decisions 6-7).

Hard claims (an unsure source, or an agree/disagree conflict that involves a High or
Medium trust source) get ONE Groq call that labels each source agree / disagree /
no_side / off_topic and must quote the sentence it relied on. A label is used only if
its quote is found verbatim (after normalisation) in that source's chunks; otherwise
the tier-1 NLI label stands. Budget: VERASCOPE_TIER2_MAX calls per run (default 15).
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import unicodedata

import credibility
import evidence_cache

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1024
CHUNK_PROMPT_CAP = 400
QUOTE_MIN = 12
MAX_RETRIES = 3
STANCES = ["agree", "disagree", "no_side", "off_topic"]
TIER1_MAP = {"agree": "agree", "disagree": "disagree", "unsure": "no_side"}

log = logging.getLogger("adjudicate")

SYSTEM_PROMPT = """\
You judge whether each numbered source supports the claim.
Labels: agree (the source states the claim is true), disagree (states it is false
or states an incompatible fact), no_side (relevant but doesn't settle it),
off_topic (about something else, e.g. a different event, place, study, or a
replica/namesake). Quote the exact sentence you relied on, copied character for
character from that source. Return one object per numbered source.
"""

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tier2_stances",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "integer"},
                            "stance": {"type": "string", "enum": STANCES},
                            "quote": {"type": "string"},
                        },
                        "required": ["source_id", "stance", "quote"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["sources"],
            "additionalProperties": False,
        },
    },
}


def _live(sources):
    return [s for s in sources if not s.get("failed") and s.get("tier1")]


def needs_tier2(sources):
    live = _live(sources)
    if any(s["tier1"] == "unsure" for s in live):
        return True
    sides = {t: [s for s in live if s["tier1"] == t] for t in ("agree", "disagree")}
    if not (sides["agree"] and sides["disagree"]):
        return False
    return any(credibility.trust(s.get("url", ""))["tier"] in ("High", "Medium")
               for s in sides["agree"] + sides["disagree"])


def build_messages(claim, sources):
    rows = [f'[{s["source_id"]}] {s.get("domain", "")}: "{s["best_chunk"][:CHUNK_PROMPT_CAP]}"'
            for s in _live(sources)]
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: {claim}\n\nSources:\n" + "\n".join(rows)}]


_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"',
                         "–": "-", "—": "-", "−": "-"})
_WS = re.compile(r"\s+")


def _norm(text):
    t = unicodedata.normalize("NFKC", text or "").translate(_QUOTES).casefold()
    return _WS.sub(" ", t).strip()


def verify_quote(quote, source):
    q = _norm(quote)
    while True:  # strip surrounding quotes / ellipses ("..." and NFKC'd "…")
        s = q.strip(" \"'").removeprefix("...").removesuffix("...").strip(" \"'")
        if s == q:
            break
        q = s
    if len(q) < QUOTE_MIN:
        return False
    return q in _norm(" ".join(source.get("chunks") or []))


class TpmPacer:
    """Rolling 60 s window of estimated tokens; wait while the next call would push
    the window over 90 % of tpm (e6.py rule). clock/sleep injectable for tests."""

    def __init__(self, tpm=None, clock=time.monotonic, sleep=asyncio.sleep):
        self.tpm = tpm or int(os.environ.get("GROQ_TPM", "8000"))
        self.clock, self.sleep, self.calls = clock, sleep, []

    async def wait(self, est):
        while True:
            now = self.clock()
            self.calls = [(t, n) for t, n in self.calls if now - t < 60]
            used = sum(n for _, n in self.calls)
            if not self.calls or used + est <= 0.9 * self.tpm:
                self.calls.append((now, est))
                return
            await self.sleep(60 - (now - self.calls[0][0]) + 0.01)


def _is_429(e):
    return getattr(e, "status_code", None) == 429


def _retry_after(e):
    try:
        return float(e.response.headers.get("retry-after"))
    except Exception:
        return 20.0


async def _groq_call(client, messages, pacer, sleep):
    est = sum(len(m["content"]) for m in messages) // 4 + MAX_TOKENS
    for attempt in range(MAX_RETRIES + 1):
        await pacer.wait(est)
        try:
            resp = await client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, response_format=_SCHEMA,
                reasoning_effort="low", max_completion_tokens=MAX_TOKENS)
        except Exception as e:
            if _is_429(e) and attempt < MAX_RETRIES:
                await sleep(_retry_after(e))
                continue
            raise
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return {"content": json.loads(resp.choices[0].message.content or "{}"), "usage": usage}


def _tier1(s):
    if s.get("failed") or not s.get("tier1"):
        return s
    return {**s, "stance": TIER1_MAP[s["tier1"]], "decided_by": "nli", "quote": None}


def merge(sources, llm, stats):
    by_id = {s["source_id"]: s for s in sources}
    verdicts = {}
    for r in (llm or {}).get("sources", []):
        if r.get("source_id") not in by_id:
            stats["invalid_ids"] += 1
            continue
        verdicts[r["source_id"]] = r
    out = []
    for s in sources:
        r = verdicts.get(s["source_id"])
        base = _tier1(s)
        if r is None or s.get("failed") or not s.get("tier1"):
            out.append(base)
        elif r.get("stance") not in STANCES:
            stats["invalid_stances"] += 1
            out.append(base)
        elif not verify_quote(r.get("quote"), s):
            stats["quote_failures"] += 1
            out.append(base)
        else:
            out.append({**s, "stance": r["stance"], "decided_by": "llm", "quote": r["quote"]})
    return out


def _new_stats():
    return dict.fromkeys(["tier2_calls", "tier2_skipped", "tier2_errors", "quote_failures",
                          "invalid_ids", "invalid_stances", "tokens_in", "tokens_out"], 0)


async def run(items, budget=None, client=None, pacer=None, sleep=asyncio.sleep):
    """items: [{"claim_text", "sources": [...]}] in line order -> (items, stats).
    Each source gains stance / decided_by / quote (failed sources untouched)."""
    budget = int(os.environ.get("VERASCOPE_TIER2_MAX", "15")) if budget is None else budget
    stats = _new_stats()
    pacer = pacer or TpmPacer(sleep=sleep)
    out = []
    for item in items:
        sources = item["sources"]
        if not needs_tier2(sources):
            out.append({**item, "sources": merge(sources, None, stats)})
            continue
        if stats["tier2_calls"] >= budget:
            stats["tier2_skipped"] += 1
            out.append({**item, "sources": merge(sources, None, stats)})
            continue
        if client is None:
            from groq import AsyncGroq
            client = AsyncGroq(max_retries=0, api_key=os.environ.get("GROQ_API_KEY", "sk-test-placeholder"))
        messages = build_messages(item["claim_text"], sources)
        parts = {"endpoint": "groq.chat", "model": MODEL, "messages": messages,
                 "schema": _SCHEMA["json_schema"], "max_tokens": MAX_TOKENS,
                 "reasoning_effort": "low"}
        stats["tier2_calls"] += 1
        try:
            res = await evidence_cache.aget_or_call(
                "tier2", parts, lambda: _groq_call(client, messages, pacer, sleep))
        except evidence_cache.CacheMiss:
            raise
        except Exception as e:
            log.warning("tier2 failed for %r: %r", item["claim_text"], e)
            stats["tier2_errors"] += 1
            out.append({**item, "sources": merge(sources, None, stats)})
            continue
        u = res.get("usage") or {}
        stats["tokens_in"] += u.get("prompt_tokens") or 0
        stats["tokens_out"] += u.get("completion_tokens") or 0
        out.append({**item, "sources": merge(sources, res["content"], stats), "tier2": res["content"]})
    return out, stats


def _selfcheck():
    src = {"source_id": 1, "chunks": ["The tower stands on the Champ de Mars in Paris."]}
    assert verify_quote("“the tower stands on the  Champ de Mars”", src)
    assert not verify_quote("Paris", src)
    srcs = [{"source_id": 1, "url": "https://a.org", "tier1": "agree", "best_chunk": "x"},
            {"source_id": 2, "url": "https://b.org", "tier1": "agree", "best_chunk": "y"}]
    out, st = asyncio.run(run([{"claim_text": "c", "sources": srcs}]))
    assert st["tier2_calls"] == 0 and out[0]["sources"][0]["decided_by"] == "nli"
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
