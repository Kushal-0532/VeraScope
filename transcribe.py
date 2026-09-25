#!/usr/bin/env python3
"""Groq-hosted transcription. Single file, no chunking (that's Phase 03).

-> [(start, end, text)] in time order, the exact tuple shape group_lines()
already consumes.
"""

import asyncio
import logging
import os
import sys
import time

import diarize_demo as dd

MAX_CONCURRENCY = 4

log = logging.getLogger("transcribe")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_key():
    if not os.environ.get("GROQ_API_KEY"):
        die("no GROQ_API_KEY: set it in ~/.env or the environment "
            "(https://console.groq.com/keys)")


async def transcribe_file(client, path, language=None):
    """Real worker. Takes an injected client so Phase 03 can share one across
    chunks and tests can monkeypatch it."""
    kwargs = dict(
        file=(os.path.basename(path), open(path, "rb").read()),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        temperature=0.0,
    )
    if language:
        kwargs["language"] = language

    resp = await client.audio.transcriptions.create(**kwargs)
    data = resp.model_dump() if hasattr(resp, "model_dump") else resp

    raw_words = data.get("words")
    if not raw_words:
        raw_words = [w for seg in (data.get("segments") or [])
                     for w in (seg.get("words") or [])]

    words = []
    for w in raw_words or []:
        text = w["word"].strip()
        if text:
            words.append((float(w["start"]), float(w["end"]), text))
    # Crosstalk in the source audio makes Groq's word order go briefly
    # non-monotonic at overlap points (two speakers' words interleaved out of
    # start-time order). group_lines() requires time-ordered input (fusion
    # contract) so we restore it here; stable sort preserves original order
    # for words with identical start times.
    words.sort(key=lambda w: w[0])
    return words


async def transcribe_one(path, language=None):
    """Thin convenience wrapper: builds its own client, calls transcribe_file.
    Used by tests and one-shot callers only."""
    from dotenv import load_dotenv
    from groq import AsyncGroq
    load_dotenv(os.path.expanduser("~/.env"))  # a real env var still wins
    _require_key()
    client = AsyncGroq(max_retries=6)
    return await transcribe_file(client, path, language)


def _forced_fail_indices():
    # ponytail: test hook
    raw = os.environ.get("VERASCOPE_FAIL_CHUNKS", "")
    return {int(i) for i in raw.split(",") if i.strip()}


async def transcribe_chunks(chunks, language=None, concurrency=MAX_CONCURRENCY):
    """chunks: [(path, start_offset_s), ...]. Returns (words, failures):
    words is a list of per-chunk word lists, in chunk order, each already
    shifted into global time. failures is [(index, start, end, repr(exc))]
    for any chunk that raised after SDK retries were exhausted.
    """
    # ponytail: DEBUG on the groq logger dumps raw multipart request bytes
    # (the whole audio file) per request in this SDK version — useless and
    # enormous. Our own per-chunk INFO log below covers timing/429 visibility
    # instead; bump to DEBUG by hand if SDK internals are ever actually needed.
    sem = asyncio.Semaphore(concurrency)
    from groq import AsyncGroq
    # api_key falls back to a placeholder so offline tests (which monkeypatch
    # transcribe_file and never touch the client) don't need a real key.
    client = AsyncGroq(max_retries=6, api_key=os.environ.get("GROQ_API_KEY", "sk-test-placeholder"))
    fail_indices = _forced_fail_indices()

    async def one(i, path, offset):
        async with sem:
            t0 = time.perf_counter()
            if i in fail_indices:
                raise RuntimeError(f"forced failure via VERASCOPE_FAIL_CHUNKS for chunk {i}")
            words = await transcribe_file(client, path, language)
            log.info("chunk %d: %.1fs, %d words", i, time.perf_counter() - t0, len(words))
            return [(s + offset, e + offset, t) for s, e, t in words]

    results = await asyncio.gather(
        *(one(i, p, o) for i, (p, o) in enumerate(chunks)), return_exceptions=True
    )

    chunk_span = None
    if len(chunks) > 1:
        chunk_span = chunks[1][1] - chunks[0][1]

    words, failures = [], []
    for i, (r, (path, offset)) in enumerate(zip(results, chunks)):
        if isinstance(r, Exception):
            end = offset + chunk_span if chunk_span is not None else offset
            gap_text = f"[transcription unavailable {dd.ts(offset)}–{dd.ts(end)}]"
            words.append([(offset, end, gap_text)])
            failures.append((i, offset, end, repr(r)))
        else:
            words.append(r)
    return words, failures


def _selfcheck():
    global transcribe_file
    orig = transcribe_file

    async def fake(client, path, language=None):
        await asyncio.sleep(0.01)
        return [(1.0, 2.0, path.split("/")[-1])]

    transcribe_file = fake
    chunks = [(f"/c{i}.flac", i * 590.0) for i in range(20)]
    words, fails = asyncio.run(transcribe_chunks(chunks, concurrency=4))
    flat = [w for ws in words for w in ws]
    assert [w[0] for w in flat] == [1.0 + i * 590.0 for i in range(20)]
    assert not fails

    async def flaky(client, path, language=None):
        if "/c7." in path:
            raise RuntimeError("simulated failure")
        return [(1.0, 2.0, "ok")]

    transcribe_file = flaky
    words, fails = asyncio.run(transcribe_chunks(chunks, concurrency=4))
    flat = [w for ws in words for w in ws]
    assert len(fails) == 1 and fails[0][0] == 7
    gap = [w for w in flat if "unavailable" in w[2]]
    assert len(gap) == 1
    assert dd.group_lines(flat, [(0, 20000, "A")])

    transcribe_file = orig
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        die("only --selfcheck is implemented as a CLI entry point")
