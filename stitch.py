#!/usr/bin/env python3
"""Merge per-chunk word lists into one continuous timeline, deduping the
overlap seams. Pure function: list in, list out. No I/O, no network.
"""

import logging
import re
import sys
from difflib import SequenceMatcher

from audio_prep import OVERLAP_S

MIN_MATCH = 3

log = logging.getLogger("stitch")


def _norm(text):
    return re.sub(r"^\W+|\W+$", "", text.lower())


def stitch(chunk_word_lists, overlap_s=OVERLAP_S):
    """chunk_word_lists: [[(start, end, text), ...], ...] one list per chunk,
    already in global time (chunk N+1's timestamps overlap the tail of chunk N
    by overlap_s seconds). Returns one flat [(start, end, text)] list.
    """
    chunks = [list(c) for c in chunk_word_lists if c is not None]
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks[0]

    result = list(chunks[0])
    for idx in range(1, len(chunks)):
        prev = result
        cur = chunks[idx]
        if not prev or not cur:
            result = result + cur
            continue

        # T = global start offset of chunk `cur`. Chunk-local time 0 maps to
        # global T, so cur's earliest word timestamp is exactly T.
        T = min(s for s, _e, _t in cur)

        tail = [(s, e, t) for s, e, t in prev if s >= T]
        head = [(s, e, t) for s, e, t in cur if s <= T + overlap_s]

        tail_norm = [_norm(t) for _, _, t in tail]
        head_norm = [_norm(t) for _, _, t in head]

        match_len = 0
        match_a = match_b = 0
        if tail_norm and head_norm:
            sm = SequenceMatcher(None, tail_norm, head_norm, autojunk=False)
            m = sm.find_longest_match(0, len(tail_norm), 0, len(head_norm))
            match_len = m.size
            match_a, match_b = m.a, m.b

        if match_len >= MIN_MATCH:
            # keep prev up to (not including) the matched tail region
            keep_prev = [w for w in prev if w[0] < tail[match_a][0]] if tail else prev
            # keep cur from the matched head region onward (cur's copy wins)
            keep_cur = [w for w in cur if w[0] >= head[match_b][0]] if head else cur
            result = keep_prev + keep_cur
        else:
            midpoint = T + overlap_s / 2
            log.info("hard cut at %.2fs (no match >= %d tokens found in overlap)",
                      midpoint, MIN_MATCH)
            keep_prev = [w for w in prev if w[0] < midpoint]
            # a word straddling the cut (e.g. a wide gap-marker tuple) belongs
            # to cur's timeline even if it started before the midpoint
            keep_cur = [w for w in cur if w[0] >= midpoint or w[1] > midpoint]
            result = keep_prev + keep_cur

    # A seam match can land on an earlier occurrence of a repeated phrase
    # (dense phrase repetition inside one overlap window), leaving keep_prev's
    # tail chronologically later than keep_cur's head. Same symptom class as
    # the Whisper timestamp jitter fixed in transcribe.py; same fix: a stable
    # sort by start time restores the fusion contract's time-ordering
    # requirement instead of hard-crashing the whole pipeline on one seam.
    result.sort(key=lambda w: w[0])
    return result


def _selfcheck():
    def W(spec):
        return [(s, s + 0.4, t) for s, t in spec]

    a = W([(586.0, "the"), (587.0, "quick"), (588.0, "brown"), (589.0, "fox"), (590.0, "jumps")])
    b = W([(590.0, "the"), (591.0, "quick"), (592.0, "brown"), (593.0, "fox"), (594.0, "jumps"), (595.0, "again")])
    out = stitch([W([(0.0, "hello")]) + a, b], overlap_s=10)
    words = [t for _, _, t in out]
    assert words.count("quick") == 1, f"duplicate survived: {words}"
    assert out == sorted(out, key=lambda w: w[0]), "not monotonic"

    c = W([(590.0, "completely"), (591.0, "different"), (592.0, "words")])
    out2 = stitch([a, c], overlap_s=10)
    assert [t for _, _, t in out2], "hard-cut path produced nothing"
    starts2 = [w[0] for w in out2]
    assert starts2 == sorted(starts2), "hard cut not monotonic"

    assert stitch([]) == []
    one = W([(0.0, "solo")])
    assert stitch([one]) == one, "single-chunk path altered output"

    # two-token match is below MIN_MATCH -> hard cut, not a false alignment
    d = W([(586.0, "xx"), (587.0, "yy"), (588.0, "zz"), (589.0, "fox"), (590.0, "jumps")])
    e = W([(590.0, "fox"), (591.0, "jumps"), (592.0, "qq"), (593.0, "rr")])
    out3 = stitch([d, e], overlap_s=10)
    starts3 = [w[0] for w in out3]
    assert starts3 == sorted(starts3)

    gap = [(600.0, 1200.0, "[transcription unavailable 10:00–20:00]")]
    out4 = stitch([a, gap])
    assert any("unavailable" in t for _, _, t in out4)

    # 3+ chunk chain: content from chunks before the second-to-last must survive.
    # Regression check for a bug where each fold recomputed from the raw
    # previous chunk instead of the running result, silently dropping
    # everything before chunks[-2].
    f = W([(0.0, "alpha"), (1.0, "bravo"), (2.0, "charlie"), (3.0, "delta"), (4.0, "echo")])
    g = W([(2.0, "charlie"), (3.0, "delta"), (4.0, "echo"), (5.0, "foxtrot"), (6.0, "golf")])
    h = W([(4.0, "echo"), (5.0, "foxtrot"), (6.0, "golf"), (7.0, "hotel"), (8.0, "india")])
    out5 = stitch([f, g, h], overlap_s=3)
    words5 = [t for _, _, t in out5]
    assert "alpha" in words5 and "bravo" in words5, f"chunk 0 content dropped: {words5}"
    assert out5 == sorted(out5, key=lambda w: w[0]), "3-chunk chain not monotonic"

    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        print("error: only --selfcheck is implemented as a CLI entry point", file=sys.stderr)
        sys.exit(1)
