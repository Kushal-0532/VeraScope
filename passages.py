"""Best passage + tier-1 stance per source (Phase 31, Prior Decision 5).

select(claim_text, sources) scores every chunk of every source in ONE scorer call,
keeps each source's most decisive chunk (max |s - 0.5|, tie -> higher s) and labels it
agree (s >= 0.85) / disagree (s <= 0.15) / unsure. A source that can't be scored is
`failed`, never disagree.
"""

import sys

AGREE_T = 0.85
DISAGREE_T = 0.15


def tier1(s):
    return "agree" if s >= AGREE_T else "disagree" if s <= DISAGREE_T else "unsure"


def _best(scores):
    return max(range(len(scores)), key=lambda i: (abs(scores[i] - 0.5), scores[i]))


def select(claim_text, sources, scorer=None):
    if scorer is None:
        from verify_claims import score_pairs as scorer
    out = [dict(s) for s in sources]
    flat = [(si, c) for si, s in enumerate(out) for c in (s.get("chunks") or []) if c and c.strip()]
    for s in out:
        if not any(c and c.strip() for c in (s.get("chunks") or [])):
            s["failed"] = "no text"
    if not flat:
        return out
    try:
        scores = list(scorer(claim_text, [c for _, c in flat]))
        if len(scores) != len(flat):
            raise ValueError(f"scorer returned {len(scores)} for {len(flat)}")
    except Exception as e:  # Constitution 4: a failure is never a disagreement
        for s in out:
            if "failed" not in s:
                s["failed"] = f"nli error: {type(e).__name__}"
        return out
    per = {}
    for (si, c), sc in zip(flat, scores):
        per.setdefault(si, []).append((c, sc))
    for si, pairs in per.items():
        b = _best([sc for _, sc in pairs])
        s = out[si]
        s["best_chunk"], s["nli_score"] = pairs[b]
        s["tier1"] = tier1(s["nli_score"])
    return out


def _selfcheck():
    src = [{"source_id": 1, "chunks": ["a", "b", "c"]}, {"source_id": 2, "chunks": []}]
    calls = []
    out = select("x", src, lambda c, ch: calls.append(ch) or [0.71, 0.97, 0.40])
    assert len(calls) == 1 and out[0]["best_chunk"] == "b" and out[0]["tier1"] == "agree"
    assert out[1]["failed"] == "no text" and "tier1" not in out[1]
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
