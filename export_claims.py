#!/usr/bin/env python3
"""Wk1 eval: run recordings through Stage 1 + Stage 2, dump every claim to CSV.

  python export_claims.py rec1.wav rec2.mp3 rec3_artifact.json [-o claims.csv] [--no-diarize] [--local-nli]

Audio runs the hosted pipeline (Groq + pyannoteAI, no GPU). A *_artifact.json
from make_artifact.py is reused as-is (no re-spend of Tavily/HF credits).
One row per (claim, line) occurrence.
"""
import asyncio
import csv
import json
import os
import re
import sys
import threading

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

NO_DIARIZE = False
COLS = ["recording", "line", "timestamp", "speaker", "claim",
        "evidence snippet", "evidence url", "model verdict", "confidence"]


def _best_evidence(ev):
    # ponytail: deciding snippet = strongest NLI signal (|score-0.5| max)
    return max(ev, key=lambda e: abs(e["score"] - 0.5)) if ev else None


def rows_for(name, lines, claims, verdicts):
    from retrieve_evidence import normalize
    from diarize_demo import ts
    for c in claims:
        lid = c["line_id"]
        recs = verdicts.get(lid, verdicts.get(str(lid), []))  # json keys are str
        rec = next((r for r in recs if normalize(r["claim_text"]) == normalize(c["claim_text"])), None)
        best = _best_evidence(rec["evidence_used"]) if rec else None
        yield {
            "recording": name,
            "line": lid,
            "timestamp": ts(c.get("start") or lines[lid][0]),
            "speaker": c.get("speaker", ""),
            "claim": c["claim_text"],
            "evidence snippet": best["snippet"] if best else ("" if rec else "(retrieval failed)"),
            "evidence url": best["url"] if best else "",
            "model verdict": rec["verdict"] if rec else "unclear",
            "confidence": f"{rec['confidence']:.2f}" if rec else "",
        }


def split_sentences(lines):
    """Without diarization, lines only break on >1.5 s pauses (one ran 400 s).
    Re-cut at sentence ends; timestamps interpolated by character offset."""
    out = []
    for start, end, spk, text, *rest in lines:
        parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p]
        pos, span = 0, (end - start) / max(len(text), 1)
        for p in parts:
            out.append((start + pos * span, start + (pos + len(p)) * span, spk, p, *rest))
            pos += len(p) + 1
    return out


def use_local_nli():
    """Same model as the hosted call, run on this machine (HF credits out).
    Single-label zero-shot = softmax([contra, entail])[entail], as hosted;
    see experiments/nli.py."""
    from transformers import pipeline
    import verify_claims as vc
    pipe, lock = pipeline("zero-shot-classification", model=vc.NLI_MODEL, device=-1), threading.Lock()

    def call(claim, snip):
        with lock:  # ponytail: one CPU model, serialize; per-thread pipes if this is the bottleneck
            return pipe(snip, candidate_labels=[claim], hypothesis_template="{}")["scores"][0]
    vc._call_model = call


def _patient(mod, name, err_names):
    """Wrap mod.<name> (an async API call) to retry rate-limit errors with backoff."""
    orig = getattr(mod, name)

    async def patient(*a, **k):
        for i in range(7):
            try:
                return await orig(*a, **k)
            except Exception as e:
                if type(e).__name__ not in err_names or i == 6:
                    raise
                await asyncio.sleep(2 ** i * 5)  # 5 s .. 160 s
    setattr(mod, name, patient)


def rate_limit_backoff():
    """Free-tier limits: Tavily dev key ~100 RPM, Groq gpt-oss-20b 8000 TPM.
    A 48-min recording trips both, so retry and run detect batches one at a time."""
    import functools
    import claim_detect
    import retrieve_evidence
    import verify_pipeline as vp
    _patient(retrieve_evidence, "_search_one", {"UsageLimitExceededError"})
    _patient(claim_detect, "_detect_batch", {"RateLimitError"})
    vp.detect_claims = functools.partial(claim_detect.detect_claims, concurrency=1)


def run_one(path):
    cache = path if path.endswith(".json") else os.path.splitext(path)[0] + "_artifact.json"
    if path.endswith(".json"):
        a = json.load(open(path))
        if "claims" in a:
            return a["lines"], a["claims"], a["verdicts"]
        lines = a["lines"]
    else:
        import audio_prep
        import diarize_demo as dd
        if NO_DIARIZE:  # ponytail: one speaker for whole file; real diarization when pyannoteAI has credits
            dd.diarize = lambda *a, **k: [(0.0, 1e9, "?")]
        flac = path + ".flac"
        audio_prep.to_flac(path, flac)
        lines, failures = asyncio.run(dd.run_pipeline(
            flac, os.environ.get("PYANNOTEAI_API_KEY"), score_confidence=False))  # ponytail: acoustic conf not in CSV, skips librosa
        if failures:
            print(f"  warn: {len(failures)} chunk(s) failed to transcribe", file=sys.stderr)
        json.dump({"source": os.path.basename(path), "lines": lines}, open(cache, "w"), indent=1)
    import verify_pipeline as vp
    if path.endswith(".json") and "detected" in a:  # detect already succeeded on a previous run
        lines, claims = a["lines"], a["detected"]
    else:
        if NO_DIARIZE:
            lines = split_sentences(lines)
        import logging
        detect_fail = []  # claim_detect swallows batch errors (logs a warning); count them so a 429 can't pass as "0 claims"
        h = logging.Handler(logging.WARNING)
        h.emit = lambda r: detect_fail.append(r) if "failed" in r.getMessage() else None
        logging.getLogger("claim_detect").addHandler(h)
        claims, _ = asyncio.run(vp.detect_only(lines))
        logging.getLogger("claim_detect").removeHandler(h)
        if detect_fail:
            sys.exit(f"{path}: {len(detect_fail)} claim-detect batch failures, first: {detect_fail[0].getMessage()[:300]}")
        json.dump({"source": os.path.basename(path), "lines": lines, "detected": claims},
                  open(cache, "w"), indent=1)  # so a retrieve/verify failure doesn't re-spend Groq
    n = len(lines)
    verdicts, stats = asyncio.run(vp.verify_claims_list(claims, n))
    bad = stats["n_retrieve_failures"] + stats["n_verify_failures"]
    if bad:
        # don't cache a half-failed stage 2 (quota errors look like 'unclear'); claims stay cached
        sys.exit(f"{path}: {len(claims)} claims, {bad} retrieve/verify failures; see log. Re-run with the .json.")
    json.dump({"source": os.path.basename(path), "lines": lines, "claims": claims,
               "n_lines": n, "verdicts": verdicts}, open(cache, "w"), indent=1)
    return lines, claims, verdicts


def main(paths, out):
    total = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for p in paths:
            name = os.path.basename(p).replace("_artifact.json", "")
            rows = list(rows_for(name, *run_one(p)))
            w.writerows(rows)
            f.flush()
            total += len(rows)
            print(f"{name}: {len(rows)} claims (running total {total})")
    print(f"wrote {out}: {total} claims" + ("" if total >= 100 else "  <-- under 100, add a recording"))


def _selfcheck():
    lines = [(0.0, 2.0, "A", "Paris is in France.", 0.9), (2.0, 4.0, "B", "Moon is cheese.", 0.8)]
    claims = [{"claim_text": "Paris is in France", "line_id": 0, "speaker": "A", "start": 0.0},
              {"claim_text": "Moon is cheese", "line_id": 1, "speaker": "B", "start": 2.0}]
    ev = [{"url": "u1", "snippet": "weak", "score": 0.6}, {"url": "u2", "snippet": "strong", "score": 0.95}]
    verdicts = {"0": [{"claim_text": "Paris is in France", "verdict": "supported",
                       "confidence": 0.9, "evidence_used": ev}]}  # line 1: retrieval failed
    r = list(rows_for("t", lines, claims, verdicts))
    assert r[0]["evidence snippet"] == "strong" and r[0]["model verdict"] == "supported"
    assert r[1]["model verdict"] == "unclear" and r[1]["evidence snippet"] == "(retrieval failed)"
    s = split_sentences([(0.0, 10.0, "?", "One two. Three four!", None)])
    assert [x[3] for x in s] == ["One two.", "Three four!"] and s[0][0] == 0.0 and s[1][1] == 10.0
    print("selfcheck ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selfcheck"]:
        _selfcheck()
        sys.exit()
    NO_DIARIZE = "--no-diarize" in args
    if NO_DIARIZE:
        args.remove("--no-diarize")
    if "--local-nli" in args:
        args.remove("--local-nli")
        use_local_nli()
    out = "claims.csv"
    if "-o" in args:
        i = args.index("-o")
        out = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    rate_limit_backoff()
    main(args, out)
