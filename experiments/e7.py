"""E7 — claim dedup on the user's own videos (optional, Tier 3).

Inputs in OWN_DIR (default DATA/own_videos): media files (run through local
Stage 1, diarize_demo.run_pipeline_wx) or *.lines.json transcripts
([[start, end, speaker, text, conf], ...]). Claims come from the repo's
claim_detect.detect_claims; dedup is retrieve_evidence._dedupe, the exact
normalisation the pipeline uses before search. One basic Tavily search per
unique claim, so calls saved = claims - unique claims.
"""

import asyncio
import json
import os
from pathlib import Path

from experiments.common import DATA, Exp, log, secret

OWN_DIR = Path(os.environ.get("OWN_VIDEOS_DIR", DATA / "own_videos"))
MEDIA = {".mp4", ".mkv", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".flac"}


def main():
    exp = Exp("E7", {"own_dir": str(OWN_DIR), "dedup": "retrieve_evidence.normalize",
                     "searches_per_unique_claim": 1})
    items = sorted(p for p in OWN_DIR.glob("*") if p.suffix.lower() in MEDIA
                   or p.name.endswith(".lines.json")) if OWN_DIR.exists() else []
    if not items:
        return exp.skip(f"no media or *.lines.json in {OWN_DIR}")
    key = secret("GROQ_API_KEY")
    if not key:
        return exp.skip("GROQ_API_KEY not set: claim detection needs the Groq API")
    os.environ["GROQ_API_KEY"] = key
    import claim_detect as cd
    import diarize_demo as dd
    import retrieve_evidence as re_
    exp.dataset("own videos", "user-provided", "all", len(items), "user's own", str(OWN_DIR))
    done = exp.done()
    for p in items:
        if p.name in done:
            continue
        try:
            if p.name.endswith(".lines.json"):
                lines = [tuple(x) for x in json.loads(p.read_text())]
            else:
                lines, _ = asyncio.run(dd.run_pipeline_wx(str(p)))
            claims = asyncio.run(cd.detect_claims(lines))
            texts, backmap = re_._dedupe(claims)
            dups = [{"normalized": k, "variants": sorted({c["claim_text"] for c in claims
                                                          if re_.normalize(c["claim_text"]) == k}),
                     "line_ids": v} for k, v in backmap.items() if len(v) > 1]
            exp.add(p.name, {"n_lines": len(lines), "n_claims": len(claims),
                             "n_unique": len(texts), "searches_saved": len(claims) - len(texts),
                             "duplicates": dups})
            log(f"E7 {p.name}: {len(claims)} claims -> {len(texts)} unique")
        except Exception as e:
            exp.fail(p.name, repr(e))
    recs = list(exp.done().values())
    n, u = sum(r["n_claims"] for r in recs), sum(r["n_unique"] for r in recs)
    errs = [{"video": r["key"], **d} for r in recs for d in r["duplicates"]]
    exp.finish({"n_videos": len(recs), "claims": n, "unique": u, "searches_saved": n - u,
                "saved_fraction": (n - u) / n if n else None},
               sorted(errs, key=lambda e: -len(e["line_ids"])))


if __name__ == "__main__":
    main()
