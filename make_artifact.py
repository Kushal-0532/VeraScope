#!/usr/bin/env python3
"""Demo fallback: run Stage 1 + Stage 2 once, dump everything app.py needs to
render without any model or API. Load with VERASCOPE_ARTIFACT=<json>.

  python make_artifact.py test.mp3 [out.json]
  python make_artifact.py old_artifact.json AUDIO [out.json]   # reuse Stage 1 lines, redo Stage 2
"""
import asyncio
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

import verify_pipeline as vp


def build(src, lines, failures, t1):
    t = time.perf_counter()
    claims, n_lines = asyncio.run(vp.detect_only(lines))
    verdicts, stats = asyncio.run(vp.verify_claims_list_v2(claims, n_lines))
    return {"schema_version": 2, "source": os.path.basename(src),
            "digest": hashlib.sha256(open(src, "rb").read()).hexdigest(),  # same key app.py computes
            "lines": lines, "failures": failures, "claims": claims, "n_lines": n_lines,
            "verdicts": verdicts, "stats": stats,
            "timing": {"stage1_s": round(t1, 1), "stage2_s": round(time.perf_counter() - t, 1)}}


def main(argv):
    if argv[0].endswith(".json"):   # Stage 1 lines from an earlier artifact (any version)
        old, src = argv[0], argv[1]
        out = argv[2] if len(argv) > 2 else old
        a = json.load(open(old))
        lines, failures, t1 = [tuple(l) for l in a["lines"]], a.get("failures", []), 0.0
    else:
        import audio_prep
        import diarize_demo as dd
        src = argv[0]
        out = argv[1] if len(argv) > 1 else os.path.splitext(src)[0] + "_artifact.json"
        flac = src + ".flac"
        audio_prep.to_flac(src, flac)
        t = time.perf_counter()
        lines, failures = asyncio.run(dd.run_pipeline_wx(flac))
        t1 = time.perf_counter() - t
    a = build(src, lines, failures, t1)
    json.dump(a, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(lines)} lines, {len(a['claims'])} claims, "
          f"{ {k: n for k, n in a['stats']['label_counts'].items() if n} }")


if __name__ == "__main__":
    main(sys.argv[1:])
