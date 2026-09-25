#!/usr/bin/env python3
"""Phase 21 demo fallback: run Stage 1 + Stage 2 once, dump everything app.py
needs to render without any model or API. Load with VERASCOPE_ARTIFACT=<json>.

  python make_artifact.py test.mp3 [out.json]
"""
import asyncio
import hashlib
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

import audio_prep
import diarize_demo as dd
import verify_pipeline as vp


def main(src, out):
    raw = open(src, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()   # same key app.py computes
    flac = src + ".flac"
    audio_prep.to_flac(src, flac)
    t = time.perf_counter()
    lines, failures = asyncio.run(dd.run_pipeline_wx(flac))
    t1 = time.perf_counter() - t
    claims, n_lines = asyncio.run(vp.detect_only(lines))
    verdicts, stats = asyncio.run(vp.verify_claims_list(claims, n_lines))
    t2 = time.perf_counter() - t - t1
    json.dump({"source": os.path.basename(src), "digest": digest, "lines": lines,
               "failures": failures, "claims": claims, "n_lines": n_lines,
               "verdicts": verdicts, "stats": stats,
               "timing": {"stage1_s": round(t1, 1), "stage2_s": round(t2, 1)}},
              open(out, "w"), indent=1)
    print(f"wrote {out}: {len(lines)} lines, {len(claims)} claims, "
          f"{stats['verdict_counts']}, stage1 {t1:.0f}s stage2 {t2:.0f}s")


if __name__ == "__main__":
    src = sys.argv[1]
    main(src, sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_artifact.json")
