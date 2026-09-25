#!/usr/bin/env python3
"""Every blocked Stage 3 criterion, in one run, output shaped to paste into the
phase files. Run on the g4dn box after onbox.sh.

  python onbox_check.py            # audio.wav (26s) + audio2.wav (186s)
  python onbox_check.py FILE ...   # any files you like

Phases covered: 16 (versions/shapes/timings), 17 (criteria 1-2),
18 (diarization), 19 (fusion + frozen schema).
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/.env"))

import diarize_demo as dd
import wx_transcribe


def hdr(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def main(paths):
    hdr("Phase 16 — environment")
    import importlib.metadata as md
    for p in ("torch", "whisperx", "faster-whisper", "ctranslate2",
              "pyannote.audio", "transformers", "numpy"):
        try:
            print(f"  {p}=={md.version(p)}")
        except Exception as e:
            print(f"  {p}: {e}")
    import torch
    print(f"  device={wx_transcribe._device()} "
          f"cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    d = wx_transcribe._device()
    arch, ctype, batch = wx_transcribe._defaults(d)
    print(f"  defaults for {d}: arch={arch} compute_type={ctype} batch_size={batch}")

    failures = 0
    for path in paths:
        if not os.path.isfile(path):
            print(f"\n!! skipping {path}: not found")
            failures += 1
            continue

        hdr(f"Phase 17 — transcribe_aligned({path!r})")
        t = time.perf_counter()
        words = wx_transcribe.transcribe_aligned(path)
        wall = time.perf_counter() - t
        print(f"  {len(words)} words in {wall:.1f}s")
        print("  first 15 rows:")
        for w in words[:15]:
            print(f"    {w}")
        failures += not check("non-empty", bool(words))
        failures += not check("tuple shape (float,float,str,float|None)", all(
            isinstance(s, float) and isinstance(e, float) and isinstance(x, str)
            and (c is None or isinstance(c, float)) for s, e, x, c in words))
        failures += not check("time-ordered",
                              words == sorted(words, key=lambda w: w[0]))
        failures += not check("all scores in [0,1] or None", all(
            c is None or 0.0 <= c <= 1.0 for *_, c in words))

        hdr(f"Phase 18 — wx_diarize({path!r})")
        t = time.perf_counter()
        turns = dd.wx_diarize(path, 1, 5)
        wall_d = time.perf_counter() - t
        spk = sorted({s for *_, s in turns})
        print(f"  {len(turns)} turns, {len(spk)} speakers {spk} in {wall_d:.1f}s")
        for tn in turns[:10]:
            print(f"    {tn}")
        failures += not check("non-empty", bool(turns))
        failures += not check("tuple shape (float,float,str)", all(
            isinstance(a, float) and isinstance(b, float) and isinstance(c, str)
            for a, b, c in turns))
        failures += not check("time-ordered",
                              turns == sorted(turns, key=lambda t: t[0]))
        failures += not check("speaker count is small (<=5)", len(spk) <= 5,
                              f"{len(spk)} distinct")
        failures += not check("drops into assign_speaker unmodified",
                              dd.assign_speaker(turns[0][0], turns[0][1], turns) is not None)

        hdr(f"Phase 19 — run_pipeline_wx({path!r})")
        t = time.perf_counter()
        lines, fails = asyncio.run(dd.run_pipeline_wx(path))
        wall_p = time.perf_counter() - t
        names = sorted({l[2] for l in lines})
        print(f"  {len(lines)} lines, {len(names)} speakers, {wall_p:.1f}s total")
        for l in lines[:10]:
            print(f"    [{dd.ts(l[0])}-{dd.ts(l[1])}] {l[2]}: {l[3][:60]!r} conf={l[4]}")
        failures += not check("failures list empty", fails == [])
        failures += not check("frozen 5-tuple schema", all(
            isinstance(s, float) and isinstance(e, float)
            and (k is None or isinstance(k, str)) and isinstance(x, str)
            and (c is None or 0.0 <= c <= 1.0)
            for s, e, k, x, c in lines))
        failures += not check("lines are time-ordered",
                              lines == sorted(lines, key=lambda l: l[0]))
        dur = dd.audio_prep.probe_duration(path)
        cov = (lines[-1][1] - lines[0][0]) / dur if lines else 0
        failures += not check("coverage > 50% of duration", cov > 0.5,
                              f"{cov:.0%} of {dd.ts(dur)}")
        print(f"\n  TIMING  transcribe+align {wall:.1f}s | diarize {wall_d:.1f}s "
              f"| pipeline {wall_p:.1f}s | audio {dd.ts(dur)}")

    hdr("RESULT")
    print("  all checks passed" if not failures else f"  {failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["audio.wav", "audio2.wav"]
    sys.exit(main(args))
